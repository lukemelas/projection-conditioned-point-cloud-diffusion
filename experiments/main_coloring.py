import datetime
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import hydra
import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms import functional as TVF

import training_utils
import diffusion_utils
from dataset import get_dataset
from model import get_coloring_model, PointCloudColoringModel
from config.structured import ProjectConfig


@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):

    # Accelerator
    accelerator = Accelerator(mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu, 
        gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps)

    # Logging
    training_utils.setup_distributed_print(accelerator.is_main_process)
    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.init(project=cfg.logging.wandb_project, name=cfg.run.name, job_type=cfg.run.job, 
                   config=OmegaConf.to_container(cfg))
        wandb.run.log_code(root=hydra.utils.get_original_cwd(),
            include_fn=lambda p: any(p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')),
            exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode')))
        cfg: ProjectConfig = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f'Current working directory: {os.getcwd()}')

    # Set random seed
    training_utils.set_seed(cfg.run.seed)

    # Model
    model = get_coloring_model(cfg)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # Exponential moving average of model parameters
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage
        model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
        model_ema.to(accelerator.device)
        print('Initialized model EMA')
    else:
        model_ema = None
        print('Not using model EMA')

    # Optimizer and scheduler
    optimizer = training_utils.get_optimizer(cfg, model, accelerator)
    scheduler = training_utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint and create the initial training state
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)

    # Datasets
    dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)
    
    # Compute total training batch size
    total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps

    # Setup. Note that this does not currently work with CO3D.
    model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis = accelerator.prepare(
        model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis)

    # Type hints
    model: PointCloudColoringModel
    optimizer: torch.optim.Optimizer

    # Visualize before training
    if cfg.run.job == 'vis' or cfg.run.vis_before_training:
        visualize(
            cfg=cfg,
            model=model,
            dataloader_vis=dataloader_vis,
            accelerator=accelerator,
            identifier='init',
            num_batches=1, 
        )
        if cfg.run.job == 'vis':
            if cfg.logging.wandb and accelerator.is_main_process:
                wandb.finish()
                time.sleep(5)
            return

    # Sample from the model
    if 'sample' in cfg.run.job:
        sample(
            cfg=cfg,
            model=model,
            dataloader=dataloader_val,
            accelerator=accelerator,
        )
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
        time.sleep(5)
        return

    # Info
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    print(f'    Dataset train size: {len(dataloader_train.dataset):_}')
    print(f'    Dataset val size: {len(dataloader_train.dataset):_}')
    print(f'    Dataloader train size: {len(dataloader_train):_}')
    print(f'    Dataloader val size: {len(dataloader_val):_}')
    print(f'    Batch size per device = {cfg.dataloader.batch_size}')
    print(f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}')
    print(f'    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}')
    print(f'    Max training steps = {cfg.run.max_steps}')
    print(f'    Training state = {train_state}')

    # Infinitely loop training
    while True:
    
        # Train progress bar
        log_header = f'Epoch: [{train_state.epoch}]'
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
        metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        progress_bar: Iterable[Any] = metric_logger.log_every(dataloader_train, cfg.run.print_step_freq, 
            header=log_header)

        # Train
        for i, batch in enumerate(progress_bar):
            if (cfg.run.limit_train_batches is not None) and (i >= cfg.run.limit_train_batches): break
            model.train()

            # Gradient accumulation
            with accelerator.accumulate(model):

                # Forward
                loss = model(batch, noise_std=cfg.run.coloring_training_noise_std)

                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # grad_norm_unclipped = training_utils.compute_grad_norm(model.parameters())  # useless w/ mixed prec
                    if cfg.optimizer.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                    grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    train_state.step += 1

                # Exit if loss was NaN
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

            # Gradient accumulation
            if accelerator.sync_gradients:

                # Logging
                log_dict = {
                    'lr': optimizer.param_groups[0]["lr"],
                    'step': train_state.step,
                    'train_loss': loss_value,
                    # 'grad_norm_unclipped': grad_norm_unclipped,  # useless w/ mixed prec
                    'grad_norm_clipped': grad_norm_clipped,
                }
                metric_logger.update(**log_dict)
                if (cfg.logging.wandb and accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0):
                    wandb.log(log_dict, step=train_state.step)
            
                # Update EMA
                if cfg.ema.use_ema and train_state.step % cfg.ema.update_every == 0:
                    model_ema.update(model)

                # Save a checkpoint
                if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0):
                    checkpoint_dict = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': train_state.best_val,
                        'model_ema': model_ema.state_dict() if model_ema else {},
                        'cfg': cfg
                    }
                    checkpoint_path = 'checkpoint-latest.pth'
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

                # Visualize
                if (cfg.run.vis_freq > 0) and (train_state.step % cfg.run.vis_freq) == 0:
                    visualize(
                        cfg=cfg,
                        model=model,
                        dataloader_vis=dataloader_vis,
                        accelerator=accelerator,
                        identifier=f'{train_state.step}', 
                        num_batches=2,
                    )

                # End training after the desired number of steps/epochs
                if train_state.step >= cfg.run.max_steps:
                    print(f'Ending training at: {datetime.datetime.now()}')
                    print(f'Final train state: {train_state}')
                    
                    wandb.finish()
                    time.sleep(5)
                    return

        # Epoch complete, log it and continue training
        train_state.epoch += 1

        # Gather stats from all processes
        metric_logger.synchronize_between_processes(device=accelerator.device)
        print(f'{log_header}  Average stats --', metric_logger)


@torch.no_grad()
def visualize(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    dataloader_vis: Iterable,
    accelerator: Accelerator,
    identifier: str = '',
    num_batches: Optional[int] = None,
    output_dir: str = 'vis',
):
    from pytorch3d.vis.plotly_vis import plot_scene
    from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
    from pytorch3d.structures import Pointclouds

    # Eval mode
    model.eval()
    metric_logger = training_utils.MetricLogger(delimiter="  ")
    progress_bar: Iterable[FrameData] = metric_logger.log_every(dataloader_vis, cfg.run.print_step_freq, "Vis")

    # Output dir
    output_dir: Path = Path(output_dir)
    (output_dir / 'raw').mkdir(exist_ok=True, parents=True)
    (output_dir / 'pointclouds').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images').mkdir(exist_ok=True, parents=True)
    (output_dir / 'videos').mkdir(exist_ok=True, parents=True)
    (output_dir / 'metadata').mkdir(exist_ok=True, parents=True)

    # Visualize
    wandb_log_dict = {}
    for batch_idx, batch in enumerate(progress_bar):
        if num_batches is not None and batch_idx >= num_batches:
            break

        # Sample
        output: Pointclouds = model(batch, return_point_cloud=True)

        # Filenames
        filestr = str(output_dir / '{dir}' / f'p-{accelerator.process_index}-b-{batch_idx}-s-{{i:02d}}-{{name}}-{identifier}.{{ext}}')
        filestr_wandb = f'{{dir}}/b-{batch_idx}-{{name}}-s-{{i:02d}}-{{name}}'

        # # Save raw samples
        # filename = filestr.format(dir='raw', name='raw', i=0, ext='pth')
        # torch.save({'output': output, 'all_outputs': all_outputs, 'batch': batch}, filename)

        # Save metadata
        metadata = diffusion_utils.get_metadata(batch)
        filename = filestr.format(dir='metadata', name='metadata', i=0, ext='txt')
        Path(filename).write_text(metadata)

        # Save individual samples
        for i in range(len(output)):
            camera = batch.camera[i]
            gt_pointcloud = batch.sequence_point_cloud[i]
            pred_pointcloud = output[i]

            # Plot using plotly and pytorch3d
            fig = plot_scene({ 
                'Pred': {'pointcloud': pred_pointcloud},
                'GT': {'pointcloud': gt_pointcloud},
            }, ncols=2, viewpoint_cameras=camera, pointcloud_max_points=16_384)
            
            # Save plot
            filename = filestr.format(dir='pointclouds', name='pointclouds', i=i, ext='html')
            fig.write_html(filename)

            # Add to W&B
            filename_wandb = filestr_wandb.format(dir='pointclouds', name='pointclouds', i=i)
            wandb_log_dict[filename_wandb] = wandb.Html(open(filename), inject=False)

            # Save input images
            filename = filestr.format(dir='images', name='image_rgb', i=i, ext='png')
            TVF.to_pil_image(batch.image_rgb[i]).save(filename)

            # Add to W&B
            filename_wandb = filestr_wandb.format(dir='images', name='image_rgb', i=i)
            wandb_log_dict[filename_wandb] = wandb.Image(filename)

            # Loop
            for name, pointcloud in (('gt', gt_pointcloud), ('pred', pred_pointcloud)):
            
                # Render gt/pred point cloud from given view
                filename_image = filestr.format(dir='images', name=name, i=i, ext='png')
                filename_image_wandb = filestr_wandb.format(dir='images', name=name, i=i)
                diffusion_utils.visualize_pointcloud_batch_pytorch3d(pointclouds=pointcloud, 
                    output_file_image=filename_image, cameras=camera, scale_factor=cfg.model.scale_factor)
                wandb_log_dict[filename_image_wandb] = wandb.Image(filename_image)

                # Render gt/pred point cloud from rotating view
                filename_video = filestr.format(dir='videos', name=name, i=i, ext='mp4')
                filename_video_wandb = filestr_wandb.format(dir='videos', name=name, i=i)
                diffusion_utils.visualize_pointcloud_batch_pytorch3d(pointclouds=pointcloud, 
                    output_file_video=filename_video, num_frames=30, scale_factor=cfg.model.scale_factor)
                wandb_log_dict[filename_video_wandb] = wandb.Video(filename_video)

    # Save to W&B
    if cfg.logging.wandb and accelerator.is_local_main_process:
        wandb.log(wandb_log_dict, commit=False)

    print('Saved visualizations to: ')
    print(output_dir.absolute())


@torch.no_grad()
def sample(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    dataloader: Iterable,
    accelerator: Accelerator,
    output_dir: str = 'sample',
    num_batches: Optional[int] = None,
):
    from pytorch3d.io import IO
    from pytorch3d.structures import Pointclouds

    # Eval mode
    model.eval()

    # Output dir
    output_dir: Path = Path(output_dir)

    # PyTorch3D IO
    io = IO()

    # Check support
    if cfg.run.coloring_sample_dir is None:
        raise ValueError(cfg.run.coloring_sample_dir)
    sample_dir = Path(cfg.run.coloring_sample_dir)
    gt_dir = sample_dir / 'gt'
    preds_dir = sample_dir / 'pred'
    images_dir = sample_dir / 'images'
    metadata_dir = sample_dir / 'metadata'
    
    # Loop over sequence categories
    sequence_categories = sorted([p.stem for p in preds_dir.iterdir()])
    print(f'Found categories: {len(sequence_categories)}')
    pbar = tqdm(sequence_categories)
    for sequence_category in pbar:
        pbar.set_description(f'Sequence category {sequence_category}')

        # Get sequence names
        sequence_names = sorted([p.stem for p in (preds_dir / sequence_category).iterdir()])
        print(f'Found predicted point clouds: {len(sequence_names)}')

        # Visualize
        pbar_inner = tqdm(sequence_names)
        for idx, sequence_name in enumerate(pbar_inner):
            pbar_inner.set_description(f'Sequence name {sequence_name}')
            if num_batches is not None and idx >= num_batches:
                break
            
            # Files
            images_path = images_dir / sequence_category / f'{sequence_name}.png'
            pointcloud_path = preds_dir / sequence_category / f'{sequence_name}.ply'
            metadata_path = metadata_dir / sequence_category / f'{sequence_name}.pth'

            # Load
            pointcloud = io.load_pointcloud(pointcloud_path)  # (1, )
            image = TVF.to_tensor(Image.open(images_path)).unsqueeze(0)  # (1, 3, H, W)
            mask = (image.sum(dim=1, keepdim=True) > 0.0).float()  #  (1, H, W)  NOTE: perhaps I should do something better here
            metadata = torch.load(metadata_path, map_location='cpu')
            camera = metadata['camera'][metadata['index']]  # (1, )
            frame_number = 0  # metadata['frame_number'][metadata['index']]

            # Assemble batch (this needs dot access)
            batch = dict(frame_number=frame_number, sequence_name=sequence_name, sequence_category=sequence_category,
                         sequence_point_cloud=pointcloud.to(accelerator.device), camera=camera.to(accelerator.device),
                         image_rgb=image.to(accelerator.device), fg_probability=mask.to(accelerator.device))

            # Predict color
            output: Pointclouds = model(batch, return_point_cloud=True)

            # Save generation
            filepath = output_dir / 'pred_color' / sequence_category / f'{sequence_name}.ply'
            filepath.parent.mkdir(exist_ok=True, parents=True)
            io.save_pointcloud(data=output, path=str(filepath))

    print('Loaded samples without color from: ')
    print(sample_dir.absolute())
    print('Compare with ground truth from: ')
    print(gt_dir.absolute())
    print('Saved samples to: ')
    print(output_dir.absolute())


if __name__ == '__main__':
    main()