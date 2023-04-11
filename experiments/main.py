import datetime
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Optional

import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import functional as TVF

import training_utils
import diffusion_utils
from dataset import get_dataset
from model import get_model, ConditionalPointCloudDiffusionModel
from config.structured import ProjectConfig

try:
    import lovely_tensors
    lovely_tensors.monkey_patch()
except ImportError:
    pass  # lovely tensors is not necessary but it really is lovely, I do recommend it


torch.multiprocessing.set_sharing_strategy('file_system')


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
    model = get_model(cfg)
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

    # import pdb; pdb.set_trace()

    # Type hints
    model: ConditionalPointCloudDiffusionModel
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
    if cfg.run.job == 'sample':
        # Whether or not to use EMA parameters for sampling
        if cfg.run.sample_from_ema:
            assert model_ema is not None
            model_ema.to(accelerator.device)
            sample_context = model_ema.average_parameters
        else:
            sample_context = nullcontext
        # Sample
        with sample_context():
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
                loss = model(batch, mode='train')

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
                    model_ema.update(model.parameters())

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
    (output_dir / 'evolutions').mkdir(exist_ok=True, parents=True)
    (output_dir / 'metadata').mkdir(exist_ok=True, parents=True)

    # Visualize
    wandb_log_dict = {}
    for batch_idx, batch in enumerate(progress_bar):
        if num_batches is not None and batch_idx >= num_batches:
            break

        # Sample
        output, all_outputs = model(batch, mode='sample', return_sample_every_n_steps=100, 
            num_inference_steps=cfg.run.num_inference_steps, disable_tqdm=(not accelerator.is_main_process))
        output: Pointclouds
        all_outputs: List[Pointclouds]  # list of B Pointclouds, each with a batch size of return_sample_every_n_steps

        # Filenames
        filestr = str(output_dir / '{dir}' / f'p-{accelerator.process_index}-b-{batch_idx}-s-{{i:02d}}-{{name}}-{identifier}.{{ext}}')
        filestr_wandb = f'{{dir}}/b-{batch_idx}-{{name}}-s-{{i:02d}}-{{name}}'

        # Save raw samples
        filename = filestr.format(dir='raw', name='raw', i=0, ext='pth')
        torch.save({'output': output, 'all_outputs': all_outputs, 'batch': batch}, filename)

        # Save metadata
        metadata = diffusion_utils.get_metadata(batch)
        filename = filestr.format(dir='metadata', name='metadata', i=0, ext='txt')
        Path(filename).write_text(metadata)

        # Save individual samples
        for i in range(len(output)):
            camera = batch.camera[i]
            gt_pointcloud = batch.sequence_point_cloud[i]
            pred_pointcloud = output[i]
            pred_all_pointclouds = all_outputs[i]

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

            # Render point cloud diffusion evolution
            filename_evo = filestr.format(dir='evolutions', name='evolutions', i=i, ext='mp4')
            filename_evo_wandb = filestr.format(dir='evolutions', name='evolutions', i=i, ext='mp4')
            diffusion_utils.visualize_pointcloud_evolution_pytorch3d(
                pointclouds=pred_all_pointclouds, output_file_video=filename_evo, camera=camera)
            wandb_log_dict[filename_evo_wandb] = wandb.Video(filename_evo)

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
):
    from pytorch3d.io import IO
    from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
    from pytorch3d.structures import Pointclouds
    from tqdm import tqdm

    # Eval mode
    model.eval()
    progress_bar: Iterable[FrameData] = tqdm(dataloader, disable=(not accelerator.is_main_process))

    # Output dir
    output_dir: Path = Path(output_dir)

    # PyTorch3D IO
    io = IO()

    # Visualize
    for batch_idx, batch in enumerate(progress_bar):
        progress_bar.set_description(f'Processing batch {batch_idx:4d} / {len(dataloader):4d}')
        if cfg.run.num_sample_batches is not None and batch_idx >= cfg.run.num_sample_batches:
            break

        # Optionally produce multiple samples for each point cloud
        for sample_idx in range(cfg.run.num_samples):

            # Filestring
            filename = f'{{name}}-{sample_idx}.{{ext}}' if cfg.run.num_samples > 1 else '{name}.{ext}'
            filestr = str(output_dir / '{dir}' / '{category}' / filename)

            # Sample
            output, all_outputs = model(batch, mode='sample', return_sample_every_n_steps=10, scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps, disable_tqdm=(not accelerator.is_main_process))
            output: Pointclouds
            all_outputs: List[Pointclouds]  # list of B Pointclouds, each with a batch size of return_sample_every_n_steps

            # Save individual samples
            for i in range(len(output)):
                sequence_name = batch.sequence_name[i]
                sequence_category = batch.sequence_category[i]
                (output_dir / 'gt' / sequence_category).mkdir(exist_ok=True, parents=True)
                (output_dir / 'pred' / sequence_category).mkdir(exist_ok=True, parents=True)
                (output_dir / 'images' / sequence_category).mkdir(exist_ok=True, parents=True)
                (output_dir / 'metadata' / sequence_category).mkdir(exist_ok=True, parents=True)
                (output_dir / 'evolutions' / sequence_category).mkdir(exist_ok=True, parents=True)
                
                # Save ground truth
                io.save_pointcloud(data=batch.sequence_point_cloud[i], path=filestr.format(dir='gt', 
                    category=sequence_category, name=sequence_name, ext='ply'))
                
                # Save generation
                io.save_pointcloud(data=output[i], path=filestr.format(dir='pred', 
                    category=sequence_category, name=sequence_name, ext='ply'))

                # Save input images
                filename = filestr.format(dir='images', category=sequence_category, name=sequence_name, ext='png')
                TVF.to_pil_image(batch.image_rgb[i]).save(filename)

                # Save camera
                filename = filestr.format(dir='metadata', category=sequence_category, name=sequence_name, ext='pth')
                metadata = dict(index=i, sequence_name=batch.sequence_name, sequence_category=batch.sequence_category,
                                frame_timestamp=batch.frame_timestamp, camera=batch.camera, image_size_hw=batch.image_size_hw,
                                image_path=batch.image_path, depth_path=batch.depth_path, mask_path=batch.mask_path,
                                bbox_xywh=batch.bbox_xywh, crop_bbox_xywh=batch.crop_bbox_xywh,
                                sequence_point_cloud_path=batch.sequence_point_cloud_path, meta=batch.meta)
                torch.save(metadata, filename)

                # Save evolutions
                if cfg.run.sample_save_evolutions:
                    torch.save(all_outputs[i], filestr.format(dir='evolutions', category=sequence_category, 
                        name=sequence_name, ext='pth'))

    print('Saved samples to: ')
    print(output_dir.absolute())


if __name__ == '__main__':
    main()