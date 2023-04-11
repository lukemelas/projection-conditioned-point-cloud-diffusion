from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pytorch3d
import torch
from torch.utils.data import SequentialSampler
from omegaconf import DictConfig
from pytorch3d.implicitron.dataset.data_loader_map_provider import \
    SequenceDataLoaderMapProvider
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2, registry)
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.renderer.cameras import CamerasBase

from config.structured import CO3DConfig, DataloaderConfig, ProjectConfig
from .exclude_sequence import EXCLUDE_SEQUENCE, LOW_QUALITY_SEQUENCE
from .utils import DatasetMap


def get_dataset(cfg: ProjectConfig):
    
    if cfg.dataset.type == 'co3dv2':
        dataset_cfg: CO3DConfig = cfg.dataset
        dataloader_cfg: DataloaderConfig = cfg.dataloader

        # Exclude bad and low-quality sequences
        exclude_sequence = []
        exclude_sequence.extend(EXCLUDE_SEQUENCE.get(dataset_cfg.category, []))
        exclude_sequence.extend(LOW_QUALITY_SEQUENCE.get(dataset_cfg.category, []))
        
        # Whether to load pointclouds
        kwargs = dict(
            remove_empty_masks=True,
            n_frames_per_sequence=1,
            load_point_clouds=True,
            max_points=dataset_cfg.max_points,
            image_height=dataset_cfg.image_size,
            image_width=dataset_cfg.image_size,
            mask_images=dataset_cfg.mask_images,
            exclude_sequence=exclude_sequence,
            pick_sequence=() if dataset_cfg.restrict_model_ids is None else dataset_cfg.restrict_model_ids,
        )

        # Get dataset mapper
        dataset_map_provider_type = registry.get(JsonIndexDatasetMapProviderV2, "JsonIndexDatasetMapProviderV2")
        expand_args_fields(dataset_map_provider_type)
        dataset_map_provider = dataset_map_provider_type(
            category=dataset_cfg.category,
            subset_name=dataset_cfg.subset_name,
            dataset_root=dataset_cfg.root,
            test_on_train=False,
            only_test_set=False,
            load_eval_batches=True,
            dataset_JsonIndexDataset_args=DictConfig(kwargs),
        )

        # Get datasets
        datasets = dataset_map_provider.get_dataset_map()

        # PATCH BUG WITH POINT CLOUD LOCATIONS!
        for dataset in (datasets["train"], datasets["val"]):
            for key, ann in dataset.seq_annots.items():
                correct_point_cloud_path = Path(dataset.dataset_root) / Path(*Path(ann.point_cloud.path).parts[-3:])
                assert correct_point_cloud_path.is_file(), correct_point_cloud_path
                ann.point_cloud.path = str(correct_point_cloud_path)

        # Get dataloader mapper
        data_loader_map_provider_type = registry.get(SequenceDataLoaderMapProvider, "SequenceDataLoaderMapProvider")
        expand_args_fields(data_loader_map_provider_type)
        data_loader_map_provider = data_loader_map_provider_type(
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
        )

        # QUICK HACK: Patch the train dataset because it is not used but it throws an error
        if (len(datasets['train']) == 0 and len(datasets[dataset_cfg.eval_split]) > 0 and 
                dataset_cfg.restrict_model_ids is not None and cfg.run.job == 'sample'):
            datasets = DatasetMap(train=datasets[dataset_cfg.eval_split], val=datasets[dataset_cfg.eval_split], 
                                  test=datasets[dataset_cfg.eval_split])
            print('Note: You used restrict_model_ids and there were no ids in the train set.')

        # Get dataloaders
        dataloaders = data_loader_map_provider.get_data_loader_map(datasets)
        dataloader_train = dataloaders['train']
        dataloader_val = dataloader_vis = dataloaders[dataset_cfg.eval_split]

        # Replace validation dataloader sampler with SequentialSampler
        dataloader_val.batch_sampler.sampler = SequentialSampler(dataloader_val.batch_sampler.sampler.data_source)

        # Modify for accelerate
        dataloader_train.batch_sampler.drop_last = True
        dataloader_val.batch_sampler.drop_last = False

    else:
        raise NotImplementedError(cfg.dataset.type)
    
    return dataloader_train, dataloader_val, dataloader_vis
