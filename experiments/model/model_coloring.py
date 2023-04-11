from typing import Optional

import torch
import torch.nn.functional as F
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor

from .point_cloud_transformer_model import PointCloudTransformerModel
from .projection_model import PointCloudProjectionModel

class PointCloudColoringModel(PointCloudProjectionModel):
    
    def __init__(
        self,
        point_cloud_model: str,
        point_cloud_model_layers: int,
        point_cloud_model_embed_dim: int,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)
        
        # Checks
        if self.predict_shape or not self.predict_color:
            raise NotImplementedError('Must predict color, not shape, for coloring')

        # Create point cloud model for processing point cloud
        self.point_cloud_model = PointCloudTransformerModel(
            num_layers=point_cloud_model_layers,
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

    def _forward(
        self, 
        pc: Pointclouds,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_point_cloud: bool = False,
        noise_std: float = 0.0,
    ):

        # Normalize colors and convert to tensor
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=True)
        x_points, x_colors = x[:, :, :3], x[:, :, 3:]

        # Add noise to points. TODO: Add to config.
        x_input = x_points + torch.randn_like(x_points) * noise_std

        # Conditioning
        x_input = self.get_input_with_conditioning(x_input, camera=camera, 
            image_rgb=image_rgb, mask=mask)

        # Forward
        pred_colors = self.point_cloud_model(x_input)

        # During inference, we return the point cloud with the predicted colors
        if return_point_cloud:
            pred_pointcloud = self.tensor_to_point_cloud(
                torch.cat((x_points, pred_colors), dim=2), denormalize=True, unscale=True)
            return pred_pointcloud

        # During training, we have ground truth colors and return the loss
        loss = F.mse_loss(pred_colors, x_colors)
        return loss

    def forward(self, batch: FrameData, **kwargs):
        """A wrapper around the forward method"""
        if isinstance(batch, dict):  # fixes a bug with multiprocessing where batch becomes a dict
            batch = FrameData(**batch)  # it really makes no sense, I do not understand it
        return self._forward(
            pc=batch.sequence_point_cloud, 
            camera=batch.camera,
            image_rgb=batch.image_rgb, 
            mask=batch.fg_probability,
            **kwargs,
        )