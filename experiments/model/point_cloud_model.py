from contextlib import nullcontext

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from torch import Tensor

from .pvcnn.pvcnn import PVCNN2
from .pvcnn.pvcnn_plus_plus import PVCNN2PlusPlus
from .simple.simple_model import SimplePointModel


class PointCloudModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        model_type: str = 'pvcnn',
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        dropout: float = 0.1,
        width_multiplier: int = 1,
        voxel_resolution_multiplier: int = 1,
    ):
        super().__init__()
        self.model_type = model_type
        if self.model_type == 'pvcnn':
            self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            self.model = PVCNN2(
                embed_dim=embed_dim,
                num_classes=out_channels,
                extra_feature_channels=(in_channels - 3),
                dropout=dropout, width_multiplier=width_multiplier, 
                voxel_resolution_multiplier=voxel_resolution_multiplier
            )
            self.model.classifier[-1].bias.data.normal_(0, 1e-6)
            self.model.classifier[-1].weight.data.normal_(0, 1e-6)
        elif self.model_type == 'pvcnnplusplus':
            self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            self.model = PVCNN2PlusPlus(
                embed_dim=embed_dim,
                num_classes=out_channels,
                extra_feature_channels=(in_channels - 3),
            )
            self.model.output_projection[-1].bias.data.normal_(0, 1e-6)
            self.model.output_projection[-1].weight.data.normal_(0, 1e-6)
        elif self.model_type == 'simple':
            self.autocast_context = nullcontext()
            self.model = SimplePointModel(
                embed_dim=embed_dim,
                num_classes=out_channels,
                extra_feature_channels=(in_channels - 3),
            )
            self.model.output_projection.bias.data.normal_(0, 1e-6)
            self.model.output_projection.weight.data.normal_(0, 1e-6)
        else:
            raise NotImplementedError()

    def forward(self, inputs: Tensor, t: Tensor) -> Tensor:
        """ Receives input of shape (B, N, in_channels) and returns output
            of shape (B, N, out_channels) """
        with self.autocast_context:
            return self.model(inputs.transpose(1, 2), t).transpose(1, 2)
