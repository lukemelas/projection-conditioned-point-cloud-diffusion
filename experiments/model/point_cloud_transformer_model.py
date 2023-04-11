from typing import Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from torch import Tensor
from timm.models.vision_transformer import Attention, LayerScale, DropPath, Mlp

from .point_cloud_model import PointCloudModel


class PointCloudModelBlock(nn.Module):

    def __init__(
        self, 
        *, 
        # Point cloud model
        dim: int,
        model_type: str = 'pvcnn',
        dropout: float = 0.1,
        width_multiplier: int = 1,
        voxel_resolution_multiplier: int = 1,
        # Transformer model
        num_heads=6, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_attn=False
    ):
        super().__init__()

        # Point cloud model
        self.norm0 = norm_layer(dim)
        self.point_cloud_model = PointCloudModel(model_type=model_type, 
            in_channels=dim, out_channels=dim, embed_dim=dim, dropout=dropout, 
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.ls0 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path0 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Attention
        self.use_attn = use_attn
        if self.use_attn:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def apply_point_cloud_model(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        t = t if t is not None else torch.zeros(len(x), device=x.device, dtype=torch.long)
        return self.point_cloud_model(x, t)

    def forward(self, x: Tensor):
        x = x + self.drop_path0(self.ls0(self.apply_point_cloud_model(self.norm0(x))))
        if self.use_attn:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PointCloudTransformerModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, num_layers: int, in_channels: int = 3, out_channels: int = 3, embed_dim: int = 64, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.input_projection = nn.Linear(in_channels, embed_dim)
        self.blocks = nn.Sequential(*[PointCloudModelBlock(dim=embed_dim, **kwargs) for i in range(self.num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        """ Receives input of shape (B, N, in_channels) and returns output
            of shape (B, N, out_channels) """
        x = self.input_projection(inputs)
        x = self.blocks(x)
        x = self.output_projection(x)
        return x
