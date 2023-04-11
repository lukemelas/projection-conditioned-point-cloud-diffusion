import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from .simple_model_utils import FeedForward, BasePointModel


class SimplePointModel(BasePointModel):
    """
    A simple model that processes a point cloud by applying a series of MLPs to each point
    individually, along with some pooled global features.
    """

    def get_layers(self):
        return nn.ModuleList([FeedForward(
            d_in=(3 * self.dim), d_hidden=(4 * self.dim), d_out=self.dim,
            activation=nn.SiLU(), is_gated=True, bias1=False, bias2=False, bias_gate=False, use_layernorm=True
        ) for _ in range(self.num_layers)])

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):

        # Prepare inputs
        x, coords = self.prepare_inputs(inputs, t)

        # Model
        for layer in self.layers:
            x_pool_max, x_pool_std = self.get_global_tensors(x)
            x_input = torch.cat((x, x_pool_max, x_pool_std), dim=-1)  # (B, N, 3 * D)
            x = x + layer(x_input)  # (B, N, D_model)

        # Project
        x = self.output_projection(x)  # (B, N, D_out)
        x = torch.transpose(x, -2, -1)  # -> (B, D_out, N)

        return x


class SimpleNearestNeighborsPointModel(BasePointModel):
    """ 
    A simple model that processes a point cloud by applying a series of MLPs to each point
    individually, along with some pooled global features, and the features of its nearest
    neighbors.
    """

    def __init__(self, num_neighbors: int = 4, **kwargs):
        self.num_neighbors = num_neighbors
        super().__init__(**kwargs)
        from pytorch3d.ops import knn_points
        self.knn_points = knn_points

    def get_layers(self):
        return nn.ModuleList([FeedForward(
            d_in=((3 + self.num_neighbors) * self.dim), d_hidden=(4 * self.dim), d_out=self.dim,
            activation=nn.SiLU(), is_gated=True, bias1=False, bias2=False, bias_gate=False, use_layernorm=True
        ) for _ in range(self.num_layers)])

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):

        # Prepare inputs
        x, coords = self.prepare_inputs(inputs, t)  # (B, N, D), (B, N, 3)

        # Get nearest neighbors. Note that the first neighbor is the identity, which is convenient
        _dists, indices, _neighbors = self.knn_points(
            p1=coords, p2=coords, K=(self.num_neighbors + 1),
            return_nn=False)  # (B, N, K), (B, N, K)
        (B, N, D), (_B, _N, K) = x.shape, indices.shape

        # Model
        for layer in self.layers:
            x_neighbor = torch.stack([x_i[idx] for x_i, idx in zip(x, indices.reshape(B, N * K))]).reshape(B, N, K * D)
            x_pool_max, x_pool_std = self.get_global_tensors(x)
            x_input = torch.cat((x_neighbor, x_pool_max, x_pool_std), dim=-1)  # (B, N, (3+K)*D)
            x = x + layer(x_input)  # (B, N, D_model)

        # Project
        x = self.output_projection(x)  # (B, N, D_out)
        x = torch.transpose(x, -2, -1)  # -> (B, D_out, N)

        return x
