from typing import Any, Callable, Iterable, List, Optional, Union

import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor, nn
from torch.nn import LayerNorm

from model.pvcnn.pvcnn_utils import get_timestep_embedding


def sample_b(size: Size, sigma: float) -> Tensor:
    """Sample b matrix for fourier features

    Arguments:
        size (Size): b matrix size
        sigma (float): std of the gaussian

    Returns:
        b (Tensor): b matrix
    """
    return torch.randn(size) * sigma


@jit.script
def map_positional_encoding(v: Tensor, freq_bands: Tensor) -> Tensor:
    """Map v to positional encoding representation phi(v)

    Arguments:
        v (Tensor): input features (B, IFeatures)
        freq_bands (Tensor): frequency bands (N_freqs, )

    Returns:
        phi(v) (Tensor): fourrier features (B, 3 + (2 * N_freqs) * 3)
    """
    pe = [v]
    for freq in freq_bands:
        fv = freq * v
        pe += [torch.sin(fv), torch.cos(fv)]
    return torch.cat(pe, dim=-1)


@jit.script
def map_fourier_features(v: Tensor, b: Tensor) -> Tensor:
    """Map v to fourier features representation phi(v)

    Arguments:
        v (Tensor): input features (B, IFeatures)
        b (Tensor): b matrix (OFeatures, IFeatures)

    Returns:
        phi(v) (Tensor): fourrier features (B, 2 * Features)
    """
    PI = 3.141592653589793
    a = 2 * PI * v @ b.T
    return torch.cat((torch.sin(a), torch.cos(a)), dim=-1)


class FeatureMapping(nn.Module):
    """FeatureMapping nn.Module

    Maps v to features following transformation phi(v)

    Arguments:
        i_dim (int): input dimensions
        o_dim (int): output dimensions
    """

    def __init__(self, i_dim: int, o_dim: int) -> None:
        super().__init__()
        self.i_dim = i_dim
        self.o_dim = o_dim

    def forward(self, v: Tensor) -> Tensor:
        """FeratureMapping forward pass

        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): mapped features (B, OFeatures)
        """
        raise NotImplementedError("Forward pass not implemented yet!")


class PositionalEncoding(FeatureMapping):
    """PositionalEncoding module

    Maps v to positional encoding representation phi(v)

    Arguments:
        i_dim (int): input dimension for v
        N_freqs (int): #frequency to sample (default: 10)
    """

    def __init__(
        self,
        i_dim: int,
        N_freqs: int = 10,
    ) -> None:
        super().__init__(i_dim, 3 + (2 * N_freqs) * 3)
        self.N_freqs = N_freqs

        a, b = 1, self.N_freqs - 1
        freq_bands = 2 ** torch.linspace(a, b, self.N_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, v: Tensor) -> Tensor:
        """Map v to positional encoding representation phi(v)

        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): fourrier features (B, 3 + (2 * N_freqs) * 3)
        """
        return map_positional_encoding(v, self.freq_bands)


class FourierFeatures(FeatureMapping):

    """Fourier Features module

    Maps v to fourier features representation phi(v)

    Arguments:
        i_dim (int): input dimension for v
        features (int): output dimension (default: 256)
        sigma (float): std of the gaussian (default: 26.)
    """

    def __init__(
        self,
        i_dim: int,
        features: int = 256,
        sigma: float = 26.,
    ) -> None:
        super().__init__(i_dim, 2 * features)
        self.features = features
        self.sigma = sigma

        self.size = Size((self.features, self.i_dim))
        self.register_buffer("b", sample_b(self.size, self.sigma))

    def forward(self, v: Tensor) -> Tensor:
        """Map v to fourier features representation phi(v)

        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): fourrier features (B, 2 * Features)
        """
        return map_fourier_features(v, self.b)


class FeedForward(nn.Module):
    """ Adapted from the FeedForward layer from labmlai """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        activation: Callable = nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
        dropout: float = 0.1,
        use_layernorm: bool = False,
    ):
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_in, d_hidden, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_hidden, d_out, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_in, d_hidden, bias=bias_gate)
        # Whether to add a layernorm layer
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = LayerNorm(d_in)

    def forward(self, x: Tensor, coords: Tensor = None) -> Tensor:
        """Applies a simple feed forward layer"""
        x = self.layernorm(x) if self.use_layernorm else x
        g = self.activation(self.layer1(x))
        x = (g * self.linear_v(x)) if self.is_gated else g
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class BasePointModel(nn.Module):
    """ A base class providing useful methods for point cloud processing. """

    def __init__(
        self,
        *,
        num_classes,
        embed_dim,
        extra_feature_channels,
        dim: int = 128,
        num_layers: int = 6
    ):

        super().__init__()
        self.extra_feature_channels = extra_feature_channels
        self.timestep_embed_dim = embed_dim
        self.output_dim = num_classes
        self.dim = dim
        self.num_layers = num_layers

        # Time embedding function
        self.timestep_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(i_dim=3, N_freqs=10)
        positional_encoding_d_out = 3 + (2 * 10) * 3

        # Input projection (point coords, point coord encodings, other features, and timestep embeddings)
        self.input_projection = nn.Linear(
            in_features=(3 + positional_encoding_d_out + extra_feature_channels + self.timestep_embed_dim),
            out_features=self.dim
        )

        # Transformer layers
        self.layers = self.get_layers()

        # Output projection
        self.output_projection = nn.Linear(self.dim, self.output_dim)

    def get_layers(self):
        raise NotImplementedError('This method should be implemented by subclasses')

    def prepare_inputs(self, inputs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        The inputs have size (B, 3 + S, N), where S is the number of additional
        feature channels and N is the number of points. The timesteps t can be either 
        continuous or discrete. This model has a sort of U-Net-like structure I think, 
        which is why it first goes down and then up in terms of resolution (?)
        """

        # Embed and project timesteps
        t_emb = get_timestep_embedding(self.timestep_embed_dim, t, inputs.device)
        t_emb = self.timestep_projection(t_emb)[:, None, :].expand(-1, inputs.shape[-1], -1)  # (B, N, D_t_emb)

        # Separate input coordinates and features
        x = torch.transpose(inputs, -2, -1)  # -> (B, N, 3 + S)
        coords = x[:, :, :3]  # (B, N, 3), point coordinates

        # Positional encoding of point coords
        coords_posenc = self.positional_encoding(coords)  # (B, N, D_p_enc)

        # Project
        x = torch.cat((x, coords_posenc, t_emb), dim=2)  # (B, N, 3 + S + D_p_enc + D_t_emb)
        x = self.input_projection(x)  # (B, N, D_model)

        return x, coords

    def get_global_tensors(self, x: Tensor):
        B, N, D = x.shape
        x_pool_max = torch.max(x, dim=1, keepdim=True).values.repeat(1, N, 1)  # (B, 1, D)
        x_pool_std = torch.std(x, dim=1, keepdim=True).repeat(1, N, 1)  # (B, 1, D)
        return x_pool_max, x_pool_std

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError('This method should be implemented by subclasses')