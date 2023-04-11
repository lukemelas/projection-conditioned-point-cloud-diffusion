import torch
import torch.nn as nn

from model.pvcnn.pvcnn import PVCNN2
from model.pvcnn.pvcnn_utils import create_mlp_components
from model.simple.simple_model import SimplePointModel


class PVCNN2PlusPlus(nn.Module):
    def __init__(
        self,
        *,
        embed_dim,
        num_classes,
        extra_feature_channels,
    ):
        super().__init__()
        
        # Create models
        self.pvcnn = PVCNN2(num_classes=num_classes, embed_dim=embed_dim, extra_feature_channels=extra_feature_channels)
        self.simple_point_model = SimplePointModel(num_classes=num_classes, embed_dim=embed_dim, extra_feature_channels=extra_feature_channels, num_layers=4)
        
        # Tie timestep embeddings
        self.pvcnn.embedf = self.simple_point_model.timestep_projection

        # Remove output projections 
        self.pvcnn.classifier = nn.Identity()
        self.simple_point_model.output_projection = nn.Identity()

        # Create new output projection
        in_channels = self.pvcnn.channels_fp_features + self.simple_point_model.dim  # concatenation of outputs
        layers, _ = create_mlp_components(
            in_channels=in_channels, out_channels=[128, self.pvcnn.dropout, num_classes],
            classifier=True, dim=2, width_multiplier=self.pvcnn.width_multiplier
        )
        self.output_projection = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        x_pvcnn = self.pvcnn(inputs, t)  # (B, N, D_pvcnn)
        x_spm = self.simple_point_model(inputs, t)  # (B, N, D_spm)
        x = torch.cat((x_pvcnn, x_spm), dim=1)  # (B, N, D_pvcnn + D_spm)
        x = self.output_projection(x)  # (B, N, D_out)
        return x
