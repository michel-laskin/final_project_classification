import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """
    Feature encoder block (MLP-based) to project features to a fixed dimension.
    Pipeline: Linear -> LayerNorm -> ReLU -> Dropout
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(FeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        return self.net(x)
