"""
Configurable Feature Encoder for Optuna Architecture Search

Variable-depth MLP encoder that can be configured via hyperparameters.
This is a drop-in replacement for FeatureEncoder, used by the Optuna objective
function. The original FeatureEncoder remains untouched.
"""

import torch
import torch.nn as nn


class ConfigurableFeatureEncoder(nn.Module):
    """
    Variable-depth MLP encoder with configurable architecture.
    
    Supports:
    - 0-2 hidden layers with configurable widths
    - Activation choice: ReLU, SiLU, GELU
    - LayerNorm between layers
    - Dropout
    
    When num_hidden_layers=0, this is equivalent to the original FeatureEncoder
    (single linear projection).
    """
    
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'gelu': nn.GELU,
    }
    
    def __init__(self, input_dim, output_dim, num_hidden_layers=0,
                 hidden_dim=64, activation='silu', dropout=0.1):
        """
        Args:
            input_dim: Number of input features.
            output_dim: Embedding dimension (output size).
            num_hidden_layers: Number of hidden layers (0, 1, or 2).
            hidden_dim: Width of hidden layers.
            activation: Activation function name ('relu', 'silu', 'gelu').
            dropout: Dropout probability.
        """
        super(ConfigurableFeatureEncoder, self).__init__()
        
        act_cls = self.ACTIVATIONS.get(activation, nn.SiLU)
        
        layers = []
        
        if num_hidden_layers == 0:
            # Direct projection (matches original FeatureEncoder)
            layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                act_cls(),
                nn.Dropout(dropout),
            ])
        elif num_hidden_layers == 1:
            # input -> hidden -> output
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_cls(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                act_cls(),
                nn.Dropout(dropout),
            ])
        elif num_hidden_layers == 2:
            # input -> hidden -> hidden//2 -> output (bottleneck)
            mid_dim = max(hidden_dim // 2, output_dim)
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_cls(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, mid_dim),
                nn.LayerNorm(mid_dim),
                act_cls(),
                nn.Dropout(dropout),
                nn.Linear(mid_dim, output_dim),
                nn.LayerNorm(output_dim),
                act_cls(),
                nn.Dropout(dropout),
            ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        return self.net(x)
