import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .feature_encoder import FeatureEncoder


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.4):
        super(TemporalBlock, self).__init__()
        
        # First dilated convolution
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.SiLU()
        # Spatial Dropout1d - drops entire channels for better regularization in conv layers
        self.dropout1 = nn.Dropout1d(dropout)

        # Second dilated convolution
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.SiLU()
        self.dropout2 = nn.Dropout1d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.SiLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Padding such that output length equals input length (causal)
            # (Kernel-1) * Dilation
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape for TCN: [batch, channels, seq_len]
        return self.network(x)


class FusionModel(nn.Module):
    def __init__(self, input_dims, embedding_dim, tcn_channels, num_classes=2, dropout=0.1):
        """
        Args:
            input_dims (dict): Map of source name to input dimension e.g. {'scalars': 10, 'vectors_A': 50}
            embedding_dim (int): Dimension to project each input source to.
            tcn_channels (list): List of integers defining TCN hidden sizes.
            num_classes (int): Number of output classes (default 2).
            dropout (float): Dropout probability.
        """
        super(FusionModel, self).__init__()
        self.input_dims = input_dims
        self.embedding_dim = embedding_dim
        
        # Parallel encoders
        self.encoders = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.encoders[name] = FeatureEncoder(dim, embedding_dim, dropout=dropout)
            
        # Total dimension after concatenation
        total_fusion_dim = embedding_dim * len(input_dims)
        
        # TCN Backbone
        self.tcn = TemporalConvNet(
            num_inputs=total_fusion_dim,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        # Classifier Head
        # Maps from last TCN channel size to num_classes
        # TCN output is [Batch, Channels, Seq_Len] -> we want to classify every time step
        self.classifier = nn.Linear(tcn_channels[-1], num_classes)

    def forward(self, inputs, return_activations=False):
        """
        Args:
            inputs (dict): Dictionary of tensors, keys matching input_dims.
                           Each tensor shape: [batch, seq_len, distinct_feat_dim]
            return_activations (bool): If True, returns a dictionary of internal activations.
        """
        encoded_feats = []
        activations = {}
        
        # 1. Input Grouping & Parallel Encoding
        # Sort keys to ensure deterministic order if dict is unordered
        # Assuming inputs and self.encoders keys match.
        for name in self.encoders:
            if name not in inputs:
                raise ValueError(f"Missing input for required feature group: {name}")
            
            x = inputs[name] # [batch, seq_len, dim]
            
            out = self.encoders[name](x)
            encoded_feats.append(out)
            
            if return_activations:
                activations[f'encoder_{name}'] = out
            
        # 2. Fusion Layer
        # Concatenate along the feature dimension (dim=2)
        fused = torch.cat(encoded_feats, dim=2) # [batch, seq_len, total_fusion_dim]
        
        if return_activations:
            activations['fused'] = fused
        
        # 3. Temporal Modeling (TCN)
        # TCN expects [batch, channels, seq_len]
        fused_t = fused.transpose(1, 2)
        tcn_out = self.tcn(fused_t)
        
        # 4. Classifier
        # Transpose back to [batch, seq_len, channels] to apply Linear
        tcn_out_t = tcn_out.transpose(1, 2)
        logits = self.classifier(tcn_out_t)
        
        if return_activations:
            return logits, activations
        return logits
