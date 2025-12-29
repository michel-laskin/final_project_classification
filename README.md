# final_project_classification

Classification model for processed recordings of Heart Rate Variability (HRV) data.

## Overview

This project implements a deep learning classification model designed to analyze HRV recordings. The model uses a **Temporal Convolutional Network (TCN)** architecture combined with feature fusion to process multi-source time-series data.

## Architecture

The model consists of the following components:

- **FeatureEncoder**: MLP-based encoder that projects input features to a fixed embedding dimension using Linear → LayerNorm → GELU → Dropout pipeline.
- **TemporalConvNet**: A stack of dilated causal convolution blocks for temporal modeling, using weight normalization and residual connections.
- **FusionModel**: The main classifier that:
  1. Encodes multiple input sources in parallel
  2. Fuses encoded features via concatenation
  3. Applies temporal modeling through the TCN backbone
  4. Produces per-timestep classification logits

## Requirements

- Python 3.x
- PyTorch

## Usage

```python
from Classifier.models import FusionModel

# Define input dimensions for each feature source
input_dims = {'scalars': 10, 'vectors_A': 50}

# Initialize the model
model = FusionModel(
    input_dims=input_dims,
    embedding_dim=64,
    tcn_channels=[64, 128, 128],
    num_classes=2,
    dropout=0.1
)

# Forward pass
logits = model(inputs)  # inputs is a dict of tensors
```
