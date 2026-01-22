# HRV Analysis & Classification Pipeline

Complete end-to-end pipeline for Heart Rate Variability (HRV) analysis and classification from raw ECG signals to deep learning predictions, with interactive Plotly visualizations.

## Overview

This project implements a comprehensive HRV analysis system that:
- Loads raw ECG/cardiac signals from CSV files
- Performs signal preprocessing (filtering, peak detection)
- Divides signals into overlapping temporal windows
- Extracts physiological features (HRV metrics, statistical, wavelet)
- Encodes features using neural networks
- Processes temporal sequences with TCN (Temporal Convolutional Networks)
- Generates interactive visualizations in a modern web dashboard

##  Quick Start

### Running the Pipeline

```bash
python main.py
```

This will:
1. Load data from `recordings/` folder
2. Process the signal through the complete pipeline
3. Generate an interactive HTML dashboard (`hrv_analysis_report.html`)
4. Automatically open the dashboard in your browser

### Example Usage

```python
from main import AFClassificationPipeline
from pathlib import Path

# Initialize pipeline with custom parameters
pipeline = AFClassificationPipeline(config={
    "window_size_sec": 10,      # 10 second windows
    "window_overlap": 0.5,       # 50% overlap
    "sampling_freq": 40,         # 40 Hz sampling rate
    "embedding_dim": 128,        # Feature encoding dimension
    "tcn_channels": [64, 64, 128]
})

# Run complete analysis
csv_file = Path("recordings/sample_data.csv")
results = pipeline.run_pipeline(csv_file, plot=True)

# Access results
features = results['features_tensor']      # [num_windows, 1, num_features]
encoded = results['encoded_features']      # [num_windows, 1, embedding_dim]
tcn_output = results['tcn_output']         # [num_windows, channels, 1]
```

##  Pipeline Stages

### 1. Signal Preprocessing
- **Gaussian Averaging Filter**: Smooths the raw signal
- **Bandpass Filter**: 0.5-5.0 Hz to isolate cardiac frequencies
- **R-Peak Detection**: Identifies heartbeat peaks
- **Output**: Filtered signal + peak locations

### 2. Windowing
- Creates overlapping temporal windows from the signal
- Configurable window size and overlap percentage
- Ensures minimum peaks per window for reliable HRV analysis
- **Output**: 19 windows (for ~96s signal with 10s windows, 50% overlap)

### 3. Feature Extraction
Each window extracts **33 physiological features**:

#### HRV Time-Domain Features
- Mean RR, SDNN, RMSSD, pNN50, etc.

#### HRV Geometric Features
- Triangular Index
- Poincaré SD1/SD2

#### Statistical Features
- Mean, Std, Skewness, Kurtosis
- Percentiles, Range

#### Wavelet Features
- Multiscale Entropy (MSE)
- Wavelet coefficients statistics

### 4. Feature Encoding (FeatureEncoder)
- MLP-based encoder: `Linear → LayerNorm → GELU → Dropout`
- Projects 33 features → 128-dimensional embedding
- Learned representation for downstream tasks
- **Note**: Currently uses random weights (untrained)

### 5. Temporal Modeling (TCN)
- Temporal Convolutional Network with dilated causal convolutions
- Processes sequential embeddings: [64, 64, 128] channel architecture
- Captures temporal dependencies across windows
- **Note**: Currently uses random weights (untrained)

## 📁 Project Structure

```
final_project_classification/
├── main.py                          # Main pipeline orchestrator
├── recordings/                      # Input CSV data
│   └── Sample_Data_*.csv
├── processing/
│   ├── preprocessing.py             # Signal filtering & peak detection
│   ├── feature_extraction.py       # HRV feature extraction
│   └── windowing.py                 # Window creation & statistics
├── Models/
│   ├── feature_encoder.py          # Neural feature encoder
│   └── tcn.py                       # Temporal Convolutional Network
├── visualizations/
│   ├── plotly_plots.py             # Interactive Plotly visualizations
│   ├── html_viewer.py              # Dashboard generation
│   ├── hrv_dashboard.html          # Dashboard template
│   └── plotting.py                  # Legacy matplotlib functions
├── verifications/
│   └── example_windowing.py        # Usage examples
└── hrv_analysis_report.html        # Generated dashboard
```

## Features

✅ **Signal Processing**
- Gaussian and bandpass filtering
- Robust R-peak detection
- Quality metrics (BPM, signal duration)

✅ **Feature Extraction**
- 33 physiological features per window
- HRV time-domain and geometric metrics
- Statistical and wavelet features
- MSE complexity analysis

✅ **Windowing**
- Configurable window size and overlap
- Automatic window validation
- Per-window feature extraction

✅ **Neural Processing**
- Feature encoding with MLP
- Temporal modeling with TCN
- Ready for classification tasks

✅ **Visualization**
- Interactive Plotly dashboards
- Modern dark theme UI
- Single HTML file output
- Responsive design

## ⚙️ Configuration

Customize pipeline parameters:

```python
config = {
    # Signal Processing
    "sampling_freq": 40,           # Hz
    "low_threshold": 0.5,          # Bandpass low cutoff
    "high_threshold": 5.0,         # Bandpass high cutoff
    
    # Windowing
    "window_size_sec": 10,         # Window duration
    "window_overlap": 0.5,         # 50% overlap
    "min_peaks_per_window": 3,     # Minimum peaks
    
    # Feature Extraction
    "scales": 20,                  # MSE scales
    
    # Model Architecture
    "embedding_dim": 128,          # Encoder output dim
    "tcn_channels": [64, 64, 128], # TCN architecture
    "dropout": 0.1,
    "num_classes": 2,
    
    # Device
    "device": "cuda"               # or "cpu"
}

pipeline = AFClassificationPipeline(config=config)
```

## Output Data

After running the pipeline:

```python
results = {
    'raw_signal': np.ndarray,           # Original signal
    'filtered_signal': np.ndarray,      # Preprocessed signal
    'peaks': np.ndarray,                # Peak indices
    'features_tensor': torch.Tensor,    # [19, 1, 33] extracted features
    'feature_info': dict,               # Metadata & statistics
    'encoded_features': torch.Tensor,   # [19, 1, 128] encoded
    'tcn_output': torch.Tensor          # [19, 128, 1] TCN output
}
```

## Model Architecture

### FeatureEncoder
```
Input: [batch, 1, 33]
  ↓
Linear(33 → 128)
  ↓
LayerNorm
  ↓
GELU
  ↓
Dropout(0.1)
  ↓
Output: [batch, 1, 128]
```

### TemporalConvNet (TCN)
```
Input: [batch, 128, seq_len]
  ↓
TemporalBlock(128 → 64)  # dilation=1
  ↓
TemporalBlock(64 → 64)   # dilation=2
  ↓
TemporalBlock(64 → 128)  # dilation=4
  ↓
Output: [batch, 128, seq_len]
```

Each TemporalBlock contains:
- Dilated causal convolution
- Weight normalization
- ReLU activation
- Dropout
- Residual connection

## Requirements

```
Python >= 3.8
numpy
pandas
torch
plotly
scipy
```

Install dependencies:
```bash
pip install numpy pandas torch plotly scipy
```

---
