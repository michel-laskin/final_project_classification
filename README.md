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

### 3. Feature Encoding (FeatureEncoder)
- MLP-based encoder: `Linear → LayerNorm → SiLU → Dropout`
- Projects 146 features → 16-dimensional embedding
- Learned representation for downstream tasks


### 4. Temporal Modeling (TCN)
- Temporal Convolutional Network with dilated causal convolutions
- Captures temporal dependencies across windows

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

✅ **Feature Extraction**
- 146 physiological features per window
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
- Single HTML file output

pipeline = AFClassificationPipeline(config=config)
```

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
