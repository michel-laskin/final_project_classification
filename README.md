# Zebrafish Toxicity Classification Pipeline

End-to-end pipeline for classifying micro-toxicity exposure in zebrafish TCN architecture

## Key Features
*   **Preprocessing**: Signal filtering and R-peak detection.
*   **Feature Extraction**: HRV metrics, non-linear analysis, and windowed statistics.
*   **Deep Learning**: MLP feature encoder and TCN-based temporal modeling for classification.

## Project Structure
```
final_project_classification/
├── main.py                  # Pipeline orchestrator
├── processing/              # Signal processing & feature extraction
├── Models/
│   ├── tcn.py              # Temporal Convolutional Network
│   └── feature_encoder.py  # Feature encoding module
└── visualizations/          # Dashboard generation
```

## Usage
```bash
python main.py
```
