# Zebrafish Toxicity Classification Pipeline

End-to-end pipeline for classifying micro-toxicity exposure in zebrafish using optical signals and Temporal Convolutional Networks (TCN).

## Key Features
*   **Preprocessing**: Signal filtering and robust R-peak detection.
*   **Feature Extraction**: Comprehensive HRV metrics, non-linear analysis, and windowed statistics.
*   **Deep Learning**: TCN-based temporal modeling for accurate classification.

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
