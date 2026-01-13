"""
Main Pipeline Script for HRV Analysis

This script orchestrates the complete pipeline:
1. Load CSV data from recordings folder
2. Preprocess signal (filtering, peak detection)
3. Divide signal into overlapping windows
4. Extract features (HRV, statistical, wavelet) from each window
5. Encode features using FeatureEncoder
6. Process through TCN
7. Visualize results at each step
"""

# Fix for OpenMP duplicate library warning
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import processing modules
from processing.preprocessing import average_filter, bandpass_filter, detect_r_peaks, plot_r_peaks, extract_rr
from processing.feature_extraction import HRVFeatureExtractor, FeatureExtractor
from visualizations.plotly_plots import (plot_mse, plot_mse_heatmap, plot_comprehensive_summary, 
                                          visualize_windows, visualize_feature_heatmap)
from visualizations.html_viewer import create_tabbed_html, open_in_browser
from processing.windowing import create_windows, extract_features_from_windows, get_window_statistics

# Import model modules
from Models.feature_encoder import FeatureEncoder
from Models.tcn import TemporalConvNet


class HRVPipeline:
    """
    Complete pipeline for HRV analysis from raw CSV to model predictions.
    """
    
    def __init__(self, config=None):
        """
        Initialize pipeline with configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters. If None, uses defaults.
        """
        # Default configuration
        self.config = {
            # Signal processing
            "sampling_freq": 40,
            "averaging_type": "Gaussian",
            "gaussian_sigma": 2,
            "low_threshold": 0.5,
            "high_threshold": 5.0,
            "filter_order": 4,
            "min_rr_sec_human": 0.3,
            
            # Feature extraction
            "samp_en_m": 2,
            "samp_en_r": 0.2,
            "scales": 20,
            "d": 8,
            "p": 3,
            "x": 50.0,
            
            # Windowing parameters
            "window_size_sec": 10,
            "window_overlap": 0.5,
            "min_peaks_per_window": 3,
            
            # Model architecture
            "embedding_dim": 128,
            "tcn_channels": [64, 64, 128],
            "tcn_kernel_size": 3,
            "dropout": 0.1,
            "num_classes": 2,
            
            # Device
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # Update with user config if provided
        if config is not None:
            self.config.update(config)
        
        # Initialize feature extractors
        self.hrv_extractor = HRVFeatureExtractor(params={
            "file_path": "",  # Will be set per file
            "d": self.config["d"],
            "p": self.config["p"],
            "x": self.config["x"],
            "hz": self.config["sampling_freq"]
        })
        
        self.entropy_extractor = FeatureExtractor(params={
            "filename": "",  # Will be set per file
            "sampling_freq": self.config["sampling_freq"],
            "samp_en_m": self.config["samp_en_m"],
            "samp_en_r": self.config["samp_en_r"],
            "scales": self.config["scales"]
        })
        
        # Models will be initialized after we know feature dimensions
        self.feature_encoder = None
        self.tcn = None
        
        # Dictionary to collect all figures for tabbed HTML view
        self.figures = {}
        
        print(f"Pipeline initialized on device: {self.config['device']}")
    
    def load_csv_data(self, csv_path):
        """
        Load CSV data file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            signal: Raw signal array
        """
        print(f"\n{'='*60}")
        print(f"Loading data from: {csv_path}")
        print(f"{'='*60}")
        
        # Load CSV - assuming single column of values
        data = pd.read_csv(csv_path, header=None)
        signal = data.values.flatten()
        
        print(f"Loaded signal: {len(signal)} samples at {self.config['sampling_freq']} Hz")
        print(f"Signal duration: {len(signal) / self.config['sampling_freq']:.2f} seconds")
        print(f"Signal range: [{signal.min():.2f}, {signal.max():.2f}]")
        
        return signal
    
    def preprocess_signal(self, signal, plot=True):
        """
        Preprocess signal: filtering and peak detection.
        
        Args:
            signal: Raw input signal
            plot: Whether to plot preprocessing results
            
        Returns:
            filtered_signal: Filtered signal
            peaks: Detected peak indices
        """
        print(f"\n{'='*60}")
        print("STEP 1: Signal Preprocessing")
        print(f"{'='*60}")
        
        # Apply Gaussian filter
        print("Applying Gaussian averaging filter...")
        averaged_signal = average_filter(signal, self.config)
        
        # Apply bandpass filter
        print(f"Applying bandpass filter ({self.config['low_threshold']}-{self.config['high_threshold']} Hz)...")
        filtered_signal = bandpass_filter(averaged_signal, self.config)
        
        # Detect R-peaks
        print("Detecting R-peaks...")
        peaks = detect_r_peaks(filtered_signal, self.config)
        print(f"Detected {len(peaks)} R-peaks")
        
        if len(peaks) > 0:
            rr_intervals = extract_rr(peaks, self.config)
            mean_rr = np.mean(rr_intervals)
            bpm = 60 / mean_rr
            print(f"Average heart rate: {bpm:.1f} BPM")
        
        # Visualization
        if plot:
            # Create time axis
            t_axis = np.arange(len(signal)) / self.config['sampling_freq']
            
            # Create Plotly subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Raw Signal', 'Filtered Signal', 'Peak Detection'),
                vertical_spacing=0.1
            )
            
            # Raw signal
            fig.add_trace(go.Scatter(
                x=t_axis,
                y=signal,
                mode='lines',
                line=dict(color='#0099ff', width=1),
                opacity=0.7,
                name='Raw Signal'
            ), row=1, col=1)
            
            # Filtered signal
            fig.add_trace(go.Scatter(
                x=t_axis,
                y=filtered_signal,
                mode='lines',
                line=dict(color='#00ff88', width=1),
                name='Filtered Signal'
            ), row=2, col=1)
            
            # Filtered signal with peaks
            fig.add_trace(go.Scatter(
                x=t_axis,
                y=filtered_signal,
                mode='lines',
                line=dict(color='#00ff88', width=1),
                name='Filtered Signal',
                showlegend=False
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=t_axis[peaks],
                y=filtered_signal[peaks],
                mode='markers',
                marker=dict(color='red', size=10, symbol='star'),
                name=f'R-Peaks (n={len(peaks)})'
            ), row=3, col=1)
            
            # Update axes labels
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_xaxes(title_text="Time (s)", row=3, col=1)
            
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=3, col=1)
            
            # Update layout
            fig.update_layout(
                height=900,
                template="plotly_dark",
                showlegend=True,
                hovermode='x unified'
            )
            
            # Store figure for tabbed view
            self.figures['1. Signal Preprocessing'] = fig
        
        return filtered_signal, peaks
    
    def extract_features(self, signal, peaks, plot=True):
        """
        Extract all features from signal.
        
        Args:
            signal: Preprocessed signal
            peaks: Peak indices
            plot: Whether to plot feature extraction results
            
        Returns:
            features_tensor: Tensor of extracted features [1, 1, num_features]
            feature_info: Dictionary with feature metadata
        """
        print(f"\n{'='*60}")
        print("STEP 2: Feature Extraction")
        print(f"{'='*60}")
        
        # Create time axis
        t_axis = np.arange(len(signal)) / self.config['sampling_freq']
        
        # Extract all features using HRVFeatureExtractor
        print("Extracting HRV, statistical, and wavelet features...")
        features_tensor, rr_intervals_ms = self.hrv_extractor.extract_all_features(
            signal, peaks, t_axis, device=self.config['device']
        )
        
        num_features = features_tensor.shape[2]
        print(f"Extracted {num_features} total features")
        print(f"Feature tensor shape: {features_tensor.shape}")
        
        # Compute MSE for visualization
        mse_values = self.entropy_extractor.multiscale_entropy(rr_intervals_ms)
        
        feature_info = {
            "num_features": num_features,
            "rr_intervals_ms": rr_intervals_ms,
            "mse_values": mse_values,
            "tensor_shape": features_tensor.shape
        }
        
        # Visualization
        if plot:
            # Create Plotly subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Multiscale Entropy (MSE)', '',
                    'RR Interval Distribution', 'Sample Feature Values (First 20)',
                    'Feature Statistics', ''
                ),
                specs=[
                    [{"colspan": 2}, None],
                    [{}, {}],
                    [{"colspan": 2}, None]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.12
            )
            
            # MSE plot
            scales = np.arange(1, len(mse_values) + 1)
            fig.add_trace(go.Scatter(
                x=scales,
                y=mse_values,
                mode='lines+markers',
                line=dict(color='steelblue', width=2),
                marker=dict(size=6),
                name='MSE'
            ), row=1, col=1)
            
            # RR interval histogram
            fig.add_trace(go.Histogram(
                x=rr_intervals_ms,
                nbinsx=30,
                marker=dict(color='coral', line=dict(color='black', width=1)),
                opacity=0.7,
                name='RR Intervals'
            ), row=2, col=1)
            
            # Add mean line to histogram
            mean_rr = np.mean(rr_intervals_ms)
            fig.add_vline(
                x=mean_rr,
                line=dict(color='red', dash='dash', width=2),
                annotation_text=f'Mean: {mean_rr:.1f} ms',
                annotation_position="top right",
                row=2, col=1
            )
            
            # Feature values (first 20 features)
            feature_vals = features_tensor[0, 0, :20].cpu().numpy()
            fig.add_trace(go.Bar(
                x=list(range(len(feature_vals))),
                y=feature_vals,
                marker=dict(color='mediumseagreen'),
                opacity=0.7,
                name='Features',
                showlegend=False
            ), row=2, col=2)
            
            # Feature statistics (as text annotation)
            all_features = features_tensor[0, 0, :].cpu().numpy()
            feature_stats = {
                'Mean': np.mean(all_features),
                'Std': np.std(all_features),
                'Min': np.min(all_features),
                'Max': np.max(all_features),
                'Median': np.median(all_features)
            }
            
            stats_text = "Feature Statistics:<br><br>"
            for key, val in feature_stats.items():
                stats_text += f"{key:>10s}: {val:>10.4f}<br>"
            
            fig.add_annotation(
                text=stats_text,
                xref="x5", yref="y5",
                x=0.5, y=0.5,
                xanchor='center', yanchor='middle',
                showarrow=False,
                bgcolor='rgba(255, 228, 181, 0.5)',
                bordercolor='white',
                borderwidth=2,
                font=dict(size=12, family='monospace', color='white'),
                row=3, col=1
            )
            
            # Update axes
            fig.update_xaxes(title_text="Scale", row=1, col=1)
            fig.update_yaxes(title_text="Sample Entropy", row=1, col=1)
            
            fig.update_xaxes(title_text="RR Interval (ms)", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            
            fig.update_xaxes(title_text="Feature Index", row=2, col=2)
            fig.update_yaxes(title_text="Value", row=2, col=2)
            
            # Hide axes for stats panel
            fig.update_xaxes(visible=False, row=3, col=1)
            fig.update_yaxes(visible=False, row=3, col=1)
            
            # Update layout
            fig.update_layout(
                height=1000,
                template="plotly_dark",
                showlegend=False,
                title_text='Feature Extraction Results',
                title_font_size=16
            )
            
            # Store figure for tabbed view
            self.figures['2. Feature Extraction'] = fig
        
        return features_tensor, feature_info
    
    def extract_features_windowed(self, signal, peaks, plot=True):
        """
        Extract features from overlapping windows of the signal.
        
        Args:
            signal: Preprocessed signal
            peaks: Peak indices
            plot: Whether to plot windowing results
            
        Returns:
            features_tensor: Tensor of features [num_windows, 1, num_features]
            window_info: Dictionary with windowing metadata
        """
        print(f"\n{'='*60}")
        print("STEP 2: Windowed Feature Extraction")
        print(f"{'='*60}")
        
        # Create time axis
        t_axis = np.arange(len(signal)) / self.config['sampling_freq']
        
        # Create windows
        print(f"Creating windows (size={self.config['window_size_sec']}s, overlap={self.config['window_overlap']*100}%)...")
        windows = create_windows(
            signal=signal,
            peaks=peaks,
            t_axis=t_axis,
            window_size_sec=self.config['window_size_sec'],
            overlap=self.config['window_overlap'],
            sampling_freq=self.config['sampling_freq'],
            min_peaks_per_window=self.config['min_peaks_per_window']
        )
        
        # Get window statistics
        window_stats = get_window_statistics(windows)
        print(f"\nWindow Statistics:")
        print(f"  - Number of windows: {window_stats['num_windows']}")
        print(f"  - Average peaks per window: {window_stats['avg_peaks_per_window']:.1f}")
        print(f"  - Peak range: [{window_stats['min_peaks']}, {window_stats['max_peaks']}]")
        print(f"  - Total duration: {window_stats['total_duration']:.2f} seconds")
        
        if len(windows) == 0:
            raise ValueError("No valid windows created. Try reducing window size or overlap.")
        
        # Extract features from all windows
        features_tensor, rr_intervals_list, window_metadata = extract_features_from_windows(
            windows=windows,
            hrv_extractor=self.hrv_extractor,
            device=self.config['device'],
            verbose=True
        )
        
        num_windows = features_tensor.shape[0]
        num_features = features_tensor.shape[2]
        print(f"\nExtracted {num_features} features from each of {num_windows} windows")
        print(f"Feature tensor shape: {features_tensor.shape}")
        
        window_info = {
            "windows": windows,
            "window_stats": window_stats,
            "window_metadata": window_metadata,
            "rr_intervals_list": rr_intervals_list,
            "num_windows": num_windows,
            "num_features": num_features
        }
        
        # Visualization
        if plot:
            # 1. Visualize window segmentation (show all windows)
            fig_windows = visualize_windows(signal, t_axis, windows, max_windows_to_show=len(windows), show=False)
            self.figures['3a. Window Segmentation'] = fig_windows
            
            # 2. Feature heatmap across windows
            fig_heatmap = visualize_feature_heatmap(features_tensor, window_metadata, max_features_to_show=30, show=False)
            self.figures['3b. Feature Heatmap'] = fig_heatmap
            
            # 3. Aggregated statistics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Heart Rate Across Windows',
                    'Mean Feature Value per Window',
                    'Combined RR Interval Distribution',
                    'Windowing Summary'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
            
            # Window-wise BPM
            bpms = [60000 / np.mean(rr) if len(rr) > 0 else 0 for rr in rr_intervals_list]
            window_indices = [w['window_idx'] for w in window_metadata]
            
            fig.add_trace(go.Scatter(
                x=window_indices,
                y=bpms,
                mode='lines+markers',
                line=dict(color='steelblue', width=2),
                marker=dict(size=6),
                name='BPM'
            ), row=1, col=1)
            
            # Feature value distribution across windows
            mean_features = features_tensor.mean(dim=2).squeeze(1).cpu().numpy()
            fig.add_trace(go.Bar(
                x=window_indices,
                y=mean_features,
                marker=dict(color='coral'),
                opacity=0.7,
                name='Mean Features'
            ), row=1, col=2)
            
            # RR interval distribution (all windows combined)
            all_rr = np.concatenate(rr_intervals_list)
            fig.add_trace(go.Histogram(
                x=all_rr,
                nbinsx=30,
                marker=dict(color='mediumseagreen', line=dict(color='black', width=1)),
                opacity=0.7,
                name='RR Intervals'
            ), row=2, col=1)
            
            # Add mean line
            mean_all_rr = np.mean(all_rr)
            fig.add_vline(
                x=mean_all_rr,
                line=dict(color='red', dash='dash', width=2),
                annotation_text=f'Mean: {mean_all_rr:.1f} ms',
                annotation_position="top right",
                row=2, col=1
            )
            
            # Summary statistics
            summary_text = "Windowing Summary:<br><br>"
            summary_text += f"Windows Created:    {num_windows}<br>"
            summary_text += f"Features per Window: {num_features}<br>"
            summary_text += f"Avg. BPM:           {np.mean(bpms):.1f}<br>"
            summary_text += f"BPM Std:            {np.std(bpms):.1f}<br>"
            summary_text += f"Avg. RR Interval:   {mean_all_rr:.1f} ms<br>"
            
            fig.add_annotation(
                text=summary_text,
                xref="x4", yref="y4",
                x=0.5, y=0.5,
                xanchor='center', yanchor='middle',
                showarrow=False,
                bgcolor='rgba(255, 228, 181, 0.5)',
                bordercolor='white',
                borderwidth=2,
                font=dict(size=12, family='monospace', color='white'),
                row=2, col=2
            )
            
            # Update axes
            fig.update_xaxes(title_text="Window Index", row=1, col=1)
            fig.update_yaxes(title_text="BPM", row=1, col=1)
            
            fig.update_xaxes(title_text="Window Index", row=1, col=2)
            fig.update_yaxes(title_text="Mean Feature Value", row=1, col=2)
            
            fig.update_xaxes(title_text="RR Interval (ms)", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            
            # Hide axes for summary panel
            fig.update_xaxes(visible=False, row=2, col=2)
            fig.update_yaxes(visible=False, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                height=800,
                template="plotly_dark",
                showlegend=False,
                title_text='Windowed Feature Extraction Results',
                title_font_size=16
            )
            
            # Store figure for tabbed view
            self.figures['3c. Windowing Statistics'] = fig
        
        return features_tensor, window_info
    
    def encode_features(self, features_tensor, plot=True):
        """
        Encode features using FeatureEncoder.
        
        Args:
            features_tensor: Input feature tensor [1, 1, num_features]
            plot: Whether to plot encoding results
            
        Returns:
            encoded_features: Encoded feature tensor [1, 1, embedding_dim]
        """
        print(f"\n{'='*60}")
        print("STEP 3: Feature Encoding")
        print(f"{'='*60}")
        
        # Initialize feature encoder if not already done
        if self.feature_encoder is None:
            input_dim = features_tensor.shape[2]
            self.feature_encoder = FeatureEncoder(
                input_dim=input_dim,
                output_dim=self.config['embedding_dim'],
                dropout=self.config['dropout']
            ).to(self.config['device'])
            print(f"Initialized FeatureEncoder: {input_dim} -> {self.config['embedding_dim']}")
        
        # Encode features
        self.feature_encoder.eval()
        with torch.no_grad():
            encoded_features = self.feature_encoder(features_tensor)
        
        print(f"Encoded feature shape: {encoded_features.shape}")
        
        # Visualization
        if plot:
            # Create Plotly subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'Original Features (dim={features_tensor.shape[2]})',
                    f'Encoded Features (dim={self.config["embedding_dim"]})',
                    'Original Feature Distribution',
                    'Encoded Feature Distribution'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
            
            # Original features
            original = features_tensor[0, 0, :].cpu().numpy()
            fig.add_trace(go.Scatter(
                y=original,
                mode='lines+markers',
                line=dict(color='steelblue', width=1),
                marker=dict(size=3),
                opacity=0.7,
                name='Original'
            ), row=1, col=1)
            
            # Encoded features
            encoded = encoded_features[0, 0, :].cpu().numpy()
            fig.add_trace(go.Scatter(
                y=encoded,
                mode='lines+markers',
                line=dict(color='coral', width=1),
                marker=dict(size=3),
                opacity=0.7,
                name='Encoded'
            ), row=1, col=2)
            
            # Original distribution
            fig.add_trace(go.Histogram(
                x=original,
                nbinsx=30,
                marker=dict(color='steelblue', line=dict(color='black', width=1)),
                opacity=0.7,
                name='Original',
                showlegend=False
            ), row=2, col=1)
            
            # Encoded distribution
            fig.add_trace(go.Histogram(
                x=encoded,
                nbinsx=30,
                marker=dict(color='coral', line=dict(color='black', width=1)),
                opacity=0.7,
                name='Encoded',
                showlegend=False
            ), row=2, col=2)
            
            # Update axes
            fig.update_xaxes(title_text="Feature Index", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=1)
            
            fig.update_xaxes(title_text="Feature Index", row=1, col=2)
            fig.update_yaxes(title_text="Value", row=1, col=2)
            
            fig.update_xaxes(title_text="Value", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            
            fig.update_xaxes(title_text="Value", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            
            # Update layout
            fig.update_layout(
                height=800,
                template="plotly_dark",
                showlegend=False,
                title_text='Feature Encoding: Original vs Encoded',
                title_font_size=16
            )
            
            # Store figure for tabbed view
            self.figures['4. Feature Encoding'] = fig
        
        return encoded_features
    
    def process_tcn(self, encoded_features, plot=True):
        """
        Process encoded features through TCN.
        
        Args:
            encoded_features: Encoded feature tensor [1, 1, embedding_dim]
            plot: Whether to plot TCN results
            
        Returns:
            tcn_output: TCN output tensor [1, channels[-1], seq_len]
        """
        print(f"\n{'='*60}")
        print("STEP 4: Temporal Convolutional Network Processing")
        print(f"{'='*60}")
        
        # Initialize TCN if not already done
        if self.tcn is None:
            self.tcn = TemporalConvNet(
                num_inputs=self.config['embedding_dim'],
                num_channels=self.config['tcn_channels'],
                kernel_size=self.config['tcn_kernel_size'],
                dropout=self.config['dropout']
            ).to(self.config['device'])
            print(f"Initialized TCN: {self.config['embedding_dim']} -> {self.config['tcn_channels']}")
        
        # Process through TCN
        # TCN expects [batch, channels, seq_len]
        self.tcn.eval()
        with torch.no_grad():
            tcn_input = encoded_features.transpose(1, 2)  # [1, embedding_dim, 1]
            tcn_output = self.tcn(tcn_input)  # [1, channels[-1], 1]
        
        print(f"TCN input shape: {tcn_input.shape}")
        print(f"TCN output shape: {tcn_output.shape}")
        
        # Visualization
        if plot:
            # Create Plotly subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f'TCN Input (dim={tcn_input.shape[1]})',
                    f'TCN Output (dim={tcn_output.shape[1]})'
                ),
                vertical_spacing=0.15
            )
            
            # Input to TCN
            tcn_in = tcn_input[0, :, 0].cpu().numpy()
            fig.add_trace(go.Scatter(
                y=tcn_in,
                mode='lines+markers',
                line=dict(color='mediumseagreen', width=1),
                marker=dict(size=3),
                opacity=0.7,
                name='TCN Input'
            ), row=1, col=1)
            
            # Output from TCN
            tcn_out = tcn_output[0, :, 0].cpu().numpy()
            fig.add_trace(go.Scatter(
                y=tcn_out,
                mode='lines+markers',
                line=dict(color='darkorange', width=1),
                marker=dict(size=4),
                opacity=0.7,
                name='TCN Output'
            ), row=2, col=1)
            
            # Update axes
            fig.update_xaxes(title_text="Channel Index", row=1, col=1)
            fig.update_yaxes(title_text="Activation", row=1, col=1)
            
            fig.update_xaxes(title_text="Channel Index", row=2, col=1)
            fig.update_yaxes(title_text="Activation", row=2, col=1)
            
            # Update layout
            fig.update_layout(
                height=700,
                template="plotly_dark",
                showlegend=False,
                title_text='Temporal Convolutional Network Processing',
                title_font_size=16
            )
            
            # Store figure for tabbed view
            self.figures['5. TCN Processing'] = fig
        
        return tcn_output
    
    def run_pipeline(self, csv_path, plot=True):
        """
        Run the complete pipeline on a CSV file.
        
        Args:
            csv_path: Path to CSV file
            plot: Whether to generate plots
            
        Returns:
            results: Dictionary containing all pipeline outputs
        """
        print(f"\n{'#'*60}")
        print(f"# Starting HRV Analysis Pipeline")
        print(f"{'#'*60}")
        
        # Step 0: Load data
        signal = self.load_csv_data(csv_path)
        
        # Step 1: Preprocess
        filtered_signal, peaks = self.preprocess_signal(signal, plot=plot)
        
        # Step 2: Extract features using windowing
        features_tensor, feature_info = self.extract_features_windowed(filtered_signal, peaks, plot=plot)
        
        # Step 3: Encode features
        encoded_features = self.encode_features(features_tensor, plot=plot)
        
        # Step 4: Process through TCN
        tcn_output = self.process_tcn(encoded_features, plot=plot)
        
        results = {
            "raw_signal": signal,
            "filtered_signal": filtered_signal,
            "peaks": peaks,
            "features_tensor": features_tensor,
            "feature_info": feature_info,
            "encoded_features": encoded_features,
            "tcn_output": tcn_output
        }
        
        print(f"\n{'#'*60}")
        print(f"# Pipeline Complete!")
        print(f"{'#'*60}")
        print(f"\nPipeline Summary:")
        print(f"  - Signal samples: {len(signal)}")
        print(f"  - Detected peaks: {len(peaks)}")
        print(f"  - Number of windows: {feature_info.get('num_windows', 'N/A')}")
        print(f"  - Features per window: {feature_info.get('num_features', 'N/A')}")
        print(f"  - Encoded dimension: {self.config['embedding_dim']}")
        print(f"  - TCN output channels: {self.config['tcn_channels'][-1]}")
        
        return results


class AFClassificationPipeline:
    """
    Complete pipeline for AF classification from PPG recordings.
    """
    
    def __init__(self, config=None):
        """Initialize the classification pipeline."""
        # Default configuration
        self.config = {
            # Signal processing
            "sampling_freq": 125,  # MIMIC-PERFORM data is sampled at 125 Hz
            "averaging_type": "Gaussian",
            "gaussian_sigma": 2,
            "low_threshold": 0.5,
            "high_threshold": 5.0,
            "filter_order": 4,
            "min_rr_sec_human": 0.3,
            
            # HRV windowing
            "hrv_window_size": 200,
            "hrv_overlap": 20,
            
            # Data split
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            
            # Model architecture (moderate regularization - balanced approach)
            "embedding_dim": 128,
            "tcn_channels": [64, 64],  # Moderate size - between [32,64] and [64,64,128]
            "tcn_kernel_size": 3,
            "dropout": 0.2,  # Moderate dropout - between 0.1 and 0.4
            "num_classes": 2,
            
            # Training
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,  # Standard learning rate
            "weight_decay": 0.001,  # Light L2 regularization (10x less than before)
            "early_stopping_patience": 10,
            
            # Device
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        if config is not None:
            self.config.update(config)
        
        # Initialize feature extractor
        self.hrv_extractor = HRVFeatureExtractor(params={
            "file_path": "",
            "d": 8,
            "p": 3,
            "x": 50.0,
            "hz": self.config["sampling_freq"]
        })
        
        # Models (will be initialized later)
        self.model = None
        
        # Dictionary to collect all figures for HTML
        self.figures = {}
        
        print(f"AF Classification Pipeline initialized on device: {self.config['device']}")
    
    def load_all_recordings(self, recordings_dir):
        """Load all CSV files and assign labels based on filename."""
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print(f"{'='*60}")
        
        recordings_path = Path(recordings_dir)
        csv_files = sorted(list(recordings_path.glob("*.csv")))
        
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {recordings_path}")
        
        files_data = []
        af_count = 0
        non_af_count = 0
        
        for csv_file in csv_files:
            filename = csv_file.name
            
            # Determine label from filename
            # Check for '_non_af_' FIRST, since it contains '_af_' substring
            if '_non_af_' in filename.lower():
                label = 0
                non_af_count += 1
            elif '_af_' in filename.lower():
                label = 1
                af_count += 1
            else:
                print(f"  Warning: Could not determine label for {filename}, skipping...")
                continue
            
            files_data.append({
                'filepath': csv_file,
                'filename': filename,
                'label': label
            })
        
        print(f"Loaded {len(files_data)} files:")
        print(f"  - AF files: {af_count}")
        print(f"  - Non-AF files: {non_af_count}")
        
        return files_data
    
    def process_file_to_hrv(self, filepath):
        """Process a single CSV file to extract HRV time series."""
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Extract Time and PPG columns
        if 'Time' not in df.columns or 'PPG' not in df.columns:
            raise ValueError(f"CSV file must contain 'Time' and 'PPG' columns. Found: {df.columns.tolist()}")
        
        time = df['Time'].values
        ecg_signal = df['PPG'].values
        
        # Preprocess signal
        averaged_signal = average_filter(ecg_signal, self.config)
        filtered_signal = bandpass_filter(averaged_signal, self.config)
        peaks = detect_r_peaks(filtered_signal, self.config)
        
        # Extract RR intervals (HRV time series)
        rr_intervals = extract_rr(peaks, self.config)
        rr_intervals_ms = rr_intervals * 1000  # Convert to milliseconds
        
        return {
            'time': time,
            'ppg_signal': ecg_signal,
            'filtered_signal': filtered_signal,
            'peaks': peaks,
            'rr_intervals_ms': rr_intervals_ms
        }
    
    def create_dataset(self, files_data):
        """Create windowed dataset from all files with FILE-LEVEL split."""
        from processing.windowing import create_hrv_windows
        import numpy as np
        
        print(f"\n{'='*60}")
        print("CREATING HRV WINDOWED DATASET")
        print(f"{'='*60}")
        
        # CRITICAL: Split files FIRST, then create windows
        # This prevents data leakage where windows from same file appear in train/val/test
        
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(files_data))
        
        n_files = len(files_data)
        n_train = int(n_files * self.config['train_ratio'])
        n_val = int(n_files * self.config['val_ratio'])
        
        train_file_indices = indices[:n_train]
        val_file_indices = indices[n_train:n_train + n_val]
        test_file_indices = indices[n_train + n_val:]
        
        print(f"\nFile-level split (prevents data leakage):")
        print(f"  - Train files: {len(train_file_indices)}")
        print(f"  - Val files: {len(val_file_indices)}")
        print(f"  - Test files: {len(test_file_indices)}")
        
        # Process each set separately
        train_windows = []
        val_windows = []
        test_windows = []
        file_stats = []
        
        for i, file_info in enumerate(files_data):
            print(f"\nProcessing {i+1}/{len(files_data)}: {file_info['filename']}")
            
            # Extract HRV
            data = self.process_file_to_hrv(file_info['filepath'])
            hrv_series = data['rr_intervals_ms']
            
            print(f"  HRV length: {len(hrv_series)} intervals")
            
            # Create windows
            windows = create_hrv_windows(
                hrv_series=hrv_series,
                window_size=self.config['hrv_window_size'],
                overlap=self.config['hrv_overlap'],
                label=file_info['label']
            )
            
            print(f"  Created {len(windows)} windows")
            
            # Store file metadata
            file_stats.append({
                'filename': file_info['filename'],
                'label': file_info['label'],
                'hrv_length': len(hrv_series),
                'num_windows': len(windows),
                'num_peaks': len(data['peaks'])
            })
            
            # Add file identifier to each window
            for window in windows:
                window['source_file'] = file_info['filename']
                window['file_label'] = file_info['label']
            
            # Add to appropriate set based on file index
            if i in train_file_indices:
                train_windows.extend(windows)
            elif i in val_file_indices:
                val_windows.extend(windows)
            else:  # test
                test_windows.extend(windows)
        
        print(f"\n{'='*60}")
        print(f"Dataset created with FILE-LEVEL split:")
        
        # Count windows by label in each set
        train_af = sum(1 for w in train_windows if w['label'] == 1)
        train_non_af = sum(1 for w in train_windows if w['label'] == 0)
        val_af = sum(1 for w in val_windows if w['label'] == 1)
        val_non_af = sum(1 for w in val_windows if w['label'] == 0)
        test_af = sum(1 for w in test_windows if w['label'] == 1)
        test_non_af = sum(1 for w in test_windows if w['label'] == 0)
        
        print(f"  - Train: {len(train_windows)} windows (AF:{train_af}, Non-AF:{train_non_af})")
        print(f"  - Val: {len(val_windows)} windows (AF:{val_af}, Non-AF:{val_non_af})")
        print(f"  - Test: {len(test_windows)} windows (AF:{test_af}, Non-AF:{test_non_af})")
        
        return {
            'train_windows': train_windows,
            'val_windows': val_windows,
            'test_windows': test_windows,
            'file_stats': file_stats
        }
    
    def extract_features_from_windows(self, windows, verbose=True):
        """Extract features from HRV windows."""
        from scipy.stats import skew, kurtosis
        import numpy as np
        import torch
        
        if verbose:
            print(f"\nExtracting features from {len(windows)} windows...")
        
        features_list = []
        labels_list = []
        
        for i, window in enumerate(windows):
            hrv_window = window['hrv_window']
            
            # We need to extract features from the HRV window
            # Create dummy signal and peaks for the feature extractor
            # The HRV window IS the RR intervals, so we can compute features directly
            
            try:
                # Compute features from RR intervals
                features_dict = {}
                
                # Time-domain features
                features_dict['mean_rr'] = np.mean(hrv_window)
                features_dict['std_rr'] = np.std(hrv_window)
                features_dict['sdnn'] = self.hrv_extractor.extract_SDRR(hrv_window)
                features_dict['rmssd'] = self.hrv_extractor.extract_RMSSD(hrv_window)
                features_dict['pnn50'] = self.hrv_extractor.extract_pNNX(hrv_window)
                features_dict['median_rr'] = np.median(hrv_window)
                features_dict['min_rr'] = np.min(hrv_window)
                features_dict['max_rr'] = np.max(hrv_window)
                features_dict['range_rr'] = np.max(hrv_window) - np.min(hrv_window)
                
                # Statistical features
                features_dict['skewness'] = float(skew(hrv_window))
                features_dict['kurtosis'] = float(kurtosis(hrv_window))
                features_dict['cv'] = np.std(hrv_window) / np.mean(hrv_window) if np.mean(hrv_window) != 0 else 0
                
                # Convert to tensor
                feature_values = list(features_dict.values())
                features_tensor = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                features_list.append(features_tensor)
                labels_list.append(window['label'])
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to extract features from window {i}: {e}")
                continue
        
        if len(features_list) == 0:
            raise ValueError("Failed to extract features from any window")
        
        # Stack all features
        all_features = torch.cat(features_list, dim=0)  # [num_windows, 1, num_features]
        all_labels = torch.tensor(labels_list, dtype=torch.long)
        
        if verbose:
            print(f"Feature extraction complete:")
            print(f"  - Features shape: {all_features.shape}")
            print(f"  - Labels shape: {all_labels.shape}")
        
        return all_features, all_labels
    
    def train_model(self, dataset):
        """Train the FusionModel for classification."""
        from Models.tcn import FusionModel
        from scipy.stats import skew, kurtosis
        
        print(f"\n{'='*60}")
        print("TRAINING MODEL")
        print(f"{'='*60}")
        
        # Extract features from all datasets
        print("\nExtracting features from training set...")
        train_features, train_labels = self.extract_features_from_windows(dataset['train_windows'])
        
        print("\nExtracting features from validation set...")
        val_features, val_labels = self.extract_features_from_windows(dataset['val_windows'])
        
        print("\nExtracting features from test set...")
        test_features, test_labels = self.extract_features_from_windows(dataset['test_windows'])
        
        # Move to device
        train_features = train_features.to(self.config['device'])
        train_labels = train_labels.to(self.config['device'])
        val_features = val_features.to(self.config['device'])
        val_labels = val_labels.to(self.config['device'])
        test_features = test_features.to(self.config['device'])
        test_labels = test_labels.to(self.config['device'])
        
        # Get feature dimension
        num_features = train_features.shape[2]
        print(f"\nFeature dimension: {num_features}")
        
        # Initialize FusionModel
        # FusionModel expects a dictionary of inputs, let's adapt it
        # For simplicity, we'll use a single input source
        input_dims = {'hrv_features': num_features}
        
        self.model = FusionModel(
            input_dims=input_dims,
            embedding_dim=self.config['embedding_dim'],
            tcn_channels=self.config['tcn_channels'],
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout']
        ).to(self.config['device'])
        
        print(f"\nModel architecture:")
        print(f"  - Input dim: {num_features}")
        print(f"  - Embedding dim: {self.config['embedding_dim']}")
        print(f"  - TCN channels: {self.config['tcn_channels']}")
        print(f"  - Num classes: {self.config['num_classes']}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)  # Add L2 regularization
        )
        
        # Training loop
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print(f"Batch size: {self.config['batch_size']}")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Create batches
            num_train = len(train_features)
            indices = torch.randperm(num_train)
            
            for batch_start in range(0, num_train, self.config['batch_size']):
                batch_end = min(batch_start + self.config['batch_size'], num_train)
                batch_indices = indices[batch_start:batch_end]
                
                batch_features = train_features[batch_indices]
                batch_labels = train_labels[batch_indices]
                
                # Forward pass
                # FusionModel expects dict input
                inputs = {'hrv_features': batch_features}
                outputs = self.model(inputs)  # [batch, seq_len, num_classes]
                
                # Take the last timestep's output
                outputs = outputs[:, -1, :]  # [batch, num_classes]
                
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * len(batch_labels)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_labels).sum().item()
                train_total += len(batch_labels)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_start in range(0, len(val_features), self.config['batch_size']):
                    batch_end = min(batch_start + self.config['batch_size'], len(val_features))
                    
                    batch_features = val_features[batch_start:batch_end]
                    batch_labels = val_labels[batch_start:batch_end]
                    
                    inputs = {'hrv_features': batch_features}
                    outputs = self.model(inputs)[:, -1, :]
                    
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item() * len(batch_labels)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_labels).sum().item()
                    val_total += len(batch_labels)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}] "
                      f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Evaluate on test set
        print(f"\n{'='*60}")
        print("EVALUATING ON TEST SET")
        print(f"{'='*60}")
        
        self.model.eval()
        test_correct = 0
        test_total = 0
        test_loss_total = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_start in range(0, len(test_features), self.config['batch_size']):
                batch_end = min(batch_start + self.config['batch_size'], len(test_features))
                
                batch_features = test_features[batch_start:batch_end]
                batch_labels = test_labels[batch_start:batch_end]
                
                inputs = {'hrv_features': batch_features}
                outputs = self.model(inputs)[:, -1, :]
                
                # Calculate test loss
                loss = criterion(outputs, batch_labels)
                test_loss_total += loss.item() * len(batch_labels)
                
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == batch_labels).sum().item()
                test_total += len(batch_labels)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        test_acc = test_correct / test_total
        test_loss = test_loss_total / test_total
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        cm = confusion_matrix(all_labels, all_predictions)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'history': history,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'test_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            },
            'predictions': all_predictions,
            'true_labels': all_labels
        }
    
    def visualize_results(self, training_results, dataset):
        """Create comprehensive visualizations."""
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        history = training_results['history']
        
        # 1. Training curves - show training vs validation
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss: Training vs Validation', 'Accuracy: Training vs Validation')
        )
        
        epochs = list(range(1, len(history['val_loss']) + 1))
        
        # Training loss
        fig.add_trace(go.Scatter(
            x=epochs, y=history['train_loss'],
            mode='lines+markers', name='Train Loss',
            line=dict(color='steelblue', width=2)
        ), row=1, col=1)
        
        # Validation loss
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'],
            mode='lines+markers', name='Val Loss',
            line=dict(color='coral', width=2)
        ), row=1, col=1)
        
        # Training accuracy
        fig.add_trace(go.Scatter(
            x=epochs, y=history['train_acc'],
            mode='lines+markers', name='Train Acc',
            line=dict(color='steelblue', width=2)
        ), row=1, col=2)
        
        # Validation accuracy
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_acc'],
            mode='lines+markers', name='Val Acc',
            line=dict(color='coral', width=2)
        ), row=1, col=2)
       
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        
        fig.update_layout(
            height=500,
            template="plotly_dark",
            title_text='Model Performance (Training vs Validation across epochs)',
            showlegend=True
        )
        
        self.figures['1. Training Progress'] = fig
        
        # 2. Confusion Matrix
        cm = training_results['test_metrics']['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Non-AF', 'AF'],
            y=['Non-AF', 'AF'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix (Test Set)',
            xaxis_title='Predicted',
            yaxis_title='True',
            template="plotly_dark",
            height=500
        )
        
        self.figures['2. Confusion Matrix'] = fig
        
        # 3. Dataset statistics
        file_stats = dataset['file_stats']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Windows per File', 'HRV Length per File')
        )
        
        # Extract file numbers for cleaner labels (e.g., "af_001", "non_af_003")
        filenames = []
        for fs in file_stats:
            fname = fs['filename']
            if '_af_' in fname and '_non_af_' not in fname:
                # Extract AF file number
                num = fname.split('_af_')[1].split('_')[0]
                filenames.append(f"af_{num}")
            elif '_non_af_' in fname:
                # Extract non-AF file number
                num = fname.split('_non_af_')[1].split('_')[0]
                filenames.append(f"non_af_{num}")
            else:
                filenames.append(fname[:15])
        
        colors = ['red' if fs['label'] == 1 else 'blue' for fs in file_stats]
        
        fig.add_trace(go.Bar(
            x=filenames,
            y=[fs['num_windows'] for fs in file_stats],
            marker=dict(color=colors),
            name='Windows',
            showlegend=False,
            hovertemplate='%{x}<br>Windows: %{y}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=filenames,
            y=[fs['hrv_length'] for fs in file_stats],
            marker=dict(color=colors),
            name='HRV Length',
            showlegend=False,
            hovertemplate='%{x}<br>HRV Length: %{y}<extra></extra>'
        ), row=1, col=2)
        
        fig.update_xaxes(tickangle=90, row=1, col=1)
        fig.update_xaxes(tickangle=90, row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_layout(
            height=700,  # Increased height for better visibility
            template="plotly_dark",
            title_text='Dataset Statistics (Red=AF, Blue=Non-AF)'
        )
        
        self.figures['3. Dataset Statistics'] = fig
        
        print("Visualizations created successfully!")


def main():
    """Main entry point for AF classification pipeline."""
    # Initialize pipeline
    pipeline = AFClassificationPipeline()
    
    # Find recordings directory
    recordings_dir = Path(__file__).parent / "recordings"
    
    # Load all files
    files_data = pipeline.load_all_recordings(recordings_dir)
    
    # Create windowed dataset
    dataset = pipeline.create_dataset(files_data)
    
    # Train model
    training_results = pipeline.train_model(dataset)
    
    # Visualize results
    pipeline.visualize_results(training_results, dataset)
    
    # Generate HTML report
    print(f"\n{'='*60}")
    print("GENERATING HTML REPORT")
    print(f"{'='*60}")
    
    html_file = create_tabbed_html(
        figures_dict=pipeline.figures,
        output_path="af_classification_report.html",
        title="AF Classification Report"
    )
    
    print(f"\nReport saved to: {html_file}")
    
    # Open in browser
    open_in_browser(html_file)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

