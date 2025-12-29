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


def main():
    """
    Main entry point for the pipeline.
    """
    # Find all CSV files in recordings folder
    recordings_path = Path(__file__).parent / "recordings"
    csv_files = list(recordings_path.glob("*.csv"))
    
    if len(csv_files) == 0:
        print(f"No CSV files found in {recordings_path}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) in recordings folder:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file.name}")
    
    # Initialize pipeline
    pipeline = HRVPipeline()
    
    # Process first CSV file (or all if you want to loop)
    csv_file = csv_files[0]
    print(f"\nProcessing: {csv_file.name}")
    
    # Run complete pipeline
    results = pipeline.run_pipeline(csv_file, plot=True)
    
    # Generate tabbed HTML report with all visualizations
    print("\n" + "="*60)
    print("Generating interactive visualization report...")
    print("="*60)
    
    html_file = create_tabbed_html(
        figures_dict=pipeline.figures,
        output_path="hrv_analysis_report.html",
        title="HRV Analysis Report"
    )
    
    # Open in browser
    open_in_browser(html_file)


if __name__ == "__main__":
    main()
