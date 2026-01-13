"""
Windowing Module for Signal Processing

This module provides utilities to divide signals into overlapping windows
and extract features from each window for temporal analysis.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional


def create_windows(
    signal: np.ndarray,
    peaks: np.ndarray,
    t_axis: np.ndarray,
    window_size_sec: float,
    overlap: float = 0.5,
    sampling_freq: int = 40,
    min_peaks_per_window: int = 3
) -> List[Dict]:
    """
    Divide a signal into overlapping windows.
    
    Args:
        signal: The preprocessed signal array
        peaks: Array of peak indices in the signal
        t_axis: Time axis corresponding to the signal
        window_size_sec: Window size in seconds
        overlap: Overlap ratio between windows (0.0 to 0.99)
        sampling_freq: Sampling frequency in Hz
        min_peaks_per_window: Minimum number of peaks required per window
        
    Returns:
        List of dictionaries, each containing:
            - 'signal': signal segment for this window
            - 'peaks': peak indices relative to window start
            - 't_axis': time axis for this window
            - 'window_idx': window index
            - 'start_idx': start index in original signal
            - 'end_idx': end index in original signal
            - 'start_time': start time in seconds
            - 'end_time': end time in seconds
    """
    # Calculate window parameters
    window_size_samples = int(window_size_sec * sampling_freq)
    step_size_samples = int(window_size_samples * (1 - overlap))
    
    # Ensure minimum step size
    if step_size_samples < 1:
        step_size_samples = 1
    
    signal_length = len(signal)
    windows = []
    window_idx = 0
    
    # Create windows
    start_idx = 0
    while start_idx + window_size_samples <= signal_length:
        end_idx = start_idx + window_size_samples
        
        # Extract window signal and time axis
        window_signal = signal[start_idx:end_idx]
        window_t_axis = t_axis[start_idx:end_idx]
        
        # Find peaks within this window
        peaks_mask = (peaks >= start_idx) & (peaks < end_idx)
        window_peaks_absolute = peaks[peaks_mask]
        
        # Convert peak indices to be relative to window start
        window_peaks = window_peaks_absolute - start_idx
        
        # Only include windows with sufficient peaks
        if len(window_peaks) >= min_peaks_per_window:
            window_info = {
                'signal': window_signal,
                'peaks': window_peaks,
                't_axis': window_t_axis - window_t_axis[0],  # Normalize to start at 0
                'window_idx': window_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': t_axis[start_idx],
                'end_time': t_axis[end_idx - 1],
                'num_peaks': len(window_peaks)
            }
            windows.append(window_info)
            window_idx += 1
        
        # Move to next window
        start_idx += step_size_samples
    
    # Handle the last partial window if it exists
    if start_idx < signal_length and signal_length - start_idx >= window_size_samples // 2:
        end_idx = signal_length
        window_signal = signal[start_idx:end_idx]
        window_t_axis = t_axis[start_idx:end_idx]
        
        peaks_mask = (peaks >= start_idx) & (peaks < end_idx)
        window_peaks_absolute = peaks[peaks_mask]
        window_peaks = window_peaks_absolute - start_idx
        
        if len(window_peaks) >= min_peaks_per_window:
            window_info = {
                'signal': window_signal,
                'peaks': window_peaks,
                't_axis': window_t_axis - window_t_axis[0],
                'window_idx': window_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': t_axis[start_idx],
                'end_time': t_axis[end_idx - 1],
                'num_peaks': len(window_peaks)
            }
            windows.append(window_info)
    
    return windows


def extract_features_from_windows(
    windows: List[Dict],
    hrv_extractor,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[torch.Tensor, List[np.ndarray], List[Dict]]:
    """
    Extract features from all windows.
    
    Args:
        windows: List of window dictionaries from create_windows()
        hrv_extractor: Instance of HRVFeatureExtractor
        device: Device for tensor operations ('cpu' or 'cuda')
        verbose: Whether to print progress information
        
    Returns:
        Tuple of:
            - features_tensor: Tensor of shape [num_windows, 1, num_features]
            - rr_intervals_list: List of RR interval arrays for each window
            - window_metadata: List of metadata dicts for each window
    """
    if len(windows) == 0:
        raise ValueError("No valid windows provided for feature extraction")
    
    feature_tensors = []
    rr_intervals_list = []
    window_metadata = []
    
    if verbose:
        print(f"\nExtracting features from {len(windows)} windows...")
    
    for i, window in enumerate(windows):
        try:
            # Extract features for this window
            features_tensor, rr_intervals_ms = hrv_extractor.extract_all_features(
                signal=window['signal'],
                peak_indices=window['peaks'],
                t_axis=window['t_axis'],
                device=device
            )
            
            feature_tensors.append(features_tensor)
            rr_intervals_list.append(rr_intervals_ms)
            
            # Store metadata
            metadata = {
                'window_idx': window['window_idx'],
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'num_peaks': window['num_peaks'],
                'num_features': features_tensor.shape[2]
            }
            window_metadata.append(metadata)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(windows)} windows...")
                
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to extract features from window {window['window_idx']}: {e}")
            continue
    
    if len(feature_tensors) == 0:
        raise ValueError("Failed to extract features from any window")
    
    # Stack all feature tensors along the batch dimension
    # Each tensor is [1, 1, num_features], stack to [num_windows, 1, num_features]
    all_features = torch.cat(feature_tensors, dim=0)
    
    if verbose:
        print(f"Successfully extracted features from {len(feature_tensors)} windows")
        print(f"Feature tensor shape: {all_features.shape}")
    
    return all_features, rr_intervals_list, window_metadata


def get_window_statistics(windows: List[Dict]) -> Dict:
    """
    Calculate statistics about the windows.
    
    Args:
        windows: List of window dictionaries
        
    Returns:
        Dictionary with window statistics
    """
    if len(windows) == 0:
        return {
            'num_windows': 0,
            'avg_peaks_per_window': 0,
            'min_peaks': 0,
            'max_peaks': 0,
            'total_duration': 0
        }
    
    peak_counts = [w['num_peaks'] for w in windows]
    durations = [w['end_time'] - w['start_time'] for w in windows]
    
    stats = {
        'num_windows': len(windows),
        'avg_peaks_per_window': np.mean(peak_counts),
        'min_peaks': np.min(peak_counts),
        'max_peaks': np.max(peak_counts),
        'std_peaks': np.std(peak_counts),
        'avg_duration': np.mean(durations),
        'total_duration': windows[-1]['end_time'] - windows[0]['start_time']
    }
    
    return stats


def visualize_windows(
    signal: np.ndarray,
    t_axis: np.ndarray,
    windows: List[Dict],
    max_windows_to_show: int = 5
):
    """
    Visualize the signal divided into windows.
    
    Args:
        signal: Original signal array
        t_axis: Time axis
        windows: List of window dictionaries
        max_windows_to_show: Maximum number of windows to highlight
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the full signal
    ax.plot(t_axis, signal, 'b-', alpha=0.5, linewidth=0.8, label='Signal')
    
    # Highlight windows with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, min(len(windows), max_windows_to_show)))
    
    for i, window in enumerate(windows[:max_windows_to_show]):
        start_t = window['start_time']
        end_t = window['end_time']
        
        # Shade the window region
        ax.axvspan(start_t, end_t, alpha=0.2, color=colors[i], 
                   label=f'Window {i+1} ({window["num_peaks"]} peaks)')
        
        # Mark peaks in this window
        peak_times = window['t_axis'][window['peaks']] + window['start_time']
        peak_values = window['signal'][window['peaks']]
        ax.plot(peak_times, peak_values, 'r*', markersize=8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Signal Windowing (showing {min(len(windows), max_windows_to_show)} of {len(windows)} windows)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_hrv_windows(
    hrv_series: np.ndarray,
    window_size: int = 200,
    overlap: int = 20,
    label: int = 0
) -> List[Dict]:
    """
    Divide HRV time series (RR intervals) into overlapping windows.
    
    Args:
        hrv_series: Array of RR intervals (HRV time series)
        window_size: Number of RR intervals per window (default: 200)
        overlap: Number of overlapping RR intervals between windows (default: 20)
        label: Label for all windows (0 for non-AF, 1 for AF)
        
    Returns:
        List of dictionaries, each containing:
            - 'hrv_window': HRV values for this window
            - 'label': Label (0 or 1)
            - 'window_idx': Window index
            - 'start_idx': Start index in original HRV series
            - 'end_idx': End index in original HRV series
    """
    if len(hrv_series) < window_size:
        # Not enough data for even one window
        return []
    
    step_size = window_size - overlap
    
    # Ensure minimum step size
    if step_size < 1:
        step_size = 1
    
    windows = []
    window_idx = 0
    start_idx = 0
    
    # Create windows
    while start_idx + window_size <= len(hrv_series):
        end_idx = start_idx + window_size
        
        window_info = {
            'hrv_window': hrv_series[start_idx:end_idx],
            'label': label,
            'window_idx': window_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'window_size': window_size
        }
        windows.append(window_info)
        window_idx += 1
        
        # Move to next window
        start_idx += step_size
    
    return windows


def split_windows(
    windows: List[Dict],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split windows into train, validation, and test sets.
    
    Args:
        windows: List of window dictionaries
        train_ratio: Ratio of windows for training
        val_ratio: Ratio of windows for validation
        test_ratio: Ratio of windows for testing
        shuffle: Whether to shuffle windows before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_windows, val_windows, test_windows)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    if len(windows) == 0:
        return [], [], []
    
    # Shuffle if requested
    if shuffle:
        np.random.seed(random_seed)
        indices = np.random.permutation(len(windows))
        windows = [windows[i] for i in indices]
    
    # Calculate split indices
    n_total = len(windows)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split windows
    train_windows = windows[:n_train]
    val_windows = windows[n_train:n_train + n_val]
    test_windows = windows[n_train + n_val:]
    
    return train_windows, val_windows, test_windows


def visualize_feature_heatmap(
    features_tensor: torch.Tensor,
    window_metadata: List[Dict],
    feature_names: Optional[List[str]] = None,
    max_features_to_show: int = 30
):
    """
    Visualize features across windows as a heatmap.
    
    Args:
        features_tensor: Tensor of shape [num_windows, 1, num_features]
        window_metadata: List of metadata for each window
        feature_names: Optional list of feature names
        max_features_to_show: Maximum number of features to display
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy and squeeze
    features = features_tensor.squeeze(1).cpu().numpy()  # [num_windows, num_features]
    num_windows, num_features = features.shape
    
    # Limit features if too many
    if num_features > max_features_to_show:
        features = features[:, :max_features_to_show]
        num_features = max_features_to_show
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    im = ax.imshow(features.T, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)
    ax.set_title('Feature Values Across Windows', fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(range(num_windows))
    ax.set_xticklabels([f"{i}" for i in range(num_windows)])
    
    if feature_names and len(feature_names) == num_features:
        ax.set_yticks(range(num_features))
        ax.set_yticklabels(feature_names, fontsize=8)
    else:
        ax.set_yticks(range(0, num_features, 5))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feature Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()
