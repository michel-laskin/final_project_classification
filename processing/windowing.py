"""
Windowing Module for Signal Processing

This module provides utilities to divide signals into overlapping windows
and extract features from each window for temporal analysis.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
from processing.preprocessing import process_file_to_hrv






def create_hrv_windows(
    hrv_series: np.ndarray,
    window_size: int = 200,
    overlap: int = 20,
    label: int = 0,
    jitter_range: int = 0
) -> List[Dict]:
    """
    Divide HRV time series (RR intervals) into overlapping windows.
    
    Args:
        hrv_series: Array of RR intervals (HRV time series)
        window_size: Number of RR intervals per window (default: 200)
        overlap: Number of overlapping RR intervals between windows (default: 20)
        label: Label for all windows (0 for non-AF, 1 for AF)
        jitter_range: Maximum random shift (±) applied to window start for augmentation (default: 0)
        
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
        # Apply jitter augmentation: randomly shift window start within bounds
        if jitter_range > 0:
            jitter = np.random.randint(-jitter_range, jitter_range + 1)
            jittered_start = max(0, min(start_idx + jitter, len(hrv_series) - window_size))
        else:
            jittered_start = start_idx
        
        end_idx = jittered_start + window_size
        
        window_info = {
            'hrv_window': hrv_series[jittered_start:end_idx],
            'label': label,
            'window_idx': window_idx,
            'start_idx': jittered_start,
            'end_idx': end_idx,
            'window_size': window_size
        }
        windows.append(window_info)
        window_idx += 1
        
        # Move to next window (based on original step, not jittered position)
        start_idx += step_size
    
    return windows


def create_windowed_dataset(files_data: List[Dict], config: Dict) -> Dict:
    """
    Create windowed dataset from all files with FILE-LEVEL split.
    
    Args:
        files_data: List of file metadata dictionaries.
        config: Configuration dictionary.
        
    Returns:
        Dictionary containing train/val/test windows and stats.
    """
    print(f"\n{'='*60}")
    print("CREATING HRV WINDOWED DATASET")
    print(f"{'='*60}")
    
    # Split files FIRST, then create windows
    # This prevents data leakage where windows from same file appear in train/val/test
    
    np.random.seed(42)  # For reproducibility
    
    # Stratified Split: Separate files by label to ensure balanced sets
    control_indices = [i for i, f in enumerate(files_data) if f['label'] == 0]
    propranolol_indices = [i for i, f in enumerate(files_data) if f['label'] == 1]
    
    # Shuffle each set
    np.random.shuffle(control_indices)
    np.random.shuffle(propranolol_indices)
    
    # Calculate split sizes for each class
    n_control = len(control_indices)
    n_prop = len(propranolol_indices)
    
    n_train_c = int(n_control * config['train_ratio'])
    n_val_c = int(n_control * config['val_ratio'])
    
    n_train_p = int(n_prop * config['train_ratio'])
    n_val_p = int(n_prop * config['val_ratio'])
    
    # Create splits per class
    train_file_indices = control_indices[:n_train_c] + propranolol_indices[:n_train_p]
    val_file_indices = control_indices[n_train_c:n_train_c + n_val_c] + propranolol_indices[n_train_p:n_train_p + n_val_p]
    test_file_indices = control_indices[n_train_c + n_val_c:] + propranolol_indices[n_train_p + n_val_p:]
    
    # Shuffle final sets to mix classes
    np.random.shuffle(train_file_indices)
    np.random.shuffle(val_file_indices)
    np.random.shuffle(test_file_indices)
    
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
        data = process_file_to_hrv(file_info['filepath'], config)
        hrv_series = data['rr_intervals_ms']
        
        print(f"  HRV length: {len(hrv_series)} intervals")
        
        # Create windows
        windows = create_hrv_windows(
            hrv_series=hrv_series,
            window_size=config['hrv_window_size'],
            overlap=config['hrv_overlap'],
            label=file_info['label']
        )
        
        print(f"  Created {len(windows)} windows")
        
        # Store file metadata
        file_stats.append({
            'filename': file_info['filename'],
            'label': file_info['label'],
            'hrv_length': len(hrv_series),
            'num_windows': len(windows),
            'num_rr_intervals': len(hrv_series)
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
    train_prop = sum(1 for w in train_windows if w['label'] == 1)
    train_ctrl = sum(1 for w in train_windows if w['label'] == 0)
    val_prop = sum(1 for w in val_windows if w['label'] == 1)
    val_ctrl = sum(1 for w in val_windows if w['label'] == 0)
    test_prop = sum(1 for w in test_windows if w['label'] == 1)
    test_ctrl = sum(1 for w in test_windows if w['label'] == 0)
    
    print(f"  - Train: {len(train_windows)} windows (Propranolol:{train_prop}, Control:{train_ctrl})")
    print(f"  - Val: {len(val_windows)} windows (Propranolol:{val_prop}, Control:{val_ctrl})")
    print(f"  - Test: {len(test_windows)} windows (Propranolol:{test_prop}, Control:{test_ctrl})")
    
    return {
        'train_windows': train_windows,
        'val_windows': val_windows,
        'test_windows': test_windows,
        'file_stats': file_stats
    }
