"""
Multiclass Data Loader for 13-Class Zebrafish Drug Classification

Handles:
- Automatic class discovery from dataset folder structure
- Exclusion of specified folders
- Loading from Values.csv with full preprocessing pipeline
- Stratified file-level train/val/test splitting
- Class weight computation for weighted CrossEntropyLoss
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from processing.preprocessing import process_file_to_hrv
from processing.windowing import create_hrv_windows


# Folder to exclude from dataset
EXCLUDED_FOLDERS = [
    "4_Carbachol 100 µM - first 5 minutes after washout",
]


def discover_classes(dataset_dir: str) -> Dict[str, int]:
    """
    Discover class folders and create name→index mapping.
    
    Args:
        dataset_dir: Path to the dataset root directory.
        
    Returns:
        Dictionary mapping folder names to class indices (sorted alphabetically).
    """
    dataset_path = Path(dataset_dir)
    class_folders = sorted([
        d.name for d in dataset_path.iterdir()
        if d.is_dir() and d.name not in EXCLUDED_FOLDERS
    ])
    
    class_map = {name: idx for idx, name in enumerate(class_folders)}
    return class_map


def load_all_files(dataset_dir: str, class_map: Optional[Dict[str, int]] = None) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Load all CSV files from the dataset directory and assign labels.
    
    Args:
        dataset_dir: Path to the dataset root directory.
        class_map: Optional pre-defined class mapping. If None, auto-discovered.
        
    Returns:
        Tuple of (files_data list, class_map dict).
    """
    dataset_path = Path(dataset_dir)
    
    if class_map is None:
        class_map = discover_classes(dataset_dir)
    
    files_data = []
    class_counts = {name: 0 for name in class_map}
    
    for class_name, class_idx in class_map.items():
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Find all Values.csv files recursively
        csv_files = sorted(list(class_dir.rglob("Values.csv")))
        
        for csv_file in csv_files:
            files_data.append({
                'filepath': csv_file,
                'filename': str(csv_file.relative_to(dataset_path)),
                'label': class_idx,
                'class_name': class_name,
            })
            class_counts[class_name] += 1
    
    print(f"\nLoaded {len(files_data)} files across {len(class_map)} classes:")
    for name, count in class_counts.items():
        print(f"  [{class_map[name]:2d}] {name}: {count} files")
    
    return files_data, class_map


def compute_class_weights(files_data: List[Dict], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted CrossEntropyLoss.
    
    Args:
        files_data: List of file metadata dicts with 'label' key.
        num_classes: Total number of classes.
        
    Returns:
        Tensor of shape [num_classes] with normalized weights.
    """
    class_counts = np.zeros(num_classes)
    for f in files_data:
        class_counts[f['label']] += 1
    
    # Inverse frequency weighting
    weights = 1.0 / np.maximum(class_counts, 1)
    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)


def create_stratified_split(
    files_data: List[Dict],
    num_classes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified file-level train/val/test split indices.
    
    Args:
        files_data: List of file metadata dicts.
        num_classes: Total number of classes.
        train_ratio: Fraction of files for training.
        val_ratio: Fraction of files for validation.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    rng = np.random.RandomState(seed)
    
    # Group file indices by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, f in enumerate(files_data):
        class_indices[f['label']].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        rng.shuffle(indices)
        
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Shuffle final sets
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    
    return train_indices, val_indices, test_indices


def extract_windows_from_files(
    files_data: List[Dict],
    file_indices: List[int],
    config: Dict,
    verbose: bool = True
) -> List[Dict]:
    """
    Process files and extract windowed HRV data.
    
    Args:
        files_data: List of all file metadata.
        file_indices: Indices of files to process.
        config: Pipeline configuration dict.
        verbose: Whether to print progress.
        
    Returns:
        List of window dictionaries.
    """
    all_windows = []
    
    for i, file_idx in enumerate(file_indices):
        file_info = files_data[file_idx]
        
        try:
            # Process file through preprocessing pipeline
            data = process_file_to_hrv(file_info['filepath'], config)
            hrv_series = data['rr_intervals_ms']
            
            # Create windows
            windows = create_hrv_windows(
                hrv_series=hrv_series,
                window_size=config['hrv_window_size'],
                overlap=config.get('hrv_overlap', 180),
                label=file_info['label']
            )
            
            # Add metadata to each window
            for w in windows:
                w['source_file'] = file_info['filename']
                w['class_name'] = file_info['class_name']
            
            all_windows.extend(windows)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(file_indices)} files, "
                      f"{len(all_windows)} windows so far...")
                
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to process {file_info['filename']}: {e}")
            continue
    
    return all_windows


def prepare_full_dataset(
    dataset_dir: str,
    config: Dict,
    verbose: bool = True
) -> Dict:
    """
    Complete data preparation pipeline.
    
    Args:
        dataset_dir: Path to dataset root.
        config: Pipeline configuration.
        verbose: Print progress.
        
    Returns:
        Dictionary with train/val/test windows, class_map, class_weights, file_stats.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("LOADING 13-CLASS DATASET")
        print(f"{'='*60}")
    
    # Load files
    files_data, class_map = load_all_files(dataset_dir)
    num_classes = len(class_map)
    
    # Compute class weights
    class_weights = compute_class_weights(files_data, num_classes)
    if verbose:
        print(f"\nClass weights: {class_weights.tolist()}")
    
    # Stratified split
    train_idx, val_idx, test_idx = create_stratified_split(
        files_data, num_classes,
        train_ratio=config.get('train_ratio', 0.6),
        val_ratio=config.get('val_ratio', 0.2),
        seed=42
    )
    
    if verbose:
        print(f"\nFile-level split:")
        print(f"  Train: {len(train_idx)} files")
        print(f"  Val:   {len(val_idx)} files")
        print(f"  Test:  {len(test_idx)} files")
    
    # Extract windows
    if verbose:
        print(f"\nExtracting windows (window_size={config['hrv_window_size']}, "
              f"overlap={config.get('hrv_overlap', 180)})...")
    
    train_windows = extract_windows_from_files(files_data, train_idx, config, verbose)
    val_windows = extract_windows_from_files(files_data, val_idx, config, verbose)
    test_windows = extract_windows_from_files(files_data, test_idx, config, verbose)
    
    if verbose:
        print(f"\nWindowed dataset:")
        print(f"  Train: {len(train_windows)} windows")
        print(f"  Val:   {len(val_windows)} windows")
        print(f"  Test:  {len(test_windows)} windows")
    
    return {
        'files_data': files_data,
        'class_map': class_map,
        'class_weights': class_weights,
        'num_classes': num_classes,
        'train_windows': train_windows,
        'val_windows': val_windows,
        'test_windows': test_windows,
        'train_file_indices': train_idx,
        'val_file_indices': val_idx,
        'test_file_indices': test_idx,
    }
