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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import processing modules
from processing.preprocessing import butter_bandpass_filter, extract_rr
from processing.feature_extraction.feature_pipeline import extract_all_features_from_rr, features_dict_to_tensor, get_feature_group_indices
from processing.windowing import create_hrv_windows
from visualizations.html_viewer import create_tabbed_html, open_in_browser

# Import model modules
from Models.tcn import FusionModel
from Models.group_lasso import GroupLassoRegularizer


class ZebrafishPipeline:
    """
    Complete pipeline for AF classification from PPG recordings.
    """
    
    def __init__(self, config=None):
        """Initialize the classification pipeline."""
        # Default configuration
        self.config = {
            # Signal processing
            "sampling_freq": 40,  # Zebrafish data is sampled at 40 Hz
            "low_threshold": 0.5,
            "high_threshold": 5.0,
            "filter_order": 4,
            
            # Peak detection params for extract_rr
            "d": 12,  # min distance between peaks (samples) = 0.3s * 40Hz
            "p": 0.1,  # prominence threshold
            "hz": 40,  # sampling frequency
            
            # HRV windowing
            "hrv_window_size": 200,
            "hrv_overlap": 180,
            
            # Feature extraction parameters
            "samp_en_m": 2,
            "samp_en_r": 0.2,
            "scales": 10,
            "dfa_min_scale": 4,
            "dfa_num_scales": 20,
            "wavelet_num_scales": 12,
            
            # Model Inputs (will be auto-computed based on extracted features)
            "input_dims": None,  # Will be set after feature extraction
            
            # Model architecture
            "embedding_dim": 16,  # Reduced from 32
            "tcn_channels": [16, 32],  # Simplified: 2 layers, fewer channels
            "tcn_kernel_size": 3,
            "dropout": 0.5,  # Increased dropout
            "num_classes": 2,
            
            # Training
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.0003,  # Slower learning rate
            "weight_decay": 0.05,  # Stronger L2 regularization
            "group_lasso_lambda": 0.01,  # Group lasso regularization strength
            "early_stopping_patience": 20,
            
            # Train/val/test split ratios
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            
            # Device
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        if config is not None:
            self.config.update(config)
        
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
        # Target specific raw data file 'Values.csv' to avoid duplicates
        csv_files = sorted(list(recordings_path.rglob("Values.csv")))
        
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {recordings_path}")
        
        files_data = []
        propranolol_count = 0
        control_count = 0
        
        for csv_file in csv_files:
            filename = csv_file.name
            filepath = str(csv_file).replace('\\', '/')
            
            # Determine label from folder name
            if "Control" in filepath:
                label = 0 
                control_count += 1
            elif "Propranolol" in filepath:
                label = 1  
                propranolol_count += 1
            else:
                print(f"Skipping {filename} - label specific identifier not found")
                continue
            
            files_data.append({
                'filepath': csv_file,
                'filename': filename,
                'label': label
            })
        
        print(f"Loaded {len(files_data)} files:")
        print(f"  - Propranolol files: {propranolol_count}")
        print(f"  - Control files: {control_count}")
        
        return files_data
    
    def process_file_to_hrv(self, filepath):
        """Process a single CSV file to extract HRV time series."""
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Extract Time and Signal
        # Handle Zebrafish 'Values.csv' format (Frame, Mean)
        if 'Frame' in df.columns and 'Mean' in df.columns:
            ecg_signal = df['Mean'].values
            # Generate time from frame number and sampling frequency
            frames = df['Frame'].values
            time = frames / self.config['sampling_freq']
            
        # Handle Standard format (Time, PPG)
        elif 'Time' in df.columns and 'PPG' in df.columns:
            time = df['Time'].values
            ecg_signal = df['PPG'].values
            
        else:
            raise ValueError(f"CSV file must contain ('Frame', 'Mean') or ('Time', 'PPG') columns. Found: {df.columns.tolist()}")
        
        # Preprocess signal with bandpass filter
        filtered_signal = butter_bandpass_filter(
            ecg_signal,
            lowcut=self.config['low_threshold'],
            highcut=self.config['high_threshold'],
            fs=self.config['sampling_freq'],
            order=self.config['filter_order']
        )
        
        # Extract RR intervals (already in milliseconds)
        rr_intervals_ms = extract_rr(filtered_signal, self.config)
        
        return {
            'time': time,
            'ppg_signal': ecg_signal,
            'filtered_signal': filtered_signal,
            'rr_intervals_ms': rr_intervals_ms
        }
    
    def create_dataset(self, files_data):
        """Create windowed dataset from all files with FILE-LEVEL split."""
        
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
        
        n_train_c = int(n_control * self.config['train_ratio'])
        n_val_c = int(n_control * self.config['val_ratio'])
        
        n_train_p = int(n_prop * self.config['train_ratio'])
        n_val_p = int(n_prop * self.config['val_ratio'])
        
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
    
    def extract_features_from_windows(self, windows, verbose=True):
        """
        Extract comprehensive features from windows (time/frequency/non-linear).
        Returns tensor of shape [Num_Windows, 1, Num_Features].
        """
        features_list = []
        labels_list = []
        
        # Get feature extraction parameters
        feature_params = {
            "samp_en_m": self.config.get("samp_en_m", 2),
            "samp_en_r": self.config.get("samp_en_r", 0.2),
            "scales": self.config.get("scales", 10),
            "dfa_min_scale": self.config.get("dfa_min_scale", 4),
            "dfa_num_scales": self.config.get("dfa_num_scales", 20),
            "wavelet_num_scales": self.config.get("wavelet_num_scales", 12),
            "upsample_factor": self.config.get("upsample_factor", 1),
            "use_octave_scales": self.config.get("use_octave_scales", False)
        }
        
        if verbose:
            print(f"\nExtracting features from {len(windows)} windows...")
        
        for i, window in enumerate(windows):
            hrv_window = window['hrv_window']
            
            # Extract all features from this window
            try:
                features_dict = extract_all_features_from_rr(
                    hrv_window,
                    sampling_freq=self.config['sampling_freq'],
                    params=feature_params
                )
                
                # Convert to tensor: [1, 1, num_features]
                feature_tensor = features_dict_to_tensor(features_dict, device=self.config['device'])
                
                features_list.append(feature_tensor)
                labels_list.append(window['label'])
                
            except Exception as e:
                if verbose and i < 5:  # Only print first few errors to avoid spam
                    print(f"  Warning: Failed to extract features from window {i}: {e}")
                continue
            
            if verbose and (i+1) % 500 == 0:
                print(f"  Processed {i+1}/{len(windows)} windows...")
        
        if not features_list:
            return None, None
        
        # Stack all features
        batch_features = torch.cat(features_list, dim=0).to(self.config['device'])  # [num_windows, 1, num_features]
        batch_labels = torch.tensor(labels_list, dtype=torch.long).to(self.config['device'])
        
        if verbose:
            print(f"\nFeature extraction complete:")
            print(f"  - Total windows: {len(features_list)}")
            print(f"  - Feature tensor shape: {batch_features.shape}")
            print(f"  - Number of features per window: {batch_features.shape[2]}")
        
        return batch_features, batch_labels
    
    def train_model(self, dataset):
        """Train the FusionModel for classification."""
        
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
        
        # Auto-detect feature dimension from extracted features
        num_features = train_features.shape[2]  # Shape is [num_windows, 1, num_features]
        print(f"\nAuto-detected feature dimension: {num_features}")
        
        # ==========================================
        # FEATURE SCALING (Standardization)
        # ==========================================
        print("\nApplying StandardScaler (Fit on Train, Transform All)...")
        
        # Reshape to 2D for scaler: [N, 1, F] -> [N, F]
        train_flat = train_features.squeeze(1).cpu().numpy()
        val_flat = val_features.squeeze(1).cpu().numpy()
        test_flat = test_features.squeeze(1).cpu().numpy()
        
        # Initialize and fit scaler
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_flat)
        
        # Transform val and test (using train stats)
        val_scaled = scaler.transform(val_flat)
        test_scaled = scaler.transform(test_flat)
        
        # Convert back to tensor and reshape to [N, 1, F]
        train_features = torch.tensor(train_scaled, dtype=torch.float32).unsqueeze(1).to(self.config['device'])
        val_features = torch.tensor(val_scaled, dtype=torch.float32).unsqueeze(1).to(self.config['device'])
        test_features = torch.tensor(test_scaled, dtype=torch.float32).unsqueeze(1).to(self.config['device'])
        
        print(f"Features scaled. Mean: {scaler.mean_.mean():.4f}, Var: {scaler.var_.mean():.4f}")

        
        # Update config with detected input dimensions
        self.config['input_dims'] = {'comprehensive_features': num_features}
        
        # Initialize FusionModel with auto-detected dimensions
        input_dims = {'comprehensive_features': num_features}
        
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
        print(f"  - Dropout: {self.config['dropout']}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)  # Add L2 regularization
        )
        
        # Initialize Group Lasso regularizer
        group_lasso_lambda = self.config.get('group_lasso_lambda', 0.01)
        if group_lasso_lambda > 0:
            feature_group_indices = get_feature_group_indices()
            group_lasso_reg = GroupLassoRegularizer(
                group_indices=feature_group_indices,
                lambda_reg=group_lasso_lambda
            )
            print(f"\nGroup Lasso regularization enabled (lambda={group_lasso_lambda})")
            print(f"  Feature groups: {list(feature_group_indices.keys())}")
            for name, indices in feature_group_indices.items():
                print(f"    - {name}: {len(indices)} features")
        else:
            group_lasso_reg = None
        
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
                
                # Forward pass - FusionModel expects dict input
                inputs = {'comprehensive_features': batch_features}
                outputs = self.model(inputs)  # [batch, seq_len, num_classes]
                
                # Take the last timestep's output (seq_len=1 for static features)
                outputs = outputs[:, -1, :]  # [batch, num_classes]
                
                ce_loss = criterion(outputs, batch_labels)
                
                # Add group lasso penalty
                if group_lasso_reg is not None:
                    group_lasso_penalty = group_lasso_reg.penalty_from_model(self.model)
                    loss = ce_loss + group_lasso_penalty
                else:
                    loss = ce_loss
                
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
                    
                    inputs = {'comprehensive_features': batch_features}
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
                
                inputs = {'comprehensive_features': batch_features}
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
            x=['Control', 'Propranolol'],
            y=['Control', 'Propranolol'],
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
        
        # Use simple file index for labels
        filenames = [f"File {i+1}" for i in range(len(file_stats))]
        
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
            height=700,
            template="plotly_dark",
            title_text='Dataset Statistics (Red=Propranolol, Blue=Control)'
        )
        
        self.figures['3. Dataset Statistics'] = fig
        
        print("Visualizations created successfully!")


def main():
    """Main entry point for Zebrafish classification pipeline."""
    # Initialize pipeline
    pipeline = ZebrafishPipeline()
    
    # Find recordings directory
    recordings_dir = Path(__file__).parent / "recordings" / "propanolol_zebrafish"
    
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
        output_path="zebrafish_classification_report.html",
        title="Zebrafish Classification Report"
    )
    
    print(f"\nReport saved to: {html_file}")
    
    # Open in browser
    open_in_browser(html_file)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

