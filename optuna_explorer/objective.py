"""
Optuna Objective Function for HRV Classification Pipeline

Wraps the complete training pipeline into an objective function
that Optuna can optimize. Supports single-objective (val accuracy or val F1)
and multi-objective (both) optimization.

Includes feature caching keyed on (window_size, overlap) to avoid
redundant recomputation.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import optuna
from typing import Dict, Any, Optional, Callable, Tuple
import hashlib
import time

from processing.feature_extraction.feature_pipeline import (
    extract_all_features_from_rr,
    features_dict_to_tensor,
    get_feature_group_indices,
)
from Models.configurable_encoder import ConfigurableFeatureEncoder
from Models.tcn import TemporalConvNet
from Models.group_lasso import GroupLassoRegularizer

from optuna_explorer.search_space import sample_params, build_tcn_channels
from optuna_explorer.data_loader import (
    load_all_files,
    compute_class_weights,
    create_stratified_split,
    extract_windows_from_files,
)


class ConfigurableFusionModel(nn.Module):
    """
    FusionModel variant that uses ConfigurableFeatureEncoder.
    Built dynamically per trial based on sampled architecture hyperparameters.
    """
    
    def __init__(self, input_dim, embedding_dim, tcn_channels,
                 num_classes, encoder_num_layers=0, encoder_hidden_dim=64,
                 encoder_activation='silu', tcn_kernel_size=3, dropout=0.1):
        super().__init__()
        
        self.encoder = ConfigurableFeatureEncoder(
            input_dim=input_dim,
            output_dim=embedding_dim,
            num_hidden_layers=encoder_num_layers,
            hidden_dim=encoder_hidden_dim,
            activation=encoder_activation,
            dropout=dropout,
        )
        
        self.tcn = TemporalConvNet(
            num_inputs=embedding_dim,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
        
        self.classifier = nn.Linear(tcn_channels[-1], num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, input_dim]
        Returns:
            Logits of shape [batch, seq_len, num_classes]
        """
        # Encode features
        encoded = self.encoder(x)  # [batch, seq_len, embedding_dim]
        
        # TCN expects [batch, channels, seq_len]
        encoded_t = encoded.transpose(1, 2)
        tcn_out = self.tcn(encoded_t)  # [batch, channels, seq_len]
        
        # Classify
        tcn_out_t = tcn_out.transpose(1, 2)  # [batch, seq_len, channels]
        logits = self.classifier(tcn_out_t)  # [batch, seq_len, num_classes]
        
        return logits


class FeatureCache:
    """
    Caches extracted features keyed by (window_size, overlap) to avoid
    redundant feature extraction when only model/training params change.
    """
    
    def __init__(self, max_entries: int = 50):
        self._cache = {}
        self._max_entries = max_entries
    
    def _make_key(self, window_size: int, overlap: int) -> str:
        return f"{window_size}_{overlap}"
    
    def get(self, window_size: int, overlap: int):
        key = self._make_key(window_size, overlap)
        return self._cache.get(key, None)
    
    def put(self, window_size: int, overlap: int, data: Dict):
        key = self._make_key(window_size, overlap)
        if len(self._cache) >= self._max_entries:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = data
    
    def clear(self):
        self._cache.clear()


def _extract_features_from_windows(windows, config, device='cpu'):
    """Extract features from window list and return tensor + labels."""
    features_list = []
    labels_list = []
    
    feature_params = {
        "samp_en_m": config.get("samp_en_m", 2),
        "samp_en_r": config.get("samp_en_r", 0.2),
        "scales": config.get("scales", 10),
        "dfa_min_scale": config.get("dfa_min_scale", 4),
        "dfa_num_scales": config.get("dfa_num_scales", 20),
        "wavelet_num_scales": config.get("wavelet_num_scales", 12),
        "upsample_factor": config.get("upsample_factor", 1),
        "use_octave_scales": config.get("use_octave_scales", False),
    }
    
    for window in windows:
        hrv_window = window['hrv_window']
        try:
            features_dict = extract_all_features_from_rr(
                hrv_window,
                sampling_freq=config.get('sampling_freq', 40),
                params=feature_params,
            )
            feature_tensor = features_dict_to_tensor(features_dict, device='cpu')
            features_list.append(feature_tensor)
            labels_list.append(window['label'])
        except Exception:
            continue
    
    if not features_list:
        return None, None
    
    batch_features = torch.cat(features_list, dim=0)  # [N, 1, F]
    batch_labels = torch.tensor(labels_list, dtype=torch.long)
    
    return batch_features, batch_labels


def _prepare_features(
    files_data, train_idx, val_idx, test_idx,
    config, feature_cache, window_size, overlap
):
    """
    Extract features or retrieve from cache.
    Returns scaled feature tensors for train/val/test.
    """
    cached = feature_cache.get(window_size, overlap)
    if cached is not None:
        return cached
    
    # Update config with windowing params
    cfg = dict(config)
    cfg['hrv_window_size'] = window_size
    cfg['hrv_overlap'] = overlap
    
    # Extract windows from files
    train_windows = extract_windows_from_files(files_data, train_idx, cfg, verbose=False)
    val_windows = extract_windows_from_files(files_data, val_idx, cfg, verbose=False)
    test_windows = extract_windows_from_files(files_data, test_idx, cfg, verbose=False)
    
    # Extract features
    train_feats, train_labels = _extract_features_from_windows(train_windows, cfg)
    val_feats, val_labels = _extract_features_from_windows(val_windows, cfg)
    test_feats, test_labels = _extract_features_from_windows(test_windows, cfg)
    
    if train_feats is None or val_feats is None:
        return None
    
    # Scale features (fit on train, transform all)
    train_flat = train_feats.squeeze(1).numpy()
    val_flat = val_feats.squeeze(1).numpy()
    test_flat = test_feats.squeeze(1).numpy() if test_feats is not None else None
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat)
    val_scaled = scaler.transform(val_flat)
    test_scaled = scaler.transform(test_flat) if test_flat is not None else None
    
    result = {
        'train_features': torch.tensor(train_scaled, dtype=torch.float32).unsqueeze(1),
        'train_labels': train_labels,
        'val_features': torch.tensor(val_scaled, dtype=torch.float32).unsqueeze(1),
        'val_labels': val_labels,
        'test_features': torch.tensor(test_scaled, dtype=torch.float32).unsqueeze(1) if test_scaled is not None else None,
        'test_labels': test_labels if test_feats is not None else None,
        'num_features': train_scaled.shape[1],
        'scaler': scaler,
    }
    
    feature_cache.put(window_size, overlap, result)
    return result


def create_objective(
    dataset_dir: str,
    base_config: Dict[str, Any],
    objective_type: str = 'val_f1',
    search_space: Dict = None,
    feature_cache: Optional[FeatureCache] = None,
    callback: Optional[Callable] = None,
    stop_event=None,
):
    """
    Create an Optuna objective function.
    
    Args:
        dataset_dir: Path to dataset root.
        base_config: Base pipeline configuration.
        objective_type: 'val_accuracy', 'val_f1', or 'multi'.
        search_space: Search space config. Uses defaults if None.
        feature_cache: FeatureCache instance for caching extracted features.
        callback: Optional callable(trial_number, epoch, train_loss, val_metric)
                  for live progress updates.
        stop_event: threading.Event that signals the optimization should stop.
        
    Returns:
        Objective function suitable for study.optimize().
    """
    # Load file listing once
    files_data, class_map = load_all_files(dataset_dir)
    num_classes = len(class_map)
    class_weights = compute_class_weights(files_data, num_classes)
    
    # Create split indices once (file-level, deterministic)
    train_idx, val_idx, test_idx = create_stratified_split(
        files_data, num_classes,
        train_ratio=base_config.get('train_ratio', 0.6),
        val_ratio=base_config.get('val_ratio', 0.2),
        seed=42,
    )
    
    if feature_cache is None:
        feature_cache = FeatureCache()
    
    device = base_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    def objective(trial):
        # Check stop signal
        if stop_event is not None and stop_event.is_set():
            raise optuna.exceptions.OptunaError("Optimization stopped by user")
        
        trial_start = time.time()
        
        # 1. Sample hyperparameters
        params = sample_params(trial, search_space)
        
        # 2. Compute windowing params
        window_size = params['hrv_window_size']
        overlap = int(window_size * params['hrv_overlap_ratio'])
        
        # 3. Get or compute features
        feat_data = _prepare_features(
            files_data, train_idx, val_idx, test_idx,
            base_config, feature_cache, window_size, overlap
        )
        
        if feat_data is None:
            raise optuna.exceptions.TrialPruned("Feature extraction failed")
        
        train_features = feat_data['train_features'].to(device)
        train_labels = feat_data['train_labels'].to(device)
        val_features = feat_data['val_features'].to(device)
        val_labels = feat_data['val_labels'].to(device)
        num_features = feat_data['num_features']
        
        # 4. Build model
        tcn_channels = build_tcn_channels(params)
        
        model = ConfigurableFusionModel(
            input_dim=num_features,
            embedding_dim=params['embedding_dim'],
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            encoder_num_layers=params['encoder_num_layers'],
            encoder_hidden_dim=params['encoder_hidden_dim'],
            encoder_activation=params['encoder_activation'],
            tcn_kernel_size=params['tcn_kernel_size'],
            dropout=params['dropout'],
        ).to(device)
        
        # 5. Setup training
        weights_device = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
        )
        
        # Group Lasso
        group_lasso_lambda = params['group_lasso_lambda']
        group_lasso_reg = None
        if group_lasso_lambda > 0:
            try:
                feature_group_indices = get_feature_group_indices()
                group_lasso_reg = GroupLassoRegularizer(
                    group_indices=feature_group_indices,
                    lambda_reg=group_lasso_lambda,
                )
            except Exception:
                pass
        
        batch_size = params['batch_size']
        epochs = base_config.get('epochs', 100)
        patience = base_config.get('early_stopping_patience', 20)
        
        best_val_metric = 0.0
        patience_counter = 0
        best_val_acc = 0.0
        best_val_f1 = 0.0
        
        # 6. Training loop
        for epoch in range(epochs):
            # Check stop signal
            if stop_event is not None and stop_event.is_set():
                raise optuna.exceptions.OptunaError("Optimization stopped by user")
            
            # -- Train --
            model.train()
            train_loss = 0.0
            train_total = 0
            
            num_train = len(train_features)
            indices = torch.randperm(num_train, device=device)
            
            for batch_start in range(0, num_train, batch_size):
                batch_end = min(batch_start + batch_size, num_train)
                batch_idx = indices[batch_start:batch_end]
                
                batch_feat = train_features[batch_idx]  # [B, 1, F]
                batch_lab = train_labels[batch_idx]
                
                outputs = model(batch_feat)[:, -1, :]  # [B, C]
                ce_loss = criterion(outputs, batch_lab)
                
                # Group lasso on encoder's first linear layer
                if group_lasso_reg is not None:
                    # Access encoder's first linear layer
                    first_linear = None
                    for module in model.encoder.net:
                        if isinstance(module, nn.Linear):
                            first_linear = module
                            break
                    if first_linear is not None:
                        gl_penalty = group_lasso_reg.compute_penalty(first_linear.weight)
                        loss = ce_loss + gl_penalty
                    else:
                        loss = ce_loss
                else:
                    loss = ce_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_lab)
                train_total += len(batch_lab)
            
            train_loss /= max(train_total, 1)
            
            # -- Validate --
            model.eval()
            val_correct = 0
            val_total = 0
            all_preds = []
            all_true = []
            
            with torch.no_grad():
                for batch_start in range(0, len(val_features), batch_size):
                    batch_end = min(batch_start + batch_size, len(val_features))
                    batch_feat = val_features[batch_start:batch_end]
                    batch_lab = val_labels[batch_start:batch_end]
                    
                    outputs = model(batch_feat)[:, -1, :]
                    _, predicted = torch.max(outputs, 1)
                    
                    val_correct += (predicted == batch_lab).sum().item()
                    val_total += len(batch_lab)
                    all_preds.extend(predicted.cpu().numpy())
                    all_true.extend(batch_lab.cpu().numpy())
            
            val_acc = val_correct / max(val_total, 1)
            val_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
            
            # Determine primary metric for pruning/early stopping
            if objective_type == 'val_accuracy':
                val_metric = val_acc
            else:  # val_f1 or multi
                val_metric = val_f1
            
            # Report to Optuna for pruning
            trial.report(val_metric, epoch)
            
            # Callback for live updates
            if callback is not None:
                callback(trial.number, epoch, train_loss, val_acc, val_f1)
            
            # Pruning check
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_val_acc = val_acc
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        trial_duration = time.time() - trial_start
        
        # Store extra info in trial
        trial.set_user_attr('best_val_acc', best_val_acc)
        trial.set_user_attr('best_val_f1', best_val_f1)
        trial.set_user_attr('epochs_trained', epoch + 1)
        trial.set_user_attr('duration_seconds', trial_duration)
        
        if objective_type == 'multi':
            return best_val_acc, best_val_f1
        elif objective_type == 'val_accuracy':
            return best_val_acc
        else:
            return best_val_f1
    
    return objective, num_classes, class_map
