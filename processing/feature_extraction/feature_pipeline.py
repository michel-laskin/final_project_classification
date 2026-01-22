"""
Feature Extraction Pipeline

Unified interface for extracting all available features from RR intervals.
Combines time-domain, frequency-domain, and non-linear features.
"""

import numpy as np
import torch
from typing import Dict, Optional

from .time_domain_features import TimeDomainFeatures
from .frequency_domain_features import FrequencyDomainFeatures
from .non_linear_features import NonLinearFeatures


def extract_all_features_from_rr(
    rr_intervals: np.ndarray,
    sampling_freq: int = 40,
    params: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Extract all available features from RR intervals.
    
    This function combines:
    - Time-domain features: HRV metrics, statistical features, Hjorth parameters
    - Frequency-domain features: Wavelet transform features
    - Non-linear features: Sample entropy, MSE, DFA, bispectrum features
    
    Args:
        rr_intervals: Array of RR intervals in milliseconds.
        sampling_freq: Sampling frequency in Hz (default: 40).
        params: Optional parameter dictionary for feature extraction configuration.
        
    Returns:
        Dictionary mapping feature names to values. Contains approximately 140+ features.
    """
    if params is None:
        params = {}
    
    # Initialize feature dictionary
    all_features = {}
    
    # ===========================
    # TIME-DOMAIN FEATURES
    # ===========================
    
    # 1. Basic HRV Features (4 features)
    hrv_features = TimeDomainFeatures.calculate_hrv_features(rr_intervals)
    all_features.update(hrv_features)
    
    # 2. Statistical Features (9 features)
    stat_features = TimeDomainFeatures.calculate_statistics_vector(rr_intervals)
    for key, val in stat_features.items():
        all_features[f"RR_{key}"] = val
    
    # 3. Hjorth Parameters (6 features)
    hjorth_features = TimeDomainFeatures.calculate_generalized_hjorth(rr_intervals)
    all_features.update(hjorth_features)
    
    # 4. Additional HRV metrics
    all_features["SDRR"] = TimeDomainFeatures.extract_SDRR(rr_intervals)
    all_features["pNN20"] = TimeDomainFeatures.extract_pNNX(rr_intervals, x=20)
    all_features["pNN50"] = TimeDomainFeatures.extract_pNNX(rr_intervals, x=50)
    
    # ===========================
    # FREQUENCY-DOMAIN FEATURES  
    # ===========================
    
    # Wavelet Features (108 features: 9 stats × 12 scales)
    wavelet_params = {
        "original_fs": sampling_freq,
        "upsample_factor": params.get("upsample_factor", 1),
        "use_octave_scales": params.get("use_octave_scales", False),
        "wavelet_num_scales": params.get("wavelet_num_scales", 12)
    }
    
    try:
        wavelet_features = FrequencyDomainFeatures.calculate_wavelet_features(
            rr_intervals, wavelet_params
        )
        all_features.update(wavelet_features)
    except Exception as e:
        # If wavelet fails (e.g., sequence too short), fill with zeros
        print(f"Warning: Wavelet feature extraction failed: {e}")
        for i in range(1, 13):
            for stat in ["Mean", "Median", "Std", "Skewness", "Kurtosis", 
                        "Abs_Diff1", "Abs_Diff2", "Norm_Diff1", "Norm_Diff2"]:
                all_features[f"Wavelet_Scale{i}_{stat}"] = 0.0
    
    # ===========================
    # NON-LINEAR FEATURES
    # ===========================
    
    # 1. Sample Entropy (1 feature)
    try:
        samp_en = NonLinearFeatures.sample_entropy(
            rr_intervals,
            m=params.get("samp_en_m", 2),
            r_ratio=params.get("samp_en_r", 0.2)
        )
        all_features["SampleEntropy"] = samp_en if not np.isnan(samp_en) else 0.0
    except Exception:
        all_features["SampleEntropy"] = 0.0
    
    # 2. Multiscale Entropy (10 features by default)
    try:
        mse_values = NonLinearFeatures.multiscale_entropy(
            rr_intervals,
            max_scale=params.get("scales", 10),
            m=params.get("samp_en_m", 2),
            r_ratio=params.get("samp_en_r", 0.2)
        )
        for i, mse_val in enumerate(mse_values):
            all_features[f"MSE_Scale{i+1}"] = mse_val
    except Exception as e:
        print(f"Warning: MSE extraction failed: {e}")
        for i in range(params.get("scales", 10)):
            all_features[f"MSE_Scale{i+1}"] = 0.0
    
    # 3. DFA (1 feature: alpha exponent)
    try:
        alpha, _, _ = NonLinearFeatures.detrended_fluctuation_analysis(
            rr_intervals,
            min_scale=params.get("dfa_min_scale", 4),
            num_scales=params.get("dfa_num_scales", 20)
        )
        all_features["DFA_alpha"] = alpha
    except Exception as e:
        print(f"Warning: DFA extraction failed: {e}")
        all_features["DFA_alpha"] = 0.0
    
    # 4. Bispectrum Features (4 features)
    try:
        B_mag, _ = NonLinearFeatures.compute_bispectrum_direct(rr_intervals)
        bispectrum_features = NonLinearFeatures.extract_bispectrum_features(B_mag)
        all_features.update(bispectrum_features)
    except Exception as e:
        print(f"Warning: Bispectrum extraction failed: {e}")
        all_features["HOS_Mean"] = 0.0
        all_features["HOS_Std"] = 0.0
        all_features["HOS_Peak"] = 0.0
        all_features["HOS_Entropy"] = 0.0
    
    return all_features


def features_dict_to_tensor(features_dict: Dict[str, float], device: str = 'cpu') -> torch.Tensor:
    """
    Convert feature dictionary to tensor format.
    
    Args:
        features_dict: Dictionary mapping feature names to values.
        device: Target device ('cpu' or 'cuda').
        
    Returns:
        Tensor of shape [1, 1, num_features] suitable for model input.
    """
    # Sort keys for deterministic ordering
    feature_names = sorted(features_dict.keys())
    feature_values = [float(features_dict[k]) for k in feature_names]
    
    # Create tensor and add batch + sequence dimensions
    tensor = torch.tensor(feature_values, dtype=torch.float32, device=device)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [num_features] -> [1, 1, num_features]
    
    return tensor


def get_feature_names(params: Optional[Dict] = None) -> list:
    """
    Get the ordered list of feature names that will be extracted.
    
    Args:
        params: Optional parameter dictionary for feature extraction configuration.
        
    Returns:
        List of feature names in the order they appear in the feature vector.
    """
    # Create a dummy RR interval series and extract features
    dummy_rr = np.random.randn(200) * 50 + 500
    features = extract_all_features_from_rr(dummy_rr, params=params)
    return sorted(features.keys())


def get_feature_group_indices(params: Optional[Dict] = None) -> Dict[str, list]:
    """
    Get the indices for each feature group.
    
    Features are organized into three groups:
    - time_domain: HRV features, statistics, Hjorth parameters, pNN metrics
    - frequency_domain: Wavelet transform features
    - non_linear: Sample entropy, MSE, DFA, bispectrum features
    
    Since features are sorted alphabetically, groups are NOT contiguous.
    This function returns the actual list of indices for each group.
    
    Args:
        params: Optional parameter dictionary for feature extraction configuration.
        
    Returns:
        Dict mapping group names to list of feature indices.
    """
    feature_names = get_feature_names(params)
    
    # Define which prefixes/patterns belong to which group
    time_domain_prefixes = [
        'BPM', 'Hjorth_', 'Mean_RR', 'RMSSD', 'RR_', 'SDNN', 'SDRR', 'pNN'
    ]
    
    frequency_domain_prefixes = [
        'Wavelet_'
    ]
    
    non_linear_prefixes = [
        'DFA_', 'HOS_', 'MSE_', 'SampleEntropy'
    ]
    
    def get_group(name: str) -> str:
        """Determine which group a feature belongs to."""
        for prefix in time_domain_prefixes:
            if name.startswith(prefix):
                return 'time_domain'
        for prefix in frequency_domain_prefixes:
            if name.startswith(prefix):
                return 'frequency_domain'
        for prefix in non_linear_prefixes:
            if name.startswith(prefix):
                return 'non_linear'
        # Default to time_domain for unrecognized features
        return 'time_domain'
    
    # Collect indices for each group
    group_indices: Dict[str, list] = {
        'time_domain': [],
        'frequency_domain': [],
        'non_linear': []
    }
    
    for idx, name in enumerate(feature_names):
        group = get_group(name)
        group_indices[group].append(idx)
    
    return group_indices


