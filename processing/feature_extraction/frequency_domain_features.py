"""
Frequency Domain Feature Extraction Module

Contains all frequency-domain signal analysis features:
- Wavelet Transform features (CWT-based)
- Wavelet coefficient computation
"""

import numpy as np
import pywt
from scipy.stats import skew, kurtosis


class FrequencyDomainFeatures:
    """Class containing all frequency-domain feature extraction methods."""
    
    DEFAULT_PARAMS = {
        "original_fs": 40,
        "upsample_factor": 1,
        "use_octave_scales": False,
        "wavelet_num_scales": 12
    }
    
    @staticmethod
    def get_wavelet_coeffs(signal: np.ndarray, params: dict):
        """
        Compute wavelet coefficients using CWT with configurable scales.
        
        Supports both octave-based (frequency-matched) and linear scales.
        
        Args:
            signal: Input signal array.
            params: Parameter dictionary containing wavelet configuration.
            
        Returns:
            Tuple of (coeffs, real_frequencies):
                - coeffs: 2D array of wavelet coefficients
                - real_frequencies: Array of frequencies corresponding to scales
        """
        fs = params.get("original_fs", 40) * params.get("upsample_factor", 1)
        use_octave = params.get("use_octave_scales", False)
        num_scales = params.get("wavelet_num_scales", 12)
        
        if use_octave:
            # Octave-based scales using frequency matching
            f0 = fs / 2.0
            k_values = np.arange(1, num_scales + 1)
            target_frequencies = f0 * (2.0 ** (-k_values))
            w_center_freq = pywt.central_frequency('morl')
            scales = (w_center_freq * fs) / target_frequencies
        else:
            # Linear scales
            scales = np.arange(1, num_scales + 1)
        
        # Perform CWT
        coeffs, freqs = pywt.cwt(signal, scales, 'morl')
        
        # Calculate real frequencies in Hz
        real_frequencies = pywt.scale2frequency('morl', scales) * fs
        
        return coeffs, real_frequencies
    
    @staticmethod
    def calculate_statistics_on_coefficients(coeffs: np.ndarray) -> dict:
        """
        Calculate 9 statistical features from wavelet coefficients.
        
        Args:
            coeffs: Wavelet coefficient array.
            
        Returns:
            Dictionary with 9 statistical features.
        """
        mean_val = np.mean(coeffs)
        median_val = np.median(coeffs)
        std_val = np.std(coeffs)
        
        skew_val = skew(coeffs)
        kurt_val = kurtosis(coeffs)
        
        diff_1 = np.diff(coeffs)
        diff_2 = np.diff(coeffs, n=2)
        
        delta_x = np.mean(np.abs(diff_1)) if len(diff_1) > 0 else 0
        gamma_x = np.mean(np.abs(diff_2)) if len(diff_2) > 0 else 0
        
        norm_delta_x = delta_x / std_val if std_val != 0 else 0
        norm_gamma_x = gamma_x / std_val if std_val != 0 else 0
        
        return {
            "Mean": mean_val,
            "Median": median_val,
            "Std": std_val,
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "Abs_Diff1": delta_x,
            "Abs_Diff2": gamma_x,
            "Norm_Diff1": norm_delta_x,
            "Norm_Diff2": norm_gamma_x
        }
    
    @staticmethod
    def calculate_wavelet_features(signal: np.ndarray, params: dict = None) -> dict:
        """
        Calculate comprehensive wavelet features from signal using CWT.
        
        Extracts 9 statistical features for EACH of 12 wavelet scales.
        Total: 108 Wavelet Features.
        
        Args:
            signal: Input signal array.
            params: Parameter dictionary. If None, uses default values.
            
        Returns:
            Dictionary with 108 wavelet features (9 stats × 12 scales).
        """
        if params is None:
            params = FrequencyDomainFeatures.DEFAULT_PARAMS
        
        coeffs, _ = FrequencyDomainFeatures.get_wavelet_coeffs(signal, params)
        
        features = {}
        
        for i, scale_coeffs in enumerate(coeffs):
            scale_num = i + 1
            
            # Take the ABSOLUTE value of the coefficients for statistics
            abs_coeffs = np.abs(scale_coeffs)
            
            # Calculate all 9 statistics for this specific scale
            stats = FrequencyDomainFeatures.calculate_statistics_on_coefficients(abs_coeffs)
            
            # Add to the main dictionary with proper naming
            for stat_name, stat_val in stats.items():
                features[f"Wavelet_Scale{scale_num}_{stat_name}"] = stat_val
        
        return features
