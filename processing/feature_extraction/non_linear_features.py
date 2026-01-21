"""
Non-Linear Feature Extraction Module

Contains all non-linear signal analysis features:
- Sample Entropy
- Multiscale Entropy (MSE)
- Detrended Fluctuation Analysis (DFA)
- Higher-Order Spectral (HOS) features / Bispectrum
"""

import numpy as np
import nolds
from scipy.stats import entropy


class NonLinearFeatures:
    """Class containing all non-linear feature extraction methods."""
    
    DEFAULT_PARAMS = {
        "samp_en_m": 2,
        "samp_en_r": 0.2,
        "scales": 10,
        "dfa_min_scale": 4,
        "dfa_num_scales": 20
    }
    
    @staticmethod
    def sample_entropy(rr_intervals: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
        """
        Calculate sample entropy for RR intervals.
        
        Args:
            rr_intervals: Array of RR intervals.
            m: Template length (default 2).
            r_ratio: Similarity threshold ratio (default 0.2).
            
        Returns:
            Sample entropy value, or np.nan if undefined.
        """
        r = r_ratio * np.std(rr_intervals)  # similarity threshold
        N = len(rr_intervals)  # number of data points
        
        def count_matches(template_length):
            count = 0
            
            # Goes through all template start indices
            for i in range(N - template_length + 1):
                template_i = rr_intervals[i:i + template_length]
                
                # Compare only with j > i, meaning no self-match, and no double counting
                for j in range(i + 1, N - template_length + 1):
                    template_j = rr_intervals[j:j + template_length]
                    
                    distance = np.max(np.abs(template_i - template_j))
                    
                    if distance <= r:
                        count += 1
                        
            return count
        
        B = count_matches(m)  # matches for templates of length m
        A = count_matches(m + 1)  # matches for template of length m+1
        
        if A == 0 or B == 0:
            return np.nan  # entropy is not defined
        
        samp_en = -np.log(A / B)
        
        return samp_en
    
    @staticmethod
    def sample_entropy_nolds(rr_intervals: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
        """
        Calculate sample entropy using the nolds library.
        
        Args:
            rr_intervals: Array of RR intervals.
            m: Template length (default 2).
            r_ratio: Similarity threshold ratio (default 0.2).
            
        Returns:
            Sample entropy value from nolds.
        """
        r = r_ratio * np.std(rr_intervals)
        return nolds.sampen(rr_intervals, emb_dim=m, tolerance=r)
    
    @staticmethod
    def multiscale_entropy(rr_intervals: np.ndarray, max_scale: int = 10, m: int = 2, r_ratio: float = 0.2) -> np.ndarray:
        """
        Calculate multiscale entropy for RR intervals.
        
        Args:
            rr_intervals: Array of RR intervals.
            max_scale: Maximum scale to compute (default 10).
            m: Template length for sample entropy (default 2).
            r_ratio: Similarity threshold ratio (default 0.2).
            
        Returns:
            Array of sample entropy values for each scale.
        """
        mse_values = np.zeros(max_scale)
        
        # Goes through each scale from 1 to max_scale
        for i in range(0, max_scale):
            scale = i + 1
            new_length = len(rr_intervals) // scale  # number of values in the new vector
            
            # If there are not enough points to calculate Sample Entropy, we will leave nan
            if new_length < m + 1:
                mse_values[i] = np.nan
                continue
            
            averaged_vector = np.zeros(new_length)  # Create an average vector for the current scale
            
            # Goes through all the blocks of the original vector according to the scale
            for j in range(0, new_length):
                block_values = rr_intervals[j * scale:(j + 1) * scale]
                averaged_vector[j] = np.mean(block_values)
            
            mse_values[i] = NonLinearFeatures.sample_entropy(averaged_vector, m, r_ratio)
        
        mse_values = np.nan_to_num(mse_values, nan=0.0)  # replace nan values with zero
        
        return mse_values
    
    @staticmethod
    def detrended_fluctuation_analysis(rr_intervals: np.ndarray, min_scale: int = 4, num_scales: int = 20) -> tuple:
        """
        Perform Detrended Fluctuation Analysis on RR intervals.
        
        Args:
            rr_intervals: Array of RR intervals.
            min_scale: Minimum scale for DFA (default 4).
            num_scales: Number of scales to compute (default 20).
            
        Returns:
            Tuple of (alpha, scales, F) where:
                - alpha: DFA scaling exponent
                - scales: Array of scales used
                - F: Array of fluctuation values for each scale
        """
        rr_intervals = np.asarray(rr_intervals)
        
        # Mean-centered cumulative sum
        rr_mean = np.mean(rr_intervals)
        y = np.cumsum(rr_intervals - rr_mean)
        N = len(y)
        
        # Define scales - Log-spaced scales
        max_scale = N // 4
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales).astype(int))
        
        F = []
        
        # Detrend & Compute RMS per scale
        for s in scales:
            n_segments = N // s
            
            if n_segments < 2:
                continue
            
            rms_vals = []
            
            for i in range(n_segments):
                start = i * s
                end = start + s
                segment = y[start:end]
                
                x = np.arange(s)
                coeffs = np.polyfit(x, segment, 1)  # Linear detrending
                trend = np.polyval(coeffs, x)
                
                rms = np.sqrt(np.mean((segment - trend) ** 2))
                rms_vals.append(rms)
            
            # Average across segments
            F_s = np.sqrt(np.mean(np.square(rms_vals)))
            F.append(F_s)
        
        F = np.array(F)
        scales = scales[:len(F)]
        
        # Linear fit in log-log space
        log_scales = np.log(scales)
        log_F = np.log(F)
        
        alpha, _ = np.polyfit(log_scales, log_F, 1)
        return alpha, scales, F
    
    @staticmethod
    def compute_bispectrum_direct(signal: np.ndarray) -> tuple:
        """
        Compute the bispectrum of a signal using direct FFT method.
        
        The bispectrum is a higher-order spectrum that detects phase coupling
        and nonlinear interactions in the signal.
        
        Args:
            signal: Input signal array (1D).
            
        Returns:
            Tuple of (B_mag, B_phase):
                - B_mag: Bispectrum magnitude matrix
                - B_phase: Bispectrum phase matrix
        """
        # Ensure signal is 1D
        signal = np.asarray(signal).flatten()
        N = len(signal)
        
        # Compute FFT
        X = np.fft.fft(signal)
        
        # Initialize bispectrum matrix (only compute for positive frequencies)
        # Due to symmetry, we only need to compute half
        n_freq = N // 2 + 1
        B = np.zeros((n_freq, n_freq), dtype=complex)
        
        # Compute bispectrum B(f1, f2) = X(f1) * X(f2) * conj(X(f1+f2))
        for i in range(n_freq):
            for j in range(n_freq):
                k = (i + j) % N  # Frequency sum with wraparound
                B[i, j] = X[i] * X[j] * np.conj(X[k])
        
        # Extract magnitude and phase
        B_mag = np.abs(B)
        B_phase = np.angle(B)
        
        return B_mag, B_phase
    
    @staticmethod
    def extract_bispectrum_features(B_mag: np.ndarray) -> dict:
        """
        Extract scalar features from the bispectrum magnitude matrix.
        
        Args:
            B_mag: Bispectrum magnitude matrix (2D array).
            
        Returns:
            Dictionary with 4 HOS features:
                - HOS_Mean: Mean bispectrum magnitude
                - HOS_Std: Standard deviation of bispectrum magnitude
                - HOS_Peak: Peak (maximum) bispectrum magnitude
                - HOS_Entropy: Entropy of normalized bispectrum
        """
        # Flatten the matrix for statistical analysis
        B_flat = B_mag.flatten()
        
        # Calculate basic statistics
        mean_val = np.mean(B_flat)
        std_val = np.std(B_flat)
        peak_val = np.max(B_flat)
        
        # Calculate entropy (normalize first to create a probability distribution)
        # Add small epsilon to avoid log(0)
        B_normalized = B_flat / (np.sum(B_flat) + 1e-10)
        entropy_val = entropy(B_normalized + 1e-10)
        
        features = {
            "HOS_Mean": mean_val,
            "HOS_Std": std_val,
            "HOS_Peak": peak_val,
            "HOS_Entropy": entropy_val
        }
        
        return features
