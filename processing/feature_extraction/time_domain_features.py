"""
Time Domain Feature Extraction Module

Contains all time-domain signal analysis features:
- HRV (Heart Rate Variability) features
- Statistical features
- Hjorth parameters (mobility and complexity)
"""

import numpy as np
from scipy.stats import skew, kurtosis


class TimeDomainFeatures:
    """Class containing all time-domain feature extraction methods."""
    
    @staticmethod
    def calculate_statistics_vector(signal: np.ndarray) -> dict:
        """
        Calculate a comprehensive set of 9 statistical features from a signal.
        
        Args:
            signal: Input signal array.
            
        Returns:
            Dictionary with 9 statistical features.
        """
        mean_val = np.mean(signal)       # mu_x
        median_val = np.median(signal)   # x_bar
        std_val = np.std(signal)         # sigma_x
        
        skew_val = skew(signal)  # Skewness: Measure of the asymmetry of the probability distribution
        kurt_val = kurtosis(signal)  # Kurtosis: Measure of the "tailedness" of the probability distribution
        
        diff_1 = np.diff(signal)  # First difference (approximate derivative): X(t+1) - X(t)
        diff_2 = np.diff(signal, n=2)  # Second difference: X(t+2) - X(t) approximated by difference of differences
        
        delta_x = np.mean(np.abs(diff_1)) if len(diff_1) > 0 else 0  # Mean of absolute values of 1st difference (delta_x)
        gamma_x = np.mean(np.abs(diff_2)) if len(diff_2) > 0 else 0  # Mean of absolute values of 2nd difference (gamma_x)
        
        norm_delta_x = delta_x / std_val if std_val != 0 else 0  # Normalized 1st difference (delta_bar_x)
        norm_gamma_x = gamma_x / std_val if std_val != 0 else 0  # Normalized 2nd difference (gamma_bar_x)
        
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
    def calculate_hrv_features(rr_intervals_ms: np.ndarray) -> dict:
        """
        Calculate Heart Rate Variability (HRV) features from RR intervals.
        
        Args:
            rr_intervals_ms: Array of RR intervals in milliseconds.
            
        Returns:
            Dictionary with HRV features (BPM, RMSSD, SDNN, Mean_RR).
        """
        if len(rr_intervals_ms) > 1:
            mean_rr = np.mean(rr_intervals_ms)
            std_rr = np.std(rr_intervals_ms, ddof=1)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2))
            bpm = 60000 / mean_rr
        else:
            mean_rr = std_rr = rmssd = bpm = 0
        
        return {
            "BPM": bpm,
            "RMSSD": rmssd,
            "SDNN": std_rr,
            "Mean_RR": mean_rr
        }
    
    @staticmethod
    def calculate_hjorth_params(signal: np.ndarray) -> tuple:
        """
        Calculate Hjorth Mobility and Complexity parameters.
        
        Definitions:
        - Activity = Variance(x(t))
        - Mobility = sqrt( Activity(x'(t)) / Activity(x(t)) )
        - Complexity = Mobility(x'(t)) / Mobility(x(t))
        
        Args:
            signal: Input signal array.
            
        Returns:
            Tuple of (mobility, complexity).
        """
        # Variance (Activity) of the signal
        var_signal = np.var(signal)
        
        # First derivative (difference) and its variance
        dx = np.diff(signal)
        var_dx = np.var(dx)
        
        # Calculate Mobility
        # Mobility = sigma_prime / sigma_x
        if var_signal > 0:
            mobility = np.sqrt(var_dx / var_signal)
        else:
            mobility = 0
        
        # Calculate Complexity
        # Complexity = Mobility(dx) / Mobility(x)
        ddx = np.diff(dx)
        var_ddx = np.var(ddx)
        
        if var_dx > 0:
            mobility_dx = np.sqrt(var_ddx / var_dx)
        else:
            mobility_dx = 0
        
        if mobility > 0:
            complexity = mobility_dx / mobility
        else:
            complexity = 0
        
        return mobility, complexity
    
    @staticmethod
    def calculate_generalized_hjorth(signal: np.ndarray) -> dict:
        """
        Calculate Hjorth parameters for the original signal and its derivatives.
        Generalizes Hjorth computation across different derivative orders.
        
        Args:
            signal: Input signal array.
            
        Returns:
            Dictionary with 6 Hjorth features (mobility and complexity for orders 0, 1, 2).
        """
        features = {}
        
        # Order 0: Original Signal
        mob_0, comp_0 = TimeDomainFeatures.calculate_hjorth_params(signal)
        features['Hjorth_Mobility_Order0'] = mob_0
        features['Hjorth_Complexity_Order0'] = comp_0
        
        # Order 1: First Derivative
        dx = np.diff(signal)
        if len(dx) > 2:
            mob_1, comp_1 = TimeDomainFeatures.calculate_hjorth_params(dx)
            features['Hjorth_Mobility_Order1'] = mob_1
            features['Hjorth_Complexity_Order1'] = comp_1
        else:
            features['Hjorth_Mobility_Order1'] = 0
            features['Hjorth_Complexity_Order1'] = 0
        
        # Order 2: Second Derivative
        if len(dx) > 1:
            ddx = np.diff(dx)
            if len(ddx) > 2:
                mob_2, comp_2 = TimeDomainFeatures.calculate_hjorth_params(ddx)
                features['Hjorth_Mobility_Order2'] = mob_2
                features['Hjorth_Complexity_Order2'] = comp_2
            else:
                features['Hjorth_Mobility_Order2'] = 0
                features['Hjorth_Complexity_Order2'] = 0
        else:
            features['Hjorth_Mobility_Order2'] = 0
            features['Hjorth_Complexity_Order2'] = 0
        
        return features
    
    @staticmethod
    def extract_SDRR(rr_interval: np.ndarray) -> float:
        """
        Calculate the standard deviation of RR intervals (SDRR).
        
        Args:
            rr_interval: Array of RR intervals in ms.
            
        Returns:
            SDRR value as float.
        """
        return np.std(rr_interval, ddof=1)
    
    @staticmethod
    def extract_RMSSD(rr_interval: np.ndarray) -> float:
        """
        Calculate the root mean square of successive differences (RMSSD).
        
        Args:
            rr_interval: Array of RR intervals in ms.
            
        Returns:
            RMSSD value as float.
        """
        diff = np.diff(rr_interval)
        return np.sqrt(np.mean(diff ** 2))
    
    @staticmethod
    def extract_pNNX(rr_interval: np.ndarray, x: float = 20) -> float:
        """
        Calculate the percentage of successive RR interval differences greater than X ms.
        
        Args:
            rr_interval: Array of RR intervals in ms.
            x: Threshold in ms (default 20).
            
        Returns:
            pNNX value as float (ratio, not percentage).
        """
        diff = np.diff(rr_interval)
        return np.sum(np.abs(diff) > x) / len(diff) if len(diff) > 0 else 0
