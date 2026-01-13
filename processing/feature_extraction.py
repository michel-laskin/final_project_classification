import numpy as np
# import matplotlib.pyplot as plt  # Not needed for classification pipeline
import nolds
from scipy.stats import skew, kurtosis
import pywt  # Wavelet library
# from matplotlib.patches import Ellipse  # Not needed for classification pipeline
import torch

# PyTorch is now required for tensor outputs
TORCH_AVAILABLE = True
import nolds

class FeatureExtractor:
    """
    A class for extracting HRV features from RR intervals using entropy-based methods.
    """
    
    # Default parameters
    DEFAULT_PARAMS = {
        "filename": "Sample_Data_Intensity_Average_Raw_40_Hz.csv",
        "sampling_freq": 40,
        
        # Filtering
        "use_BPF": False,
        "use_averaging": True,
        "averaging_type": "Gaussian",
        "low_threshold": 0.5,
        "high_threshold": 15.0,
        "filter_order": 2,
        "gaussian_sigma": 2,
        
        # R-Peak Detection
        "min_rr_sec_human": 0.6,
        "min_rr_sec_zebrafish": 0.15,
        
        # Sample Entropy
        "samp_en_m": 2,
        "samp_en_r": 0.2,
        
        # Multiscale Entropy
        "scales": 10
    }
    
    def __init__(self, params: dict = None):
        """
        Initialize the FeatureExtractor with given parameters.
        
        Args:
            params: Dictionary of parameters. If None, uses default parameters.
        """
        if params is None:
            self.params = self.DEFAULT_PARAMS.copy()
        else:
            self.params = params.copy()
    
    @property
    def samp_en_m(self) -> int:
        """Template length for sample entropy."""
        return self.params["samp_en_m"]
    
    @samp_en_m.setter
    def samp_en_m(self, value: int):
        self.params["samp_en_m"] = value
    
    @property
    def samp_en_r(self) -> float:
        """Similarity threshold ratio for sample entropy."""
        return self.params["samp_en_r"]
    
    @samp_en_r.setter
    def samp_en_r(self, value: float):
        self.params["samp_en_r"] = value
    
    @property
    def scales(self) -> int:
        """Maximum scale for multiscale entropy."""
        return self.params["scales"]
    
    @scales.setter
    def scales(self, value: int):
        self.params["scales"] = value
    
    def sample_entropy(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate sample entropy for RR intervals.
        
        Args:
            rr_intervals: Array of RR intervals.
            
        Returns:
            Sample entropy value, or np.nan if undefined.
        """
        m = self.samp_en_m  # template length
        r = self.samp_en_r * np.std(rr_intervals)  # similarity threshold
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
    
    def sample_entropy_nolds(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate sample entropy using the nolds library.
        
        Args:
            rr_intervals: Array of RR intervals.
            
        Returns:
            Sample entropy value from nolds.
        """
        m = self.samp_en_m
        r = self.samp_en_r * np.std(rr_intervals)
        
        return nolds.sampen(rr_intervals, emb_dim=m, tolerance=r)

    def multiscale_entropy(self, rr_intervals: np.ndarray) -> np.ndarray:
        """
        Calculate multiscale entropy for RR intervals.
        
        Args:
            rr_intervals: Array of RR intervals.
            
        Returns:
            Array of sample entropy values for each scale.
        """
        max_scale = self.scales
        mse_values = np.zeros(max_scale)
        
        # Goes through each scale from 1 to max_scale
        for i in range(0, max_scale):
            scale = i + 1
            new_length = len(rr_intervals) // scale  # number of values in the new vector
            
            # If there are not enough points to calculate Sample Entropy, we will leave nan
            if new_length < self.samp_en_m + 1:
                mse_values[i] = np.nan
                continue
            
            averaged_vector = np.zeros(new_length)  # Create an average vector for the current scale
            
            # Goes through all the blocks of the original vector according to the scale
            for j in range(0, new_length):
                block_values = rr_intervals[j * scale:(j + 1) * scale]
                averaged_vector[j] = np.mean(block_values)
            
            mse_values[i] = self.sample_entropy(averaged_vector)
        
        mse_values = np.nan_to_num(mse_values, nan=0.0)  # replace nan values with zero
        
        return mse_values
    
    def compute_mse_grid(self, rr_intervals: np.ndarray, m_values: list, r_values: list) -> dict:
        """
        Compute MSE for multiple m and r values.
        
        Args:
            rr_intervals: Array of RR intervals.
            m_values: List of m values to test.
            r_values: List of r values to test.
            
        Returns:
            Dictionary mapping (m, r) tuples to MSE value arrays.
        """
        results = {}
        
        # Store original values to restore later
        original_m = self.samp_en_m
        original_r = self.samp_en_r
        
        for m in m_values:
            for r in r_values:
                self.samp_en_m = m
                self.samp_en_r = r
                mse_vals = self.multiscale_entropy(rr_intervals)
                results[(m, r)] = mse_vals
        
        # Restore original values
        self.samp_en_m = original_m
        self.samp_en_r = original_r
        
        return results


def _features_dict_to_tensor(features_dict: dict, device: str = 'cpu'):
    """
    Convert feature dictionary to tensor format compatible with FeatureEncoder.
    
    Args:
        features_dict: Dictionary mapping feature names to values.
        device: Target device ('cpu' or 'cuda').
        
    Returns:
        Tensor of shape [1, 1, num_features].
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available. Cannot convert to tensor.")
    
    # Sort keys for deterministic ordering
    feature_names = sorted(features_dict.keys())
    feature_values = [float(features_dict[k]) for k in feature_names]
    
    # Create tensor and add batch + sequence dimensions
    tensor = torch.tensor(feature_values, dtype=torch.float32, device=device)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [num_features] -> [1, 1, num_features]
    
    return tensor


class HRVFeatureExtractor:
    """
    A class for extracting time-domain and geometric HRV features from RR intervals.
    """
    
    DEFAULT_PARAMS = {
        "file_path": "./data/Sample_Data_Intensity_Average_Raw_40_Hz.csv",
        "d": 8,
        "p": 3,
        "hz": 40,
        "is_filtered": False,
        "lowcut": 0.5,
        "highcut": 8.0,
        "order": 2,
        "x": 20,
        "interpolation_factor": 10,
        "want_interpolation": True,
    }
    
    def __init__(self, params: dict = None):
        """
        Initialize the HRVFeatureExtractor with given parameters.
        
        Args:
            params: Dictionary of parameters. If None, uses default parameters.
        """
        if params is None:
            self.params = self.DEFAULT_PARAMS.copy()
        else:
            self.params = params.copy()
    
    @property
    def hz(self) -> int:
        """Sampling frequency in Hz."""
        return self.params["hz"]
    
    @hz.setter
    def hz(self, value: int):
        self.params["hz"] = value
    
    @property
    def x(self) -> float:
        """Threshold for pNNX calculation in ms."""
        return self.params["x"]
    
    @x.setter
    def x(self, value: float):
        self.params["x"] = value
    
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
    
    def extract_pNNX(self, rr_interval: np.ndarray) -> float:
        """
        Calculate the percentage of successive RR interval differences greater than X ms.
        
        Args:
            rr_interval: Array of RR intervals in ms.
            
        Returns:
            pNNX value as float (ratio, not percentage).
        """
        diff = np.diff(rr_interval)
        return np.sum(np.abs(diff) > self.x) / len(diff)
    
    def extract_HRV_Triangular_Index(self, rr_interval: np.ndarray, ax=None) -> float:
        """
        Calculate the HRV Triangular Index.
        
        Args:
            rr_interval: Array of RR intervals in ms.
            ax: Optional matplotlib axes for plotting histogram.
            
        Returns:
            Triangular Index value as float.
        """
        bin_width = 1000 / self.hz
        min_rr, max_rr = np.min(rr_interval), np.max(rr_interval)
        num_bins = int(np.ceil((max_rr - min_rr) / bin_width)) + 1
        bin_edges = np.linspace(min_rr, min_rr + num_bins * bin_width, num_bins + 1)
        
        hist, _ = np.histogram(rr_interval, bins=bin_edges)
        result = len(rr_interval) / np.max(hist)
        
        if ax is not None:
            bin_times_x = bin_edges[:-1]  # Left edges of bins
            ax.bar(bin_times_x, hist, width=bin_width * 0.9, align='edge', color='skyblue', edgecolor='black')
            ax.set_xlabel('RR Interval Duration (ms)')
            ax.set_ylabel('Count (Number of Beats)')
            ax.set_title(f'Triangular Index: {result:.2f}')
            ax.grid(axis='y', alpha=0.3)
        
        return result
    
    @staticmethod
    def extract_Poincare_Plot(rr_interval: np.ndarray, ax=None) -> tuple:
        """
        Calculate Poincare plot metrics (SD1 and SD2).
        
        Args:
            rr_interval: Array of RR intervals in ms.
            ax: Optional matplotlib axes for plotting.
            
        Returns:
            Tuple of (SD1, SD2) values.
        """
        x = rr_interval[:-1]
        y = rr_interval[1:]
        diff = y - x
        sd1 = np.std(diff, ddof=1) / np.sqrt(2)
        total = y + x
        sd2 = np.std(total, ddof=1) / np.sqrt(2)
        mean_rr = np.mean(rr_interval)
        
        if ax is not None:
            ax.scatter(x, y, c='blue', s=10, alpha=0.5, label='RR Intervals')
            ellipse = Ellipse(xy=(mean_rr, mean_rr),
                              width=2 * sd2, height=2 * sd1,
                              angle=45, edgecolor='red', fc='None', lw=2, label='Fitted Ellipse')
            ax.add_patch(ellipse)
            
            min_val = min(min(x), min(y)) - 10
            max_val = max(max(x), max(y)) + 10
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.3, label='Line of Identity')
            
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_xlabel('$RR_n$ (ms)')
            ax.set_ylabel('$RR_{n+1}$ (ms)')
            ax.set_title(f'Poincare Plot\nSD1={sd1:.2f}ms, SD2={sd2:.2f}ms')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        return sd1, sd2
    
    @staticmethod
    def calculate_statistical_features(mse_values: np.ndarray, device: str = 'cpu'):
        """
        Calculate statistical features from MSE values.
        
        Args:
            mse_values: Array of multiscale entropy values.
            device: Device for tensor ('cpu' or 'cuda').
            
        Returns:
            Tensor of shape [1, 1, 9] containing statistical features.
        """
        mean_val = np.mean(mse_values)  # mu_x
        median_val = np.median(mse_values)  # x_bar
        std_val = np.std(mse_values)  # sigma_x
        skew_val = skew(mse_values)  # Skewness: Measure of the asymmetry of the probability distribution
        kurt_val = kurtosis(mse_values)  # Kurtosis: Measure of the "tailedness" of the probability distribution
        
        diff_1 = np.diff(mse_values)  # First difference (approximate derivative): X(t+1) - X(t)
        diff_2 = np.diff(mse_values, n=2)  # Second difference: X(t+2) - X(t) approximated by difference of differences
        
        delta_x = np.mean(np.abs(diff_1))  # Mean of absolute values of 1st difference (delta_x)
        gamma_x = np.mean(np.abs(diff_2))  # Mean of absolute values of 2nd difference (gamma_x)
        
        norm_delta_x = delta_x / std_val if std_val != 0 else 0  # Normalized 1st difference (delta_bar_x)
        norm_gamma_x = gamma_x / std_val if std_val != 0 else 0  # Normalized 2nd difference (gamma_bar_x)
        
        features = {
            "Stat_Mean": mean_val,
            "Stat_Median": median_val,
            "Stat_Std": std_val,
            "Stat_Skewness": skew_val,
            "Stat_Kurtosis": kurt_val,
            "Stat_Abs_Diff1": delta_x,
            "Stat_Abs_Diff2": gamma_x,
            "Stat_Norm_Diff1": norm_delta_x,
            "Stat_Norm_Diff2": norm_gamma_x
        }
        
        return _features_dict_to_tensor(features, device)
    
    @staticmethod
    def calculate_wavelet_features(mse_values: np.ndarray, fixed_scales: int = 10, device: str = 'cpu'):
        """
        Calculate wavelet features from MSE values using Continuous Wavelet Transform.
        
        Args:
            mse_values: Array of multiscale entropy values.
            fixed_scales: Number of scales to use (default 10 for 20 features).
            device: Device for tensor ('cpu' or 'cuda').
            
        Returns:
            Tensor of shape [1, 1, 2*fixed_scales] containing wavelet features.
        """
        # Pad or truncate to fixed length for consistent output dimensions
        if len(mse_values) < fixed_scales:
            padded_mse = np.pad(mse_values, (0, fixed_scales - len(mse_values)), 
                               mode='constant', constant_values=0)
        else:
            padded_mse = mse_values[:fixed_scales]
        
        # Define scales for the CWT
        scales = np.arange(1, fixed_scales + 1)
        
        # Performing Continuous Wavelet Transform with Morlet wavelet
        coeffs, freqs = pywt.cwt(padded_mse, scales, 'morl')
        
        features = {}
        
        # Calculating Mean and STD for each scale
        for i, scale_coeffs in enumerate(coeffs):
            scale_num = i + 1
            abs_coeffs = np.abs(scale_coeffs)
            energy_c = abs_coeffs ** 2
            mean_c = np.mean(energy_c)
            std_c = np.std(energy_c)
            
            # Store in dictionary
            features[f"Wavelet_Scale{scale_num}_Mean"] = mean_c
            features[f"Wavelet_Scale{scale_num}_Std"] = std_c
        
        return _features_dict_to_tensor(features, device)
    
    def extract_all_features(self, signal: np.ndarray, peak_indices: np.ndarray, t_axis: np.ndarray, device: str = 'cpu'):
        """
        Extract all features from a signal including HRV, statistical, and wavelet features.
        
        Args:
            signal: The input signal array.
            peak_indices: Array of indices where peaks (R-peaks) are detected.
            t_axis: Time axis array corresponding to the signal.
            device: Device for tensor ('cpu' or 'cuda').
            
        Returns:
            Tuple of (all_features_tensor, rr_intervals_ms):
                - all_features_tensor: Tensor of shape [1, 1, total_features]
                - rr_intervals_ms: Array of RR intervals in milliseconds
        """
        # Calculate HRV (Time Domain)
        peak_times = t_axis[peak_indices]
        rr_intervals_ms = np.diff(peak_times) * 1000
        
        if len(rr_intervals_ms) > 1:
            mean_rr = np.mean(rr_intervals_ms)
            std_rr = np.std(rr_intervals_ms, ddof=1)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2))
            bpm = 60000 / mean_rr
        else:
            mean_rr = std_rr = rmssd = bpm = 0
        
        hrv_features = {
            "BPM": bpm,
            "RMSSD": rmssd,
            "SDNN": std_rr,
            "Mean_RR": mean_rr
        }
        
        # Compute MSE from RR intervals for statistical and wavelet feature extraction
        # Need to instantiate FeatureExtractor to compute MSE
        from processing.feature_extraction import FeatureExtractor
        entropy_extractor = FeatureExtractor()
        mse_values = entropy_extractor.multiscale_entropy(rr_intervals_ms)
        
        # Calculate statistical features from MSE (not raw signal)
        stat_features_tensor = self.calculate_statistical_features(mse_values, device=device)
        
        # Calculate Wavelet features from MSE (not raw signal)
        wavelet_features_tensor = self.calculate_wavelet_features(mse_values, device=device)
        
        # Convert HRV features dict to tensor
        hrv_features_tensor = _features_dict_to_tensor(hrv_features, device=device)
        
        # Concatenate all feature tensors along the feature dimension (dim=2)
        # Each tensor is [1, 1, num_features], concatenate to [1, 1, total_features]
        all_features_tensor = torch.cat([hrv_features_tensor, stat_features_tensor, wavelet_features_tensor], dim=2)
        
        return all_features_tensor, rr_intervals_ms


# =============================================================================
# Backward compatibility: Module-level functions that wrap the classes
# =============================================================================

# Default params for backward compatibility
params = FeatureExtractor.DEFAULT_PARAMS.copy()


def sample_entropy(rr_intervals, params):
    """Backward compatible wrapper for sample_entropy."""
    extractor = FeatureExtractor(params)
    return extractor.sample_entropy(rr_intervals)


def sample_entropy_nolds(rr_intervals, params):
    """Backward compatible wrapper for sample_entropy_nolds."""
    extractor = FeatureExtractor(params)
    return extractor.sample_entropy_nolds(rr_intervals)


def multiscale_entropy(rr_intervals, params):
    """Backward compatible wrapper for multiscale_entropy."""
    extractor = FeatureExtractor(params)
    return extractor.multiscale_entropy(rr_intervals)


def compute_mse_grid(rr_intervals, params, m_values, r_values):
    """Backward compatible wrapper for compute_mse_grid."""
    extractor = FeatureExtractor(params)
    return extractor.compute_mse_grid(rr_intervals, m_values, r_values)


def calculate_mse_statistical_features(mse_values):
    """Backward compatible wrapper for calculate_statistical_features."""
    return HRVFeatureExtractor.calculate_statistical_features(mse_values)


def calculate_mse_wavelet_features(mse_values):
    """Backward compatible wrapper for calculate_wavelet_features."""
    return HRVFeatureExtractor.calculate_wavelet_features(mse_values)


# Plotting functions moved to processing.plotting module
# Import them here        
# Not needed for classification pipeline - commenting out
# from visualizations.plotting import plot_mse, plot_mse_heatmap, plot_mse_vs_scale