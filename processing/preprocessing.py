import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d  
from scipy.interpolate import interp1d


def interpolated_signal(original_signal, original_fs, upsample_factor):
    """Upsample signal using cubic interpolation."""
    new_fs = original_fs * upsample_factor
    t_original = np.arange(len(original_signal)) / original_fs
    t_new = np.linspace(t_original[0], t_original[-1], len(t_original) * upsample_factor)
    interpolator = interp1d(t_original, original_signal, kind='cubic')
    signal_new = interpolator(t_new)
    return signal_new, new_fs, t_new


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def extract_rr(signal, params):
    """
    Detect R-peaks and extract RR intervals in one step.
    
    Returns RR intervals in milliseconds.
    """
    peaks, _ = find_peaks(signal, distance=params["d"], prominence=params["p"])
    rr_locations = np.diff(peaks)
    rr = rr_locations * (1 / (params["hz"]) * 1000)
    return rr