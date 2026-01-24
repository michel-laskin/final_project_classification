import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d  
from scipy.interpolate import interp1d




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


def process_file_to_hrv(filepath, config):
    """
    Process a single CSV file to extract HRV time series.
    
    Args:
        filepath: Path to the CSV file.
        config: Configuration dictionary containing processing parameters.
        
    Returns:
        Dictionary containing extracted time, signals, and RR intervals.
    """
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Extract Time and Signal
    # Handle Zebrafish 'Values.csv' format (Frame, Mean)
    if 'Frame' in df.columns and 'Mean' in df.columns:
        ecg_signal = df['Mean'].values
        # Generate time from frame number and sampling frequency
        frames = df['Frame'].values
        time = frames / config['sampling_freq']
        
    # Handle Standard format (Time, PPG)
    elif 'Time' in df.columns and 'PPG' in df.columns:
        time = df['Time'].values
        ecg_signal = df['PPG'].values
        
    else:
        raise ValueError(f"CSV file must contain ('Frame', 'Mean') or ('Time', 'PPG') columns. Found: {df.columns.tolist()}")
    
    # Preprocess signal with bandpass filter
    filtered_signal = butter_bandpass_filter(
        ecg_signal,
        lowcut=config['low_threshold'],
        highcut=config['high_threshold'],
        fs=config['sampling_freq'],
        order=config['filter_order']
    )
    
    # Extract RR intervals (already in milliseconds)
    # extract_rr uses 'd', 'p', 'hz' keys which should be in config
    rr_intervals_ms = extract_rr(filtered_signal, config)
    
    return {
        'time': time,
        'ppg_signal': ecg_signal,
        'filtered_signal': filtered_signal,
        'rr_intervals_ms': rr_intervals_ms
    }
