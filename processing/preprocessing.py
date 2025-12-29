import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d




def plot_signal(signal, title="Signal"):
    plt.figure()
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()
    
    
def average_filter(signal, params):
    if params["averaging_type"] == "Gaussian":
        filtered_signal = gaussian_filter1d(signal, sigma=params["gaussian_sigma"])
    
    return filtered_signal


def bandpass_filter(signal, params):
    fs = params["sampling_freq"]
    low_threshold = params["low_threshold"]
    high_threshold = params["high_threshold"]
    filter_order = params["filter_order"]
    
    nyquist = fs / 2
    b, a = butter(filter_order, [low_threshold/nyquist, high_threshold/nyquist], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def detect_r_peaks(filtered_signal, params):
    fs = params["sampling_freq"]
    min_distance = int(params["min_rr_sec_human"] * fs)
    
    peaks, _ = find_peaks(filtered_signal, distance = min_distance)
    
    return peaks


def plot_r_peaks(filtered_signal, r_peaks, num_peaks_to_plot = 7):
    # Full Signal Plot
    plt.figure()
    plt.plot(filtered_signal, label = "Filtered signal")
    plt.plot(r_peaks, filtered_signal[r_peaks], 'rx', label = "R Peaks")
    plt.legend()
    plt.title("R Peak Detection - Full Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()
    
    #Zoomed-in Plot
    if num_peaks_to_plot is not None and len(r_peaks) >= num_peaks_to_plot:
        start_index = r_peaks[0]
        end_index = r_peaks[num_peaks_to_plot - 1] + int((r_peaks[num_peaks_to_plot - 1] - r_peaks[0] * 0.5))
        
        plt.figure()
        plt.plot(filtered_signal[start_index:end_index], label = "Signal (Zoomed-in)")
        
        # Adjust peak indices for zoomed plot
        zoom_r_peaks = r_peaks[:num_peaks_to_plot] - start_index
        plt.plot(zoom_r_peaks, filtered_signal[r_peaks[:num_peaks_to_plot]], 'rx', label="R Peaks")
        plt.legend()
        plt.title(f"R Peak Dectection - Zoomed on first {num_peaks_to_plot} peaks")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.show()
        
        
def extract_rr(r_peaks, params):
    fs = params["sampling_freq"]
    rr_samples = np.diff(r_peaks)
    rr_intervals = rr_samples / fs
    
    return rr_intervals
    