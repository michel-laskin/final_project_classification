"""
Plotting functions for HRV feature visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_mse(mse_values: np.ndarray):
    """
    Plot multiscale entropy vs scale.
    
    Args:
        mse_values: Array of multiscale entropy values.
    """
    scales = np.arange(1, len(mse_values) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(scales, mse_values, marker='o', linestyle='-', color='b', label='MSE')
    plt.title("Multiscale Entropy (MSE) vs Scale")
    plt.xlabel("Scale")
    plt.ylabel("Sample Entropy")
    plt.xticks(scales)
    plt.legend()
    plt.show()


def plot_mse_heatmap(mse_grid: dict, m_values: list, r_values: list):
    """
    Plot heatmap of MSE for different m and r values.
    
    Args:
        mse_grid: Dictionary mapping (m, r) tuples to MSE value arrays.
        m_values: List of m values used.
        r_values: List of r values used.
    """
    mse_matrix = np.zeros((len(m_values), len(r_values)))
    
    for i, m in enumerate(m_values):
        for j, r in enumerate(r_values):
            mse_matrix[i, j] = np.mean(mse_grid[(m, r)])
    
    plt.figure(figsize=(12, 6))
    im = plt.imshow(mse_matrix, aspect='auto', origin='lower',
                    extent=[r_values[0], r_values[-1], m_values[0], m_values[-1]],
                    cmap='viridis')
    plt.colorbar(im, label='Mean MSE across scales')
    plt.xlabel('r')
    plt.ylabel('m')
    plt.title('Heatmap of Mean Multiscale Entropy (MSE) for different m and r')
    plt.show()


def plot_mse_vs_scale(mse_grid: dict, m_to_plot: list, r_to_plot: list):
    """
    Plot MSE vs Scale for selected m and r combinations.
    
    Args:
        mse_grid: Dictionary mapping (m, r) tuples to MSE value arrays.
        m_to_plot: List of m values to plot.
        r_to_plot: List of r values to plot.
    """
    plt.figure(figsize=(12, 6))
    for m in m_to_plot:
        for r in r_to_plot:
            mse_values = mse_grid[(m, r)]
            scales = np.arange(1, len(mse_values) + 1)
            plt.plot(scales, mse_values, marker='o', linestyle='-', label=f"m={m}, r={r}")
    
    plt.xlabel("Scale")
    plt.ylabel("Sample Entropy (MSE)")
    plt.title("MSE vs Scale for selected m and r combinations")
    plt.legend()
    plt.show()


def plot_hrv_triangular_index(rr_interval: np.ndarray, hz: int = 40, ax=None) -> str:
    """
    Plot HRV Triangular Index histogram.
    
    Args:
        rr_interval: Array of RR intervals in ms.
        hz: Sampling frequency in Hz.
        ax: Optional matplotlib axes for plotting histogram.
        
    Returns:
        Triangular Index value formatted as string with 2 decimal places.
    """
    bin_width = 1000 / hz
    max_rr, min_rr = max(rr_interval), min(rr_interval)
    range_hist = int((max_rr - min_rr) / bin_width)
    hist = [0] * (range_hist + 1)
    for rr in rr_interval:
        hist[int((rr - min_rr) / bin_width)] += 1
    result = len(rr_interval) / max(hist)
    
    if ax is not None:
        num_bins = len(hist)
        bin_times_x = min_rr + (np.arange(num_bins) * bin_width)
        ax.bar(bin_times_x, hist, width=bin_width * 0.9, align='edge', color='skyblue', edgecolor='black')
        ax.set_xlabel('RR Interval Duration (ms)')
        ax.set_ylabel('Count (Number of Beats)')
        ax.set_title(f'Triangular Index: {result:.2f}')
        ax.grid(axis='y', alpha=0.3)
    
    return f'{result:.2f}'


def plot_poincare(rr_interval: np.ndarray, ax=None) -> tuple:
    """
    Plot Poincare plot and calculate SD1/SD2 metrics.
    
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


def plot_interpolation_comparison(original_signal, original_fs, new_signal, new_fs, t_new):
    """
    Plots a comparison between the original signal (dots) and the interpolated signal (line).
    """
    # Recreate original time axis for visualization
    t_original = np.arange(len(original_signal)) / original_fs
    
    plt.figure(figsize=(12, 5))
    
    # Show only the first 5 seconds 
    sec_to_show = 5
    orig_limit = int(sec_to_show * original_fs)
    new_limit = int(sec_to_show * new_fs)
    
    print(f"Loaded {len(original_signal)} samples.")
    print(f"Loaded {len(new_signal)} samples.")
    print(f"Original FS: {original_fs} Hz -> New FS: {new_fs} Hz")

    plt.plot(t_original[:orig_limit], original_signal[:orig_limit], 'ro', label='Original', markersize=4)
    plt.plot(t_new[:new_limit], new_signal[:new_limit], 'b-', label='Interpolated', alpha=0.6)
    
    plt.title(f'Interpolation Check (First {sec_to_show} seconds)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_interpolated_filtered_signal(interpolated_filtered_signal):
    plt.figure(figsize=(10, 4))
    plt.plot(interpolated_filtered_signal)
    plt.title("Filtered ECG Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(interpolated_filtered_signal)
    plt.xlim(0, 800) 
    plt.title("Zoomed Signal (First 800 Samples)")
    plt.show()


def plot_comprehensive_summary(raw_signal, interpolated_signal, filtered_signal, peaks, t_new, original_fs, new_fs, features):
    """
    Plots a 3-panel summary:
    1. Raw vs Interpolated Signal
    2. Filtered Signal with Peaks
    3. Tachogram with HRV Statistics
    """
    
    t_original = np.arange(len(raw_signal)) / original_fs
    
    plt.figure(figsize=(12, 10))

    # Subplot 1: Raw vs Interpolated 
    plt.subplot(3, 1, 1)
    # Define zoom window (first 5 seconds)
    samples_to_show = int(5 * new_fs) 
    orig_samples_to_show = int(5 * original_fs)

    plt.plot(t_original[:orig_samples_to_show], raw_signal[:orig_samples_to_show],'o', label='Original (40Hz)', color='red', markersize=4)
    plt.plot(t_new[:samples_to_show], interpolated_signal[:samples_to_show],'-', label=f'Interpolated ({int(new_fs)}Hz)', color='blue', alpha=0.5)
    
    plt.title('Raw Signal & Interpolation (First 5 Seconds)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Filtered Signal & Peaks
    plt.subplot(3, 1, 2)
    plt.plot(t_new, filtered_signal, label='Filtered Signal (BPF 0.5-8Hz)', color='green')
    plt.plot(t_new[peaks], filtered_signal[peaks], "x", color='red', markersize=8, label='Detected Peaks')
    
    plt.title(f'Filtered Signal & Peaks (Total Peaks: {len(peaks)})')
    plt.xlim(0, 10) 
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    # Subplot 3: Tachogram (RR Intervals)
    plt.subplot(3, 1, 3)
    if len(peaks) > 1:
        rr_intervals_ms = np.diff(t_new[peaks]) * 1000 
        plt.plot(rr_intervals_ms, color='purple', marker='.', linestyle='-')
        plt.title('RR Intervals (Tachogram)')
        plt.xlabel('Beat Number')
        plt.ylabel('RR Interval [ms]')

        info_text = (f"Mean RR: {features['Mean_RR']:.1f} ms\n"
                     f"Est. HR: {features['BPM']:.1f} BPM\n"
                     f"RMSSD: {features['RMSSD']:.1f} ms")
        
        plt.text(0.02, 0.05, info_text, transform=plt.gca().transAxes, 
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
