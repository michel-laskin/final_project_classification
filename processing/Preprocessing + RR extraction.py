"""
RR Extraction Pipeline with Plotly Visualization
Processes all signals from all drug/dose classes and generates interactive HTML visualizations.
"""

import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
import plotly.subplots as sp
import plotly.graph_objects as go


# ==============================================================================
# DEFAULT FILTER PARAMETERS
# ==============================================================================

DEFAULT_PARAMS = {
    "sampling_freq": 40,
    
    # Filtering
    "use_BPF": False,
    "use_averaging": True,
    "averaging_type": "Gaussian",
    "low_threshold": 0.5,
    "high_threshold": 15.0,
    "filter_order": 2,
    "gaussian_sigma": 2,

    "cut_segments": None,
    "cut_segment_start_sec": None,
    "cut_segment_end_sec": None,
    
    # R-Peak Detection
    "min_distance_samples": None,  # Will be calculated dynamically
    "peak_prominence": 0.1,  # Can be adjusted per signal
    "peak_threshold": None,
    "peak_height": None,
    "local_height_windows": None,
    "local_height_start_sec":None,
    "local_height_end_sec":None,
    "local_peak_height":None,
    "min_rr_sec_zebrafish": 0.15,
    
    # RR Interval Output
    "save_rr_intervals": True,  # Save RR interval vectors to NPY
    "save_rr_summary": True  # Save summary report of all RR intervals
}

# OVERRIDE DICTIONARY - Specify custom parameters for specific signals
# Format: "Signal_Name" or "Class_Name": { parameter_overrides }
OVERRIDE_PARAMS = {
     
     "1_Isoproterenol 1 µM/Larva 07": {
        "peak_height": 1150
    },
    "1_Isoproterenol 1 µM/Larva 08": {
        "peak_height": 1350
    },

     "1_Isoproterenol 10 µM/Larva 09": {
        "peak_height": 1350,
        "gaussian_sigma": 2.5
    },

    "1_Isoproterenol 10 µM/Larva 11": {
        "local_height_windows": [
        (34.5, 35, 1500),
        (56, 58, 1570),
        (160, 161.8, 1700),
        (252, 255, 1280),
        (282, 284, 1670)
        ],
        "cut_segment_start_sec": 80.45,
        "cut_segment_end_sec": 82.10,
        "gaussian_sigma": 2.65
    },

    "2_Atropine 1 µM/Larva 12": {
        "cut_segment_start_sec": 50.225,
        "cut_segment_end_sec": 56.55
    },

    "2_Atropine 1 µM/Larva 22": {
        "local_height_start_sec": 68,
        "local_height_end_sec": 98,
        "local_peak_height": 2500,
        "gaussian_sigma": 2.5
    },

     "3_Propranolol 1 µM/Larva 05": {
        "cut_segment_start_sec": 202.725,
        "cut_segment_end_sec": 204.9
    },

    "3_Propranolol 1 µM/Larva 06": {
        "cut_segment_start_sec": 105.8,
        "cut_segment_end_sec": 153.14
    },

    "3_Propranolol 1 µM/Larva 14": {
        "local_height_start_sec": 49,
        "local_height_end_sec": 77,
        "local_peak_height": 2170
    },

    "3_Propranolol 1 µM/Larva 15": {
        "peak_height": 1100
    },

    "3_Propranolol 1 µM/Larva 18": {
        "local_height_start_sec": 150,
        "local_height_end_sec": 210,
        "local_peak_height": 1500,
        "gaussian_sigma": 2.3
    },

    "3_Propranolol 1 µM/Larva 25": {
        "local_height_start_sec": 161,
        "local_height_end_sec": 261,
        "local_peak_height": 1990,
        "gaussian_sigma": 3
    },

    "3_Propranolol 1 µM/Larva 35": {
        "cut_segment_start_sec": 36.02,
        "cut_segment_end_sec": 38.55
    },

     "3_Propranolol 10 µM/Larva 08": {
        "peak_height": 1600
    },

    "3_Propranolol 10 µM/Larva 20": {
    "local_height_windows": [
        (162, 164, 1000),
        (234, 239, 900)
    ]
    },

    "3_Propranolol 100 µM/Larva 01": {
        "cut_segments": [
            (53.65, 56.08),
            (124.125, 126.33),
            (229.65, 231.0)
        ],
        "gaussian_sigma": 3
    },

    "3_Propranolol 100 µM/Larva 02": {
        "gaussian_sigma": 4
    },

    "3_Propranolol 100 µM/Larva 04": {
        "local_height_start_sec": 74,
        "local_height_end_sec": 78,
        "local_peak_height": 700
    },

    "3_Propranolol 100 µM/Larva 10": {
        "peak_height": 845
    },

    "3_Propranolol 100 µM/Larva 12": {
        "peak_height": 1100
    },

    "3_Propranolol 100 µM/Larva 15": {
        "peak_height": 1100,
        "local_height_start_sec": 66,
        "local_height_end_sec": 70,
        "local_peak_height": 1250
    },

    "3_Propranolol 100 µM/Larva 21": {
        "peak_height": 700
    },

    "3_Propranolol 100 µM/Larva 22": {
        "cut_segments": [
            (5.50, 7.92)
        ],
        "gaussian_sigma": 2.7
    },

    "3_Propranolol 100 µM/Larva 23": {
        "peak_height": 800
    },

    "3_Propranolol 100 µM/Larva 24": {
        "peak_height": 1500
    },

    "3_Propranolol 100 µM/Larva 26": {
        "cut_segments": [
            (83.59, 84.97)
        ],
        "gaussian_sigma": 2.5
    },

    "4_Carbachol 10 µM/Larva 03": {
    "peak_height": 1000
    },

    "4_Carbachol 10 µM/Larva 05": {
        "local_height_start_sec": 72,
        "local_height_end_sec": 76,
        "local_peak_height": 975
    },

    "4_Carbachol 10 µM/Larva 09": {
        "local_height_start_sec": 118,
        "local_height_end_sec": 122,
        "local_peak_height": 1670
    },

    "4_Carbachol 10 µM/Larva 14": {
        "peak_height": 1300
    },

    "4_Carbachol 100 µM - second 5 minutes after washout/Larva 04": {
    "peak_height": 750
    },

    "4_Carbachol 100 µM - second 5 minutes after washout/Larva 16": {
        "peak_height": 1630
    },

    "4_Carbachol 100 µM - second 5 minutes after washout/Larva 19": {
        "gaussian_sigma": 3.2
    },

    "4_Carbachol 100 µM - second 5 minutes after washout/Larva 20": {
        "peak_height": 1150
    },

    "Control/1_Control/Larva 06": {
        "cut_segment_start_sec": 278.48,
        "cut_segment_end_sec": 281.43,
        "gaussian_sigma": 2.5
    },

    "Control/2_Control/Larva 12": {
        "peak_height": 450,
        "gaussian_sigma": 3
    },

    "Control/2_Control/Larva 19": {
        "gaussian_sigma": 3
    },

    "Control/2_Control/Larva 24": {
        "peak_height": 1580,
        "gaussian_sigma": 2.5
    },

    "Control/3_Control/Larva 13": {
        "local_height_start_sec": 130,
        "local_height_end_sec": 135,
        "local_peak_height": 1400,
        "gaussian_sigma": 2.8
    },

    "Control/4_Control/Larva 05": {
        "gaussian_sigma": 3
    },

    "Control/4_Control/Larva 12": {
        "gaussian_sigma": 3
    },

    "Control/4_Control/Larva 14": {
        "gaussian_sigma": 3
    },

    "Control/4_Control/Larva 23": {
        "peak_height": 850
    },

    "Control/6_Control/Larva 13": {
        "peak_height": 1350
    },

    "Control/6_Control/Larva 18": {
        "peak_height": 2250
    },

    "Control/6_Control/Larva 29": {
        "peak_height": 1100
    },

    "Control/7_Control/Larva 04": {
        "peak_height": 1000
    },

    "Control/7_Control/Larva 09": {
        "peak_height": 1100,
        "cut_segment_start_sec": 122.22,
        "cut_segment_end_sec": 123.8
    },

    "Control/7_Control/Larva 17": {
        "peak_height": 1500
    }

}

# ==============================================================================
# FILTERING FUNCTIONS
# ==============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def apply_gaussian_filter(data, sigma):
    """Apply Gaussian filter."""
    return gaussian_filter1d(data, sigma=sigma)


def save_rr_intervals_to_npy(rr_intervals, output_dir, larva_name):
    """
    Save RR interval vector to NPY file.
    
    Returns:
        Path to saved NPY file
    """
    npy_filename = f"{larva_name}_RR_intervals.npy"
    npy_path = os.path.join(output_dir, npy_filename)
    
    np.save(npy_path, rr_intervals)
    return npy_path

def filter_signal(raw_signal, params):
    """
    Apply filtering to raw signal based on parameters.
    
    Returns:
        Filtered signal
    """
    filtered = raw_signal.copy()
    
    # Apply Bandpass Filter if enabled
    if params.get("use_BPF", False):
        filtered = butter_bandpass_filter(
            filtered,
            lowcut=params["low_threshold"],
            highcut=params["high_threshold"],
            fs=params["sampling_freq"],
            order=params["filter_order"]
        )
    
    # Apply Gaussian smoothing if enabled
    if params.get("use_averaging", False) and params.get("averaging_type") == "Gaussian":
        filtered = apply_gaussian_filter(filtered, sigma=params["gaussian_sigma"])
    
    return filtered


# ==============================================================================
# PEAK DETECTION & RR EXTRACTION
# ==============================================================================

def detect_peaks_and_extract_rr(filtered_signal, time, params):
    """
    Detect R-peaks and extract RR intervals.
    
    Returns:
        Tuple of (peak_indices, rr_intervals_ms)
    """
    # Calculate minimum distance between peaks
    min_distance = params.get("min_distance_samples")
    if min_distance is None:
        # Calculate from minimum RR interval
        min_rr_sec = params.get("min_rr_sec_zebrafish", 0.15)
        min_distance = int(min_rr_sec * params["sampling_freq"])
    
    # Detect peaks on the full signal
    peaks = find_peaks(
        filtered_signal,
        distance=min_distance,
        prominence=params.get("peak_prominence", 0.1),
        height=params.get("peak_height", None),
        threshold=params.get("peak_threshold", None)
    )[0]

    # Support multiple local height windows
    local_height_windows = params.get("local_height_windows", None)
    
     # Support a single local window too
    if local_height_windows is None:
        local_start = params.get("local_height_start_sec", None)
        local_end = params.get("local_height_end_sec", None)
        local_peak_height = params.get("local_peak_height", None)

        if local_start is not None and local_end is not None and local_peak_height is not None:
            local_height_windows = [(local_start, local_end, local_peak_height)]

    # Apply local window filtering if requested
    if local_height_windows is not None and len(peaks) > 0:
        peak_times = time[peaks]
        keep_mask = np.ones(len(peaks), dtype=bool)

        for start_sec, end_sec, min_height in local_height_windows:
            in_window_mask = (peak_times >= start_sec) & (peak_times <= end_sec)
            failing_mask = in_window_mask & (filtered_signal[peaks] < min_height)
            keep_mask[failing_mask] = False

        peaks = peaks[keep_mask]

    # Extract RR intervals
    if len(peaks) > 1:
        rr_locations = np.diff(peaks)
        rr_intervals_ms = rr_locations * (1000 / params["sampling_freq"])
    else:
        rr_intervals_ms = np.array([])

    return peaks, rr_intervals_ms


# ==============================================================================
# PLOTLY VISUALIZATION
# ==============================================================================

def create_rr_visualization(time, raw_signal, filtered_signal, peaks, params, filename):
    """
    Create interactive Plotly visualization with subplots:
    1. Raw signal
    2. Filtered signal
    3. Raw + Filtered overlay
    
    All plots include detected peaks marked.
    """
    
    # Create subplots
    fig = sp.make_subplots(
        rows=3, cols=1,
        subplot_titles=("Raw Signal", "Filtered Signal", "Raw + Filtered (Overlay)"),
        vertical_spacing=0.12
    )
    
    # Convert peaks to time domain
    peak_times = time[peaks] if len(peaks) > 0 else np.array([])
    peak_values_raw = raw_signal[peaks] if len(peaks) > 0 else np.array([])
    peak_values_filtered = filtered_signal[peaks] if len(peaks) > 0 else np.array([])
    
    # Row 1: Raw Signal
    fig.add_trace(
        go.Scatter(
            x=time, y=raw_signal,
            mode='lines',
            name='Raw Signal',
            line=dict(color='blue', width=1.5),
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Value:</b> %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add peaks to raw signal
    if len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_times, y=peak_values_raw,
                mode='markers',
                name='Detected Peaks',
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate='<b>Peak Time:</b> %{x:.2f}s<br><b>Value:</b> %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Row 2: Filtered Signal
    fig.add_trace(
        go.Scatter(
            x=time, y=filtered_signal,
            mode='lines',
            name='Filtered Signal',
            line=dict(color='green', width=1.5),
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Value:</b> %{y:.4f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add peaks to filtered signal
    if len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_times, y=peak_values_filtered,
                mode='markers',
                name='Detected Peaks',
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate='<b>Peak Time:</b> %{x:.2f}s<br><b>Value:</b> %{y:.4f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Row 3: Raw + Filtered Overlay
    fig.add_trace(
        go.Scatter(
            x=time, y=raw_signal,
            mode='lines',
            name='Raw Signal',
            line=dict(color='blue', width=1.5, dash='dash'),
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Raw:</b> %{y:.4f}<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=filtered_signal,
            mode='lines',
            name='Filtered Signal',
            line=dict(color='green', width=1.5),
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Filtered:</b> %{y:.4f}<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Add peaks to overlay
    if len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_times, y=peak_values_filtered,
                mode='markers',
                name='Detected Peaks',
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate='<b>Peak Time:</b> %{x:.2f}s<br><b>Value:</b> %{y:.4f}<extra></extra>',
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=3, col=1)
    
    fig.update_layout(
        title_text=f"RR Extraction Analysis - {filename}",
        height=1000,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


# ==============================================================================
# FILE PROCESSING
# ==============================================================================

def normalize_larva_name(larva_name):
    """
    Normalize larva folder names like:
    'Larva 10 - 01.04.25' -> 'Larva 10'
    """
    return larva_name.split(" - ")[0].strip()


def normalize_class_name(class_name):
    """
    Normalize class names for override matching.
    """
    return class_name.strip().lower()


def get_signal_specific_params(signal_name, class_name, params):
    """
    Get parameter overrides for a specific signal if they exist.

   Matching is done after normalization:
    - class_name: case-insensitive
    - larva_name: date suffix removed
    """
    final_params = params.copy()

    normalized_signal_name = normalize_larva_name(signal_name)
    normalized_class_name = normalize_class_name(class_name)
    
    # Check for class-level overrides
    if class_name in OVERRIDE_PARAMS:
        final_params.update(OVERRIDE_PARAMS[class_name])
    
    # Check for signal-level overrides (takes precedence)
    if signal_name in OVERRIDE_PARAMS:
        final_params.update(OVERRIDE_PARAMS[signal_name])
    
    # check for class + signal override (highest priority)
    combined_key = f"{class_name}/{signal_name}"
    if combined_key in OVERRIDE_PARAMS:
        final_params.update(OVERRIDE_PARAMS[combined_key])

    # Check for overrides with normalized matching (for flexibility)
    for key, overrides in OVERRIDE_PARAMS.items():
        if "/" not in key:
            continue

        key_parts = key.split("/")
        override_signal_name = key_parts[-1].strip()
        override_class_name = "/".join(key_parts[:-1]).strip()

        if (
            normalize_class_name(override_class_name) == normalized_class_name
            and normalize_larva_name(override_signal_name) == normalized_signal_name
        ):
            final_params.update(overrides)

    return final_params


def cut_signal_segment_and_stitch(time, raw_signal, params):
    """
    Cut one or more segments from the signal and stitch the remaining parts
    together with time continuity and amplitude offset correction.

    Returns:
        stitched_time, stitched_signal
    """

    cut_segments = params.get("cut_segments", None)
    
    # Allow single cut too
    if cut_segments is None:
        start_sec = params.get("cut_segment_start_sec", None)
        end_sec = params.get("cut_segment_end_sec", None)

        if start_sec is None or end_sec is None:
            return time, raw_signal

        cut_segments = [(start_sec, end_sec)]

    stitched_time = time.copy()
    stitched_signal = raw_signal.copy()

    for start_sec, end_sec in sorted(cut_segments, reverse=True):
        before_mask = stitched_time < start_sec
        after_mask = stitched_time > end_sec

        if not np.any(before_mask) or not np.any(after_mask):
            continue

        time_before = stitched_time[before_mask]
        signal_before = stitched_signal[before_mask]

        time_after = stitched_time[after_mask]
        signal_after = stitched_signal[after_mask]

        removed_duration = end_sec - start_sec
        stitched_time_after = time_after - removed_duration

        offset = signal_before[-1] - signal_after[0]
        stitched_signal_after = signal_after + offset

        stitched_time = np.concatenate([time_before, stitched_time_after])
        stitched_signal = np.concatenate([signal_before, stitched_signal_after])

    return stitched_time, stitched_signal


def process_single_file(filepath, class_name, larva_name, params, output_dir):
    """
    Process a single Values.csv file:
    1. Load the signal
    2. Apply filtering
    3. Detect peaks
    4. Create visualization
    5. Save HTML
    """
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        
        if 'Frame' not in df.columns or 'Mean' not in df.columns:
            print(f"  ✗ Skipping {filepath}: Missing 'Frame' or 'Mean' columns")
            return None
        
        # Extract signal and time
        raw_signal = df['Mean'].values
        frames = df['Frame'].values
        time = frames / params['sampling_freq']
        
        # Normalize larva name for override matching
        normalized_larva_name = normalize_larva_name(larva_name)

        # Get signal-specific parameters with overrides
        final_params = get_signal_specific_params(normalized_larva_name, class_name, params)

        # Cut problematic segment if needed and stitch signal
        time, raw_signal = cut_signal_segment_and_stitch(time, raw_signal, final_params)
        
        # Apply filtering
        filtered_signal = filter_signal(raw_signal, final_params)
        
        # Detect peaks and extract RR
        peaks, rr_intervals = detect_peaks_and_extract_rr(filtered_signal, time, final_params)
        
        # Create visualization
        html_filename = f"{larva_name}_processed.html"
        fig = create_rr_visualization(time, raw_signal, filtered_signal, peaks, final_params, html_filename)
        
        # Save HTML
        output_path = os.path.join(output_dir, html_filename)
        fig.write_html(output_path)
        
        # Save RR intervals to NPY if enabled
        rr_npy_path = None
        if final_params.get("save_rr_intervals", True) and len(rr_intervals) > 0:
            rr_npy_path = save_rr_intervals_to_npy(rr_intervals, output_dir, larva_name)
        
        # Print summary
        print(f"  ✓ {larva_name}: {len(peaks)} peaks detected, {len(rr_intervals)} RR intervals extracted")
        
        return {
            'file': filepath,
            'class': class_name,
            'larva': larva_name,
            'num_peaks': len(peaks),
            'num_rr_intervals': len(rr_intervals),
            'rr_intervals': rr_intervals,
            'output_html': output_path,
            'output_npy': rr_npy_path
        }
        
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {str(e)}")
        return None


def process_all_signals(dataset_root, output_root="Processing results"):
    """
    Process all signals in the dataset:
    1. Iterate through all class folders
    2. For each class, iterate through all larva subfolders
    3. Process Values.csv from each larva
    4. Save HTML to class-specific output folder
    5. Special handling for "control" folder with subdirectories
    """
    
    # Create output root directory
    output_path = os.path.join(os.path.dirname(dataset_root), output_root)
    os.makedirs(output_path, exist_ok=True)
    
    results_summary = []
    
    # Iterate through all class folders
    for class_folder in sorted(os.listdir(dataset_root)):
        class_path = os.path.join(dataset_root, class_folder)
        
        if not os.path.isdir(class_path):
            continue
        
        print(f"\nProcessing class: {class_folder}")
        
        # Special handling for "control" folder
        if class_folder.lower() == "control":
            # Create main control output directory
            control_output = os.path.join(output_path, class_folder)
            os.makedirs(control_output, exist_ok=True)
            
            # Iterate through control subfolders (1_Control, 2_Control, etc.)
            for control_subfolder in sorted(os.listdir(class_path)):
                control_subfolder_path = os.path.join(class_path, control_subfolder)
                
                if not os.path.isdir(control_subfolder_path):
                    continue
                
                # Create output subfolder for this control group
                control_subfolder_output = os.path.join(control_output, control_subfolder)
                os.makedirs(control_subfolder_output, exist_ok=True)
                
                print(f"  Processing {control_subfolder}...")
                
                # Iterate through larva folders inside control subfolders
                for larva_folder in sorted(os.listdir(control_subfolder_path)):
                    larva_path = os.path.join(control_subfolder_path, larva_folder)
                    
                    if not os.path.isdir(larva_path):
                        continue
                    
                    # Look for Values.csv
                    values_file = os.path.join(larva_path, "Values.csv")
                    
                    if os.path.exists(values_file):
                        # Extract larva name
                        larva_name = larva_folder.strip()
                        
                        print(f"    Processing {larva_name}...")
                        
                        # Process the file
                        result = process_single_file(
                            values_file,
                            f"{class_folder}/{control_subfolder}",
                            larva_name,
                            DEFAULT_PARAMS,
                            control_subfolder_output
                        )
                        
                        if result:
                            results_summary.append(result)
                    else:
                        print(f"    ⊘ {larva_folder}: Values.csv not found")
        
        # Standard processing for other classes
        else:
            # Create class output directory
            class_output = os.path.join(output_path, class_folder)
            os.makedirs(class_output, exist_ok=True)
            
            # Iterate through larva subfolders
            for larva_folder in sorted(os.listdir(class_path)):
                larva_path = os.path.join(class_path, larva_folder)
                
                if not os.path.isdir(larva_path):
                    continue
                
                # Look for Values.csv
                values_file = os.path.join(larva_path, "Values.csv")
                
                if os.path.exists(values_file):
                    # Extract larva name (remove leading/trailing spaces and special chars)
                    larva_name = larva_folder.strip()
                    
                    print(f"  Processing {larva_name}...")
                    
                    # Process the file
                    result = process_single_file(
                        values_file,
                        class_folder,
                        larva_name,
                        DEFAULT_PARAMS,
                        class_output
                    )
                    
                    if result:
                        results_summary.append(result)
                else:
                    print(f"  ⊘ {larva_folder}: Values.csv not found")
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Processed {len(results_summary)} files successfully")
    print(f"Output saved to: {output_path}")
    print(f"\nClass folders created:")
    for class_folder in sorted(os.listdir(output_path)):
        class_output_path = os.path.join(output_path, class_folder)
        if os.path.isdir(class_output_path):
            if class_folder.lower() == "control":
                # For control folder, count files recursively
                total_html = 0
                total_npy = 0
                for subdir in os.listdir(class_output_path):
                    subdir_path = os.path.join(class_output_path, subdir)
                    if os.path.isdir(subdir_path):
                        html_count = len([f for f in os.listdir(subdir_path) if f.endswith('.html')])
                        npy_count = len([f for f in os.listdir(subdir_path) if f.endswith('_RR_intervals.npy')])
                        total_html += html_count
                        total_npy += npy_count
                        print(f"  • {class_folder}/{subdir}: {html_count} HTML, {npy_count} NPY files")
                print(f"  • {class_folder} (total): {total_html} HTML, {total_npy} NPY files")
            else:
                num_html = len([f for f in os.listdir(class_output_path) if f.endswith('.html')])
                num_npy = len([f for f in os.listdir(class_output_path) if f.endswith('_RR_intervals.npy')])
                print(f"  • {class_folder}: {num_html} HTML, {num_npy} NPY files")
    
    # Save summary report of all RR intervals if enabled
    if DEFAULT_PARAMS.get("save_rr_summary", True) and results_summary:
        summary_df_list = []
        for result in results_summary:
            if result['num_rr_intervals'] > 0:
                rr_data = {
                    'Class': result['class'],
                    'Larva': result['larva'],
                    'Num_Peaks': result['num_peaks'],
                    'Num_RR_Intervals': result['num_rr_intervals'],
                    'Mean_RR_ms': np.mean(result['rr_intervals']),
                    'Std_RR_ms': np.std(result['rr_intervals']),
                    'Min_RR_ms': np.min(result['rr_intervals']),
                    'Max_RR_ms': np.max(result['rr_intervals'])
                }
                summary_df_list.append(rr_data)
        
        if summary_df_list:
            summary_df = pd.DataFrame(summary_df_list)
            summary_path = os.path.join(output_path, "RR_intervals_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary report saved to: {summary_path}")
    
    return results_summary


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Define paths
    DATASET_ROOT = os.path.join(
        os.path.dirname(__file__),
        "..",
        "dataset"
    )
    
    print("RR Extraction Pipeline")
    print("="*80)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Default filter parameters:")
    for key, value in DEFAULT_PARAMS.items():
        if not key.startswith('_'):
            print(f"  • {key}: {value}")
    
    if OVERRIDE_PARAMS:
        print(f"\nSignal-specific overrides:")
        for signal, overrides in OVERRIDE_PARAMS.items():
            print(f"  • {signal}: {overrides}")
    else:
        print("\nNo signal-specific overrides configured")
    
    # Process all signals
    results = process_all_signals(DATASET_ROOT)
    
    print("\nDone!")