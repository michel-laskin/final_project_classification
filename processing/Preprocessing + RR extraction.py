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
    
    # R-Peak Detection
    "min_distance_samples": None,  # Will be calculated dynamically
    "peak_prominence": 0.1,  # Can be adjusted per signal
    "min_rr_sec_zebrafish": 0.15,
    
    # RR Interval Output
    "save_rr_intervals": True,  # Save RR interval vectors to NPY
    "save_rr_summary": True  # Save summary report of all RR intervals
}

# OVERRIDE DICTIONARY - Specify custom parameters for specific signals
# Format: "Signal_Name" or "Class_Name": { parameter_overrides }
OVERRIDE_PARAMS = {
    # Example: "Larva_01": {"gaussian_sigma": 3, "peak_prominence": 0.15}
    # You can add overrides here as needed during analysis
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

def detect_peaks_and_extract_rr(filtered_signal, params):
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
    
    # Detect peaks
    peaks = find_peaks(
        filtered_signal,
        distance=min_distance,
        prominence=params.get("peak_prominence", 0.1)
    )[0]
    
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

def get_signal_specific_params(signal_name, class_name, params):
    """
    Get parameter overrides for a specific signal if they exist.
    
    Priority:
    1. Signal-specific overrides
    2. Class-specific overrides
    3. Default parameters
    """
    final_params = params.copy()
    
    # Check for class-level overrides
    if class_name in OVERRIDE_PARAMS:
        final_params.update(OVERRIDE_PARAMS[class_name])
    
    # Check for signal-level overrides (takes precedence)
    if signal_name in OVERRIDE_PARAMS:
        final_params.update(OVERRIDE_PARAMS[signal_name])
    
    return final_params


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
        
        # Get signal-specific parameters with overrides
        final_params = get_signal_specific_params(larva_name, class_name, params)
        
        # Apply filtering
        filtered_signal = filter_signal(raw_signal, final_params)
        
        # Detect peaks and extract RR
        peaks, rr_intervals = detect_peaks_and_extract_rr(filtered_signal, final_params)
        
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