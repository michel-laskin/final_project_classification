"""
Plotly-based plotting functions for HRV feature visualization.
These functions return Plotly figure objects instead of displaying matplotlib plots.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_mse(mse_values: np.ndarray) -> go.Figure:
    """
    Create Plotly figure for multiscale entropy vs scale.
    
    Args:
        mse_values: Array of multiscale entropy values.
        
    Returns:
        Plotly Figure object.
    """
    scales = np.arange(1, len(mse_values) + 1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scales,
        y=mse_values,
        mode='lines+markers',
        name='MSE',
        line=dict(color='#00d4ff', width=2),
        marker=dict(size=8, color='#00d4ff')
    ))
    
    fig.update_layout(
        title="Multiscale Entropy (MSE) vs Scale",
        xaxis_title="Scale",
        yaxis_title="Sample Entropy",
        template="plotly_dark",
        hovermode='x unified',
        font=dict(size=12)
    )
    
    return fig


def plot_mse_heatmap(mse_grid: dict, m_values: list, r_values: list) -> go.Figure:
    """
    Create Plotly heatmap of MSE for different m and r values.
    
    Args:
        mse_grid: Dictionary mapping (m, r) tuples to MSE value arrays.
        m_values: List of m values used.
        r_values: List of r values used.
        
    Returns:
        Plotly Figure object.
    """
    mse_matrix = np.zeros((len(m_values), len(r_values)))
    
    for i, m in enumerate(m_values):
        for j, r in enumerate(r_values):
            mse_matrix[i, j] = np.mean(mse_grid[(m, r)])
    
    fig = go.Figure(data=go.Heatmap(
        z=mse_matrix,
        x=r_values,
        y=m_values,
        colorscale='Viridis',
        colorbar=dict(title='Mean MSE')
    ))
    
    fig.update_layout(
        title='Heatmap of Mean Multiscale Entropy (MSE) for different m and r',
        xaxis_title='r',
        yaxis_title='m',
        template="plotly_dark"
    )
    
    return fig


def plot_mse_vs_scale(mse_grid: dict, m_to_plot: list, r_to_plot: list) -> go.Figure:
    """
    Create Plotly figure for MSE vs Scale for selected m and r combinations.
    
    Args:
        mse_grid: Dictionary mapping (m, r) tuples to MSE value arrays.
        m_to_plot: List of m values to plot.
        r_to_plot: List of r values to plot.
        
    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    color_idx = 0
    
    for m in m_to_plot:
        for r in r_to_plot:
            mse_values = mse_grid[(m, r)]
            scales = np.arange(1, len(mse_values) + 1)
            
            fig.add_trace(go.Scatter(
                x=scales,
                y=mse_values,
                mode='lines+markers',
                name=f"m={m}, r={r}",
                line=dict(color=colors[color_idx % len(colors)], width=2),
                marker=dict(size=6)
            ))
            color_idx += 1
    
    fig.update_layout(
        title="MSE vs Scale for selected m and r combinations",
        xaxis_title="Scale",
        yaxis_title="Sample Entropy (MSE)",
        template="plotly_dark",
        hovermode='x unified',
        legend=dict(x=1.05, y=1)
    )
    
    return fig


def plot_hrv_triangular_index(rr_interval: np.ndarray, hz: int = 40) -> go.Figure:
    """
    Create Plotly histogram for HRV Triangular Index.
    
    Args:
        rr_interval: Array of RR intervals in ms.
        hz: Sampling frequency in Hz.
        
    Returns:
        Plotly Figure object.
    """
    bin_width = 1000 / hz
    max_rr, min_rr = max(rr_interval), min(rr_interval)
    range_hist = int((max_rr - min_rr) / bin_width)
    hist = [0] * (range_hist + 1)
    
    for rr in rr_interval:
        hist[int((rr - min_rr) / bin_width)] += 1
    
    result = len(rr_interval) / max(hist)
    
    num_bins = len(hist)
    bin_times_x = min_rr + (np.arange(num_bins) * bin_width)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_times_x,
        y=hist,
        width=bin_width * 0.9,
        marker=dict(
            color='#00d4ff',
            line=dict(color='#ffffff', width=1)
        ),
        name='Count'
    ))
    
    fig.update_layout(
        title=f'HRV Triangular Index: {result:.2f}',
        xaxis_title='RR Interval Duration (ms)',
        yaxis_title='Count (Number of Beats)',
        template="plotly_dark",
        showlegend=False
    )
    
    return fig


def plot_poincare(rr_interval: np.ndarray) -> go.Figure:
    """
    Create Plotly Poincare plot with SD1/SD2 metrics.
    
    Args:
        rr_interval: Array of RR intervals in ms.
        
    Returns:
        Plotly Figure object.
    """
    x = rr_interval[:-1]
    y = rr_interval[1:]
    diff = y - x
    sd1 = np.std(diff, ddof=1) / np.sqrt(2)
    total = y + x
    sd2 = np.std(total, ddof=1) / np.sqrt(2)
    mean_rr = np.mean(rr_interval)
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(color='#00d4ff', size=6, opacity=0.6),
        name='RR Intervals'
    ))
    
    # Line of identity
    min_val = min(min(x), min(y)) - 10
    max_val = max(max(x), max(y)) + 10
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=1),
        name='Line of Identity',
        opacity=0.3
    ))
    
    # Ellipse (approximated with points)
    theta = np.linspace(0, 2*np.pi, 100)
    # Rotate by 45 degrees
    angle_rad = np.radians(45)
    ellipse_x = sd2 * np.cos(theta) * np.cos(angle_rad) - sd1 * np.sin(theta) * np.sin(angle_rad) + mean_rr
    ellipse_y = sd2 * np.cos(theta) * np.sin(angle_rad) + sd1 * np.sin(theta) * np.cos(angle_rad) + mean_rr
    
    fig.add_trace(go.Scatter(
        x=ellipse_x,
        y=ellipse_y,
        mode='lines',
        line=dict(color='red', width=2),
        name='Fitted Ellipse'
    ))
    
    fig.update_layout(
        title=f'Poincaré Plot<br>SD1={sd1:.2f}ms, SD2={sd2:.2f}ms',
        xaxis_title='RR<sub>n</sub> (ms)',
        yaxis_title='RR<sub>n+1</sub> (ms)',
        template="plotly_dark",
        hovermode='closest',
        xaxis=dict(range=[min_val, max_val]),
        yaxis=dict(range=[min_val, max_val], scaleanchor="x", scaleratio=1)
    )
    
    return fig


def plot_interpolation_comparison(original_signal, original_fs, new_signal, new_fs, t_new) -> go.Figure:
    """
    Create Plotly comparison between original and interpolated signal.
    
    Args:
        original_signal: Original signal array.
        original_fs: Original sampling frequency.
        new_signal: Interpolated signal array.
        new_fs: New sampling frequency.
        t_new: Time array for new signal.
        
    Returns:
        Plotly Figure object.
    """
    t_original = np.arange(len(original_signal)) / original_fs
    
    # Show only the first 5 seconds
    sec_to_show = 5
    orig_limit = int(sec_to_show * original_fs)
    new_limit = int(sec_to_show * new_fs)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t_original[:orig_limit],
        y=original_signal[:orig_limit],
        mode='markers',
        marker=dict(color='red', size=4),
        name='Original'
    ))
    
    fig.add_trace(go.Scatter(
        x=t_new[:new_limit],
        y=new_signal[:new_limit],
        mode='lines',
        line=dict(color='#00d4ff', width=2),
        opacity=0.6,
        name='Interpolated'
    ))
    
    fig.update_layout(
        title=f'Interpolation Check (First {sec_to_show} seconds)',
        xaxis_title='Time [sec]',
        yaxis_title='Amplitude',
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def plot_interpolated_filtered_signal(interpolated_filtered_signal) -> tuple:
    """
    Create two Plotly figures: full signal and zoomed view.
    
    Args:
        interpolated_filtered_signal: Filtered signal array.
        
    Returns:
        Tuple of (full_fig, zoomed_fig).
    """
    # Full signal
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=interpolated_filtered_signal,
        mode='lines',
        line=dict(color='#00ff88', width=1),
        name='Filtered Signal'
    ))
    fig1.update_layout(
        title="Filtered PPG Signal",
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        template="plotly_dark"
    )
    
    # Zoomed signal
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=interpolated_filtered_signal[:800],
        mode='lines',
        line=dict(color='#00ff88', width=1),
        name='Filtered Signal'
    ))
    fig2.update_layout(
        title="Zoomed Signal (First 800 Samples)",
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        template="plotly_dark"
    )
    
    return fig1, fig2


def plot_comprehensive_summary(raw_signal, interpolated_signal, filtered_signal, peaks, t_new, 
                               original_fs, new_fs, features) -> go.Figure:
    """
    Create comprehensive 3-panel summary plot using Plotly subplots.
    
    Args:
        raw_signal: Raw signal array.
        interpolated_signal: Interpolated signal array.
        filtered_signal: Filtered signal array.
        peaks: Peak indices array.
        t_new: Time array for new signal.
        original_fs: Original sampling frequency.
        new_fs: New sampling frequency.
        features: Dictionary of HRV features.
        
    Returns:
        Plotly Figure object with subplots.
    """
    t_original = np.arange(len(raw_signal)) / original_fs
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Raw Signal & Interpolation (First 5 Seconds)',
            f'Filtered Signal & Peaks (Total Peaks: {len(peaks)})',
            'RR Intervals (Tachogram)'
        ),
        vertical_spacing=0.1
    )
    
    # Subplot 1: Raw vs Interpolated
    samples_to_show = int(5 * new_fs)
    orig_samples_to_show = int(5 * original_fs)
    
    fig.add_trace(go.Scatter(
        x=t_original[:orig_samples_to_show],
        y=raw_signal[:orig_samples_to_show],
        mode='markers',
        marker=dict(color='red', size=4),
        name=f'Original ({original_fs}Hz)',
        legendgroup='group1'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=t_new[:samples_to_show],
        y=interpolated_signal[:samples_to_show],
        mode='lines',
        line=dict(color='#00d4ff', width=2),
        opacity=0.5,
        name=f'Interpolated ({int(new_fs)}Hz)',
        legendgroup='group1'
    ), row=1, col=1)
    
    # Subplot 2: Filtered Signal & Peaks
    # Show first 10 seconds
    ten_sec_limit = int(10 * new_fs)
    peaks_in_range = peaks[peaks < ten_sec_limit]
    
    fig.add_trace(go.Scatter(
        x=t_new[:ten_sec_limit],
        y=filtered_signal[:ten_sec_limit],
        mode='lines',
        line=dict(color='#00ff88', width=1.5),
        name='Filtered Signal (BPF 0.5-8Hz)',
        legendgroup='group2'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=t_new[peaks_in_range],
        y=filtered_signal[peaks_in_range],
        mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name='Detected Peaks',
        legendgroup='group2'
    ), row=2, col=1)
    
    # Subplot 3: Tachogram
    if len(peaks) > 1:
        rr_intervals_ms = np.diff(t_new[peaks]) * 1000
        
        fig.add_trace(go.Scatter(
            y=rr_intervals_ms,
            mode='lines+markers',
            line=dict(color='#bf00ff', width=2),
            marker=dict(size=4),
            name='RR Intervals',
            legendgroup='group3'
        ), row=3, col=1)
        
        # Add text annotation for stats
        info_text = (f"Mean RR: {features['Mean_RR']:.1f} ms<br>"
                    f"Est. HR: {features['BPM']:.1f} BPM<br>"
                    f"RMSSD: {features['RMSSD']:.1f} ms")
        
        fig.add_annotation(
            text=info_text,
            xref="x3", yref="y3",
            x=0.02, y=0.95,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='white',
            borderwidth=1,
            font=dict(size=10, color='white'),
            row=3, col=1
        )
    
    # Update axes
    fig.update_xaxes(title_text="Time [sec]", row=1, col=1)
    fig.update_xaxes(title_text="Time [sec]", row=2, col=1)
    fig.update_xaxes(title_text="Beat Number", row=3, col=1)
    
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="RR Interval [ms]", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=1000,
        template="plotly_dark",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def visualize_windows(signal, t_axis, windows, max_windows_to_show=5, show=True):
    """
    Visualize the signal divided into windows using Plotly.
    
    Args:
        signal: Original signal array.
        t_axis: Time axis.
        windows: List of window dictionaries.
        max_windows_to_show: Maximum number of windows to highlight.
        show: Whether to display the figure immediately.
        
    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    
    # Plot the full signal
    fig.add_trace(go.Scatter(
        x=t_axis,
        y=signal,
        mode='lines',
        line=dict(color='#0099ff', width=1),
        opacity=0.5,
        name='Signal'
    ))
    
    # Highlight windows with different colors
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
    
    for i, window in enumerate(windows[:max_windows_to_show]):
        start_t = window['start_time']
        end_t = window['end_time']
        color_idx = i % len(colors)
        
        # Add shaded region for window
        fig.add_vrect(
            x0=start_t, x1=end_t,
            fillcolor=colors[color_idx],
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text=f"W{i+1} ({window['num_peaks']}p)",
            annotation_position="top left"
        )
        
        # Mark peaks in this window
        peak_times = window['t_axis'][window['peaks']] + window['start_time']
        peak_values = window['signal'][window['peaks']]
        fig.add_trace(go.Scatter(
            x=peak_times,
            y=peak_values,
            mode='markers',
            marker=dict(color='red', size=8, symbol='star'),
            name=f'Window {i+1} peaks',
            showlegend=(i == 0)  # Only show one legend entry for peaks
        ))
    
    fig.update_layout(
        title=f'Signal Windowing (showing {min(len(windows), max_windows_to_show)} of {len(windows)} windows)',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        template='plotly_dark',
        height=600,
        hovermode='x unified'
    )
    
    if show:
        fig.show()
    return fig


def visualize_feature_heatmap(features_tensor, window_metadata, feature_names=None, max_features_to_show=30, show=True):
    """
    Visualize features across windows as a heatmap using Plotly.
    
    Args:
        features_tensor: Tensor of shape [num_windows, 1, num_features].
        window_metadata: List of metadata for each window.
        feature_names: Optional list of feature names.
        max_features_to_show: Maximum number of features to display.
        show: Whether to display the figure immediately.
        
    Returns:
        Plotly Figure object.
    """
    import torch
    
    # Convert to numpy and squeeze
    features = features_tensor.squeeze(1).cpu().numpy()  # [num_windows, num_features]
    num_windows, num_features = features.shape
    
    # Limit features if too many
    if num_features > max_features_to_show:
        features = features[:, :max_features_to_show]
        num_features = max_features_to_show
    
    # Window labels
    window_labels = [f"W{i}" for i in range(num_windows)]
    
    # Feature labels
    if feature_names and len(feature_names) >= num_features:
        feature_labels = feature_names[:num_features]
    else:
        feature_labels = [f"F{i}" for i in range(num_features)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=features.T,  # Transpose so features are on Y-axis
        x=window_labels,
        y=feature_labels,
        colorscale='Viridis',
        colorbar=dict(title='Feature Value')
    ))
    
    fig.update_layout(
        title='Feature Values Across Windows',
        xaxis_title='Window Index',
        yaxis_title='Feature Index',
        template='plotly_dark',
        height=800,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')  # Reverse Y-axis to match matplotlib convention
    )
    
    if show:
        fig.show()
    return fig
