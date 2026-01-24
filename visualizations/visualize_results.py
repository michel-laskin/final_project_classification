"""
Visualization Script for Training Results

This script allows users to select a results.json file from a file explorer
and generates an HTML report with visualizations.
"""

import json
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import plotly.graph_objects as go
import webbrowser
from plotly.subplots import make_subplots


def load_results(results_path: str) -> dict:
    """Load results from a JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_visualizations(results: dict) -> dict:
    """Create Plotly visualizations from results data."""
    figures = {}
    
    history = results['training_history']
    
    # 1. Training curves - show training vs validation
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss: Training vs Validation', 'Accuracy: Training vs Validation')
    )
    
    epochs = list(range(1, len(history['val_loss']) + 1))
    
    # Training loss
    fig.add_trace(go.Scatter(
        x=epochs, y=history['train_loss'],
        mode='lines+markers', name='Train Loss',
        line=dict(color='orange', width=2),
        marker=dict(size=6, color='orange')
    ), row=1, col=1)
    
    # Validation loss
    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_loss'],
        mode='lines+markers', name='Val Loss',
        line=dict(color='purple', width=2),
        marker=dict(size=6, color='purple')
    ), row=1, col=1)
    
    # Training accuracy
    fig.add_trace(go.Scatter(
        x=epochs, y=history['train_acc'],
        mode='lines+markers', name='Train Acc',
        line=dict(color='orange', width=2),
        marker=dict(size=6, color='orange')
    ), row=1, col=2)
    
    # Validation accuracy
    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_acc'],
        mode='lines+markers', name='Val Acc',
        line=dict(color='purple', width=2),
        marker=dict(size=6, color='purple')
    ), row=1, col=2)
   
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    fig.update_layout(
        height=500,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        title_text='Model Performance (Training vs Validation across epochs)',
        showlegend=True,
        xaxis=dict(showgrid=True, gridcolor='#ecf0f1', zeroline=True, zerolinecolor='#bdc3c7', linecolor='black'),
        yaxis=dict(showgrid=True, gridcolor='#ecf0f1', zeroline=True, zerolinecolor='#bdc3c7', linecolor='black'),
        xaxis2=dict(showgrid=True, gridcolor='#ecf0f1', zeroline=True, zerolinecolor='#bdc3c7', linecolor='black'),
        yaxis2=dict(showgrid=True, gridcolor='#ecf0f1', zeroline=True, zerolinecolor='#bdc3c7', linecolor='black'),
    )
    
    figures['1. Training Progress'] = fig
    
    # 2. Confusion Matrix
    cm = np.array(results['test_metrics']['confusion_matrix'])
    
    # Swap columns: Control (0), Propranolol (1) -> Propranolol (0), Control (1)
    # Original columns: [Pred Control, Pred Prop]
    # New columns: [Pred Prop, Pred Control]
    cm_swapped = cm[:, [1, 0]]
    
    # Calculate row-wise percentages (percentage of actual class)
    cm_percentages = cm_swapped.astype('float') / cm_swapped.sum(axis=1)[:, np.newaxis] * 100
    
    # Create text labels with count and percentage
    cm_text = [[f"{cm_swapped[i][j]}<br>({cm_percentages[i][j]:.1f}%)" 
                for j in range(len(cm_swapped[i]))] 
               for i in range(len(cm_swapped))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_percentages,  # Use percentages for color intensity
        x=['Propranolol', 'Control'],
        y=['Control', 'Propranolol'],
        colorscale='Blues',
        text=cm_text,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True,
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Percentage: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confusion Matrix (Test Set)',
        xaxis_title='Predicted',
        yaxis_title='True',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(linecolor='black', type='category'),
        yaxis=dict(linecolor='black', type='category', autorange='reversed'),
        height=500
    )
    
    figures['2. Confusion Matrix'] = fig
    
    # 3. Dataset statistics
    file_stats = results['dataset_stats']['file_stats']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Windows per File', 'HRV Length per File')
    )
    
    # Use simple file index for labels
    filenames = [f"File {i+1}" for i in range(len(file_stats))]
    
    colors = ['#27ae60' if fs['label'] == 1 else '#2980b9' for fs in file_stats]
    
    fig.add_trace(go.Bar(
        x=filenames,
        y=[fs['num_windows'] for fs in file_stats],
        marker=dict(color=colors),
        name='Windows',
        showlegend=False,
        hovertemplate='%{x}<br>Windows: %{y}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=filenames,
        y=[fs['hrv_length'] for fs in file_stats],
        marker=dict(color=colors),
        name='HRV Length',
        showlegend=False,
        hovertemplate='%{x}<br>HRV Length: %{y}<extra></extra>'
    ), row=1, col=2)
    
    fig.update_xaxes(tickangle=90, row=1, col=1)
    fig.update_xaxes(tickangle=90, row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_layout(
        height=700,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        title_text='Dataset Statistics (Green=Propranolol, Blue=Control)',
        xaxis=dict(showgrid=True, gridcolor='#ecf0f1', linecolor='black'),
        yaxis=dict(showgrid=True, gridcolor='#ecf0f1', linecolor='black'),
        xaxis2=dict(showgrid=True, gridcolor='#ecf0f1', linecolor='black'),
        yaxis2=dict(showgrid=True, gridcolor='#ecf0f1', linecolor='black'),
    )
    
    figures['3. Dataset Statistics'] = fig
    
    # 4. Model Configuration Summary
    config = results['config']
    test_metrics = results['test_metrics']
    
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "table"}], [{"type": "table"}]],
        vertical_spacing=0.1,
        subplot_titles=("Test Metrics", "Model Configuration")
    )

    # Metrics Table
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#ecf0f1',
            align='left',
            font=dict(color='black', size=12),
            line_color='#bdc3c7'
        ),
        cells=dict(
            values=[
                ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                [f"{test_metrics['accuracy']:.4f}", f"{test_metrics['precision']:.4f}", f"{test_metrics['recall']:.4f}", f"{test_metrics['f1']:.4f}"]
            ],
            fill_color='white',
            align='left',
            font=dict(color='black', size=12),
            line_color='#bdc3c7',
            height=30
        )
    ), row=1, col=1)

    # Config Table
    config_keys = [
        'Embedding Dim', 'TCN Channels', 'Dropout', 'Learning Rate', 
        'Batch Size', 'Weight Decay', 'HRV Window Size', 'HRV Overlap'
    ]
    config_values = [
        str(config.get('embedding_dim', 'N/A')),
        str(config.get('tcn_channels', 'N/A')),
        str(config.get('dropout', 'N/A')),
        str(config.get('learning_rate', 'N/A')),
        str(config.get('batch_size', 'N/A')),
        str(config.get('weight_decay', 'N/A')),
        str(config.get('hrv_window_size', 'N/A')),
        str(config.get('hrv_overlap', 'N/A'))
    ]

    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Parameter</b>', '<b>Value</b>'],
            fill_color='#ecf0f1',
            align='left',
            font=dict(color='black', size=12),
            line_color='#bdc3c7'
        ),
        cells=dict(
            values=[config_keys, config_values],
            fill_color='white',
            align='left',
            font=dict(color='black', size=12),
            line_color='#bdc3c7',
            height=30
        )
    ), row=2, col=1)
    
    fig.update_layout(
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        title_text=f"Training Session: {results['timestamp']}"
    )
    
    figures['4. Model Config & Metrics'] = fig
    
    return figures


def create_tabbed_html(figures_dict, output_path="report.html", title="Analysis Report"):
    """
    Create a modern, light-themed HTML report with tabbed navigation.
    
    Args:
        figures_dict: Dictionary of {tab_name: plotly_figure}
        output_path: Output HTML file path
        title: Report title
    
    Returns:
        Path to the created HTML file
    """
    
    # Generate tab buttons and content
    tab_buttons = []
    tab_contents = []
    
    for i, (tab_name, fig) in enumerate(figures_dict.items()):
        is_active = "active" if i == 0 else ""
        display = "block" if i == 0 else "none"
        
        # Create tab button
        tab_buttons.append(f'''
            <button class="tab-btn {is_active}" onclick="openTab(event, 'tab{i}')">{tab_name}</button>
        ''')
        
        # Convert figure to HTML div (just the div, not full page)
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
        
        tab_contents.append(f'''
            <div id="tab{i}" class="tab-content" style="display: {display};">
                {fig_html}
            </div>
        ''')
    
    # Build the full HTML with light academic theme
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Times New Roman', 'Georgia', serif;
            background: #ffffff;
            min-height: 100vh;
            color: #1a1a1a;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            margin-bottom: 20px;
            border-bottom: 2px solid #333;
        }}
        
        h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: #1a1a1a;
        }}
        
        .subtitle {{
            color: #555;
            font-size: 0.95rem;
            margin-top: 8px;
            font-style: italic;
        }}
        
        .tab-nav {{
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 20px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        
        .tab-btn {{
            background: #ffffff;
            border: 1px solid #ccc;
            color: #333;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            font-family: 'Segoe UI', sans-serif;
            border-radius: 4px;
            transition: all 0.2s ease;
        }}
        
        .tab-btn:hover {{
            background: #e8e8e8;
            border-color: #999;
        }}
        
        .tab-btn.active {{
            background: #2c3e50;
            color: #fff;
            border-color: #2c3e50;
        }}
        
        .tab-content {{
            background: #ffffff;
            border-radius: 4px;
            padding: 20px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            min-height: 500px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Interactive Analysis Dashboard</p>
        </header>
        
        <nav class="tab-nav">
            {''.join(tab_buttons)}
        </nav>
        
        {''.join(tab_contents)}
    </div>
    
    <script>
        function openTab(evt, tabId) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.style.display = 'none';
            }});
            
            // Remove active class from all buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Show selected tab and mark button as active
            document.getElementById(tabId).style.display = 'block';
            evt.currentTarget.classList.add('active');
            
            // Trigger Plotly resize to fix any layout issues
            window.dispatchEvent(new Event('resize'));
        }}
        
        // Trigger resize on page load to ensure plots render correctly
        window.onload = function() {{
            window.dispatchEvent(new Event('resize'));
        }};
    </script>
</body>
</html>'''
    
    # Write to file
    output_file = Path(output_path)
    output_file.write_text(html_content, encoding='utf-8')
    
    print(f"HTML report created: {output_file.absolute()}")
    return str(output_file.absolute())


def open_in_browser(html_path):
    """Open the HTML file in the default web browser."""
    file_path = Path(html_path).absolute()
    webbrowser.open(f'file://{file_path}')
    print(f"Opened in browser: {file_path}")


def select_results_file() -> str:
    """Open file dialog to select a results.json file."""
    # Hide the root tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Set initial directory to results folder
    initial_dir = Path(__file__).parent.parent / "results"
    if not initial_dir.exists():
        initial_dir = Path(__file__).parent.parent
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select a results.json file",
        initialdir=str(initial_dir),
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path


def main():
    """Main entry point for visualization script."""
    print("="*60)
    print("TRAINING RESULTS VISUALIZER")
    print("="*60)
    
    # Open file picker
    print("\nOpening file picker...")
    results_path = select_results_file()
    
    if not results_path:
        print("No file selected. Exiting.")
        return
    
    print(f"Selected: {results_path}")
    
    # Load results
    print("\nLoading results...")
    results = load_results(results_path)
    print(f"Loaded results from session: {results['timestamp']}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    figures = create_visualizations(results)
    print(f"Created {len(figures)} visualizations")
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    results_dir = Path(results_path).parent
    output_path = results_dir / "report.html"
    
    html_file = create_tabbed_html(
        figures_dict=figures,
        output_path=str(output_path),
        title=f"Zebrafish Classification Report - {results['timestamp']}"
    )
    
    print(f"Report saved to: {html_file}")
    
    # Open in browser
    open_in_browser(html_file)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
