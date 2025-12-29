"""
Interactive HTML Dashboard Generator for HRV Analysis
Creates a modern, dark-themed, tabbed interface for all visualizations
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def generate_dashboard_html(figures_dict: dict, output_file: str = "hrv_dashboard.html"):
    """
    Generate an interactive HTML dashboard with tabs for different plot categories.
    
    Args:
        figures_dict: Dictionary mapping tab names to Plotly figure objects or lists of figures.
                     Example: {
                         'MSE Analysis': [fig1, fig2, fig3],
                         'HRV Metrics': [fig4, fig5],
                         'Signal Processing': fig6
                     }
        output_file: Path to output HTML file.
    """
    
    # Custom CSS for dark, modern styling
    custom_css = """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(20, 24, 44, 0.8);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            text-align: center;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        .dashboard-header h1 {
            font-size: 2.5em;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            margin-bottom: 10px;
        }
        
        .dashboard-header p {
            font-size: 1.1em;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 300;
        }
        
        .tabs {
            display: flex;
            background: rgba(10, 14, 39, 0.6);
            padding: 10px 20px 0 20px;
            overflow-x: auto;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        .tab-button {
            background: transparent;
            border: none;
            color: #a0a0a0;
            padding: 15px 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.3s ease;
            border-radius: 8px 8px 0 0;
            margin-right: 5px;
            white-space: nowrap;
            position: relative;
            overflow: hidden;
        }
        
        .tab-button:before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        
        .tab-button:hover {
            color: #ffffff;
            background: rgba(102, 126, 234, 0.1);
        }
        
        .tab-button.active {
            color: #ffffff;
            background: rgba(102, 126, 234, 0.2);
            font-weight: 600;
        }
        
        .tab-button.active:before {
            transform: scaleX(1);
        }
        
        .tab-content {
            display: none;
            padding: 30px;
            animation: fadeIn 0.5s ease;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .plot-container {
            background: rgba(30, 34, 54, 0.6);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .plot-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
        }
        
        .plot-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #ffffff;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #808080;
            font-size: 0.9em;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(10, 14, 39, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.5);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.8);
        }
    </style>
    """
    
    # JavaScript for tab functionality
    custom_js = """
    <script>
        function openTab(evt, tabName) {
            // Hide all tab content
            var tabcontent = document.getElementsByClassName("tab-content");
            for (var i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove('active');
            }
            
            // Remove active class from all buttons
            var tabbuttons = document.getElementsByClassName("tab-button");
            for (var i = 0; i < tabbuttons.length; i++) {
                tabbuttons[i].classList.remove('active');
            }
            
            // Show current tab and mark button as active
            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');
        }
        
        // Open first tab by default
        window.onload = function() {
            document.querySelector('.tab-button').click();
        };
    </script>
    """
    
    # Start building HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='utf-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>HRV Analysis Dashboard</title>",
        "    <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        custom_css,
        "</head>",
        "<body>",
        "    <div class='dashboard-container'>",
        "        <div class='dashboard-header'>",
        "            <h1>🫀 HRV Analysis Dashboard</h1>",
        "            <p>Interactive Heart Rate Variability Visualization</p>",
        "        </div>",
        "        <div class='tabs'>"
    ]
    
    # Create tab buttons
    for i, tab_name in enumerate(figures_dict.keys()):
        safe_tab_name = tab_name.replace(" ", "_").replace("-", "_")
        html_parts.append(
            f"            <button class='tab-button' onclick='openTab(event, \"{safe_tab_name}\")'>{tab_name}</button>"
        )
    
    html_parts.append("        </div>")
    
    # Create tab content
    for tab_name, figures in figures_dict.items():
        safe_tab_name = tab_name.replace(" ", "_").replace("-", "_")
        html_parts.append(f"        <div id='{safe_tab_name}' class='tab-content'>")
        
        # Handle single figure or list of figures
        if not isinstance(figures, list):
            figures = [figures]
        
        for idx, fig in enumerate(figures):
            if fig is not None:
                # Generate unique div ID
                div_id = f"{safe_tab_name}_plot_{idx}"
                html_parts.append(f"            <div class='plot-container'>")
                html_parts.append(f"                <div id='{div_id}'></div>")
                html_parts.append("            </div>")
        
        html_parts.append("        </div>")
    
    # Add footer
    html_parts.extend([
        "        <div class='footer'>",
        "            <p>Generated with Plotly | Interactive HRV Analysis Platform</p>",
        "        </div>",
        "    </div>",
        custom_js
    ])
    
    # Add Plotly plot creation scripts
    html_parts.append("    <script>")
    
    for tab_name, figures in figures_dict.items():
        safe_tab_name = tab_name.replace(" ", "_").replace("-", "_")
        
        if not isinstance(figures, list):
            figures = [figures]
        
        for idx, fig in enumerate(figures):
            if fig is not None:
                div_id = f"{safe_tab_name}_plot_{idx}"
                
                # Convert figure to JSON
                fig_json = fig.to_json()
                
                html_parts.append(f"""
        var plotData_{div_id} = {fig_json};
        Plotly.newPlot('{div_id}', plotData_{div_id}.data, plotData_{div_id}.layout, {{responsive: true}});
                """)
    
    html_parts.extend([
        "    </script>",
        "</body>",
        "</html>"
    ])
    
    # Write to file
    html_content = "\n".join(html_parts)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Dashboard generated successfully: {output_file}")
    return output_file


# Example usage function
def create_sample_dashboard():
    """
    Create a sample dashboard with example data.
    This demonstrates how to use the dashboard generator.
    """
    from processing import plotly_plots
    
    # Generate sample data
    np.random.seed(42)
    
    # Sample MSE data
    mse_values = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6])
    
    # Sample RR intervals
    rr_intervals = np.random.normal(800, 50, 100)
    
    # Sample signal data
    t = np.linspace(0, 10, 1000)
    original_signal = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, 1000)
    interpolated_signal = np.sin(2 * np.pi * t) + np.random.normal(0, 0.05, 1000)
    
    # Create figures
    mse_fig = plotly_plots.plot_mse(mse_values)
    triangular_fig = plotly_plots.plot_hrv_triangular_index(rr_intervals)
    poincare_fig = plotly_plots.plot_poincare(rr_intervals)
    interp_fig = plotly_plots.plot_interpolation_comparison(
        original_signal, 100, interpolated_signal, 100, t
    )
    
    # Organize into tabs
    figures_dict = {
        'MSE Analysis': [mse_fig],
        'HRV Metrics': [triangular_fig, poincare_fig],
        'Signal Processing': [interp_fig]
    }
    
    # Generate dashboard
    generate_dashboard_html(figures_dict, "hrv_dashboard.html")


if __name__ == "__main__":
    # Run example if executed directly
    create_sample_dashboard()
