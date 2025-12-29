"""
HTML Viewer for Plotly Visualizations
Creates a single HTML page with tabbed interface for all plots.
Uses the template from hrv_dashboard.html
"""
import plotly.io as pio
from pathlib import Path
import webbrowser
from datetime import datetime


def create_tabbed_html(figures_dict, output_path="visualizations_report.html", title="HRV Analysis Dashboard"):
    """
    Create a single HTML file with all Plotly figures organized in tabs.
    Uses the existing hrv_dashboard.html template for styling.
    
    Args:
        figures_dict: Dictionary where keys are tab names and values are Plotly figure objects
        output_path: Path to save the HTML file
        title: Title of the report
    
    Returns:
        Path to the created HTML file
    """
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create tab buttons HTML
    tab_buttons_html = ""
    tab_content_html = ""
    plot_scripts = ""
    
    for idx, (tab_name, fig) in enumerate(figures_dict.items()):
        if fig is None:
            continue
            
        active_class = "active" if idx == 0 else ""
        tab_id = tab_name.replace(' ', '_').replace('.', '')
        
        # Tab button
        tab_buttons_html += f'''
            <button class='tab-button {active_class}' onclick='openTab(event, "{tab_id}")'>{tab_name}</button>'''
        
        # Tab content with plot container
        tab_content_html += f'''
        <div id='{tab_id}' class='tab-content {"active" if idx == 0 else ""}'>
            <div class='plot-container'>
                <div id='plot_{tab_id}'></div>
            </div>
        </div>'''
        
        # Convert figure to JSON and create Plotly script
        fig_json = fig.to_json()
        plot_scripts += f'''
        var plotData_{tab_id} = {fig_json};
        Plotly.newPlot('plot_{tab_id}', plotData_{tab_id}.data, plotData_{tab_id}.layout, {{responsive: true}});
        '''
    
    # Complete HTML structure using hrv_dashboard.html template
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>{title}</title>
    <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .dashboard-container {{
            max-width: 1600px;
            margin: 0 auto;
            background: rgba(20, 24, 44, 0.8);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }}
        
        .dashboard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            text-align: center;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }}
        
        .dashboard-header h1 {{
            font-size: 2.5em;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            margin-bottom: 10px;
        }}
        
        .dashboard-header p {{
            font-size: 1.1em;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 300;
        }}
        
        .tabs {{
            display: flex;
            background: rgba(10, 14, 39, 0.6);
            padding: 10px 20px 0 20px;
            overflow-x: auto;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
            flex-wrap: wrap;
        }}
        
        .tab-button {{
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
            margin-bottom: -2px;
            white-space: nowrap;
            position: relative;
            overflow: hidden;
        }}
        
        .tab-button:before {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }}
        
        .tab-button:hover {{
            color: #ffffff;
            background: rgba(102, 126, 234, 0.1);
        }}
        
        .tab-button.active {{
            color: #ffffff;
            background: rgba(102, 126, 234, 0.2);
            font-weight: 600;
        }}
        
        .tab-button.active:before {{
            transform: scaleX(1);
        }}
        
        .tab-content {{
            display: none;
            padding: 30px;
            animation: fadeIn 0.5s ease;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .plot-container {{
            background: rgba(30, 34, 54, 0.6);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .plot-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #808080;
            font-size: 0.9em;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(10, 14, 39, 0.5);
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: rgba(102, 126, 234, 0.5);
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(102, 126, 234, 0.8);
        }}
    </style>
    
</head>
<body>
    <div class='dashboard-container'>
        <div class='dashboard-header'>
            <h1>🫀 {title}</h1>
            <p>Interactive Heart Rate Variability Visualization • Generated on {timestamp}</p>
        </div>
        <div class='tabs'>
            {tab_buttons_html}
        </div>
        {tab_content_html}
        <div class='footer'>
            <p>Generated with Plotly | Interactive HRV Analysis Platform</p>
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {{
            // Hide all tab content
            var tabcontent = document.getElementsByClassName("tab-content");
            for (var i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].classList.remove('active');
            }}
            
            // Remove active class from all buttons
            var tabbuttons = document.getElementsByClassName("tab-button");
            for (var i = 0; i < tabbuttons.length; i++) {{
                tabbuttons[i].classList.remove('active');
            }}
            
            // Show current tab and mark button as active
            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');
        }}
    </script>
    
    <script>
        {plot_scripts}
    </script>
</body>
</html>'''
    
    # Write to file
    output_file = Path(output_path)
    output_file.write_text(html_content, encoding='utf-8')
    
    return output_file


def open_in_browser(html_path):
    """
    Open the HTML file in the default browser.
    
    Args:
        html_path: Path to the HTML file
    """
    html_path = Path(html_path).resolve()
    webbrowser.open(f'file:///{html_path}')
    print(f"\n{'='*60}")
    print(f"✓ Visualizations opened in browser!")
    print(f"✓ File: {html_path}")
    print(f"{'='*60}")
