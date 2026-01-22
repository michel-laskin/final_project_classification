"""
HTML Viewer Module - Creates dark-themed interactive HTML reports with Plotly.
"""

import webbrowser
from pathlib import Path


def create_tabbed_html(figures_dict, output_path="report.html", title="Analysis Report"):
    """
    Create a modern, dark-themed HTML report with tabbed navigation.
    
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
    
    # Build the full HTML
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
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            margin-bottom: 20px;
        }}
        
        h1 {{
            font-size: 2rem;
            font-weight: 300;
            letter-spacing: 2px;
            color: #fff;
            text-transform: uppercase;
        }}
        
        .subtitle {{
            color: #888;
            font-size: 0.9rem;
            margin-top: 8px;
            letter-spacing: 1px;
        }}
        
        .tab-nav {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 20px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}
        
        .tab-btn {{
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #888;
            padding: 12px 24px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            letter-spacing: 0.5px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }}
        
        .tab-btn:hover {{
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
            border-color: rgba(255, 255, 255, 0.2);
        }}
        
        .tab-btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            border-color: transparent;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .tab-content {{
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        
        /* Override Plotly background for dark theme */
        .js-plotly-plot .plotly .main-svg {{
            background: transparent !important;
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
