"""
Dark Theme & Styling Constants for Optuna Explorer GUI

Premium dark theme with cyan/coral accent colors.
"""

# Color palette
COLORS = {
    'bg_dark': '#0d1117',
    'bg_main': '#161b22',
    'bg_surface': '#1c2333',
    'bg_card': '#21262d',
    'bg_hover': '#292e36',
    'bg_input': '#0d1117',
    'border': '#30363d',
    'border_focus': '#00bcd4',
    'text_primary': '#f0f6fc',
    'text_secondary': '#8b949e',
    'text_muted': '#6e7681',
    'accent_cyan': '#00bcd4',
    'accent_cyan_dim': '#006670',
    'accent_coral': '#e94560',
    'accent_green': '#00e676',
    'accent_orange': '#f0883e',
    'accent_purple': '#bc8cff',
    'accent_yellow': '#e3b341',
    'scrollbar': '#30363d',
    'scrollbar_hover': '#484f58',
}

# Matplotlib dark style for embedded plots
MPL_STYLE = {
    'figure.facecolor': COLORS['bg_surface'],
    'axes.facecolor': COLORS['bg_card'],
    'axes.edgecolor': COLORS['border'],
    'axes.labelcolor': COLORS['text_secondary'],
    'text.color': COLORS['text_primary'],
    'xtick.color': COLORS['text_secondary'],
    'ytick.color': COLORS['text_secondary'],
    'grid.color': COLORS['border'],
    'grid.alpha': 0.3,
    'legend.facecolor': COLORS['bg_card'],
    'legend.edgecolor': COLORS['border'],
    'legend.fontsize': 9,
}

# PyQtGraph theme
PYQTGRAPH_BG = COLORS['bg_card']
PYQTGRAPH_FG = COLORS['text_primary']


def get_stylesheet() -> str:
    """Generate the complete QSS stylesheet for the application."""
    c = COLORS
    return f"""
    /* === Global === */
    QMainWindow, QWidget {{
        background-color: {c['bg_main']};
        color: {c['text_primary']};
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 13px;
    }}
    
    /* === Menu Bar === */
    QMenuBar {{
        background-color: {c['bg_dark']};
        color: {c['text_primary']};
        border-bottom: 1px solid {c['border']};
        padding: 2px;
    }}
    QMenuBar::item:selected {{
        background-color: {c['bg_hover']};
        border-radius: 4px;
    }}
    QMenu {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 4px;
    }}
    QMenu::item:selected {{
        background-color: {c['accent_cyan_dim']};
        border-radius: 4px;
    }}
    
    /* === Tab Widget === */
    QTabWidget::pane {{
        border: 1px solid {c['border']};
        border-radius: 8px;
        background-color: {c['bg_surface']};
        margin-top: -1px;
    }}
    QTabBar::tab {{
        background-color: {c['bg_card']};
        color: {c['text_secondary']};
        border: 1px solid {c['border']};
        border-bottom: none;
        padding: 8px 20px;
        margin-right: 2px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        min-width: 100px;
    }}
    QTabBar::tab:selected {{
        background-color: {c['bg_surface']};
        color: {c['accent_cyan']};
        border-bottom: 2px solid {c['accent_cyan']};
        font-weight: bold;
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {c['bg_hover']};
        color: {c['text_primary']};
    }}
    
    /* === Buttons === */
    QPushButton {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {c['bg_hover']};
        border-color: {c['accent_cyan']};
    }}
    QPushButton:pressed {{
        background-color: {c['accent_cyan_dim']};
    }}
    QPushButton:disabled {{
        color: {c['text_muted']};
        border-color: {c['border']};
    }}
    QPushButton#primary {{
        background-color: {c['accent_cyan_dim']};
        border-color: {c['accent_cyan']};
        color: {c['text_primary']};
    }}
    QPushButton#primary:hover {{
        background-color: {c['accent_cyan']};
        color: {c['bg_dark']};
    }}
    QPushButton#danger {{
        border-color: {c['accent_coral']};
        color: {c['accent_coral']};
    }}
    QPushButton#danger:hover {{
        background-color: {c['accent_coral']};
        color: {c['text_primary']};
    }}
    QPushButton#success {{
        border-color: {c['accent_green']};
        color: {c['accent_green']};
    }}
    QPushButton#success:hover {{
        background-color: {c['accent_green']};
        color: {c['bg_dark']};
    }}
    
    /* === Inputs === */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {c['bg_input']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 6px 10px;
        selection-background-color: {c['accent_cyan_dim']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {c['accent_cyan']};
    }}
    QComboBox {{
        background-color: {c['bg_input']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 6px 10px;
        min-height: 20px;
    }}
    QComboBox:focus {{
        border-color: {c['accent_cyan']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        selection-background-color: {c['accent_cyan_dim']};
    }}
    
    /* === Labels === */
    QLabel {{
        color: {c['text_primary']};
    }}
    QLabel#heading {{
        font-size: 18px;
        font-weight: bold;
        color: {c['accent_cyan']};
    }}
    QLabel#subheading {{
        font-size: 14px;
        color: {c['text_secondary']};
    }}
    QLabel#metric {{
        font-size: 24px;
        font-weight: bold;
        color: {c['accent_green']};
    }}
    
    /* === Group Box === */
    QGroupBox {{
        background-color: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 20px;
        font-weight: bold;
        color: {c['accent_cyan']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 4px 12px;
        color: {c['accent_cyan']};
    }}
    
    /* === Table === */
    QTableWidget, QTableView {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        gridline-color: {c['border']};
        selection-background-color: {c['accent_cyan_dim']};
    }}
    QHeaderView::section {{
        background-color: {c['bg_surface']};
        color: {c['text_secondary']};
        border: none;
        border-bottom: 1px solid {c['border']};
        border-right: 1px solid {c['border']};
        padding: 6px 8px;
        font-weight: bold;
    }}
    
    /* === Scrollbar === */
    QScrollBar:vertical {{
        background-color: {c['bg_main']};
        width: 10px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {c['scrollbar']};
        border-radius: 5px;
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {c['scrollbar_hover']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background-color: {c['bg_main']};
        height: 10px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {c['scrollbar']};
        border-radius: 5px;
        min-width: 20px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background-color: {c['scrollbar_hover']};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    
    /* === Progress Bar === */
    QProgressBar {{
        background-color: {c['bg_input']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        text-align: center;
        color: {c['text_primary']};
        min-height: 22px;
    }}
    QProgressBar::chunk {{
        background-color: {c['accent_cyan']};
        border-radius: 5px;
    }}
    
    /* === Status Bar === */
    QStatusBar {{
        background-color: {c['bg_dark']};
        color: {c['text_secondary']};
        border-top: 1px solid {c['border']};
    }}
    
    /* === Checkbox === */
    QCheckBox {{
        color: {c['text_primary']};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {c['border']};
        border-radius: 4px;
        background-color: {c['bg_input']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c['accent_cyan']};
        border-color: {c['accent_cyan']};
    }}
    
    /* === Splitter === */
    QSplitter::handle {{
        background-color: {c['border']};
    }}
    
    /* === ToolTip === */
    QToolTip {{
        background-color: {c['bg_card']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 4px;
        padding: 4px;
    }}
    """
