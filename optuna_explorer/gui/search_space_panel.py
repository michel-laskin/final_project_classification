"""
Search Space Panel — Configure Hyperparameter Ranges

Tab 2 of the Optuna Explorer GUI.
Provides a form-based editor for enabling/disabling parameters
and adjusting their search ranges.
"""

import copy
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QFormLayout, QScrollArea,
    QLineEdit, QDoubleSpinBox, QSpinBox, QFrame,
)
from PyQt6.QtCore import Qt
from optuna_explorer.search_space import DEFAULT_SEARCH_SPACE, get_groups
from optuna_explorer.gui.theme import COLORS


class SearchSpacePanel(QWidget):
    """Interactive editor for the Optuna search space configuration."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = copy.deepcopy(DEFAULT_SEARCH_SPACE)
        self._widgets = {}  # name -> dict of widgets
        self._setup_ui()
    
    def _setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Search Space Configuration")
        title.setObjectName("heading")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        reset_btn = QPushButton("↻ Reset Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        header_layout.addWidget(reset_btn)
        
        outer_layout.addLayout(header_layout)
        
        desc = QLabel("Enable/disable parameters and adjust ranges. Disabled parameters use their default values.")
        desc.setObjectName("subheading")
        desc.setWordWrap(True)
        outer_layout.addWidget(desc)
        
        # Scrollable area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(12)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        
        # Group parameters
        groups = get_groups()
        for group_name in groups:
            group_box = QGroupBox(group_name)
            group_layout = QFormLayout(group_box)
            group_layout.setSpacing(10)
            group_layout.setContentsMargins(16, 24, 16, 12)
            
            for param_name, spec in self._config.items():
                if spec['group'] != group_name:
                    continue
                
                row_widget = self._create_param_row(param_name, spec)
                group_layout.addRow(row_widget)
            
            scroll_layout.addWidget(group_box)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        outer_layout.addWidget(scroll)
    
    def _create_param_row(self, name, spec):
        """Create a row widget for a single parameter."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        widgets = {}
        
        # Enable checkbox
        cb = QCheckBox()
        cb.setChecked(spec.get('enabled', True))
        cb.stateChanged.connect(lambda state, n=name: self._on_toggle(n, state))
        layout.addWidget(cb)
        widgets['enabled'] = cb
        
        # Parameter name + description
        name_label = QLabel(f"<b>{name}</b>")
        name_label.setMinimumWidth(180)
        name_label.setToolTip(spec.get('description', ''))
        layout.addWidget(name_label)
        
        # Type label
        type_label = QLabel(f"[{spec['type']}]")
        type_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        type_label.setFixedWidth(80)
        layout.addWidget(type_label)
        
        param_type = spec['type']
        
        if param_type in ('int',):
            # Min/Max spinboxes
            layout.addWidget(QLabel("Min:"))
            min_spin = QSpinBox()
            min_spin.setRange(0, 10000)
            min_spin.setValue(spec['low'])
            layout.addWidget(min_spin)
            widgets['low'] = min_spin
            
            layout.addWidget(QLabel("Max:"))
            max_spin = QSpinBox()
            max_spin.setRange(0, 10000)
            max_spin.setValue(spec['high'])
            layout.addWidget(max_spin)
            widgets['high'] = max_spin
            
        elif param_type in ('float', 'log_float'):
            layout.addWidget(QLabel("Min:"))
            min_spin = QDoubleSpinBox()
            min_spin.setDecimals(6)
            min_spin.setRange(0.000001, 1000)
            min_spin.setValue(spec['low'])
            layout.addWidget(min_spin)
            widgets['low'] = min_spin
            
            layout.addWidget(QLabel("Max:"))
            max_spin = QDoubleSpinBox()
            max_spin.setDecimals(6)
            max_spin.setRange(0.000001, 1000)
            max_spin.setValue(spec['high'])
            layout.addWidget(max_spin)
            widgets['high'] = max_spin
            
        elif param_type == 'categorical':
            layout.addWidget(QLabel("Choices:"))
            choices_input = QLineEdit()
            choices_str = ", ".join(str(c) for c in spec['choices'])
            choices_input.setText(choices_str)
            choices_input.setMinimumWidth(200)
            layout.addWidget(choices_input, 1)
            widgets['choices'] = choices_input
        
        # Default value
        default_label = QLabel(f"(default: {spec['default']})")
        default_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        layout.addWidget(default_label)
        
        layout.addStretch()
        
        self._widgets[name] = widgets
        return row
    
    def _on_toggle(self, name, state):
        """Handle parameter enable/disable toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self._config[name]['enabled'] = enabled
        
        # Visually dim disabled params
        widgets = self._widgets.get(name, {})
        for key, w in widgets.items():
            if key != 'enabled':
                w.setEnabled(enabled)
    
    def _reset_defaults(self):
        """Reset all parameters to defaults."""
        self._config = copy.deepcopy(DEFAULT_SEARCH_SPACE)
        
        for name, spec in self._config.items():
            widgets = self._widgets.get(name, {})
            
            if 'enabled' in widgets:
                widgets['enabled'].setChecked(spec.get('enabled', True))
            
            if 'low' in widgets:
                widgets['low'].setValue(spec['low'])
            if 'high' in widgets:
                widgets['high'].setValue(spec['high'])
            if 'choices' in widgets:
                choices_str = ", ".join(str(c) for c in spec['choices'])
                widgets['choices'].setText(choices_str)
    
    def get_search_space(self):
        """
        Read current search space configuration from the UI.
        
        Returns:
            Deep copy of the search space config with UI values applied.
        """
        config = copy.deepcopy(self._config)
        
        for name, spec in config.items():
            widgets = self._widgets.get(name, {})
            
            if 'enabled' in widgets:
                spec['enabled'] = widgets['enabled'].isChecked()
            
            if 'low' in widgets:
                spec['low'] = widgets['low'].value()
            if 'high' in widgets:
                spec['high'] = widgets['high'].value()
            
            if 'choices' in widgets:
                text = widgets['choices'].text().strip()
                if text:
                    # Parse choices - try to preserve types
                    raw_choices = [c.strip() for c in text.split(',')]
                    parsed = []
                    for c in raw_choices:
                        if not c:
                            continue
                        try:
                            parsed.append(int(c))
                        except ValueError:
                            try:
                                parsed.append(float(c))
                            except ValueError:
                                parsed.append(c)
                    spec['choices'] = parsed
        
        return config
