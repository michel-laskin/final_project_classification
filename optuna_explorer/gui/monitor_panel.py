"""
Monitor Panel — Live Trial Progress and Controls

Tab 3 of the Optuna Explorer GUI.
Shows real-time optimization progress with live chart,
trial table, current/best trial cards, and start/pause/stop controls.
"""

import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QGridLayout, QFrame, QSplitter, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
from optuna_explorer.gui.theme import COLORS, PYQTGRAPH_BG, PYQTGRAPH_FG


class MetricCard(QFrame):
    """A styled card displaying a metric label and value."""
    
    def __init__(self, title, value="—", parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: {COLORS['bg_card']}; "
            f"border: 1px solid {COLORS['border']}; "
            f"border-radius: 8px; padding: 12px;"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; border: none;"
        )
        layout.addWidget(self.title_label)
        
        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet(
            f"color: {COLORS['accent_cyan']}; font-size: 20px; "
            f"font-weight: bold; border: none;"
        )
        layout.addWidget(self.value_label)
    
    def set_value(self, value, color=None):
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(
                f"color: {color}; font-size: 20px; font-weight: bold; border: none;"
            )


class MonitorPanel(QWidget):
    """Live monitoring panel for optimization progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._trial_values = []
        self._trial_states = []
        self._best_so_far = []
        self._start_time = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # === Top Bar: Progress + Controls ===
        top_bar = QHBoxLayout()
        
        # Progress bar
        progress_container = QVBoxLayout()
        progress_label_row = QHBoxLayout()
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        progress_label_row.addWidget(self.progress_label)
        
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        progress_label_row.addStretch()
        progress_label_row.addWidget(self.time_label)
        progress_container.addLayout(progress_label_row)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_container.addWidget(self.progress_bar)
        top_bar.addLayout(progress_container, 1)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("success")
        self.start_btn.setMinimumHeight(44)
        self.start_btn.setMinimumWidth(100)
        btn_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setObjectName("primary")
        self.pause_btn.setMinimumHeight(44)
        self.pause_btn.setMinimumWidth(100)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setMinimumHeight(44)
        self.stop_btn.setMinimumWidth(100)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        top_bar.addLayout(btn_layout)
        layout.addLayout(top_bar)
        
        # === Metric Cards Row ===
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        
        self.card_completed = MetricCard("Trials Completed", "0")
        self.card_pruned = MetricCard("Trials Pruned", "0")
        self.card_best_value = MetricCard("Best Value", "—")
        self.card_best_trial = MetricCard("Best Trial #", "—")
        self.card_current = MetricCard("Current Trial", "—")
        
        cards_row.addWidget(self.card_completed)
        cards_row.addWidget(self.card_pruned)
        cards_row.addWidget(self.card_best_value)
        cards_row.addWidget(self.card_best_trial)
        cards_row.addWidget(self.card_current)
        
        layout.addLayout(cards_row)
        
        # === Main Content: Chart + Table ===
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Live chart (pyqtgraph)
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        chart_header = QLabel("Optimization History")
        chart_header.setStyleSheet(
            f"color: {COLORS['accent_cyan']}; font-weight: bold; font-size: 14px;"
        )
        chart_layout.addWidget(chart_header)
        
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(PYQTGRAPH_BG)
        self.plot_widget.setLabel('left', 'Objective Value',
                                  color=COLORS['text_secondary'])
        self.plot_widget.setLabel('bottom', 'Trial Number',
                                  color=COLORS['text_secondary'])
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.getAxis('left').setTextPen(COLORS['text_secondary'])
        self.plot_widget.getAxis('bottom').setTextPen(COLORS['text_secondary'])
        
        # Scatter for all trials
        self.scatter_complete = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(COLORS['accent_cyan']),
            size=8, symbol='o',
        )
        self.plot_widget.addItem(self.scatter_complete)
        
        # Scatter for pruned trials
        self.scatter_pruned = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(color=(100, 100, 100, 120)),
            size=6, symbol='x',
        )
        self.plot_widget.addItem(self.scatter_pruned)
        
        # Best-so-far line
        self.best_line = pg.PlotDataItem(
            pen=pg.mkPen(color=COLORS['accent_green'], width=2),
        )
        self.plot_widget.addItem(self.best_line)
        
        chart_layout.addWidget(self.plot_widget)
        splitter.addWidget(chart_container)
        
        # Trial table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        table_header = QLabel("Trial History")
        table_header.setStyleSheet(
            f"color: {COLORS['accent_cyan']}; font-weight: bold; font-size: 14px;"
        )
        table_layout.addWidget(table_header)
        
        self.trial_table = QTableWidget()
        self.trial_table.setColumnCount(7)
        self.trial_table.setHorizontalHeaderLabels([
            "#", "State", "Value", "Val Acc", "Val F1", "Epochs", "Duration"
        ])
        self.trial_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.trial_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.trial_table.setAlternatingRowColors(True)
        self.trial_table.setSortingEnabled(True)
        self.trial_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.trial_table)
        
        splitter.addWidget(table_container)
        splitter.setSizes([400, 300])
        
        layout.addWidget(splitter, 1)
    
    def set_total_trials(self, total):
        """Set the total expected trials."""
        self.progress_bar.setMaximum(total)
    
    def start_timer(self):
        """Mark the start time for elapsed time tracking."""
        self._start_time = time.time()
    
    def update_progress(self, completed, total, pruned=0):
        """Update the progress bar and labels."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(completed)
        self.progress_label.setText(f"Trial {completed} / {total}")
        
        self.card_completed.set_value(str(completed), COLORS['accent_cyan'])
        self.card_pruned.set_value(str(pruned), COLORS['accent_orange'])
        
        # Update elapsed time
        if self._start_time:
            elapsed = time.time() - self._start_time
            elapsed_str = self._format_time(elapsed)
            
            if completed > 0:
                avg_time = elapsed / completed
                remaining = avg_time * (total - completed)
                eta_str = self._format_time(remaining)
                self.time_label.setText(f"Elapsed: {elapsed_str}  |  ETA: {eta_str}")
            else:
                self.time_label.setText(f"Elapsed: {elapsed_str}")
    
    def update_current_trial(self, trial_number):
        """Update the current trial card."""
        self.card_current.set_value(f"#{trial_number}", COLORS['accent_purple'])
    
    def update_best(self, value, trial_number):
        """Update the best trial cards."""
        if value is not None:
            self.card_best_value.set_value(f"{value:.4f}", COLORS['accent_green'])
            self.card_best_trial.set_value(f"#{trial_number}", COLORS['accent_green'])
    
    def add_trial_result(self, trial_number, state, value, val_acc, val_f1, 
                         epochs, duration):
        """Add a completed trial to the chart and table."""
        # Update chart data
        if state == 'COMPLETE' and value is not None:
            self._trial_values.append((trial_number, value))
            self._trial_states.append('complete')
            
            # Update best-so-far
            if not self._best_so_far or value > self._best_so_far[-1][1]:
                self._best_so_far.append((trial_number, value))
            else:
                self._best_so_far.append(
                    (trial_number, self._best_so_far[-1][1])
                )
        elif state == 'PRUNED':
            self._trial_states.append('pruned')
        
        self._update_chart()
        
        # Update table
        row = self.trial_table.rowCount()
        self.trial_table.insertRow(row)
        
        items = [
            str(trial_number),
            state,
            f"{value:.4f}" if value is not None else "—",
            f"{val_acc:.4f}" if val_acc is not None else "—",
            f"{val_f1:.4f}" if val_f1 is not None else "—",
            str(epochs) if epochs else "—",
            f"{duration:.1f}s" if duration else "—",
        ]
        
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Color state column
            if col == 1:
                if state == 'COMPLETE':
                    item.setForeground(
                        pg.mkColor(COLORS['accent_green'])
                    )
                elif state == 'PRUNED':
                    item.setForeground(
                        pg.mkColor(COLORS['accent_orange'])
                    )
                elif state == 'FAIL':
                    item.setForeground(
                        pg.mkColor(COLORS['accent_coral'])
                    )
            
            self.trial_table.setItem(row, col, item)
        
        # Auto-scroll to bottom
        self.trial_table.scrollToBottom()
    
    def _update_chart(self):
        """Refresh the pyqtgraph chart with current data."""
        # Complete trials
        complete_data = [(t, v) for (t, v), s in 
                         zip(self._trial_values, self._trial_states) 
                         if s == 'complete']
        if complete_data:
            x = [d[0] for d in complete_data]
            y = [d[1] for d in complete_data]
            self.scatter_complete.setData(x=x, y=y)
        
        # Best-so-far line
        if self._best_so_far:
            x = [d[0] for d in self._best_so_far]
            y = [d[1] for d in self._best_so_far]
            self.best_line.setData(x=x, y=y)
    
    def load_existing_trials(self, trials):
        """Load trials from an existing study for display."""
        self._trial_values.clear()
        self._trial_states.clear()
        self._best_so_far.clear()
        self.trial_table.setRowCount(0)
        
        for t in trials:
            state = t.state.name
            value = t.value if hasattr(t, 'value') else None
            val_acc = t.user_attrs.get('best_val_acc', None)
            val_f1 = t.user_attrs.get('best_val_f1', None)
            epochs = t.user_attrs.get('epochs_trained', None)
            duration = t.user_attrs.get('duration_seconds', None)
            
            self.add_trial_result(
                t.number, state, value, val_acc, val_f1, epochs, duration
            )
    
    def set_running_state(self, running):
        """Update button states for running/stopped."""
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
    
    def reset(self):
        """Reset all display state."""
        self._trial_values.clear()
        self._trial_states.clear()
        self._best_so_far.clear()
        self._start_time = None
        
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        self.time_label.setText("")
        self.trial_table.setRowCount(0)
        
        self.card_completed.set_value("0")
        self.card_pruned.set_value("0")
        self.card_best_value.set_value("—")
        self.card_best_trial.set_value("—")
        self.card_current.set_value("—")
        
        self.scatter_complete.clear()
        self.scatter_pruned.clear()
        self.best_line.clear()
    
    @staticmethod
    def _format_time(seconds):
        """Format seconds into H:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
