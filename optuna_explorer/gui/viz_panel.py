"""
Visualization Panel — Post-hoc Analysis of Completed Studies

Tab 4 of the Optuna Explorer GUI.
Provides matplotlib-based visualizations embedded in PyQt6:
- Optimization History
- Parameter Importance
- Parallel Coordinates
- Slice Plots
- Contour Plot
- Pareto Front (multi-objective)
"""

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QComboBox, QFrame, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt
import optuna

from optuna_explorer.gui.theme import COLORS, MPL_STYLE


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding in PyQt6."""
    
    def __init__(self, width=8, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor(MPL_STYLE['figure.facecolor'])
        super().__init__(self.fig)


class PlotTab(QWidget):
    """Base widget for a single visualization tab with toolbar."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Toolbar row
        toolbar_row = QHBoxLayout()
        
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet(
            f"background-color: {COLORS['bg_card']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 4px;"
        )
        toolbar_row.addWidget(self.toolbar)
        toolbar_row.addStretch()
        
        self.refresh_btn = QPushButton("↻ Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        toolbar_row.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton("💾 Export PNG")
        self.export_btn.clicked.connect(self.export_png)
        toolbar_row.addWidget(self.export_btn)
        
        layout.addLayout(toolbar_row)
        layout.addWidget(self.canvas, 1)
        
        self._study = None
    
    def set_study(self, study):
        self._study = study
    
    def refresh(self):
        """Override in subclass to redraw the plot."""
        pass
    
    def export_png(self):
        if self.canvas.fig:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Plot", "", "PNG Files (*.png)"
            )
            if path:
                self.canvas.fig.savefig(
                    path, dpi=150, bbox_inches='tight',
                    facecolor=self.canvas.fig.get_facecolor()
                )
    
    def _apply_style(self, ax):
        """Apply dark theme to a matplotlib axes."""
        ax.set_facecolor(MPL_STYLE['axes.facecolor'])
        ax.tick_params(colors=MPL_STYLE['xtick.color'])
        ax.xaxis.label.set_color(MPL_STYLE['axes.labelcolor'])
        ax.yaxis.label.set_color(MPL_STYLE['axes.labelcolor'])
        ax.title.set_color(MPL_STYLE['text.color'])
        for spine in ax.spines.values():
            spine.set_edgecolor(MPL_STYLE['axes.edgecolor'])
        ax.grid(True, alpha=0.2, color=MPL_STYLE['grid.color'])


class OptimizationHistoryTab(PlotTab):
    """Optimization history: objective value vs trial number."""
    
    def refresh(self):
        if self._study is None:
            return
        
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._apply_style(ax)
        
        trials = self._study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        
        if completed:
            x = [t.number for t in completed]
            y = [t.value for t in completed]
            ax.scatter(x, y, c=COLORS['accent_cyan'], s=20, alpha=0.7, 
                      label='Completed', zorder=3)
            
            # Best-so-far line
            best_y = []
            current_best = -float('inf')
            for v in y:
                current_best = max(current_best, v)
                best_y.append(current_best)
            
            # Sort by trial number for proper line
            sorted_pairs = sorted(zip(x, best_y))
            bx = [p[0] for p in sorted_pairs]
            by = [p[1] for p in sorted_pairs]
            ax.plot(bx, by, color=COLORS['accent_green'], linewidth=2,
                   label='Best so far', zorder=4)
        
        if pruned:
            px = [t.number for t in pruned]
            # For pruned trials, show the last reported value
            py = []
            for t in pruned:
                if t.intermediate_values:
                    py.append(max(t.intermediate_values.values()))
                else:
                    py.append(0)
            ax.scatter(px, py, c='gray', s=15, alpha=0.4, marker='x',
                      label='Pruned', zorder=2)
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization History')
        ax.legend(facecolor=COLORS['bg_card'], edgecolor=COLORS['border'],
                 labelcolor=COLORS['text_primary'])
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class ParamImportanceTab(PlotTab):
    """Parameter importance using fANOVA."""
    
    def refresh(self):
        if self._study is None:
            return
        
        completed = [t for t in self._study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 5:
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            self._apply_style(ax)
            ax.text(0.5, 0.5, 'Need ≥5 completed trials',
                   transform=ax.transAxes, ha='center', va='center',
                   color=COLORS['text_secondary'], fontsize=14)
            self.canvas.draw()
            return
        
        try:
            importances = optuna.importance.get_param_importances(self._study)
        except Exception as e:
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            self._apply_style(ax)
            ax.text(0.5, 0.5, f'Error: {e}',
                   transform=ax.transAxes, ha='center', va='center',
                   color=COLORS['accent_coral'], fontsize=12)
            self.canvas.draw()
            return
        
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._apply_style(ax)
        
        names = list(importances.keys())
        values = list(importances.values())
        
        # Reverse for horizontal bar chart (most important at top)
        names = names[::-1]
        values = values[::-1]
        
        colors = [COLORS['accent_cyan'] if v > 0.1 else COLORS['accent_cyan_dim'] 
                 for v in values]
        
        bars = ax.barh(range(len(names)), values, color=colors, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importance')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', color=COLORS['text_secondary'],
                   fontsize=9)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class ParallelCoordinatesTab(PlotTab):
    """Parallel coordinates plot of parameters colored by objective."""
    
    def refresh(self):
        if self._study is None:
            return
        
        completed = [t for t in self._study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            self._apply_style(ax)
            ax.text(0.5, 0.5, 'Need ≥2 completed trials',
                   transform=ax.transAxes, ha='center', va='center',
                   color=COLORS['text_secondary'], fontsize=14)
            self.canvas.draw()
            return
        
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._apply_style(ax)
        
        # Get all param names from first completed trial
        param_names = list(completed[0].params.keys())
        if not param_names:
            return
        
        # Build data matrix (normalized to [0, 1])
        n_params = len(param_names)
        values = np.zeros((len(completed), n_params))
        obj_values = np.array([t.value for t in completed])
        
        # Maps for categorical params
        cat_maps = {}
        
        for j, name in enumerate(param_names):
            col_vals = [t.params.get(name, 0) for t in completed]
            
            # Check if categorical (string values)
            if any(isinstance(v, str) for v in col_vals):
                unique_vals = sorted(set(col_vals))
                cat_maps[name] = unique_vals
                col_vals = [unique_vals.index(v) for v in col_vals]
            
            col_vals = np.array(col_vals, dtype=float)
            vmin, vmax = col_vals.min(), col_vals.max()
            if vmax > vmin:
                values[:, j] = (col_vals - vmin) / (vmax - vmin)
            else:
                values[:, j] = 0.5
        
        # Normalize objective for coloring
        obj_min, obj_max = obj_values.min(), obj_values.max()
        if obj_max > obj_min:
            obj_norm = (obj_values - obj_min) / (obj_max - obj_min)
        else:
            obj_norm = np.ones_like(obj_values) * 0.5
        
        # Plot
        cmap = plt.cm.RdYlGn
        for i in range(len(completed)):
            color = cmap(obj_norm[i])
            ax.plot(range(n_params), values[i], color=color, alpha=0.4, linewidth=1)
        
        # Axes labels
        ax.set_xticks(range(n_params))
        ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Parallel Coordinates (colored by objective)')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap, 
            norm=plt.Normalize(vmin=obj_min, vmax=obj_max)
        )
        sm.set_array([])
        cbar = self.canvas.fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Objective Value', color=COLORS['text_secondary'])
        cbar.ax.yaxis.set_tick_params(color=COLORS['text_secondary'])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS['text_secondary'])
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class SlicePlotsTab(PlotTab):
    """Grid of slice plots: objective vs each parameter."""
    
    def refresh(self):
        if self._study is None:
            return
        
        completed = [t for t in self._study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            self._apply_style(ax)
            ax.text(0.5, 0.5, 'Need ≥2 completed trials',
                   transform=ax.transAxes, ha='center', va='center',
                   color=COLORS['text_secondary'], fontsize=14)
            self.canvas.draw()
            return
        
        param_names = list(completed[0].params.keys())
        n = len(param_names)
        
        if n == 0:
            return
        
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        
        self.canvas.fig.clear()
        
        obj_values = [t.value for t in completed]
        
        for idx, name in enumerate(param_names):
            ax = self.canvas.fig.add_subplot(nrows, ncols, idx + 1)
            self._apply_style(ax)
            
            param_vals = [t.params.get(name, 0) for t in completed]
            
            # Handle categorical
            if any(isinstance(v, str) for v in param_vals):
                unique_vals = sorted(set(param_vals))
                x_vals = [unique_vals.index(v) for v in param_vals]
                ax.scatter(x_vals, obj_values, c=COLORS['accent_cyan'],
                          s=15, alpha=0.6)
                ax.set_xticks(range(len(unique_vals)))
                ax.set_xticklabels(unique_vals, rotation=45, fontsize=7)
            else:
                ax.scatter(param_vals, obj_values, c=COLORS['accent_cyan'],
                          s=15, alpha=0.6)
            
            ax.set_xlabel(name, fontsize=8)
            if idx % ncols == 0:
                ax.set_ylabel('Objective', fontsize=8)
            ax.tick_params(labelsize=7)
        
        self.canvas.fig.suptitle('Slice Plots', color=COLORS['text_primary'],
                                fontsize=14)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class ContourPlotTab(PlotTab):
    """2D contour plot for user-selected parameter pairs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Add parameter selectors before the canvas
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("X Param:"))
        self.x_combo = QComboBox()
        self.x_combo.setMinimumWidth(150)
        selector_layout.addWidget(self.x_combo)
        
        selector_layout.addWidget(QLabel("Y Param:"))
        self.y_combo = QComboBox()
        self.y_combo.setMinimumWidth(150)
        selector_layout.addWidget(self.y_combo)
        
        plot_btn = QPushButton("Plot")
        plot_btn.setObjectName("primary")
        plot_btn.clicked.connect(self.refresh)
        selector_layout.addWidget(plot_btn)
        selector_layout.addStretch()
        
        # Insert selector before the canvas
        self.layout().insertLayout(1, selector_layout)
    
    def set_study(self, study):
        super().set_study(study)
        self._update_param_combos()
    
    def _update_param_combos(self):
        if self._study is None:
            return
        
        completed = [t for t in self._study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return
        
        param_names = list(completed[0].params.keys())
        
        # Only use numeric params
        numeric_params = []
        for name in param_names:
            vals = [t.params.get(name) for t in completed]
            if all(isinstance(v, (int, float)) for v in vals if v is not None):
                numeric_params.append(name)
        
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(numeric_params)
        self.y_combo.addItems(numeric_params)
        
        if len(numeric_params) >= 2:
            self.y_combo.setCurrentIndex(1)
    
    def refresh(self):
        if self._study is None:
            return
        
        x_param = self.x_combo.currentText()
        y_param = self.y_combo.currentText()
        
        if not x_param or not y_param or x_param == y_param:
            return
        
        completed = [t for t in self._study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 3:
            return
        
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._apply_style(ax)
        
        x_vals = np.array([t.params[x_param] for t in completed])
        y_vals = np.array([t.params[y_param] for t in completed])
        obj_vals = np.array([t.value for t in completed])
        
        scatter = ax.scatter(x_vals, y_vals, c=obj_vals, cmap='RdYlGn',
                           s=30, alpha=0.8, edgecolors=COLORS['border'],
                           linewidth=0.5, zorder=3)
        
        # Try contour if enough unique points
        try:
            from scipy.interpolate import griddata
            
            xi = np.linspace(x_vals.min(), x_vals.max(), 50)
            yi = np.linspace(y_vals.min(), y_vals.max(), 50)
            xi, yi = np.meshgrid(xi, yi)
            
            zi = griddata((x_vals, y_vals), obj_vals, (xi, yi), method='cubic')
            
            ax.contourf(xi, yi, zi, levels=15, cmap='RdYlGn', alpha=0.3, zorder=1)
            ax.contour(xi, yi, zi, levels=15, colors=COLORS['border'],
                      alpha=0.3, linewidths=0.5, zorder=2)
        except Exception:
            pass  # Fall back to scatter only
        
        cbar = self.canvas.fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Objective Value', color=COLORS['text_secondary'])
        cbar.ax.yaxis.set_tick_params(color=COLORS['text_secondary'])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS['text_secondary'])
        
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f'Contour: {x_param} vs {y_param}')
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class ParetoFrontTab(PlotTab):
    """Pareto front visualization for multi-objective studies."""
    
    def refresh(self):
        if self._study is None:
            return
        
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._apply_style(ax)
        
        completed = [t for t in self._study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed) < 2:
            ax.text(0.5, 0.5, 'Need ≥2 completed trials',
                   transform=ax.transAxes, ha='center', va='center',
                   color=COLORS['text_secondary'], fontsize=14)
            self.canvas.draw()
            return
        
        # Check if multi-objective
        if hasattr(completed[0], 'values') and completed[0].values and len(completed[0].values) >= 2:
            all_vals = np.array([t.values for t in completed])
            
            # Get Pareto front trials
            try:
                best_trials = self._study.best_trials
                pareto_nums = set(t.number for t in best_trials)
            except Exception:
                pareto_nums = set()
            
            pareto_mask = np.array([t.number in pareto_nums for t in completed])
            
            # Plot all trials
            ax.scatter(all_vals[~pareto_mask, 0], all_vals[~pareto_mask, 1],
                      c=COLORS['text_muted'], s=20, alpha=0.4, label='Dominated')
            
            # Plot Pareto front
            if pareto_mask.any():
                pareto_vals = all_vals[pareto_mask]
                # Sort by first objective for line
                sort_idx = np.argsort(pareto_vals[:, 0])
                pareto_sorted = pareto_vals[sort_idx]
                
                ax.scatter(pareto_sorted[:, 0], pareto_sorted[:, 1],
                          c=COLORS['accent_cyan'], s=50, zorder=5,
                          edgecolors=COLORS['accent_green'], linewidth=1.5,
                          label='Pareto Optimal')
                ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                       color=COLORS['accent_green'], linewidth=1.5,
                       alpha=0.7, zorder=4)
            
            ax.set_xlabel('Val Accuracy')
            ax.set_ylabel('Val F1 (macro)')
            ax.set_title('Pareto Front')
            ax.legend(facecolor=COLORS['bg_card'], edgecolor=COLORS['border'],
                     labelcolor=COLORS['text_primary'])
        else:
            # Single objective - show acc vs f1 from user attrs
            accs = []
            f1s = []
            for t in completed:
                acc = t.user_attrs.get('best_val_acc')
                f1 = t.user_attrs.get('best_val_f1')
                if acc is not None and f1 is not None:
                    accs.append(acc)
                    f1s.append(f1)
            
            if accs:
                ax.scatter(accs, f1s, c=COLORS['accent_cyan'], s=20, alpha=0.6)
                ax.set_xlabel('Val Accuracy')
                ax.set_ylabel('Val F1 (macro)')
                ax.set_title('Accuracy vs F1 (single objective)')
            else:
                ax.text(0.5, 0.5, 'No acc/f1 data available',
                       transform=ax.transAxes, ha='center', va='center',
                       color=COLORS['text_secondary'])
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class VizPanel(QWidget):
    """Main visualization panel with sub-tabs for different plot types."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._study = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Visualizations")
        title.setObjectName("heading")
        header.addWidget(title)
        header.addStretch()
        
        refresh_all_btn = QPushButton("↻ Refresh All")
        refresh_all_btn.setObjectName("primary")
        refresh_all_btn.clicked.connect(self.refresh_all)
        header.addWidget(refresh_all_btn)
        layout.addLayout(header)
        
        # Sub-tabs
        self.sub_tabs = QTabWidget()
        
        self.history_tab = OptimizationHistoryTab()
        self.importance_tab = ParamImportanceTab()
        self.parallel_tab = ParallelCoordinatesTab()
        self.slice_tab = SlicePlotsTab()
        self.contour_tab = ContourPlotTab()
        self.pareto_tab = ParetoFrontTab()
        
        self.sub_tabs.addTab(self.history_tab, "📈 History")
        self.sub_tabs.addTab(self.importance_tab, "🎯 Importance")
        self.sub_tabs.addTab(self.parallel_tab, "═ Parallel Coords")
        self.sub_tabs.addTab(self.slice_tab, "📊 Slice Plots")
        self.sub_tabs.addTab(self.contour_tab, "🗺 Contour")
        self.sub_tabs.addTab(self.pareto_tab, "⚡ Pareto / Acc vs F1")
        
        layout.addWidget(self.sub_tabs, 1)
    
    def set_study(self, study):
        """Set the study for all sub-tabs."""
        self._study = study
        self.history_tab.set_study(study)
        self.importance_tab.set_study(study)
        self.parallel_tab.set_study(study)
        self.slice_tab.set_study(study)
        self.contour_tab.set_study(study)
        self.pareto_tab.set_study(study)
    
    def refresh_all(self):
        """Refresh all visualization tabs."""
        if self._study is None:
            return
        
        self.history_tab.refresh()
        self.importance_tab.refresh()
        self.parallel_tab.refresh()
        self.slice_tab.refresh()
        self.contour_tab.refresh()
        self.pareto_tab.refresh()
    
    def refresh_current(self):
        """Refresh only the currently visible tab."""
        if self._study is None:
            return
        
        current = self.sub_tabs.currentWidget()
        if hasattr(current, 'refresh'):
            current.refresh()
