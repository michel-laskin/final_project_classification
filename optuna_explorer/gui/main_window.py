"""
Main Window — Optuna Explorer GUI

Central QMainWindow with tab layout, menu bar, status bar,
and optimization worker thread orchestration.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import threading
import copy
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QMenuBar, QMenu,
    QMessageBox, QApplication, QLabel, QHBoxLayout, QWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QAction
import optuna

from optuna_explorer.study_manager import StudyManager
from optuna_explorer.search_space import DEFAULT_SEARCH_SPACE

# Lazy imports for torch-dependent modules (deferred to avoid DLL issues at startup)
_objective_module = None

def _get_objective_module():
    global _objective_module
    if _objective_module is None:
        from optuna_explorer import objective as _mod
        _objective_module = _mod
    return _objective_module

from optuna_explorer.gui.theme import get_stylesheet, COLORS
from optuna_explorer.gui.study_panel import StudyPanel
from optuna_explorer.gui.search_space_panel import SearchSpacePanel
from optuna_explorer.gui.monitor_panel import MonitorPanel
from optuna_explorer.gui.viz_panel import VizPanel


class OptimizationWorker(QThread):
    """
    Background worker thread that runs Optuna study.optimize().
    
    Emits signals for GUI updates after each trial and epoch.
    """
    
    trial_completed = pyqtSignal(dict)  # trial result info
    epoch_update = pyqtSignal(dict)     # epoch progress info
    optimization_done = pyqtSignal()    # all trials finished
    error_occurred = pyqtSignal(str)    # error message
    
    def __init__(self, study, objective_fn, n_trials, parent=None):
        super().__init__(parent)
        self.study = study
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self._stop_event = threading.Event()
    
    def run(self):
        try:
            # Callback for after each trial
            def trial_callback(study, trial):
                if self._stop_event.is_set():
                    study.stop()
                    return
                
                info = {
                    'trial_number': trial.number,
                    'state': trial.state.name,
                    'value': trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None,
                    'val_acc': trial.user_attrs.get('best_val_acc'),
                    'val_f1': trial.user_attrs.get('best_val_f1'),
                    'epochs': trial.user_attrs.get('epochs_trained'),
                    'duration': trial.user_attrs.get('duration_seconds'),
                    'params': trial.params,
                }
                self.trial_completed.emit(info)
            
            self.study.optimize(
                self.objective_fn,
                n_trials=self.n_trials,
                callbacks=[trial_callback],
                show_progress_bar=False,
            )
            
        except Exception as e:
            if "stopped by user" not in str(e).lower():
                self.error_occurred.emit(str(e))
        finally:
            self.optimization_done.emit()
    
    def stop(self):
        self._stop_event.set()
    
    @property
    def stop_event(self):
        return self._stop_event


class MainWindow(QMainWindow):
    """Main application window for Optuna Explorer."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Optuna Explorer — HRV Classification Pipeline")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # State
        self.study_manager = StudyManager()
        self.feature_cache = None  # Lazy init when optimization starts
        self.current_study = None
        self.current_study_name = None
        self.current_objective_type = None
        self.worker = None
        self.total_trials = 0
        self.completed_trials = 0
        self.pruned_trials = 0
        
        # Build UI
        self._setup_menu()
        self._setup_tabs()
        self._setup_statusbar()
        self._connect_signals()
        
        # Apply theme
        self.setStyleSheet(get_stylesheet())
    
    def _setup_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Study", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Study", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Study menu
        study_menu = menubar.addMenu("Study")
        
        start_action = QAction("Start Optimization", self)
        start_action.setShortcut("F5")
        start_action.triggered.connect(self._on_start)
        study_menu.addAction(start_action)
        
        pause_action = QAction("Pause", self)
        pause_action.setShortcut("F6")
        pause_action.triggered.connect(self._on_pause)
        study_menu.addAction(pause_action)
        
        stop_action = QAction("Stop", self)
        stop_action.setShortcut("F7")
        stop_action.triggered.connect(self._on_stop)
        study_menu.addAction(stop_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        refresh_viz = QAction("Refresh Visualizations", self)
        refresh_viz.setShortcut("F9")
        refresh_viz.triggered.connect(self._refresh_viz)
        view_menu.addAction(refresh_viz)
    
    def _setup_tabs(self):
        self.tabs = QTabWidget()
        
        self.study_panel = StudyPanel(self.study_manager)
        self.search_space_panel = SearchSpacePanel()
        self.monitor_panel = MonitorPanel()
        self.viz_panel = VizPanel()
        
        self.tabs.addTab(self.study_panel, "📋 Study")
        self.tabs.addTab(self.search_space_panel, "🔧 Search Space")
        self.tabs.addTab(self.monitor_panel, "📡 Monitor")
        self.tabs.addTab(self.viz_panel, "📊 Visualizations")
        
        self.setCentralWidget(self.tabs)
    
    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.status_study = QLabel("No study loaded")
        self.status_study.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.statusbar.addWidget(self.status_study)
        
        self.status_progress = QLabel("")
        self.status_progress.setStyleSheet(f"color: {COLORS['accent_cyan']};")
        self.statusbar.addPermanentWidget(self.status_progress)
    
    def _connect_signals(self):
        # Study panel signals
        self.study_panel.study_created.connect(self._on_study_created)
        self.study_panel.study_loaded.connect(self._on_study_loaded)
        self.study_panel.study_resumed.connect(self._on_study_resumed)
        
        # Monitor panel buttons
        self.monitor_panel.start_btn.clicked.connect(self._on_start)
        self.monitor_panel.pause_btn.clicked.connect(self._on_pause)
        self.monitor_panel.stop_btn.clicked.connect(self._on_stop)
    
    # ==================== Study Lifecycle ====================
    
    @pyqtSlot(str, str, int)
    def _on_study_created(self, name, objective_type, n_trials):
        """Handle new study creation."""
        try:
            search_space = self.search_space_panel.get_search_space()
            
            study = self.study_manager.create_study(
                name=name,
                objective_type=objective_type,
                total_trials=n_trials,
                search_space_config=search_space,
            )
            
            self.current_study = study
            self.current_study_name = name
            self.current_objective_type = objective_type
            self.total_trials = n_trials
            self.completed_trials = 0
            self.pruned_trials = 0
            
            # Update UI
            self.status_study.setText(f"Study: {name} ({objective_type})")
            self.monitor_panel.reset()
            self.monitor_panel.set_total_trials(n_trials)
            
            # Switch to monitor tab
            self.tabs.setCurrentIndex(2)
            
            QMessageBox.information(
                self, "Study Created",
                f"Study '{name}' created successfully.\n"
                f"Objective: {objective_type}\n"
                f"Trials: {n_trials}\n\n"
                f"Press Start to begin optimization."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create study:\n{e}")
    
    @pyqtSlot(str)
    def _on_study_loaded(self, name):
        """Handle loading an existing study for viewing."""
        try:
            study = self.study_manager.load_study(name)
            meta = self.study_manager.get_study_info(name)
            
            self.current_study = study
            self.current_study_name = name
            self.current_objective_type = meta.get('objective_type', 'val_f1')
            self.total_trials = meta.get('total_trials', 500)
            
            # Load existing trials into monitor
            self.monitor_panel.reset()
            self.monitor_panel.set_total_trials(self.total_trials)
            self.monitor_panel.load_existing_trials(study.trials)
            
            completed = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
            pruned = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED]
            
            self.completed_trials = len(completed)
            self.pruned_trials = len(pruned)
            
            total_run = len(study.trials)
            self.monitor_panel.update_progress(
                total_run, self.total_trials, len(pruned)
            )
            
            if completed and self.current_objective_type != 'multi':
                self.monitor_panel.update_best(
                    study.best_value, study.best_trial.number
                )
            
            # Load into viz panel
            self.viz_panel.set_study(study)
            self.viz_panel.refresh_all()
            
            self.status_study.setText(f"Study: {name} ({self.current_objective_type})")
            self.status_progress.setText(f"{total_run} trials loaded")
            
            # Switch to viz tab
            self.tabs.setCurrentIndex(3)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load study:\n{e}")
    
    @pyqtSlot(str, int)
    def _on_study_resumed(self, name, additional_trials):
        """Handle resuming an existing study."""
        try:
            study = self.study_manager.load_study(name)
            meta = self.study_manager.get_study_info(name)
            
            self.current_study = study
            self.current_study_name = name
            self.current_objective_type = meta.get('objective_type', 'val_f1')
            
            # Update total trials
            existing = len(study.trials)
            self.total_trials = existing + additional_trials
            
            # Update metadata
            self.study_manager.update_state(name, 'running')
            
            # Load existing trials
            self.monitor_panel.reset()
            self.monitor_panel.set_total_trials(self.total_trials)
            self.monitor_panel.load_existing_trials(study.trials)
            
            completed = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
            pruned = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED]
            self.completed_trials = len(completed)
            self.pruned_trials = len(pruned)
            self.monitor_panel.update_progress(
                existing, self.total_trials, len(pruned)
            )
            
            if completed and self.current_objective_type != 'multi':
                self.monitor_panel.update_best(
                    study.best_value, study.best_trial.number
                )
            
            self.status_study.setText(f"Study: {name} (resumed)")
            
            # Switch to monitor and auto-start
            self.tabs.setCurrentIndex(2)
            
            # Start optimization with additional trials
            self._start_optimization(additional_trials)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resume study:\n{e}")
    
    # ==================== Optimization Controls ====================
    
    def _on_start(self):
        """Start optimization."""
        if self.current_study is None:
            QMessageBox.warning(self, "No Study",
                              "Create or load a study first.")
            return
        
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running",
                              "Optimization is already running.")
            return
        
        remaining = self.total_trials - len(self.current_study.trials)
        if remaining <= 0:
            QMessageBox.information(self, "Complete",
                                   "All planned trials have been completed.")
            return
        
        self._start_optimization(remaining)
    
    def _start_optimization(self, n_trials):
        """Launch the optimization worker thread."""
        try:
            # Get dataset path and search space
            dataset_path = self.study_panel.get_dataset_path()
            if not dataset_path:
                dataset_path = "dataset"
            
            # Make path relative to project root if needed
            project_root = Path(__file__).parent.parent
            full_dataset_path = project_root / dataset_path
            if not full_dataset_path.exists():
                full_dataset_path = Path(dataset_path)
            
            # Get search space config
            search_space = self.search_space_panel.get_search_space()
            
            # Base config for the pipeline
            base_config = {
                'sampling_freq': 40,
                'low_threshold': 0.5,
                'high_threshold': 5.0,
                'filter_order': 4,
                'p': 0.1,
                'hz': 40,
                'hrv_window_size': 200,
                'hrv_overlap': 180,
                'samp_en_m': 2,
                'samp_en_r': 0.2,
                'scales': 10,
                'dfa_min_scale': 4,
                'dfa_num_scales': 20,
                'wavelet_num_scales': 12,
                'epochs': 100,
                'early_stopping_patience': 20,
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'device': 'cuda' if _get_objective_module().torch.cuda.is_available() else 'cpu',
            }
            
            # Epoch callback for live updates
            def epoch_callback(trial_num, epoch, train_loss, val_acc, val_f1):
                # This runs in the worker thread, so we can't directly update UI
                # The trial_completed signal handles the final result
                pass
            
            # Create stop event
            stop_event = threading.Event()
            
            # Lazy-init feature cache and import objective module
            obj_mod = _get_objective_module()
            if self.feature_cache is None:
                self.feature_cache = obj_mod.FeatureCache()
            
            # Create objective
            objective_fn, num_classes, class_map = obj_mod.create_objective(
                dataset_dir=str(full_dataset_path),
                base_config=base_config,
                objective_type=self.current_objective_type,
                search_space=search_space,
                feature_cache=self.feature_cache,
                callback=epoch_callback,
                stop_event=stop_event,
            )
            
            # Create and start worker
            self.worker = OptimizationWorker(
                study=self.current_study,
                objective_fn=objective_fn,
                n_trials=n_trials,
            )
            self.worker._stop_event = stop_event
            self.worker.trial_completed.connect(self._on_trial_completed)
            self.worker.optimization_done.connect(self._on_optimization_done)
            self.worker.error_occurred.connect(self._on_error)
            
            self.worker.start()
            
            # Update UI state
            self.monitor_panel.set_running_state(True)
            self.monitor_panel.start_timer()
            self.study_manager.update_state(self.current_study_name, 'running')
            self.status_progress.setText("Optimizing...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error",
                               f"Failed to start optimization:\n{e}")
            import traceback
            traceback.print_exc()
    
    def _on_pause(self):
        """Pause the optimization."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.monitor_panel.pause_btn.setEnabled(False)
            self.status_progress.setText("Pausing...")
            self.study_manager.update_state(self.current_study_name, 'paused')
    
    def _on_stop(self):
        """Stop the optimization."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Stop",
                "Stop optimization? Completed trials are saved.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.status_progress.setText("Stopping...")
                self.study_manager.update_state(self.current_study_name, 'paused')
    
    # ==================== Worker Callbacks ====================
    
    @pyqtSlot(dict)
    def _on_trial_completed(self, info):
        """Handle a completed trial from the worker thread."""
        state = info['state']
        
        if state == 'COMPLETE':
            self.completed_trials += 1
        elif state == 'PRUNED':
            self.pruned_trials += 1
        
        total_run = self.completed_trials + self.pruned_trials
        
        # Update monitor
        self.monitor_panel.add_trial_result(
            trial_number=info['trial_number'],
            state=state,
            value=info.get('value'),
            val_acc=info.get('val_acc'),
            val_f1=info.get('val_f1'),
            epochs=info.get('epochs'),
            duration=info.get('duration'),
        )
        
        self.monitor_panel.update_progress(
            total_run, self.total_trials, self.pruned_trials
        )
        
        self.monitor_panel.update_current_trial(info['trial_number'] + 1)
        
        # Update best
        if state == 'COMPLETE' and self.current_objective_type != 'multi':
            try:
                self.monitor_panel.update_best(
                    self.current_study.best_value,
                    self.current_study.best_trial.number,
                )
            except Exception:
                pass
        
        # Update status bar
        self.status_progress.setText(
            f"Trial {total_run}/{self.total_trials} | "
            f"Completed: {self.completed_trials} | Pruned: {self.pruned_trials}"
        )
        
        # Auto-refresh viz every 10 trials
        if total_run % 10 == 0:
            self.viz_panel.set_study(self.current_study)
            self.viz_panel.refresh_current()
    
    @pyqtSlot()
    def _on_optimization_done(self):
        """Handle optimization completion."""
        self.monitor_panel.set_running_state(False)
        
        total_run = len(self.current_study.trials)
        
        if total_run >= self.total_trials:
            self.study_manager.update_state(self.current_study_name, 'completed')
            self.status_progress.setText(f"Completed — {total_run} trials")
        else:
            self.study_manager.update_state(self.current_study_name, 'paused')
            self.status_progress.setText(f"Paused — {total_run}/{self.total_trials}")
        
        # Refresh visualizations
        self.viz_panel.set_study(self.current_study)
        self.viz_panel.refresh_all()
        
        # Refresh study list
        self.study_panel._refresh_studies()
        
        self.worker = None
    
    @pyqtSlot(str)
    def _on_error(self, message):
        """Handle optimization error."""
        QMessageBox.critical(self, "Optimization Error", message)
    
    def _refresh_viz(self):
        """Refresh visualizations from menu."""
        if self.current_study:
            self.viz_panel.set_study(self.current_study)
            self.viz_panel.refresh_all()
    
    # ==================== Window Events ====================
    
    def closeEvent(self, event):
        """Handle window close — stop optimization if running."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Quit",
                "Optimization is still running. Stop and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            
            self.worker.stop()
            self.worker.wait(5000)
        
        event.accept()
