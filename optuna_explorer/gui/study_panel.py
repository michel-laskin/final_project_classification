"""
Study Panel — Create, Load, and Resume Optuna Studies

Tab 1 of the Optuna Explorer GUI.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QSpinBox, QGroupBox, QFormLayout,
    QFileDialog, QMessageBox, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from optuna_explorer.gui.theme import COLORS


class StudyPanel(QWidget):
    """Panel for creating new studies and loading/resuming existing ones."""
    
    # Signals
    study_created = pyqtSignal(str, str, int)    # name, objective_type, n_trials
    study_loaded = pyqtSignal(str)                # study_name
    study_resumed = pyqtSignal(str, int)          # study_name, additional_trials
    
    def __init__(self, study_manager, parent=None):
        super().__init__(parent)
        self.study_manager = study_manager
        self._setup_ui()
        self._refresh_studies()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # === Create New Study Section ===
        create_group = QGroupBox("Create New Study")
        create_layout = QFormLayout(create_group)
        create_layout.setSpacing(12)
        create_layout.setContentsMargins(16, 24, 16, 16)
        
        # Study name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. baseline_search_v1")
        self.name_input.setMinimumWidth(300)
        create_layout.addRow("Study Name:", self.name_input)
        
        # Objective type
        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "Val F1 (macro) — Recommended",
            "Val Accuracy",
            "Multi-Objective (Accuracy + F1)",
        ])
        create_layout.addRow("Objective:", self.objective_combo)
        
        # Dataset path
        dataset_row = QHBoxLayout()
        self.dataset_input = QLineEdit()
        self.dataset_input.setText("dataset")
        self.dataset_input.setMinimumWidth(250)
        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(self.dataset_input)
        dataset_row.addWidget(browse_btn)
        create_layout.addRow("Dataset Path:", dataset_row)
        
        # Total trials
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 10000)
        self.trials_spin.setValue(500)
        self.trials_spin.setSingleStep(50)
        create_layout.addRow("Total Trials:", self.trials_spin)
        
        # Create button
        create_btn = QPushButton("✦  Create Study")
        create_btn.setObjectName("primary")
        create_btn.setMinimumHeight(40)
        create_btn.clicked.connect(self._on_create)
        create_layout.addRow("", create_btn)
        
        layout.addWidget(create_group)
        
        # === Separator ===
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {COLORS['border']};")
        sep.setFixedHeight(1)
        layout.addWidget(sep)
        
        # === Load / Resume Study Section ===
        load_group = QGroupBox("Load / Resume Existing Study")
        load_layout = QVBoxLayout(load_group)
        load_layout.setSpacing(12)
        load_layout.setContentsMargins(16, 24, 16, 16)
        
        # Study selector
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Select Study:"))
        self.study_combo = QComboBox()
        self.study_combo.setMinimumWidth(300)
        self.study_combo.currentTextChanged.connect(self._on_study_selected)
        selector_row.addWidget(self.study_combo, 1)
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setMaximumWidth(100)
        refresh_btn.clicked.connect(self._refresh_studies)
        selector_row.addWidget(refresh_btn)
        load_layout.addLayout(selector_row)
        
        # Study info card
        self.info_label = QLabel("No study selected")
        self.info_label.setObjectName("subheading")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            f"background-color: {COLORS['bg_input']}; "
            f"border: 1px solid {COLORS['border']}; "
            f"border-radius: 6px; padding: 12px;"
        )
        self.info_label.setMinimumHeight(80)
        load_layout.addWidget(self.info_label)
        
        # Resume controls
        resume_row = QHBoxLayout()
        resume_row.addWidget(QLabel("Additional Trials:"))
        self.resume_trials_spin = QSpinBox()
        self.resume_trials_spin.setRange(1, 10000)
        self.resume_trials_spin.setValue(100)
        self.resume_trials_spin.setSingleStep(50)
        resume_row.addWidget(self.resume_trials_spin)
        
        self.resume_btn = QPushButton("▶  Resume Study")
        self.resume_btn.setObjectName("success")
        self.resume_btn.setMinimumHeight(40)
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self._on_resume)
        resume_row.addWidget(self.resume_btn)
        
        self.view_btn = QPushButton("📊 View Results")
        self.view_btn.setMinimumHeight(40)
        self.view_btn.setEnabled(False)
        self.view_btn.clicked.connect(self._on_view)
        resume_row.addWidget(self.view_btn)
        
        load_layout.addLayout(resume_row)
        layout.addWidget(load_group)
        
        # Spacer
        layout.addStretch()
    
    def _get_objective_type(self) -> str:
        idx = self.objective_combo.currentIndex()
        return ['val_f1', 'val_accuracy', 'multi'][idx]
    
    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if path:
            self.dataset_input.setText(path)
    
    def _on_create(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a study name.")
            return
        
        # Sanitize name for filename
        name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        if self.study_manager.study_exists(name):
            QMessageBox.warning(
                self, "Error",
                f"Study '{name}' already exists. Choose a different name."
            )
            return
        
        objective_type = self._get_objective_type()
        n_trials = self.trials_spin.value()
        
        self.study_created.emit(name, objective_type, n_trials)
    
    def _refresh_studies(self):
        self.study_combo.clear()
        studies = self.study_manager.list_studies()
        
        if not studies:
            self.study_combo.addItem("(no studies found)")
            self.resume_btn.setEnabled(False)
            self.view_btn.setEnabled(False)
            return
        
        for s in studies:
            name = s.get('name', 'unknown')
            n_trials = s.get('n_trials', 0)
            self.study_combo.addItem(f"{name} ({n_trials} trials)")
        
        self._on_study_selected()
    
    def _on_study_selected(self):
        text = self.study_combo.currentText()
        if not text or text.startswith("(no"):
            self.info_label.setText("No study selected")
            self.resume_btn.setEnabled(False)
            self.view_btn.setEnabled(False)
            return
        
        # Extract study name from combo text
        name = text.rsplit(' (', 1)[0]
        
        try:
            info = self.study_manager.get_study_info(name)
            
            lines = [
                f"<b>Name:</b> {info.get('name', 'N/A')}",
                f"<b>Objective:</b> {info.get('objective_type', 'N/A')}",
                f"<b>Created:</b> {info.get('created_at', 'N/A')[:19]}",
                f"<b>Trials:</b> {info.get('n_completed', 0)} completed, "
                f"{info.get('n_pruned', 0)} pruned, "
                f"{info.get('n_failed', 0)} failed",
                f"<b>Planned Total:</b> {info.get('total_trials', 'N/A')}",
            ]
            
            if info.get('best_value') is not None:
                lines.append(f"<b>Best Value:</b> <span style='color: {COLORS['accent_green']};'>"
                           f"{info['best_value']:.4f}</span>")
            
            if info.get('state'):
                state_color = {
                    'running': COLORS['accent_green'],
                    'paused': COLORS['accent_orange'],
                    'completed': COLORS['accent_cyan'],
                    'created': COLORS['text_secondary'],
                }.get(info['state'], COLORS['text_secondary'])
                lines.append(f"<b>State:</b> <span style='color: {state_color};'>"
                           f"{info['state'].upper()}</span>")
            
            self.info_label.setText("<br>".join(lines))
            self.resume_btn.setEnabled(True)
            self.view_btn.setEnabled(info.get('n_completed', 0) > 0)
            
        except Exception as e:
            self.info_label.setText(f"Error loading study info: {e}")
            self.resume_btn.setEnabled(False)
            self.view_btn.setEnabled(False)
    
    def _on_resume(self):
        text = self.study_combo.currentText()
        if not text or text.startswith("(no"):
            return
        name = text.rsplit(' (', 1)[0]
        additional = self.resume_trials_spin.value()
        self.study_resumed.emit(name, additional)
    
    def _on_view(self):
        text = self.study_combo.currentText()
        if not text or text.startswith("(no"):
            return
        name = text.rsplit(' (', 1)[0]
        self.study_loaded.emit(name)
    
    def get_dataset_path(self) -> str:
        return self.dataset_input.text().strip()
