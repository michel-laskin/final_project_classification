"""
Optuna Explorer — Entry Point

Launch the GUI application.

Usage:
    python -m optuna_explorer.run
    or
    python optuna_explorer/run.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# ============================================================
# Fix for Windows PyTorch DLL loading (WinError 1114)
# Must run BEFORE any 'import torch' anywhere in the process.
# ============================================================
if os.name == 'nt':
    def _fix_torch_dll():
        """
        Fix PyTorch DLL loading on Windows (WinError 1114).
        
        Uses three strategies:
        1. Prepend torch/lib to PATH env var
        2. Use os.add_dll_directory() as backup  
        3. Add CUDA toolkit to PATH if available
        """
        import site
        import ctypes
        
        # Find torch install location
        search_dirs = []
        
        # User site-packages (pip install --user)
        user_sp = site.getusersitepackages()
        if isinstance(user_sp, str):
            user_sp = [user_sp]
        for sp in user_sp:
            search_dirs.append(Path(sp) / 'torch' / 'lib')
        
        # Global site-packages
        for sp in site.getsitepackages():
            search_dirs.append(Path(sp) / 'torch' / 'lib')
        
        paths_to_add = []
        
        for torch_lib in search_dirs:
            if torch_lib.exists():
                paths_to_add.append(str(torch_lib))
                # torch/bin often has additional DLLs
                torch_bin = torch_lib.parent / 'bin'
                if torch_bin.exists():
                    paths_to_add.append(str(torch_bin))
                break
        
        # Add NVIDIA CUDA toolkit if installed
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            cuda_bin = Path(cuda_path) / 'bin'
            if cuda_bin.exists():
                paths_to_add.append(str(cuda_bin))
        
        # Strategy 1: Prepend to PATH (most reliable on Windows)
        if paths_to_add:
            current_path = os.environ.get('PATH', '')
            new_entries = ';'.join(paths_to_add)
            os.environ['PATH'] = new_entries + ';' + current_path
        
        # Strategy 2: Also use add_dll_directory as backup
        for p in paths_to_add:
            try:
                os.add_dll_directory(p)
            except OSError:
                pass
    
    try:
        _fix_torch_dll()
    except Exception:
        pass  # Best effort

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from optuna_explorer.gui.main_window import MainWindow


def main():
    # Suppress Optuna logs in GUI mode (they go to the console)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # High DPI support — MUST be set before creating QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Optuna Explorer")
    app.setOrganizationName("HRV Classification")
    
    # Set default font (explicit point size > 0)
    font = QFont("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
