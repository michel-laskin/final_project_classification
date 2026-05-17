"""
Study Manager for Optuna Studies

Handles study lifecycle: create, load, resume, list.
Uses SQLite storage for persistence.
Stores sidecar metadata JSON alongside each study database.
"""

import json
import optuna
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


# Default storage directory (relative to project root)
DEFAULT_STORAGE_DIR = Path(__file__).parent / "studies"


class StudyManager:
    """Manages Optuna studies with SQLite persistence and metadata."""
    
    def __init__(self, storage_dir: str = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _db_path(self, name: str) -> Path:
        return self.storage_dir / f"{name}.db"
    
    def _meta_path(self, name: str) -> Path:
        return self.storage_dir / f"{name}_meta.json"
    
    def _storage_url(self, name: str) -> str:
        db = self._db_path(name)
        return f"sqlite:///{db.as_posix()}"
    
    def create_study(
        self,
        name: str,
        objective_type: str = 'val_f1',
        total_trials: int = 500,
        search_space_config: Dict = None,
    ) -> optuna.Study:
        """
        Create a new Optuna study with SQLite storage.
        
        Args:
            name: Study name (used as filename).
            objective_type: 'val_accuracy', 'val_f1', or 'multi'.
            total_trials: Planned total number of trials.
            search_space_config: Search space configuration to save.
            
        Returns:
            Created Optuna Study object.
        """
        storage_url = self._storage_url(name)
        
        # Configure directions based on objective type
        if objective_type == 'multi':
            directions = ['maximize', 'maximize']
            direction = None
        else:
            directions = None
            direction = 'maximize'
        
        # Sampler
        sampler = optuna.samplers.TPESampler(seed=42)
        
        # Pruner - MedianPruner for early termination of bad trials
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=15,
            interval_steps=1,
        )
        
        if direction is not None:
            study = optuna.create_study(
                study_name=name,
                storage=storage_url,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=False,
            )
        else:
            study = optuna.create_study(
                study_name=name,
                storage=storage_url,
                directions=directions,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=False,
            )
        
        # Save metadata
        meta = {
            'name': name,
            'objective_type': objective_type,
            'total_trials': total_trials,
            'created_at': datetime.now().isoformat(),
            'search_space': search_space_config,
            'state': 'created',
        }
        self._save_meta(name, meta)
        
        return study
    
    def load_study(self, name: str) -> optuna.Study:
        """
        Load an existing study from SQLite storage.
        
        Args:
            name: Study name.
            
        Returns:
            Loaded Optuna Study object.
        """
        storage_url = self._storage_url(name)
        meta = self._load_meta(name)
        
        objective_type = meta.get('objective_type', 'val_f1')
        
        # Recreate sampler and pruner
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=15,
            interval_steps=1,
        )
        
        if objective_type == 'multi':
            study = optuna.load_study(
                study_name=name,
                storage=storage_url,
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.load_study(
                study_name=name,
                storage=storage_url,
                sampler=sampler,
                pruner=pruner,
            )
        
        return study
    
    def list_studies(self) -> List[Dict[str, Any]]:
        """
        List all available studies in the storage directory.
        
        Returns:
            List of metadata dictionaries for each study.
        """
        studies = []
        for meta_file in sorted(self.storage_dir.glob("*_meta.json")):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                
                # Load study to get current trial count
                name = meta['name']
                db_path = self._db_path(name)
                if db_path.exists():
                    storage_url = self._storage_url(name)
                    try:
                        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
                        if summaries:
                            s = summaries[0]
                            meta['n_trials'] = s.n_trials
                            meta['best_value'] = None
                            if hasattr(s, 'best_trial') and s.best_trial is not None:
                                meta['best_value'] = s.best_trial.value
                    except Exception:
                        meta['n_trials'] = 0
                        meta['best_value'] = None
                
                studies.append(meta)
            except Exception:
                continue
        
        return studies
    
    def get_study_info(self, name: str) -> Dict[str, Any]:
        """Get detailed info about a study."""
        meta = self._load_meta(name)
        
        try:
            study = self.load_study(name)
            trials = study.trials
            
            completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
            failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
            
            meta['n_completed'] = len(completed)
            meta['n_pruned'] = len(pruned)
            meta['n_failed'] = len(failed)
            meta['n_total_run'] = len(trials)
            
            if completed:
                objective_type = meta.get('objective_type', 'val_f1')
                if objective_type != 'multi':
                    meta['best_value'] = study.best_value
                    meta['best_trial_number'] = study.best_trial.number
                    meta['best_params'] = study.best_trial.params
                else:
                    # Multi-objective: show best trials from pareto front
                    best_trials = study.best_trials
                    meta['n_pareto'] = len(best_trials)
                    meta['best_value'] = None
            else:
                meta['best_value'] = None
        except Exception as e:
            meta['error'] = str(e)
        
        return meta
    
    def update_state(self, name: str, state: str):
        """Update study state (created, running, paused, completed)."""
        meta = self._load_meta(name)
        meta['state'] = state
        meta['updated_at'] = datetime.now().isoformat()
        self._save_meta(name, meta)
    
    def study_exists(self, name: str) -> bool:
        """Check if a study with the given name exists."""
        return self._db_path(name).exists()
    
    def _save_meta(self, name: str, meta: Dict):
        meta_path = self._meta_path(name)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
    
    def _load_meta(self, name: str) -> Dict:
        meta_path = self._meta_path(name)
        if not meta_path.exists():
            return {'name': name}
        with open(meta_path, 'r') as f:
            return json.load(f)
