"""Run manager: creates run directories, manages metadata, persists artifacts."""

import os
import json
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime


class RunManager:
    """Manages run directories and artifacts."""
    
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)
    
    def create_run(self, run_id: str) -> Path:
        """Create a new run directory."""
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (run_dir / "artifacts").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "curves").mkdir(exist_ok=True)
        (run_dir / "tables").mkdir(exist_ok=True)
        
        return run_dir
    
    def get_run_dir(self, run_id: str) -> Path:
        """Get run directory."""
        return self.runs_dir / run_id
    
    def save_config(self, run_id: str, config_path: str) -> None:
        """Copy config to run directory."""
        run_dir = self.get_run_dir(run_id)
        config_file = run_dir / "config.yaml"
        shutil.copy(config_path, config_file)
    
    def save_artifact(self, run_id: str, artifact_path: str, dest_name: Optional[str] = None) -> None:
        """Save artifact to run directory."""
        run_dir = self.get_run_dir(run_id)
        artifact_dir = run_dir / "artifacts"
        
        if dest_name is None:
            dest_name = os.path.basename(artifact_path)
        
        dest_path = artifact_dir / dest_name
        
        if os.path.isfile(artifact_path):
            shutil.copy(artifact_path, dest_path)
        elif os.path.isdir(artifact_path):
            shutil.copytree(artifact_path, dest_path, dirs_exist_ok=True)
    
    def save_checkpoint(self, run_id: str, checkpoint_path: str, name: str = "model.pt") -> None:
        """Save model checkpoint."""
        run_dir = self.get_run_dir(run_id)
        checkpoint_dir = run_dir / "checkpoints"
        dest_path = checkpoint_dir / name
        shutil.copy(checkpoint_path, dest_path)
    
    def list_runs(self) -> list:
        """List all runs."""
        return [d.name for d in self.runs_dir.iterdir() if d.is_dir()]
