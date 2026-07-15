"""
utils/experiment_tracker.py
===========================
Lightweight experiment tracking utility.
Logs hyperparameters, metrics, and artifacts to a local JSON-based directory,
with optional integration to MLflow if installed.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger("villu_pattu.experiment_tracker")


class ExperimentTracker:
    def __init__(self, experiment_name: str, tracking_dir: str | Path = "outputs/experiments"):
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir) / experiment_name
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = str(uuid.uuid4())[:8]
        self.run_dir = self.tracking_dir / f"run_{self.run_id}_{int(time.time())}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.artifacts: list[str] = []
        
        # Try MLflow
        self.use_mlflow = False
        try:
            import mlflow
            self.mlflow = mlflow
            self.use_mlflow = True
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_id)
            logger.info(f"MLflow active for experiment: {self.experiment_name}")
        except ImportError:
            logger.info("MLflow not installed, using local JSON tracking only.")
            
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.params.update(params)
        if self.use_mlflow:
            self.mlflow.log_params(params)
            
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log performance metrics."""
        self.metrics.update(metrics)
        if self.use_mlflow:
            self.mlflow.log_metrics(metrics, step=step)
            
    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file artifact."""
        local_path = Path(local_path)
        if local_path.exists():
            self.artifacts.append(str(local_path))
            if self.use_mlflow:
                self.mlflow.log_artifact(str(local_path))
                
    def save_run(self) -> None:
        """Save the run to local disk and end MLflow run."""
        run_data = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "timestamp": time.time(),
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts
        }
        
        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(run_data, f, indent=2)
            
        logger.info(f"Run {self.run_id} saved locally to {self.run_dir}")
        
        if self.use_mlflow:
            self.mlflow.end_run()
