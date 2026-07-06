"""
api/model_manager.py
=====================
Model Manager facade that acts as a singleton for loading and serving predictions
from classical ML, CNN, or CRNN models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.predict import TalaPredictor
from utils.config_loader import get_config
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.model_manager", level="INFO")


class ModelManager:
    """Singleton model management class."""

    _instance: Optional[ModelManager] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, models_dir: Optional[str | Path] = None) -> None:
        # Avoid re-initialization if already loaded
        if hasattr(self, "initialized") and self.initialized:
            return

        try:
            cfg = get_config()
            self.models_dir = Path(cfg.models.saved_dir)
        except Exception:
            self.models_dir = Path(models_dir or "models/saved_models")

        self.predictor = TalaPredictor(model_dir=self.models_dir)
        self.active_model_type = "auto"
        self.initialized = True
        logger.info(f"ModelManager initialised with directory: {self.models_dir}")

    def get_info(self) -> Dict[str, Any]:
        """Get information about the best available model."""
        # Try loading metadata from best classical first
        try:
            meta_path = self.models_dir / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                return {
                    "model_type": "classical_ml",
                    "best_model_name": meta.get("best_model"),
                    "classes": meta.get("classes", []),
                    "accuracy": meta.get("test_accuracy", 0.0),
                    "n_features": meta.get("n_features"),
                    "training_time_seconds": meta.get("training_time_seconds")
                }
        except Exception as exc:
            logger.debug(f"Failed loading classical metadata: {exc}")

        # Fallback to CNN metadata if classical doesn't exist
        try:
            cnn_meta_path = self.models_dir / "cnn_metadata.json"
            if cnn_meta_path.exists():
                with open(cnn_meta_path, "r") as f:
                    meta = json.load(f)
                return {
                    "model_type": "cnn",
                    "classes": meta.get("classes", []),
                    "accuracy": meta.get("test_accuracy", 0.0),
                    "training_time_seconds": meta.get("training_time_seconds")
                }
        except Exception as exc:
            logger.debug(f"Failed loading CNN metadata: {exc}")

        return {
            "model_type": "none",
            "classes": ["Adi", "Rupaka", "Misra_Chapu", "Khanda_Chapu", "Other"],
            "accuracy": 0.0,
        }

    def predict_audio(self, audio_path: str | Path, model_type: str = "auto") -> Optional[Dict[str, Any]]:
        """Run predictions on the given audio path."""
        return self.predictor.predict(audio_path, model_type=model_type)
