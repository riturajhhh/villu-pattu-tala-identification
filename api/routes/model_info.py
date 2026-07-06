"""
api/routes/model_info.py
=========================
FastAPI route for returning information about the active classifier model.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.model_manager import ModelManager
from api.schemas import ModelInfoSchema

router = APIRouter()
model_manager = ModelManager()


@router.get("/model-info", response_model=ModelInfoSchema)
def get_model_info():
    """Retrieve details about the current trained models and classes."""
    info = model_manager.get_info()
    return {
        "model_type": info.get("model_type", "none"),
        "best_model_name": info.get("best_model_name"),
        "classes": info.get("classes", []),
        "accuracy": info.get("accuracy", 0.0),
        "n_features": info.get("n_features"),
        "training_time_seconds": info.get("training_time_seconds")
    }
