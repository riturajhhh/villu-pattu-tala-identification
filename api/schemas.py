"""
api/schemas.py
==============
Pydantic schemas for request validation and response serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictionResultSchema(BaseModel):
    """Schema for POST /predict response."""
    predicted_tala: str
    confidence: float
    bpm: float
    pulse_clarity: float
    duration: float
    model_used: str
    top_predictions: List[Dict[str, Any]]
    beat_positions: List[float]


class UploadResponseSchema(BaseModel):
    """Schema for POST /upload response."""
    file_id: str
    filename: str
    size_bytes: int
    duration_seconds: float
    message: str


class HistoryRecordSchema(BaseModel):
    """Schema for single record in GET /history."""
    id: int
    filename: str
    predicted_tala: str
    confidence: float
    bpm: float
    model_used: str
    timestamp: str


class ModelInfoSchema(BaseModel):
    """Schema for GET /model-info."""
    model_type: str
    best_model_name: Optional[str] = None
    classes: List[str]
    accuracy: float
    n_features: Optional[int] = None
    training_time_seconds: Optional[float] = None


class FeedbackSubmitSchema(BaseModel):
    """Schema for POST /feedback."""
    file_id: str
    filename: str
    original_tala: str
    corrected_tala: str
    confidence: float
    notes: Optional[str] = ""


class FeedbackStatsSchema(BaseModel):
    """Schema for GET /feedback/stats."""
    total_feedbacks: int
    corrections_by_tala: Dict[str, int]
    recent_corrections: List[Dict[str, Any]]

