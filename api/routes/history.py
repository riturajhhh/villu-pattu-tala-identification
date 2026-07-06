"""
api/routes/history.py
======================
FastAPI route for retrieving the log history of predictions.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.database import PredictionHistory, get_db
from api.schemas import HistoryRecordSchema

router = APIRouter()


@router.get("/history", response_model=List[HistoryRecordSchema])
def get_prediction_history(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Retrieve the recent prediction log history."""
    records = db.query(PredictionHistory).order_by(
        PredictionHistory.timestamp.desc()
    ).limit(limit).all()

    output = []
    for r in records:
        output.append({
            "id": r.id,
            "filename": r.filename,
            "predicted_tala": r.predicted_tala,
            "confidence": r.confidence,
            "bpm": r.bpm,
            "model_used": r.model_used,
            "timestamp": r.timestamp.isoformat()
        })
    return output
