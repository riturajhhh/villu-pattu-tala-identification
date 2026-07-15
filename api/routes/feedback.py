"""
api/routes/feedback.py
=======================
FastAPI route for submitting expert feedback on predictions.
This enables an active learning loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.database import ExpertFeedback, get_db
from api.schemas import FeedbackSubmitSchema, FeedbackStatsSchema
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.api.feedback", level="INFO")
router = APIRouter()


@router.post("/feedback")
def submit_feedback(
    feedback: FeedbackSubmitSchema,
    db: Session = Depends(get_db)
):
    """Submit a correction for a previous prediction."""
    try:
        new_feedback = ExpertFeedback(
            file_id=feedback.file_id,
            filename=feedback.filename,
            original_tala=feedback.original_tala,
            corrected_tala=feedback.corrected_tala,
            confidence=feedback.confidence,
            notes=feedback.notes
        )
        db.add(new_feedback)
        db.commit()
        logger.info(f"Feedback logged: {feedback.original_tala} -> {feedback.corrected_tala}")
        return {"message": "Feedback submitted successfully."}
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to submit feedback: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save feedback.")


@router.get("/feedback/stats", response_model=FeedbackStatsSchema)
def get_feedback_stats(db: Session = Depends(get_db)):
    """Retrieve statistics on expert feedback."""
    try:
        total = db.query(ExpertFeedback).count()
        
        # Group by corrected tala
        counts = db.query(
            ExpertFeedback.corrected_tala, 
            func.count(ExpertFeedback.id)
        ).group_by(ExpertFeedback.corrected_tala).all()
        
        corrections_dict = {tala: count for tala, count in counts}
        
        recent = db.query(ExpertFeedback).order_by(
            ExpertFeedback.timestamp.desc()
        ).limit(10).all()
        
        recent_list = [
            {
                "file_id": r.file_id,
                "filename": r.filename,
                "original_tala": r.original_tala,
                "corrected_tala": r.corrected_tala,
                "timestamp": r.timestamp.isoformat()
            } for r in recent
        ]
        
        return {
            "total_feedbacks": total,
            "corrections_by_tala": corrections_dict,
            "recent_corrections": recent_list
        }
    except Exception as exc:
        logger.error(f"Failed to get feedback stats: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch feedback stats.")
