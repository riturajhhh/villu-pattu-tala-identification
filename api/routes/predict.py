"""
api/routes/predict.py
======================
FastAPI route for running inference predictions on uploaded audio clips.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.database import PredictionHistory, get_db
from api.model_manager import ModelManager
from api.schemas import PredictionResultSchema
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.api.predict", level="INFO")
router = APIRouter()
model_manager = ModelManager()


@router.post("/predict", response_model=PredictionResultSchema)
async def predict_tala(
    file: UploadFile = File(...),
    model_type: str = "auto",
    db: Session = Depends(get_db)
):
    """Predict the Tala of the uploaded audio file.

    Parameters
    ----------
    file:
        Uploaded audio file (WAV, MP3, FLAC).
    model_type:
        Model architecture to use: ``'classical'``, ``'cnn'``, ``'crnn'``, or ``'auto'``.
    """
    suffix = Path(file.filename).suffix
    if suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported: WAV, MP3, FLAC, OGG."
        )

    try:
        from utils.config_loader import get_config
        cfg = get_config()
        max_size_mb = float(cfg.api.max_upload_size_mb)
    except Exception:
        max_size_mb = 50.0

    try:
        file.file.seek(0, 2)
        size_bytes = file.file.tell()
        file.file.seek(0)
    except Exception:
        size_bytes = 0

    if max_size_mb > 0 and size_bytes > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_bytes / (1024 * 1024):.1f} MB). Maximum allowed size is {max_size_mb} MB."
        )

    # Save to a temporary file for analysis
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Run prediction
        res = model_manager.predict_audio(tmp_path, model_type=model_type)
        if res is None:
            raise HTTPException(status_code=500, detail="Inference prediction failed.")

        # Log prediction to DB
        history_record = PredictionHistory(
            filename=file.filename,
            predicted_tala=res["predicted_tala"],
            confidence=res["confidence"],
            bpm=res["bpm"],
            pulse_clarity=res["pulse_clarity"],
            duration=res["duration"],
            model_used=res["model_used"]
        )
        db.add(history_record)
        db.commit()

        return {
            "predicted_tala": res["predicted_tala"],
            "confidence": res["confidence"],
            "bpm": res["bpm"],
            "pulse_clarity": res["pulse_clarity"],
            "duration": res["duration"],
            "model_used": res["model_used"],
            "top_predictions": res["top_predictions"],
            "beat_positions": res["beat_positions"]
        }

    except Exception as exc:
        logger.error(f"Error during API prediction: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        # Clean up temporary file
        if tmp_path.exists():
            os.unlink(tmp_path)


@router.post("/predict/by-id", response_model=PredictionResultSchema)
def predict_tala_by_id(
    file_id: str,
    model_type: str = "auto",
    db: Session = Depends(get_db)
):
    """Predict the Tala of a file that was already uploaded/downloaded by its reference ID."""
    try:
        from utils.config_loader import get_config
        cfg = get_config()
        upload_dir = Path(cfg.api.upload_dir)
    except Exception:
        upload_dir = Path("outputs/uploads")

    # Find file with matching file_id
    matches = list(upload_dir.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File ID reference not found.")

    audio_path = matches[0]

    try:
        # Run prediction
        res = model_manager.predict_audio(audio_path, model_type=model_type)
        if res is None:
            raise HTTPException(status_code=500, detail="Inference prediction failed.")

        # Log prediction to DB
        history_record = PredictionHistory(
            filename=audio_path.name,
            predicted_tala=res["predicted_tala"],
            confidence=res["confidence"],
            bpm=res["bpm"],
            pulse_clarity=res["pulse_clarity"],
            duration=res["duration"],
            model_used=res["model_used"]
        )
        db.add(history_record)
        db.commit()

        return {
            "predicted_tala": res["predicted_tala"],
            "confidence": res["confidence"],
            "bpm": res["bpm"],
            "pulse_clarity": res["pulse_clarity"],
            "duration": res["duration"],
            "model_used": res["model_used"],
            "top_predictions": res["top_predictions"],
            "beat_positions": res["beat_positions"]
        }

    except Exception as exc:
        logger.error(f"Error during ID-based prediction: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

