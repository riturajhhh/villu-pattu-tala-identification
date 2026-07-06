"""
api/routes/features.py
=======================
FastAPI route for extracting features and generating base64 visualization plots.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.visualizer import (
    plot_beat_tracking,
    plot_mel_spectrogram,
    plot_mfcc_heatmap,
    plot_onset_envelope,
    plot_waveform,
)
from feature_extraction.feature_extractor import FeatureExtractor
from preprocessing.audio_preprocessor import create_preprocessor
from utils.config_loader import get_config
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.api.features", level="INFO")
router = APIRouter()
extractor = FeatureExtractor()
preprocessor = create_preprocessor()

try:
    cfg = get_config()
    UPLOAD_DIR = Path(cfg.api.upload_dir)
except Exception:
    UPLOAD_DIR = Path("outputs/uploads")


def _fig_to_base64(fig) -> str:
    """Helper to convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return encoded


@router.get("/features")
def extract_and_visualize_features(file_id: str):
    """Extract features and generate base64 visualization plots for a file.

    Parameters
    ----------
    file_id:
        Unique reference ID returned during upload.
    """
    # Find file with matching file_id
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File ID reference not found.")

    audio_path = matches[0]

    try:
        # Preprocess waveform
        y, sr = preprocessor.preprocess(audio_path)

        # Extract features
        features = extractor.extract_from_waveform(y, sr)
        if features is None:
            raise HTTPException(status_code=500, detail="Feature extraction failed.")

        # Generate plots
        fig_wave = plot_waveform(y, sr)
        
        # Beat tracking overlay
        from feature_extraction.rhythm_features import extract_tempo_and_beats
        rhythm_info = extract_tempo_and_beats(y, sr)
        beat_times = rhythm_info.get("beat_positions", [])
        fig_beats = plot_beat_tracking(y, sr, beat_times)

        # Mel Spectrogram
        fig_mel = plot_mel_spectrogram(y, sr)

        # Onset Envelope
        fig_onset = plot_onset_envelope(y, sr)

        # MFCC
        fig_mfcc = plot_mfcc_heatmap(y, sr)

        # Build response
        clean_features = {
            k: float(v) for k, v in features.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }

        return {
            "file_id": file_id,
            "bpm": clean_features.get("tempo_bpm", 0.0),
            "pulse_clarity": clean_features.get("pulse_clarity", 0.0),
            "duration": clean_features.get("duration", 0.0),
            "features": clean_features,
            "plots": {
                "waveform": _fig_to_base64(fig_wave),
                "beat_tracking": _fig_to_base64(fig_beats),
                "mel_spectrogram": _fig_to_base64(fig_mel),
                "onset_envelope": _fig_to_base64(fig_onset),
                "mfcc": _fig_to_base64(fig_mfcc),
            }
        }

    except Exception as exc:
        logger.error(f"Failed feature extraction visualization: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
