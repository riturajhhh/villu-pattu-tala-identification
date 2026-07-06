"""
api/routes/upload.py
=====================
FastAPI route for uploading audio files and returning unique handles.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import UploadResponseSchema
from utils.config_loader import get_config
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.api.upload", level="INFO")
router = APIRouter()

try:
    cfg = get_config()
    UPLOAD_DIR = Path(cfg.api.upload_dir)
except Exception:
    UPLOAD_DIR = Path("outputs/uploads")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=UploadResponseSchema)
async def upload_audio_file(file: UploadFile = File(...)):
    """Upload an audio clip and return a unique file reference ID."""
    suffix = Path(file.filename).suffix
    if suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Only WAV, MP3, FLAC, and OGG are supported."
        )

    try:
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

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{suffix}"

    try:
        content = await file.read()
        size_bytes = len(content)

        with open(save_path, "wb") as f:
            f.write(content)

        # Estimate duration safely using librosa
        import librosa
        try:
            duration = float(librosa.get_duration(path=str(save_path)))
        except Exception:
            duration = 0.0

        logger.info(f"File uploaded successfully: {file.filename} -> {save_path.name} ({duration:.2f}s)")

        return {
            "file_id": file_id,
            "filename": file.filename,
            "size_bytes": size_bytes,
            "duration_seconds": duration,
            "message": "Upload successful"
        }

    except Exception as exc:
        logger.error(f"Failed saving uploaded file: {exc}")
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")


@router.post("/upload/youtube", response_model=UploadResponseSchema)
async def upload_youtube_link(url: str):
    """Download audio from a YouTube URL and return a unique file reference ID."""
    from utils.youtube_downloader import download_youtube_audio
    import librosa

    file_id = str(uuid.uuid4())
    logger.info(f"Received YouTube download request for URL: {url}")

    try:
        # Download audio as wav
        wav_path = download_youtube_audio(url, UPLOAD_DIR, file_id)
        if wav_path is None or not wav_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Failed to download YouTube audio. Please ensure the URL is valid and ffmpeg is installed."
            )

        # Get size and duration
        size_bytes = wav_path.stat().st_size
        try:
            duration = float(librosa.get_duration(path=str(wav_path)))
        except Exception:
            duration = 0.0

        return {
            "file_id": file_id,
            "filename": f"youtube_{file_id}.wav",
            "size_bytes": size_bytes,
            "duration_seconds": duration,
            "message": "YouTube audio downloaded and extracted successfully."
        }

    except HTTPException as he:
        raise he
    except Exception as exc:
        logger.error(f"Failed during YouTube download route: {exc}")
        raise HTTPException(status_code=500, detail=f"YouTube download failed: {exc}")

