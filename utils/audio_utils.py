"""
utils/audio_utils.py
====================
Shared audio utility helpers for the Villu Pattu Tala Identification System.
Handles format detection, duration checking, and conversion wrappers.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def is_supported_audio(path: str | Path) -> bool:
    """Return True if the file extension is a supported audio format."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def get_audio_format(path: str | Path) -> str:
    """Return the lowercase extension without the leading dot."""
    return Path(path).suffix.lower().lstrip(".")


# ---------------------------------------------------------------------------
# Duration / integrity checks
# ---------------------------------------------------------------------------

def get_duration(path: str | Path, sr: int = 22050) -> float:
    """Return the duration of an audio file in seconds.

    Uses librosa for broad format support.

    Parameters
    ----------
    path:
        Path to the audio file.
    sr:
        Target sample rate (used during load).

    Returns
    -------
    float
        Duration in seconds.  Returns 0.0 on error.
    """
    try:
        import librosa
        duration = librosa.get_duration(path=str(path))
        return float(duration)
    except Exception:
        return 0.0


def validate_audio_file(
    path: str | Path,
    min_duration: float = 0.5,
    max_duration: float = 300.0,
    raise_on_fail: bool = True,
) -> Tuple[bool, str]:
    """Validate that an audio file is readable and within duration bounds.

    Parameters
    ----------
    path:
        Path to the audio file.
    min_duration:
        Minimum acceptable duration in seconds.
    max_duration:
        Maximum acceptable duration in seconds.
    raise_on_fail:
        If True, raise ``ValueError`` on validation failure.

    Returns
    -------
    (is_valid, message)
    """
    path = Path(path)

    if not path.exists():
        msg = f"File not found: {path}"
        if raise_on_fail:
            raise FileNotFoundError(msg)
        return False, msg

    if not is_supported_audio(path):
        msg = f"Unsupported format '{path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}"
        if raise_on_fail:
            raise ValueError(msg)
        return False, msg

    duration = get_duration(path)

    if duration < min_duration:
        msg = f"Audio too short ({duration:.2f}s). Minimum is {min_duration}s."
        if raise_on_fail:
            raise ValueError(msg)
        return False, msg

    if duration > max_duration:
        msg = f"Audio too long ({duration:.2f}s). Maximum is {max_duration}s."
        if raise_on_fail:
            raise ValueError(msg)
        return False, msg

    return True, "OK"


# ---------------------------------------------------------------------------
# Bytes helpers (for API / Streamlit upload)
# ---------------------------------------------------------------------------

def bytes_to_numpy(
    audio_bytes: bytes,
    sr: int = 22050,
    mono: bool = True,
    file_format: str = "wav",
) -> Tuple[np.ndarray, int]:
    """Load raw audio bytes into a NumPy array.

    Parameters
    ----------
    audio_bytes:
        Raw bytes of an audio file (WAV, MP3, FLAC, etc.).
    sr:
        Target sample rate for resampling.
    mono:
        If True, convert to mono.
    file_format:
        Hint for the audio format (e.g., ``'mp3'``).

    Returns
    -------
    (y, sr)
        Tuple of (waveform array, sample rate).
    """
    import librosa

    # Write to a temp file because librosa needs a path or file-like object
    suffix = f".{file_format}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, loaded_sr = librosa.load(tmp_path, sr=sr, mono=mono)
    finally:
        os.unlink(tmp_path)

    return y, loaded_sr


def numpy_to_wav_bytes(y: np.ndarray, sr: int = 22050) -> bytes:
    """Convert a NumPy waveform array to WAV bytes.

    Parameters
    ----------
    y:
        Audio waveform (float32 in [-1, 1]).
    sr:
        Sample rate.

    Returns
    -------
    bytes
        WAV-encoded bytes.
    """
    import soundfile as sf

    buf = io.BytesIO()
    # Clamp to [-1, 1] before encoding
    y_clamped = np.clip(y, -1.0, 1.0)
    sf.write(buf, y_clamped, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def convert_to_wav(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    sr: int = 22050,
) -> Path:
    """Convert any supported audio file to WAV.

    Parameters
    ----------
    input_path:
        Source audio file.
    output_path:
        Destination WAV path.  If None, places ``.wav`` next to the input.
    sr:
        Target sample rate.

    Returns
    -------
    Path
        Path to the converted WAV file.
    """
    import librosa
    import soundfile as sf

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    output_path = Path(output_path)

    y, _ = librosa.load(str(input_path), sr=sr, mono=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y, sr, subtype="PCM_16")
    return output_path


# ---------------------------------------------------------------------------
# Waveform utilities
# ---------------------------------------------------------------------------

def trim_leading_trailing_silence(
    y: np.ndarray,
    sr: int = 22050,
    top_db: int = 30,
) -> np.ndarray:
    """Remove leading and trailing silence from a waveform.

    Parameters
    ----------
    y:
        Input waveform.
    sr:
        Sample rate (unused but kept for API consistency).
    top_db:
        Silence threshold in dB below peak.

    Returns
    -------
    np.ndarray
        Trimmed waveform.
    """
    import librosa

    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def pad_or_truncate(y: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """Pad (with zeros) or truncate waveform to *target_duration* seconds.

    Parameters
    ----------
    y:
        Input waveform.
    sr:
        Sample rate.
    target_duration:
        Desired duration in seconds.

    Returns
    -------
    np.ndarray
        Fixed-length waveform.
    """
    target_samples = int(target_duration * sr)
    if len(y) >= target_samples:
        return y[:target_samples]
    # Pad with zeros
    padding = target_samples - len(y)
    return np.pad(y, (0, padding), mode="constant")


def split_into_segments(
    y: np.ndarray,
    sr: int,
    segment_duration: float = 5.0,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """Split a waveform into overlapping segments.

    Parameters
    ----------
    y:
        Input waveform.
    sr:
        Sample rate.
    segment_duration:
        Length of each segment in seconds.
    overlap:
        Fraction of overlap between consecutive segments (0–1).

    Returns
    -------
    list[np.ndarray]
        List of segment arrays.
    """
    segment_len = int(segment_duration * sr)
    hop_len = int(segment_len * (1.0 - overlap))
    segments = []
    start = 0
    while start + segment_len <= len(y):
        segments.append(y[start : start + segment_len])
        start += hop_len
    # Keep a final partial segment if it's at least 50% of segment_len
    if start < len(y) and (len(y) - start) >= segment_len * 0.5:
        segments.append(pad_or_truncate(y[start:], sr, segment_duration))
    return segments


# ---------------------------------------------------------------------------
# Amplitude / RMS helpers
# ---------------------------------------------------------------------------

def compute_rms(y: np.ndarray) -> float:
    """Root-mean-square energy of a waveform."""
    return float(np.sqrt(np.mean(y ** 2)))


def peak_normalize(y: np.ndarray) -> np.ndarray:
    """Normalise waveform to peak amplitude of 1.0."""
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y
    return y / peak
