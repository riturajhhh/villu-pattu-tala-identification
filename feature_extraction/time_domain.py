"""
feature_extraction/time_domain.py
==================================
Time-domain audio feature extractors for Tala identification.

Features
--------
- Zero Crossing Rate (ZCR)
- RMS Energy
- Amplitude Envelope
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def extract_zcr(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Compute Zero Crossing Rate statistics.

    ZCR is the rate at which the signal changes sign.  High ZCR suggests
    percussive / noisy content; low ZCR suggests tonal content.

    Parameters
    ----------
    y:
        Mono waveform (float32).
    frame_length:
        Frame length in samples.
    hop_length:
        Hop between frames.

    Returns
    -------
    dict with keys: ``zcr_mean``, ``zcr_std``, ``zcr_median``,
    ``zcr_max``, ``zcr_min``
    """
    import librosa

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]

    return {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "zcr_median": float(np.median(zcr)),
        "zcr_max": float(np.max(zcr)),
        "zcr_min": float(np.min(zcr)),
    }


def extract_rms_energy(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Compute Root Mean Square (RMS) energy statistics.

    RMS measures overall loudness per frame.  Rhythmic music shows periodic
    peaks in RMS that correspond to beats.

    Parameters
    ----------
    y:
        Mono waveform.
    frame_length:
        Frame length.
    hop_length:
        Hop size.

    Returns
    -------
    dict with keys: ``rms_mean``, ``rms_std``, ``rms_median``,
    ``rms_max``, ``rms_min``, ``rms_dynamic_range``
    """
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_median": float(np.median(rms)),
        "rms_max": float(np.max(rms)),
        "rms_min": float(np.min(rms)),
        "rms_dynamic_range": float(np.max(rms) - np.min(rms)),
    }


def extract_amplitude_envelope(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Compute amplitude envelope statistics.

    The amplitude envelope is the maximum absolute value in each frame.
    It captures the macro-level dynamics of the recording.

    Parameters
    ----------
    y:
        Mono waveform.
    frame_length:
        Frame length.
    hop_length:
        Hop size.

    Returns
    -------
    dict with keys: ``env_mean``, ``env_std``, ``env_median``,
    ``env_max``, ``env_min``
    """
    n_frames = 1 + (len(y) - frame_length) // hop_length
    envelope = np.zeros(max(1, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(y))
        envelope[i] = np.max(np.abs(y[start:end]))

    return {
        "env_mean": float(np.mean(envelope)),
        "env_std": float(np.std(envelope)),
        "env_median": float(np.median(envelope)),
        "env_max": float(np.max(envelope)),
        "env_min": float(np.min(envelope)),
    }


def extract_all_time_domain(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Extract all time-domain features and return as a single flat dict."""
    features: Dict[str, float] = {}
    features.update(extract_zcr(y, frame_length, hop_length))
    features.update(extract_rms_energy(y, frame_length, hop_length))
    features.update(extract_amplitude_envelope(y, frame_length, hop_length))
    return features
