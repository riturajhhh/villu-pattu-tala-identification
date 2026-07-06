"""
feature_extraction/frequency_domain.py
========================================
Frequency-domain audio feature extractors for Tala identification.

Features
--------
- MFCC (Mel-Frequency Cepstral Coefficients) + delta + delta-delta
- Chroma STFT
- Spectral Centroid, Bandwidth, Contrast, Roll-off
- Mel Spectrogram (summary stats)
- Constant-Q Transform (CQT)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# MFCC
# ---------------------------------------------------------------------------

def extract_mfcc(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    include_deltas: bool = True,
) -> Dict[str, float]:
    """Extract MFCC coefficients (mean + std) and optionally Δ / ΔΔ.

    Returns up to 40×2 (MFCC) + 40×2 (Δ) + 40×2 (ΔΔ) = 240 features.
    """
    import librosa

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    features: Dict[str, float] = {}

    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    if include_deltas:
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        for i in range(n_mfcc):
            features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
            features[f"mfcc_delta_{i+1}_std"] = float(np.std(mfcc_delta[i]))
            features[f"mfcc_delta2_{i+1}_mean"] = float(np.mean(mfcc_delta2[i]))
            features[f"mfcc_delta2_{i+1}_std"] = float(np.std(mfcc_delta2[i]))

    return features


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

def extract_chroma(
    y: np.ndarray,
    sr: int = 22050,
    n_chroma: int = 12,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Extract Chroma STFT features (12 pitch classes).

    Chroma features capture harmonic / pitch-class content.  Useful for
    Villu Pattu because the vocal tonal patterns differ across Talas.
    """
    import librosa

    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length
    )

    features: Dict[str, float] = {}
    for i in range(n_chroma):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    return features


# ---------------------------------------------------------------------------
# Spectral descriptors
# ---------------------------------------------------------------------------

def extract_spectral_features(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Extract spectral shape descriptors.

    Includes
    --------
    - Spectral Centroid  — "brightness" of the sound
    - Spectral Bandwidth — spread around the centroid
    - Spectral Contrast  — peak-to-valley contrast in sub-bands
    - Spectral Roll-off  — frequency below which 85% of energy is concentrated
    - Spectral Flatness  — tonal vs. noisy (0 = tonal, 1 = noisy)
    """
    import librosa

    features: Dict[str, float] = {}

    # --- Centroid ---
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # --- Bandwidth ---
    bw = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    features["spectral_bandwidth_mean"] = float(np.mean(bw))
    features["spectral_bandwidth_std"] = float(np.std(bw))

    # --- Contrast ---
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    for i in range(contrast.shape[0]):
        features[f"spectral_contrast_{i+1}_mean"] = float(np.mean(contrast[i]))
        features[f"spectral_contrast_{i+1}_std"] = float(np.std(contrast[i]))

    # --- Roll-off ---
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    # --- Flatness ---
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))

    return features


# ---------------------------------------------------------------------------
# Mel Spectrogram (summary statistics)
# ---------------------------------------------------------------------------

def extract_mel_spectrogram_stats(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: int = 8000,
) -> Dict[str, float]:
    """Extract summary statistics from the Mel Spectrogram.

    For CNN training the full 2D mel spectrogram is used (see
    ``feature_extractor.py``), but for classical ML we flatten to scalar stats.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    features: Dict[str, float] = {}
    features["mel_mean"] = float(np.mean(mel_db))
    features["mel_std"] = float(np.std(mel_db))
    features["mel_max"] = float(np.max(mel_db))
    features["mel_min"] = float(np.min(mel_db))
    features["mel_median"] = float(np.median(mel_db))

    # Per-band means (first 20 bands as representative)
    for i in range(min(20, n_mels)):
        features[f"mel_band_{i+1}_mean"] = float(np.mean(mel_db[i]))

    return features


# ---------------------------------------------------------------------------
# Constant-Q Transform
# ---------------------------------------------------------------------------

def extract_cqt(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    fmin: float = 32.7,  # C1
    n_bins: int = 84,     # 7 octaves × 12 semitones
) -> Dict[str, float]:
    """Extract Constant-Q Transform (CQT) summary statistics.

    CQT provides logarithmically-spaced frequency resolution, well-suited
    for musical analysis.
    """
    import librosa

    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

    features: Dict[str, float] = {}
    features["cqt_mean"] = float(np.mean(cqt_db))
    features["cqt_std"] = float(np.std(cqt_db))
    features["cqt_max"] = float(np.max(cqt_db))
    features["cqt_min"] = float(np.min(cqt_db))

    return features


# ---------------------------------------------------------------------------
# Harmonic-Percussive Separation features
# ---------------------------------------------------------------------------

def extract_harmonic_percussive(
    y: np.ndarray,
    sr: int = 22050,
) -> Dict[str, float]:
    """Separate harmonic and percussive components and compute their ratio.

    Percussive dominance indicates strong rhythmic (beat) content, which is
    characteristic of different Tala patterns.
    """
    import librosa

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    h_rms = float(np.sqrt(np.mean(y_harmonic ** 2)))
    p_rms = float(np.sqrt(np.mean(y_percussive ** 2)))

    features: Dict[str, float] = {}
    features["harmonic_rms"] = h_rms
    features["percussive_rms"] = p_rms
    features["hp_ratio"] = h_rms / (p_rms + 1e-8)
    features["percussive_fraction"] = p_rms / (h_rms + p_rms + 1e-8)

    return features


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def extract_all_frequency_domain(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmax: int = 8000,
) -> Dict[str, float]:
    """Extract all frequency-domain features and return as a flat dict."""
    features: Dict[str, float] = {}
    features.update(extract_mfcc(y, sr, n_mfcc, n_fft, hop_length))
    features.update(extract_chroma(y, sr, n_fft=n_fft, hop_length=hop_length))
    features.update(extract_spectral_features(y, sr, n_fft, hop_length))
    features.update(extract_mel_spectrogram_stats(y, sr, n_mels, n_fft, hop_length, fmax))
    features.update(extract_cqt(y, sr, hop_length))
    features.update(extract_harmonic_percussive(y, sr))
    return features
