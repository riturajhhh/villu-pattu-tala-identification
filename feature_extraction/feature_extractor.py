"""
feature_extraction/feature_extractor.py
========================================
Unified feature extraction facade for the Villu Pattu Tala system.

Coordinates all sub-extractors and returns either:
- A flat feature dictionary (for classical ML)
- A Mel spectrogram image (for CNN / CRNN deep learning)

Usage
-----
    from feature_extraction.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    features_dict = extractor.extract("path/to/audio.wav")
    mel_image     = extractor.extract_mel_image("path/to/audio.wav")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger("villu_pattu.features")


class FeatureExtractor:
    """Unified feature extraction engine.

    Parameters
    ----------
    cfg:
        Config object (from ``utils.config_loader``).  If None, defaults used.
    sample_rate:
        Target sample rate.  Overridden by cfg if provided.
    """

    # Keys in the feature dict that are NOT scalar floats (excluded from ML)
    _NON_SCALAR_KEYS = {"beat_positions", "path", "filename", "tala"}

    def __init__(self, cfg=None, sample_rate: int = 22050) -> None:
        if cfg is not None:
            try:
                self.sr = int(cfg.audio.sample_rate)
                self.hop_length = int(cfg.audio.hop_length)
                self.n_fft = int(cfg.audio.n_fft)
                self.n_mfcc = int(cfg.audio.n_mfcc)
                self.n_mels = int(cfg.audio.n_mels)
                self.fmax = int(cfg.audio.fmax)
                self.frame_length = int(cfg.audio.frame_length)
            except Exception:
                self._set_defaults(sample_rate)
        else:
            self._set_defaults(sample_rate)

    def _set_defaults(self, sr: int) -> None:
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
        self.n_mfcc = 40
        self.n_mels = 128
        self.fmax = 8000
        self.frame_length = 2048

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract(self, path: str | Path) -> Optional[Dict[str, Any]]:
        """Extract all features from an audio file.

        Parameters
        ----------
        path:
            Path to WAV / MP3 / FLAC file.

        Returns
        -------
        dict
            Flat dictionary of features + metadata.  Returns None on error.
        """
        from preprocessing.audio_preprocessor import create_preprocessor
        from feature_extraction.time_domain import extract_all_time_domain
        from feature_extraction.frequency_domain import extract_all_frequency_domain
        from feature_extraction.rhythm_features import extract_all_rhythm_features

        path = Path(path)

        try:
            # 1. Preprocess
            preprocessor = create_preprocessor()
            y, sr = preprocessor.preprocess(path)

            if len(y) < self.n_fft:
                logger.warning(f"Audio too short after preprocessing: {path.name}")
                return None

            # 2. Time-domain features
            features: Dict[str, Any] = {}
            features.update(
                extract_all_time_domain(y, self.frame_length, self.hop_length)
            )

            # 3. Frequency-domain features
            features.update(
                extract_all_frequency_domain(
                    y, sr, self.n_mfcc, self.n_fft,
                    self.hop_length, self.n_mels, self.fmax
                )
            )

            # 4. Rhythm features
            features.update(
                extract_all_rhythm_features(y, sr, self.hop_length)
            )

            # 5. Metadata (kept for tracking; excluded from ML)
            features["duration"] = float(len(y) / sr)

            logger.debug(f"Extracted {len(features)} features from {path.name}")
            return features

        except Exception as exc:
            logger.error(f"Feature extraction failed for {path.name}: {exc}")
            return None

    def extract_from_waveform(
        self,
        y: np.ndarray,
        sr: int,
    ) -> Optional[Dict[str, Any]]:
        """Extract features from an already-loaded waveform array.

        Parameters
        ----------
        y:
            Preprocessed mono waveform (float32, normalised).
        sr:
            Sample rate.

        Returns
        -------
        dict or None
        """
        from feature_extraction.time_domain import extract_all_time_domain
        from feature_extraction.frequency_domain import extract_all_frequency_domain
        from feature_extraction.rhythm_features import extract_all_rhythm_features

        try:
            if len(y) < self.n_fft:
                return None

            features: Dict[str, Any] = {}
            features.update(
                extract_all_time_domain(y, self.frame_length, self.hop_length)
            )
            features.update(
                extract_all_frequency_domain(
                    y, sr, self.n_mfcc, self.n_fft,
                    self.hop_length, self.n_mels, self.fmax
                )
            )
            features.update(
                extract_all_rhythm_features(y, sr, self.hop_length)
            )
            features["duration"] = float(len(y) / sr)
            return features

        except Exception as exc:
            logger.error(f"Feature extraction from waveform failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Mel spectrogram image for CNN
    # ------------------------------------------------------------------

    def extract_mel_image(
        self,
        path: str | Path,
        target_shape: Tuple[int, int] = (128, 128),
    ) -> Optional[np.ndarray]:
        """Extract a Mel spectrogram as a 2D numpy array (image).

        Parameters
        ----------
        path:
            Path to audio file.
        target_shape:
            (n_mels, time_frames) target shape.  The spectrogram is
            resized/padded to this shape for CNN input.

        Returns
        -------
        np.ndarray of shape ``target_shape`` (float32, dB-scaled).
        """
        import librosa

        from preprocessing.audio_preprocessor import create_preprocessor

        try:
            preprocessor = create_preprocessor()
            y, sr = preprocessor.preprocess(path)

            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=target_shape[0],
                n_fft=self.n_fft, hop_length=self.hop_length, fmax=self.fmax,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Resize to target shape
            mel_db = self._resize_2d(mel_db, target_shape)

            return mel_db.astype(np.float32)

        except Exception as exc:
            logger.error(f"Mel image extraction failed for {path}: {exc}")
            return None

    def extract_mel_image_from_waveform(
        self,
        y: np.ndarray,
        sr: int,
        target_shape: Tuple[int, int] = (128, 128),
    ) -> Optional[np.ndarray]:
        """Extract a Mel spectrogram image from a waveform array."""
        import librosa

        try:
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=target_shape[0],
                n_fft=self.n_fft, hop_length=self.hop_length, fmax=self.fmax,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = self._resize_2d(mel_db, target_shape)
            return mel_db.astype(np.float32)
        except Exception as exc:
            logger.error(f"Mel image extraction from waveform failed: {exc}")
            return None

    @staticmethod
    def _resize_2d(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize a 2D array to target_shape via padding or truncation."""
        h, w = arr.shape
        th, tw = target_shape

        # Handle height
        if h >= th:
            arr = arr[:th, :]
        else:
            arr = np.pad(arr, ((0, th - h), (0, 0)), mode="constant")

        # Handle width
        h, w = arr.shape
        if w >= tw:
            arr = arr[:, :tw]
        else:
            arr = np.pad(arr, ((0, 0), (0, tw - w)), mode="constant")

        return arr

    # ------------------------------------------------------------------
    # Utility: get scalar feature names for ML
    # ------------------------------------------------------------------

    def get_feature_names(self, sample_features: Dict[str, Any]) -> list[str]:
        """Return sorted list of scalar feature names (for ML model input).

        Excludes non-scalar keys like ``beat_positions``, ``path``, etc.
        """
        return sorted([
            k for k, v in sample_features.items()
            if isinstance(v, (int, float, np.integer, np.floating))
            and k not in self._NON_SCALAR_KEYS
        ])
