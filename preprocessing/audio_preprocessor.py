"""
preprocessing/audio_preprocessor.py
=====================================
Full audio preprocessing pipeline for the Villu Pattu Tala Identification System.

Steps (all configurable)
------------------------
1. Load          — WAV / MP3 / FLAC → float32 NumPy array
2. To Mono       — stereo → mono
3. Resample      — to target sample rate (default 22050 Hz)
4. Trim Silence  — remove leading/trailing silence
5. Noise Reduce  — spectral subtraction noise floor removal
6. Normalize     — peak normalisation to [-1, 1]
7. Pad/Truncate  — to fixed target duration (optional)

Usage
-----
    from preprocessing.audio_preprocessor import AudioPreprocessor

    preprocessor = AudioPreprocessor()
    y, sr = preprocessor.preprocess("path/to/file.wav")
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger("villu_pattu.preprocess")


class AudioPreprocessor:
    """Full preprocessing pipeline for Villu Pattu audio recordings.

    Parameters
    ----------
    sample_rate:
        Target sample rate (Hz).  All audio is resampled to this.
    top_db:
        Silence threshold in dB below peak for trimming.
    reduce_noise:
        Whether to apply spectral subtraction noise reduction.
    normalize:
        Whether to apply peak normalization.
    target_duration:
        If set (seconds), pad or truncate to this length.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        top_db: int = 30,
        reduce_noise: bool = True,
        normalize: bool = True,
        target_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.top_db = top_db
        self._reduce_noise = reduce_noise
        self._normalize = normalize
        self.target_duration = target_duration
        self.max_duration = max_duration

    # ------------------------------------------------------------------
    # Individual pipeline steps
    # ------------------------------------------------------------------

    def load(
        self,
        path: str | Path,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load an audio file into a float32 NumPy array.

        Parameters
        ----------
        path:
            Path to WAV, MP3, or FLAC file.
        offset:
            Start reading at this many seconds from the beginning.
        duration:
            If set, read only this many seconds.

        Returns
        -------
        (y, sr)
            Waveform and sample rate of the *loaded* file (before resampling).
        """
        import librosa

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.debug(f"Loading: {path.name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(
                str(path),
                sr=None,        # Load at native SR; we resample separately
                mono=False,     # Keep channels; we convert to mono separately
                offset=offset,
                duration=duration if duration is not None else self.max_duration,
            )
        logger.debug(f"Loaded {len(y) if y.ndim == 1 else y.shape[1]} samples at {sr} Hz")
        return y, sr

    def to_mono(self, y: np.ndarray) -> np.ndarray:
        """Convert a stereo (or multi-channel) waveform to mono.

        Parameters
        ----------
        y:
            Waveform. Shape: ``(samples,)`` for mono, ``(channels, samples)`` for multi.
        """
        if y.ndim == 1:
            return y.astype(np.float32)
        # Average all channels
        return y.mean(axis=0).astype(np.float32)

    def resample(self, y: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample waveform to ``self.sample_rate``.

        Parameters
        ----------
        y:
            Mono waveform.
        orig_sr:
            Original sample rate.
        """
        import librosa

        if orig_sr == self.sample_rate:
            return y.astype(np.float32)
        logger.debug(f"Resampling {orig_sr} → {self.sample_rate} Hz")
        return librosa.resample(y, orig_sr=orig_sr, target_sr=self.sample_rate).astype(np.float32)

    def trim_silence(self, y: np.ndarray) -> np.ndarray:
        """Remove leading and trailing silence.

        Parameters
        ----------
        y:
            Mono waveform at ``self.sample_rate``.
        """
        import librosa

        y_trimmed, _ = librosa.effects.trim(y, top_db=self.top_db)
        logger.debug(
            f"Silence trimmed: {len(y)} → {len(y_trimmed)} samples "
            f"({len(y_trimmed)/self.sample_rate:.2f}s)"
        )
        return y_trimmed.astype(np.float32)

    def noise_reduce(self, y: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction noise reduction.

        Uses the ``noisereduce`` library when available, falls back to a
        simple spectral subtraction otherwise.

        Parameters
        ----------
        y:
            Mono waveform.
        """
        try:
            import noisereduce as nr

            y_reduced = nr.reduce_noise(
                y=y,
                sr=self.sample_rate,
                prop_decrease=0.75,        # Reduce noise by 75%
                stationary=False,          # Non-stationary noise model
                n_fft=2048,
            )
            return y_reduced.astype(np.float32)
        except ImportError:
            logger.debug("noisereduce not installed — using simple spectral subtraction")
            return self._simple_spectral_subtraction(y)

    def _simple_spectral_subtraction(self, y: np.ndarray) -> np.ndarray:
        """Basic spectral subtraction using first 0.5s as noise profile."""
        n_fft = 2048
        hop = 512
        # Estimate noise spectrum from first 0.5s
        n_noise = min(int(0.5 * self.sample_rate), len(y) // 4)
        noise_profile = y[:n_noise] if n_noise > 0 else y

        # Short-time Fourier transform
        stft = np.array([
            np.fft.rfft(y[i:i+n_fft] * np.hanning(n_fft))
            for i in range(0, len(y) - n_fft, hop)
        ])
        noise_stft = np.array([
            np.fft.rfft(noise_profile[i:i+n_fft] * np.hanning(n_fft))
            for i in range(0, len(noise_profile) - n_fft, hop)
            if i + n_fft <= len(noise_profile)
        ])

        if len(noise_stft) == 0:
            return y

        noise_mag = np.mean(np.abs(noise_stft), axis=0)
        cleaned_frames = []
        for frame in stft:
            mag = np.abs(frame)
            phase = np.angle(frame)
            mag_clean = np.maximum(mag - noise_mag * 1.5, 0)
            cleaned_frames.append(mag_clean * np.exp(1j * phase))

        # Reconstruct signal via overlap-add
        y_clean = np.zeros(len(y), dtype=np.float32)
        window = np.hanning(n_fft)
        for i, frame in enumerate(cleaned_frames):
            start = i * hop
            end = min(start + n_fft, len(y))
            reconstructed = np.fft.irfft(frame)[:end - start]
            y_clean[start:end] += (reconstructed * window[:end - start]).astype(np.float32)

        return np.clip(y_clean, -1.0, 1.0)

    def normalize(self, y: np.ndarray) -> np.ndarray:
        """Peak-normalise waveform to [-1, 1]."""
        peak = np.max(np.abs(y))
        if peak < 1e-8:
            logger.warning("Near-silent audio; skipping normalization")
            return y.astype(np.float32)
        return (y / peak).astype(np.float32)

    def pad_or_truncate(self, y: np.ndarray) -> np.ndarray:
        """Pad with zeros or truncate to ``self.target_duration`` seconds."""
        if self.target_duration is None:
            return y
        target_samples = int(self.target_duration * self.sample_rate)
        if len(y) >= target_samples:
            return y[:target_samples]
        return np.pad(y, (0, target_samples - len(y))).astype(np.float32)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        path: str | Path,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Run the full preprocessing pipeline on an audio file.

        Parameters
        ----------
        path:
            Path to audio file (WAV/MP3/FLAC).
        offset:
            Start time offset in seconds.
        duration:
            Read only this many seconds (None = full file).

        Returns
        -------
        (y, sr)
            Preprocessed mono waveform and the target sample rate.
        """
        logger.info(f"Preprocessing: {Path(path).name}")

        # Step 1: Load
        y, orig_sr = self.load(path, offset=offset, duration=duration)

        # Step 2: To mono
        y = self.to_mono(y)

        # Step 3: Resample
        y = self.resample(y, orig_sr)

        # Step 4: Trim silence
        y = self.trim_silence(y)

        if len(y) < 512:
            raise ValueError(f"Audio too short after silence trimming ({len(y)} samples)")

        # Step 5: Noise reduction
        if self._reduce_noise:
            y = self.noise_reduce(y)

        # Step 6: Normalize
        if self._normalize:
            y = self.normalize(y)

        # Step 7: Pad / truncate
        y = self.pad_or_truncate(y)

        logger.info(
            f"Done. Duration: {len(y)/self.sample_rate:.2f}s | "
            f"SR: {self.sample_rate} Hz | Shape: {y.shape}"
        )
        return y, self.sample_rate

    def preprocess_bytes(
        self,
        audio_bytes: bytes,
        file_format: str = "wav",
    ) -> Tuple[np.ndarray, int]:
        """Preprocess raw audio bytes (e.g., from an API upload).

        Parameters
        ----------
        audio_bytes:
            Raw bytes of a WAV / MP3 / FLAC file.
        file_format:
            Hint for the audio format.

        Returns
        -------
        (y, sr)
        """
        import tempfile
        import os

        suffix = f".{file_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.preprocess(tmp_path)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_preprocessor(cfg=None) -> AudioPreprocessor:
    """Create an ``AudioPreprocessor`` from the project config.

    Parameters
    ----------
    cfg:
        Config object from ``utils.config_loader.get_config()``.
        If None, defaults are used.
    """
    if cfg is None:
        try:
            from utils.config_loader import get_config
            cfg = get_config()
        except Exception:
            return AudioPreprocessor()

    try:
        max_duration = float(cfg.audio.max_duration)
    except Exception:
        max_duration = 60.0

    return AudioPreprocessor(
        sample_rate=cfg.audio.sample_rate,
        top_db=cfg.audio.silence_top_db,
        reduce_noise=False,
        normalize=True,
        target_duration=None,  # Don't fix duration at preprocessing stage
        max_duration=max_duration,
    )
