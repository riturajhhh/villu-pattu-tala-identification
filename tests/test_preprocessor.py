"""
tests/test_preprocessor.py
==========================
Unit tests for the AudioPreprocessor pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.audio_preprocessor import AudioPreprocessor


def test_audio_preprocessor_mono_conversion():
    """Test that multi-channel inputs are correctly merged to mono."""
    preprocessor = AudioPreprocessor()
    
    # Simulate a stereo waveform (2 channels, 1000 samples)
    stereo_y = np.random.uniform(-1, 1, (2, 1000)).astype(np.float32)
    
    mono_y = preprocessor.to_mono(stereo_y)
    
    assert mono_y.ndim == 1
    assert len(mono_y) == 1000
    assert np.allclose(mono_y, stereo_y.mean(axis=0))


def test_audio_preprocessor_normalization():
    """Test peak normalisation scaling."""
    preprocessor = AudioPreprocessor()
    
    # Create unnormalized waveform with peak amplitude 0.5
    y = np.random.uniform(-0.5, 0.5, 1000).astype(np.float32)
    y[500] = 0.5
    y[501] = -0.5
    
    y_norm = preprocessor.normalize(y)
    
    assert np.max(np.abs(y_norm)) == pytest.approx(1.0)
    assert y_norm.dtype == np.float32


def test_audio_preprocessor_pad_or_truncate():
    """Test padding/truncation duration formatting."""
    preprocessor = AudioPreprocessor(sample_rate=100, target_duration=5.0)  # 500 samples
    
    # Under-length input
    short_y = np.ones(300, dtype=np.float32)
    padded = preprocessor.pad_or_truncate(short_y)
    assert len(padded) == 500
    assert np.all(padded[300:] == 0.0)
    
    # Over-length input
    long_y = np.ones(700, dtype=np.float32)
    truncated = preprocessor.pad_or_truncate(long_y)
    assert len(truncated) == 500


def test_audio_preprocessor_max_duration(tmp_path):
    """Test that max_duration parameter limits the loaded audio duration."""
    import soundfile as sf

    # Create a 5-second synthetic audio file (sample rate 1000 Hz, total 5000 samples)
    sr = 1000
    y = np.sin(2 * np.pi * 10 * np.linspace(0, 5.0, 5000, endpoint=False)).astype(np.float32)

    test_file = tmp_path / "test_max_dur.wav"
    sf.write(str(test_file), y, sr)

    # Instantiate AudioPreprocessor with max_duration of 2.0 seconds
    preprocessor = AudioPreprocessor(sample_rate=sr, max_duration=2.0, reduce_noise=False)

    # Preprocess the file
    y_proc, sr_proc = preprocessor.preprocess(test_file)

    # The processed signal duration should be capped at 2.0 seconds (2000 samples)
    assert len(y_proc) <= 2000
    assert sr_proc == sr
