"""
tests/test_feature_extractor.py
================================
Unit tests for feature extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.feature_extractor import FeatureExtractor


def test_feature_extractor_waveform():
    """Test that feature extraction from raw array output has correct types and columns."""
    extractor = FeatureExtractor(sample_rate=22050)
    
    # 2 seconds of synthetic signal
    sr = 22050
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    # 120 BPM clicks (every 0.5 seconds)
    y = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    
    features = extractor.extract_from_waveform(y, sr)
    
    assert features is not None
    assert isinstance(features, dict)
    
    # Check for expected feature keys
    assert "tempo_bpm" in features
    assert "zcr_mean" in features
    assert "rms_mean" in features
    assert "mfcc_1_mean" in features
    assert "chroma_1_mean" in features
    
    # Retrieve flat features and verify list
    feature_names = extractor.get_feature_names(features)
    assert len(feature_names) > 0
    assert "beat_positions" not in feature_names  # should be excluded
