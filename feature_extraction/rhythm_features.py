"""
feature_extraction/rhythm_features.py
=======================================
Rhythm and tempo feature extractors — the **core** of Tala identification.

Features
--------
- Tempo (BPM) via librosa.beat.beat_track
- Beat positions and beat intervals
- Onset strength envelope
- Tempogram (Fourier-based)
- Rhythm histogram (normalised beat-interval distribution)
- Pulse clarity score
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tempo & beat tracking
# ---------------------------------------------------------------------------

def extract_tempo_and_beats(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict[str, object]:
    """Extract tempo (BPM) and beat positions.

    Parameters
    ----------
    y:
        Mono waveform.
    sr:
        Sample rate.
    hop_length:
        Hop size for onset analysis.

    Returns
    -------
    dict with keys:
        ``tempo_bpm``, ``beat_count``, ``beat_interval_mean``,
        ``beat_interval_std``, ``beat_interval_cv``,
        ``beat_positions`` (list of seconds),
        ``beat_regularity``
    """
    import librosa

    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units="frames"
    )
    # Handle librosa versions that return tempo as an array
    tempo_val = float(np.atleast_1d(tempo)[0])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    features: Dict[str, object] = {}
    features["tempo_bpm"] = tempo_val
    features["beat_count"] = len(beat_times)

    if len(beat_times) >= 2:
        intervals = np.diff(beat_times)
        features["beat_interval_mean"] = float(np.mean(intervals))
        features["beat_interval_std"] = float(np.std(intervals))
        features["beat_interval_cv"] = float(np.std(intervals) / (np.mean(intervals) + 1e-8))
        # Regularity: how consistent are beat intervals (0 = chaotic, 1 = perfect)
        features["beat_regularity"] = float(
            1.0 - min(1.0, np.std(intervals) / (np.mean(intervals) + 1e-8))
        )
    else:
        features["beat_interval_mean"] = 0.0
        features["beat_interval_std"] = 0.0
        features["beat_interval_cv"] = 0.0
        features["beat_regularity"] = 0.0

    features["beat_positions"] = beat_times.tolist()  # kept for visualization

    return features


# ---------------------------------------------------------------------------
# Onset strength
# ---------------------------------------------------------------------------

def extract_onset_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Extract onset strength envelope statistics.

    Onset strength captures the presence of note/beat attacks and is critical
    for Tala identification (percussive vs melodic patterns).
    """
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    features: Dict[str, float] = {}
    features["onset_strength_mean"] = float(np.mean(onset_env))
    features["onset_strength_std"] = float(np.std(onset_env))
    features["onset_strength_max"] = float(np.max(onset_env))
    features["onset_strength_median"] = float(np.median(onset_env))

    # Number of significant onsets (peaks in onset envelope)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, units="frames"
    )
    features["onset_count"] = len(onset_frames)

    # Onset rate (onsets per second)
    duration = len(y) / sr
    features["onset_rate"] = len(onset_frames) / max(duration, 0.1)

    return features


# ---------------------------------------------------------------------------
# Tempogram
# ---------------------------------------------------------------------------

def extract_tempogram_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Extract Fourier tempogram statistics.

    The tempogram shows tempo variations over time.  For a well-defined Tala,
    the tempogram should show a strong peak at the tala tempo.
    """
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )

    features: Dict[str, float] = {}
    features["tempogram_mean"] = float(np.mean(tempogram))
    features["tempogram_std"] = float(np.std(tempogram))
    features["tempogram_max"] = float(np.max(tempogram))

    # Dominant tempo from tempogram (BPM of strongest autocorrelation peak)
    ac_global = np.mean(tempogram, axis=1)
    if len(ac_global) > 1:
        # Ignore lag 0
        bpm_candidates = librosa.tempo_frequencies(len(ac_global), sr=sr, hop_length=hop_length)
        # Skip first element (inf BPM)
        ac_trimmed = ac_global[1:]
        bpm_trimmed = bpm_candidates[1:]
        if len(ac_trimmed) > 0:
            peak_idx = np.argmax(ac_trimmed)
            features["tempogram_dominant_bpm"] = float(bpm_trimmed[peak_idx])
            features["tempogram_peak_strength"] = float(ac_trimmed[peak_idx])
        else:
            features["tempogram_dominant_bpm"] = 0.0
            features["tempogram_peak_strength"] = 0.0
    else:
        features["tempogram_dominant_bpm"] = 0.0
        features["tempogram_peak_strength"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Rhythm histogram (beat interval distribution)
# ---------------------------------------------------------------------------

def extract_rhythm_histogram(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_bins: int = 16,
) -> Dict[str, float]:
    """Compute a normalised rhythm histogram from beat intervals.

    The rhythm histogram captures the distribution of inter-beat intervals
    (IOIs), which encodes the tala's characteristic grouping pattern.

    Parameters
    ----------
    n_bins:
        Number of histogram bins over the IOI range.

    Returns
    -------
    dict with ``rhythm_hist_0`` through ``rhythm_hist_{n_bins-1}``
    plus ``rhythm_hist_entropy`` (Shannon entropy of the distribution).
    """
    import librosa

    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units="frames"
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    features: Dict[str, float] = {}

    if len(beat_times) < 3:
        for i in range(n_bins):
            features[f"rhythm_hist_{i}"] = 0.0
        features["rhythm_hist_entropy"] = 0.0
        return features

    intervals = np.diff(beat_times)

    # Normalise intervals to range [0, 1] relative to the median IOI
    median_ioi = np.median(intervals)
    if median_ioi < 1e-5:
        median_ioi = 1.0
    normalised = intervals / (2.0 * median_ioi)  # [0, ~1] for "normal" intervals
    normalised = np.clip(normalised, 0, 2.0)

    hist, _ = np.histogram(normalised, bins=n_bins, range=(0, 2.0), density=True)
    hist_norm = hist / (hist.sum() + 1e-8)

    for i in range(n_bins):
        features[f"rhythm_hist_{i}"] = float(hist_norm[i])

    # Shannon entropy of the rhythm histogram (higher = more diverse rhythms)
    h = hist_norm + 1e-10
    entropy = float(-np.sum(h * np.log2(h)))
    features["rhythm_hist_entropy"] = entropy

    return features


# ---------------------------------------------------------------------------
# Pulse clarity
# ---------------------------------------------------------------------------

def extract_pulse_clarity(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Estimate pulse clarity — how clearly a rhythmic pulse is perceived.

    Higher pulse clarity indicates a strong, well-defined Tala pattern.

    Method: Compute the autocorrelation of the onset envelope and measure
    the strength of the strongest periodic peak relative to the mean.
    """
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Autocorrelation of onset envelope
    ac = np.correlate(onset_env, onset_env, mode="full")
    ac = ac[len(ac) // 2:]  # Keep positive lags only

    if len(ac) < 2:
        return {"pulse_clarity": 0.0, "pulse_strength": 0.0}

    # Normalise
    ac = ac / (ac[0] + 1e-10)

    # Find the strongest peak (ignoring lag 0)
    # Search in range corresponding to 40–200 BPM
    min_lag = int(60.0 / 200.0 * sr / hop_length)
    max_lag = int(60.0 / 40.0 * sr / hop_length)
    max_lag = min(max_lag, len(ac) - 1)
    min_lag = max(min_lag, 1)

    if min_lag >= max_lag:
        return {"pulse_clarity": 0.0, "pulse_strength": 0.0}

    search_region = ac[min_lag:max_lag]
    peak_idx = np.argmax(search_region)
    peak_val = float(search_region[peak_idx])

    # Pulse clarity = ratio of peak to mean autocorrelation
    mean_ac = float(np.mean(search_region))
    clarity = peak_val / (mean_ac + 1e-10)

    return {
        "pulse_clarity": float(min(clarity, 10.0)),  # cap at 10
        "pulse_strength": peak_val,
    }


# ---------------------------------------------------------------------------
# Cyclic Tempogram (Tala-gram) — phase alignment features
# ---------------------------------------------------------------------------

def extract_cyclic_tempogram_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Compute cyclic tempogram features for Tala-specific phase alignment.

    Folds the onset strength envelope into assumed cycle lengths corresponding
    to each known Tala (5, 6, 7, 8 beats). Measures how well the peaks in the
    onset envelope align across repeated cycles, producing a *phase alignment
    score* per cycle length and selecting the best-matching cycle.

    This is a **domain-specific feature** designed for Indian folk meters where
    standard Western beat-tracking often fails — the cyclic structure captures
    the grouping patterns (e.g., 3+4 for Misra Chapu) that distinguish Talas.

    Returns
    -------
    dict with keys:
        ``cyclic_align_5``, ``cyclic_align_6``, ``cyclic_align_7``,
        ``cyclic_align_8``, ``best_cycle_length``, ``best_cycle_score``
    """
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    if len(onset_env) < 2:
        return {
            "cyclic_align_5": 0.0, "cyclic_align_6": 0.0,
            "cyclic_align_7": 0.0, "cyclic_align_8": 0.0,
            "best_cycle_length": 0, "best_cycle_score": 0.0,
        }

    # Estimate tempo to convert beat counts → frame counts
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo_val = float(np.atleast_1d(tempo)[0])
    if tempo_val < 30:
        tempo_val = 120.0  # fallback

    beat_period_frames = int((60.0 / tempo_val) * sr / hop_length)

    features: Dict[str, float] = {}
    cycle_scores: Dict[int, float] = {}

    for n_beats in (5, 6, 7, 8):
        cycle_len = beat_period_frames * n_beats
        if cycle_len < 2 or cycle_len >= len(onset_env):
            features[f"cyclic_align_{n_beats}"] = 0.0
            cycle_scores[n_beats] = 0.0
            continue

        # Fold the onset envelope into cycles of length cycle_len
        n_full_cycles = len(onset_env) // cycle_len
        if n_full_cycles < 2:
            features[f"cyclic_align_{n_beats}"] = 0.0
            cycle_scores[n_beats] = 0.0
            continue

        usable = onset_env[:n_full_cycles * cycle_len]
        folded = usable.reshape(n_full_cycles, cycle_len)

        # Phase alignment = mean correlation of each cycle with the average template
        avg_template = folded.mean(axis=0)
        template_norm = np.linalg.norm(avg_template)

        if template_norm < 1e-8:
            score = 0.0
        else:
            correlations = []
            for row in folded:
                row_norm = np.linalg.norm(row)
                if row_norm < 1e-8:
                    correlations.append(0.0)
                else:
                    correlations.append(
                        float(np.dot(row, avg_template) / (row_norm * template_norm))
                    )
            score = float(np.mean(correlations))

        features[f"cyclic_align_{n_beats}"] = score
        cycle_scores[n_beats] = score

    # Best cycle
    if cycle_scores:
        best_cycle = max(cycle_scores, key=cycle_scores.get)
        features["best_cycle_length"] = float(best_cycle)
        features["best_cycle_score"] = cycle_scores[best_cycle]
    else:
        features["best_cycle_length"] = 0.0
        features["best_cycle_score"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def extract_all_rhythm_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> Dict[str, object]:
    """Extract all rhythm-related features and return as a flat dict.

    Note: ``beat_positions`` is a list (not float) — it should be excluded
    when building the feature matrix for ML, but kept for visualization.
    """
    features: Dict[str, object] = {}
    features.update(extract_tempo_and_beats(y, sr, hop_length))
    features.update(extract_onset_features(y, sr, hop_length))
    features.update(extract_tempogram_features(y, sr, hop_length))
    features.update(extract_rhythm_histogram(y, sr, hop_length))
    features.update(extract_pulse_clarity(y, sr, hop_length))
    features.update(extract_cyclic_tempogram_features(y, sr, hop_length))
    return features
