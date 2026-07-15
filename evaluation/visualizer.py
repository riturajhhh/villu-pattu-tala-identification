"""
evaluation/visualizer.py
========================
Audio feature visualization library for the Villu Pattu Tala system.
Generates publication-quality waveforms, spectrograms, beat tracking overlays,
MFCC heatmaps, and onset envelopes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_waveform(
    y: np.ndarray,
    sr: int = 22050,
    title: str = "Audio Waveform",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot the amplitude envelope over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.figure

    time_axis = np.linspace(0, len(y) / sr, len(y))
    ax.plot(time_axis, y, color="#1DB954", alpha=0.8, linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, time_axis[-1])
    plt.tight_layout()
    return fig


def plot_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a standard power spectrogram (dB scale)."""
    import librosa
    import librosa.display

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)

    img = librosa.display.specshow(
        stft_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax,
        cmap="magma",
    )
    ax.set_title("Power Spectrogram", fontsize=12, fontweight="bold", pad=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512,
    fmax: int = 8000,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot the Mel-scaled spectrogram (dB scale)."""
    import librosa
    import librosa.display

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmax=fmax,
        ax=ax,
        cmap="viridis",
    )
    ax.set_title("Mel Spectrogram", fontsize=12, fontweight="bold", pad=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_beat_tracking(
    y: np.ndarray,
    sr: int,
    beat_times: List[float],
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot the waveform with detected beats overlaid as vertical dotted lines."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.figure

    time_axis = np.linspace(0, len(y) / sr, len(y))
    ax.plot(time_axis, y, color="#555555", alpha=0.4, linewidth=0.5)

    # Plot vertical lines for each beat
    for idx, bt in enumerate(beat_times):
        label = "Detected Beats" if idx == 0 else ""
        ax.axvline(bt, color="#E74C3C", linestyle="--", alpha=0.8, linewidth=1.2, label=label)

    ax.set_title(f"Beat Tracking Overlay ({len(beat_times)} Beats)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_xlim(0, time_axis[-1])
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_onset_envelope(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot the onset strength envelope over time."""
    import librosa
    import librosa.display

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.figure

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    ax.plot(times, onset_env, color="#3498DB", alpha=0.9, linewidth=1.5)
    ax.set_title("Onset Strength Envelope", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Onset Strength", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, times[-1])
    plt.tight_layout()
    return fig


def plot_mfcc_heatmap(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    hop_length: int = 512,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a heatmap of Mel-Frequency Cepstral Coefficients over time."""
    import librosa
    import librosa.display

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    img = librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        ax=ax,
        cmap="coolwarm",
    )
    ax.set_title(f"MFCC Heatmap (First {n_mfcc} Coefficients)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Coefficients", fontsize=10)
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    return fig


def plot_chromagram(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a Chromagram showing pitch class distribution over time."""
    import librosa
    import librosa.display

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    img = librosa.display.specshow(
        chroma,
        y_axis='chroma',
        x_axis='time',
        ax=ax,
        cmap='coolwarm'
    )
    ax.set_title('Chromagram (Pitch Class)', fontsize=12, fontweight="bold", pad=10)
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    return fig


def plot_tempogram(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a Fourier Tempogram showing tempo variations over time."""
    import librosa
    import librosa.display

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)

    img = librosa.display.specshow(
        tempogram,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='tempo',
        cmap='magma',
        ax=ax
    )
    ax.set_title('Fourier Tempogram', fontsize=12, fontweight="bold", pad=10)
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    return fig
