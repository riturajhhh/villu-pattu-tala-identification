"""
preprocessing/preprocess.py
============================
CLI script to batch-preprocess all audio files in the dataset.

Usage
-----
    python -m preprocessing.preprocess
    python -m preprocessing.preprocess --root dataset --output dataset/preprocessed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.audio_preprocessor import AudioPreprocessor, create_preprocessor
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.preprocess_cli", level="INFO")

SUPPORTED = {".wav", ".mp3", ".flac", ".ogg"}


def batch_preprocess(
    root: str | Path,
    output_dir: str | Path | None = None,
    sr: int = 22050,
) -> None:
    """Preprocess all audio files under *root* and save cleaned WAVs.

    Parameters
    ----------
    root:
        Dataset root with tala sub-folders.
    output_dir:
        Where to write preprocessed files.  If None, saves in-place
        with ``_clean`` suffix.
    sr:
        Target sample rate.
    """
    import soundfile as sf

    root = Path(root)
    preprocessor = create_preprocessor()

    audio_files = [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED
    ]

    logger.info(f"Found {len(audio_files)} audio files under {root}")

    success, failed = 0, 0

    for idx, audio_path in enumerate(audio_files, 1):
        try:
            y, out_sr = preprocessor.preprocess(audio_path)

            if output_dir:
                # Preserve sub-folder structure
                rel = audio_path.relative_to(root)
                out_path = Path(output_dir) / rel.with_suffix(".wav")
            else:
                out_path = audio_path.with_name(
                    audio_path.stem + "_clean.wav"
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), y, out_sr, subtype="PCM_16")
            success += 1

        except Exception as exc:
            logger.warning(f"  ✗ Failed: {audio_path.name} — {exc}")
            failed += 1

        if idx % 25 == 0:
            logger.info(f"  Progress: {idx}/{len(audio_files)}")

    logger.info(f"\nDone. Success: {success} | Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch preprocess audio files")
    parser.add_argument("--root", default="dataset", help="Dataset root")
    parser.add_argument("--output", default=None, help="Output directory (None = in-place)")
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate")
    args = parser.parse_args()

    batch_preprocess(root=args.root, output_dir=args.output, sr=args.sr)
