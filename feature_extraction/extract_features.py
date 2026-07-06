"""
feature_extraction/extract_features.py
========================================
CLI script to extract features from all audio files in the dataset
and write the combined result to a CSV.

Usage
-----
    python -m feature_extraction.extract_features
    python -m feature_extraction.extract_features --root dataset --output dataset/features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.feature_extractor import FeatureExtractor
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.extract_features", level="INFO")

SUPPORTED = {".wav", ".mp3", ".flac", ".ogg"}


def extract_all(
    root: str | Path,
    output_csv: str | Path = "dataset/features.csv",
    force: bool = False,
) -> pd.DataFrame:
    """Extract features for every audio file under ``root``.

    Parameters
    ----------
    root:
        Dataset root with tala sub-folders.
    output_csv:
        Path to write the features CSV.
    force:
        If True, re-extract even if the CSV already exists.

    Returns
    -------
    pd.DataFrame
        Combined feature dataframe.
    """
    root = Path(root)
    output_csv = Path(output_csv)

    if output_csv.exists() and not force:
        logger.info(f"Features CSV already exists: {output_csv}. Use --force to re-extract.")
        return pd.read_csv(output_csv)

    extractor = FeatureExtractor()

    # Walk all tala sub-folders
    tala_dirs = [
        d for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith((".", "_", "splits"))
    ]

    if not tala_dirs:
        logger.error(f"No tala folders found in {root}")
        return pd.DataFrame()

    logger.info(f"Found {len(tala_dirs)} tala classes: {[d.name for d in tala_dirs]}")

    rows = []
    total_files = 0
    success_count = 0

    for tala_dir in tala_dirs:
        tala_name = tala_dir.name
        audio_files = [
            f for f in sorted(tala_dir.rglob("*"))
            if f.is_file() and f.suffix.lower() in SUPPORTED
        ]
        logger.info(f"  {tala_name}: {len(audio_files)} files")
        total_files += len(audio_files)

        for idx, audio_path in enumerate(audio_files, 1):
            features = extractor.extract(audio_path)
            if features is not None:
                features["path"] = str(audio_path)
                features["filename"] = audio_path.name
                features["tala"] = tala_name
                rows.append(features)
                success_count += 1

            if idx % 25 == 0:
                logger.info(f"    {idx}/{len(audio_files)} processed")

    df = pd.DataFrame(rows)
    logger.info(f"\nExtracted features for {success_count}/{total_files} files")

    if df.empty:
        logger.error("No features extracted!")
        return df

    # Drop non-scalar columns for the ML-ready CSV
    # (beat_positions is a list, not a scalar)
    if "beat_positions" in df.columns:
        df = df.drop(columns=["beat_positions"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Features saved to: {output_csv}")
    logger.info(f"Feature columns: {len(df.columns)}")
    logger.info(f"Class distribution:\n{df['tala'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for all dataset audio")
    parser.add_argument("--root", default="dataset", help="Dataset root")
    parser.add_argument("--output", default="dataset/features.csv", help="Output CSV path")
    parser.add_argument("--force", action="store_true", help="Force re-extraction")
    args = parser.parse_args()

    extract_all(root=args.root, output_csv=args.output, force=args.force)
