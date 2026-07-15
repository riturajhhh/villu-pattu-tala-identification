"""
training/train.py
==================
Master training engine for the Villu Pattu Tala Identification system.
Sequentially runs:
1. Classical ML Training (train_classical.py)
2. CNN Deep Learning (train_cnn.py)
3. CRNN Deep Learning (train_crnn.py)

Then displays a comprehensive comparison table comparing all architectures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_classical import train_all_models
from training.train_cnn import train_cnn
from training.train_crnn import train_crnn
from utils.experiment_tracker import ExperimentTracker
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.train_master", level="INFO")


def run_master_training(
    dataset_root: str | Path = "dataset",
    features_csv: str | Path = "dataset/features.csv",
    output_dir: str | Path = "models/saved_models",
    epochs: int = 25,  # Moderate default for full pipeline execution
    batch_size: int = 32,
) -> None:
    """Run all training pipelines and print comparison table."""
    dataset_root = Path(dataset_root)
    features_csv = Path(features_csv)
    output_dir = Path(output_dir)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("MASTER MODEL TRAINING ENGINE — VILLU PATTU TALA SYSTEM")
    logger.info("=" * 60)

    tracker = ExperimentTracker(experiment_name="villu_pattu_tala_models")
    tracker.log_params({
        "dataset_root": str(dataset_root),
        "epochs": epochs,
        "batch_size": batch_size,
    })

    # 1. Train Classical Models
    classical_meta = None
    try:
        classical_meta = train_all_models(
            features_csv=features_csv,
            output_dir=output_dir,
            test_size=0.15,
        )
    except Exception as exc:
        logger.error(f"Classical training failed: {exc}")

    # 2. Train CNN Model
    cnn_meta = None
    try:
        cnn_meta = train_cnn(
            dataset_root=dataset_root,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
        )
    except Exception as exc:
        logger.error(f"CNN training failed: {exc}")

    # 3. Train CRNN Model
    crnn_meta = None
    try:
        crnn_meta = train_crnn(
            dataset_root=dataset_root,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
        )
    except Exception as exc:
        logger.error(f"CRNN training failed: {exc}")

    # --- Display Summary Comparison ---
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model/Architecture':<35s} | {'Test Accuracy':>15s}")
    logger.info("-" * 60)

    comparison_rows = []

    # Classical summary
    if classical_meta and "results" in classical_meta:
        for r in classical_meta["results"]:
            name = f"Classical: {r['model']}"
            acc = r["accuracy"]
            comparison_rows.append({"model": name, "accuracy": acc})
            logger.info(f"{name:<35s} | {acc:>15.4%}")

    # CNN summary
    if cnn_meta:
        name = "Deep Learning: CNN (Mel Spectrogram)"
        acc = cnn_meta["test_accuracy"]
        comparison_rows.append({"model": name, "accuracy": acc})
        logger.info(f"{name:<35s} | {acc:>15.4%}")

    # CRNN summary
    if crnn_meta:
        name = "Deep Learning: CRNN (CNN + LSTM)"
        acc = crnn_meta["test_accuracy"]
        comparison_rows.append({"model": name, "accuracy": acc})
        logger.info(f"{name:<35s} | {acc:>15.4%}")

    # Save master comparison
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(output_dir / "model_comparison.csv", index=False)

    total_time = time.time() - start_time
    logger.info(f"\nTotal Pipeline Execution Time: {total_time/60:.1f} minutes")
    logger.info("=" * 60)
    
    # Track results
    tracker.log_metrics({row["model"]: row["accuracy"] for row in comparison_rows})
    tracker.log_artifact(output_dir / "model_comparison.csv")
    tracker.save_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master training engine for Tala ID")
    parser.add_argument("--dataset", default="dataset", help="Dataset root dir")
    parser.add_argument("--features", default="dataset/features.csv", help="Features CSV path")
    parser.add_argument("--output", default="models/saved_models", help="Model saved directory")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs for DL models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DL models")
    args = parser.parse_args()

    run_master_training(
        dataset_root=args.dataset,
        features_csv=args.features,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
