"""
evaluation/evaluate.py
=======================
Evaluation pipeline for the Villu Pattu Tala Identification System.
Computes and saves performance metrics, confusion matrices, classification reports,
and learning curves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger("villu_pattu.evaluate", level="INFO")


def generate_evaluation_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_dir: str | Path = "outputs/reports",
    plots_dir: str | Path = "outputs/plots",
) -> Dict[str, Any]:
    """Generate and save comprehensive classification performance metrics.

    Parameters
    ----------
    y_true:
        True class labels (indices or strings).
    y_pred:
        Predicted class labels (indices or strings).
    classes:
        List of all class names.
    output_dir:
        Directory to save reports (CSV, JSON).
    plots_dir:
        Directory to save plots (PNG).
    """
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation reports...")

    # Classification Report
    report_dict = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )
    report_txt = classification_report(y_true, y_pred, target_names=classes)

    # Save textual report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report_txt)

    # Save JSON report
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="dark")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title("Tala Classification Confusion Matrix")
    plt.ylabel("Actual Tala")
    plt.xlabel("Predicted Tala")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # Calculate overall metrics
    accuracy = float(np.mean(y_true == y_pred))
    logger.info(f"Evaluation Accuracy: {accuracy:.4%}")

    return {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def plot_dl_history(
    history_json: str | Path,
    plots_dir: str | Path = "outputs/plots",
) -> None:
    """Plot deep learning training loss and accuracy curves."""
    history_json = Path(history_json)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not history_json.exists():
        logger.warning(f"History file not found: {history_json}")
        return

    with open(history_json, "r") as f:
        hist = json.load(f)

    epochs = range(1, len(hist["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["train_loss"], label="Train Loss", color="#1f77b4")
    plt.plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["train_acc"], label="Train Acc", color="#2ca02c")
    plt.plot(epochs, hist["val_acc"], label="Val Acc", color="#d62728")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plots_dir / f"{history_json.stem}_curves.png", dpi=300)
    plt.close()
    logger.info(f"Loss/Accuracy curves saved to {plots_dir}")


if __name__ == "__main__":
    # Test execution
    y_t = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_p = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2])
    generate_evaluation_reports(y_t, y_p, ["Adi", "Rupaka", "Misra_Chapu"])
