"""
training/train_classical.py
=============================
Classical ML model training and comparison for Tala identification.

Models implemented
------------------
- Random Forest (scikit-learn)
- XGBoost
- SVM (RBF kernel, calibrated)
- KNN (distance-weighted)
- Voting Ensemble (soft voting)

Outputs
-------
- Best model saved as ``models/saved_models/best_classical_model.pkl``
- Label encoder as ``models/saved_models/label_encoder.pkl``
- Comparison table and confusion matrix
- Model metadata JSON

Usage
-----
    python -m training.train_classical
    python -m training.train_classical --features dataset/features.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

warnings.filterwarnings("ignore")

logger = setup_logger("villu_pattu.train_classical", level="INFO")

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Feature columns to exclude (non-numeric or metadata)
# ---------------------------------------------------------------------------

EXCLUDE_COLS = {"path", "filename", "tala", "beat_positions", "duration"}


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> Dict[str, Any]:
    """Return a dictionary of model name → model instance."""
    models: Dict[str, Any] = {}

    # Random Forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    # XGBoost
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        logger.warning("XGBoost not installed — skipping")

    # SVM (RBF kernel, calibrated for probability estimates)
    models["svm_rbf"] = CalibratedClassifierCV(
        SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            random_state=SEED,
        ),
        cv=3,
    )

    # KNN (distance-weighted)
    models["knn"] = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        algorithm="auto",
        n_jobs=-1,
    )

    # Soft-Voting Ensemble (RF + SVM + KNN)
    models["voting_ensemble"] = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=SEED, n_jobs=-1
            )),
            ("svm", CalibratedClassifierCV(
                SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=SEED),
                cv=3
            )),
            ("knn", KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1
            )),
        ],
        voting="soft",
    )

    return models


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_features(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """Load features CSV and return X, y arrays."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {csv_path}. "
            "Run `python -m feature_extraction.extract_features` first."
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")

    if "tala" not in df.columns:
        raise ValueError("CSV must contain a 'tala' column")

    # Select only numeric feature columns
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    logger.info(f"Feature columns: {len(feature_cols)}")

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["tala"])

    logger.info(f"Classes: {le.classes_.tolist()}")
    logger.info(f"Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    return X, y, feature_cols, le


def train_all_models(
    features_csv: str | Path = "dataset/features.csv",
    output_dir: str | Path = "models/saved_models",
    test_size: float = 0.15,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Train all classical ML models, evaluate, and save the best.

    Returns
    -------
    dict with ``best_model_name``, ``best_accuracy``, ``results`` (list of dicts).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("CLASSICAL ML TRAINING — Villu Pattu Tala System")
    logger.info("=" * 60)

    # Load data
    X, y, feature_cols, le = load_features(features_csv)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = get_models()
    best_acc = 0.0
    best_name = ""
    best_model = None
    results: List[Dict[str, Any]] = []

    logger.info(f"\n{'Model':<25s} {'Accuracy':>10s} {'F1':>10s} {'Precision':>10s} {'Recall':>10s} {'Time':>8s}")
    logger.info("-" * 75)

    for name, clf in models.items():
        t0 = time.time()

        try:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            elapsed = time.time() - t0

            tag = ""
            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_model = clf
                tag = " ★"

            results.append({
                "model": name,
                "accuracy": float(acc),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "time_seconds": float(elapsed),
            })

            logger.info(
                f"  {name:<23s} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f} {elapsed:>7.1f}s{tag}"
            )

        except Exception as exc:
            logger.error(f"  {name}: FAILED — {exc}")

    if best_model is None:
        logger.error("All models failed!")
        return {"best_model_name": None, "best_accuracy": 0, "results": []}

    # --- Save best model ---
    logger.info(f"\n{'=' * 60}")
    logger.info(f"WINNER: {best_name} (accuracy: {best_acc:.4f})")
    logger.info(f"{'=' * 60}")

    # Final classification report
    y_pred_final = best_model.predict(X_test_scaled)
    logger.info("\nClassification Report:")
    report = classification_report(y_test, y_pred_final, target_names=le.classes_)
    logger.info(f"\n{report}")

    cm = confusion_matrix(y_test, y_pred_final)
    logger.info("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    logger.info(f"\n{cm_df}")

    # Cross-validation on full dataset
    logger.info(f"\n{cv_folds}-Fold Cross-Validation on full dataset ...")
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring="accuracy")
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Save artifacts ---
    # Build a full pipeline: scaler + model
    full_pipeline = Pipeline([
        ("scaler", scaler),
        ("clf", best_model),
    ])

    # Refit on full training data
    full_pipeline.fit(X_train, y_train)

    joblib.dump(full_pipeline, output_dir / "best_classical_model.pkl")
    joblib.dump(le, output_dir / "label_encoder.pkl")
    joblib.dump(feature_cols, output_dir / "feature_columns.pkl")

    # Metadata
    metadata = {
        "model_type": "classical_ml",
        "best_model": best_name,
        "test_accuracy": float(best_acc),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "classes": le.classes_.tolist(),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "results": results,
        "training_time_seconds": float(time.time() - start_time),
    }

    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nArtifacts saved to: {output_dir}/")
    logger.info(f"Total training time: {time.time() - start_time:.1f}s")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classical ML models for Tala ID")
    parser.add_argument("--features", default="dataset/features.csv", help="Features CSV")
    parser.add_argument("--output", default="models/saved_models", help="Model output dir")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    train_all_models(
        features_csv=args.features,
        output_dir=args.output,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )
