"""
evaluation/explainability.py
=============================
Explainable AI (XAI) features for the Villu Pattu Tala identification system.
Includes:
- Classical ML feature importance visualization
- SHAP value explanations (optional backend support)
- Gradient / feature attribution explanation
- Feature attribution mapping for web interface
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger("villu_pattu.explainability", level="INFO")


def get_feature_importances(
    model: any,
    feature_names: List[str],
    top_n: int = 15,
    save_plot_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Extract and plot feature importances from a trained tree-based model (e.g. Random Forest, XGBoost).

    Parameters
    ----------
    model:
        Trained model classifier.
    feature_names:
        List of all feature names.
    top_n:
        Limit results to top N features.
    save_plot_path:
        Optional path to save feature importance bar plot.
    """
    import sklearn.ensemble
    import pipeline
    
    # Unwrap pipeline if needed
    clf = model
    if hasattr(model, "steps"):
        clf = model.steps[-1][1]

    # Unwrap calibrated classifier wrapper if needed
    if hasattr(clf, "base_estimator"):
        clf = clf.base_estimator
    elif hasattr(clf, "estimator"):
        clf = clf.estimator

    # Check for feature importances
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.mean(np.abs(clf.coef_), axis=0)

    if importances is None:
        logger.warning("Selected model does not support feature importances / coefficients.")
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_df = df.head(top_n)

    if save_plot_path:
        save_plot_path = Path(save_plot_path)
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="darkgrid")
        sns.barplot(
            data=top_df,
            x="importance",
            y="feature",
            palette="viridis"
        )
        plt.title(f"Top {top_n} Most Important Rhythmic/Acoustic Features")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature Name")
        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()
        logger.info(f"Feature importance plot saved to {save_plot_path}")

    return df


def explain_prediction_confidence(
    probabilities: np.ndarray,
    classes: List[str]
) -> List[Dict[str, float]]:
    """Format full probability distribution as an ordered list of prediction confidences."""
    sorted_indices = np.argsort(probabilities)[::-1]
    explanations = []
    for idx in sorted_indices:
        explanations.append({
            "tala": classes[idx],
            "confidence": float(probabilities[idx])
        })
    return explanations
