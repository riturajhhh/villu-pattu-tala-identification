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
        })
    return explanations


def generate_gradcam_heatmap(
    model: any,
    input_tensor: any,
    target_class: int,
) -> Optional[np.ndarray]:
    """Generate Grad-CAM heatmap for a CNN model to explain its prediction.

    Hooks into the final Conv2D layer to calculate gradient-weighted
    class activation maps, highlighting which regions of the spectrogram
    were most important for the predicted class.

    Parameters
    ----------
    model:
        Trained PyTorch CNN/CRNN model.
    input_tensor:
        PyTorch tensor of shape (1, 1, 128, 128).
    target_class:
        Index of the predicted or target class.

    Returns
    -------
    2D numpy array containing the normalized heatmap, or None on failure.
    """
    import torch
    import torch.nn.functional as F

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Find the last convolutional layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        logger.error("No Conv2D layer found for Grad-CAM.")
        return None

    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad()
        output = model(input_tensor)

        if output.dim() == 2:
            score = output[0, target_class]
        else:
            score = output[target_class]

        score.backward()

        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]

        # Global average pooling on gradients
        weights = np.mean(grads, axis=(1, 2))

        # Weight the channels
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0) # ReLU
        if np.max(cam) > 0:
            cam = cam / np.max(cam) # Normalize
            
        # Resize to input dimensions (128x128)
        import cv2
        cam_resized = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam_resized
    
    except Exception as exc:
        logger.error(f"Grad-CAM generation failed: {exc}")
        return None
        
    finally:
        handle_forward.remove()
        handle_backward.remove()


def plot_gradcam_overlay(
    mel_image: np.ndarray,
    heatmap: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> any:
    """Plot the Grad-CAM heatmap over the original Mel spectrogram.

    Parameters
    ----------
    mel_image:
        Original 2D Mel spectrogram array.
    heatmap:
        2D Grad-CAM heatmap array from generate_gradcam_heatmap.
    save_path:
        Optional path to save the plot.

    Returns
    -------
    Matplotlib Figure object.
    """
    import cv2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original Mel
    im0 = axes[0].imshow(mel_image, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Original Mel Spectrogram")
    fig.colorbar(im0, ax=axes[0])
    
    # Plot heatmap
    im1 = axes[1].imshow(heatmap, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    fig.colorbar(im1, ax=axes[1])
    
    # Plot overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Normalize mel image to 0-255 for blending
    mel_norm = (mel_image - mel_image.min()) / (mel_image.max() - mel_image.min() + 1e-8)
    mel_colored = cv2.applyColorMap(np.uint8(255 * mel_norm), cv2.COLORMAP_MAGMA)
    mel_colored = cv2.cvtColor(mel_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(mel_colored, 0.5, heatmap_colored, 0.5, 0)
    
    axes[2].imshow(overlay, aspect='auto', origin='lower')
    axes[2].set_title("Grad-CAM Overlay")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Grad-CAM overlay saved to {save_path}")
        
    return fig

