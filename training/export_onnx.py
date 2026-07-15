"""
training/export_onnx.py
========================
Export trained PyTorch models (CNN / CRNN) to ONNX format for optimised
production inference. ONNX models run significantly faster via
onnxruntime and are portable across hardware backends.

Usage
-----
    python -m training.export_onnx --model-type cnn
    python -m training.export_onnx --model-type crnn --models-dir models/saved_models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger("villu_pattu.export_onnx", level="INFO")


def export_to_onnx(
    model_type: str = "cnn",
    models_dir: str | Path = "models/saved_models",
    output_path: Optional[str | Path] = None,
    opset_version: int = 14,
) -> Optional[Path]:
    """Convert a trained PyTorch model to ONNX format.

    Parameters
    ----------
    model_type:
        ``'cnn'`` or ``'crnn'``.
    models_dir:
        Directory containing the trained ``.pt`` weights and metadata JSON.
    output_path:
        Explicit output path. If None, saves alongside the ``.pt`` file.
    opset_version:
        ONNX opset version (default 14, widely supported).

    Returns
    -------
    Path to the exported ``.onnx`` file, or None on failure.
    """
    import torch

    models_dir = Path(models_dir)

    # --- Resolve model artifacts ---
    if model_type == "cnn":
        from training.train_cnn import build_cnn_model
        meta_path = models_dir / "cnn_metadata.json"
        weights_path = models_dir / "cnn_model.pt"
        default_out = models_dir / "cnn_model.onnx"
        build_fn = build_cnn_model
    elif model_type == "crnn":
        from training.train_crnn import build_crnn_model
        meta_path = models_dir / "crnn_metadata.json"
        weights_path = models_dir / "crnn_model.pt"
        default_out = models_dir / "crnn_model.onnx"
        build_fn = build_crnn_model
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None

    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        return None

    if not meta_path.exists():
        logger.error(f"Metadata file not found: {meta_path}")
        return None

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)
    n_classes = meta.get("n_classes", len(meta.get("classes", [])))

    # Build and load model
    model = build_fn(n_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    # Create dummy input (batch=1, channel=1, 128x128 mel spectrogram)
    dummy_input = torch.randn(1, 1, 128, 128)

    # Export
    out_path = Path(output_path) if output_path else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["mel_spectrogram"],
            output_names=["logits"],
            dynamic_axes={
                "mel_spectrogram": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        logger.info(f"Successfully exported {model_type.upper()} model to: {out_path}")
        logger.info(f"  Classes: {meta.get('classes', [])}")
        logger.info(f"  ONNX opset version: {opset_version}")
        logger.info(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")
        return out_path

    except Exception as exc:
        logger.error(f"ONNX export failed: {exc}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch Tala model to ONNX")
    parser.add_argument(
        "--model-type", default="cnn", choices=["cnn", "crnn"],
        help="Which model to export"
    )
    parser.add_argument(
        "--models-dir", default="models/saved_models",
        help="Directory containing trained model files"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output ONNX file path (optional)"
    )
    parser.add_argument(
        "--opset", type=int, default=14,
        help="ONNX opset version"
    )
    args = parser.parse_args()

    result = export_to_onnx(
        model_type=args.model_type,
        models_dir=args.models_dir,
        output_path=args.output,
        opset_version=args.opset,
    )

    if result:
        print(f"\n✓ ONNX model exported to: {result}")
    else:
        print("\n✗ Export failed. Check logs for details.")
        sys.exit(1)
