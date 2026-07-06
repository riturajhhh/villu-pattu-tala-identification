"""
training/train_cnn.py
======================
CNN model training on Mel Spectrograms for Tala identification.

Architecture
------------
Input:  1 × 128 × 128  (Mel spectrogram image)
Conv2D (32, 3×3) + BN + ReLU + MaxPool(2×2)
Conv2D (64, 3×3) + BN + ReLU + MaxPool(2×2)
Conv2D (128, 3×3) + BN + ReLU + MaxPool(2×2)
Global Average Pool → FC(128) → ReLU → Dropout → FC(n_classes) → Softmax

Usage
-----
    python -m training.train_cnn
    python -m training.train_cnn --dataset dataset --epochs 50 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger("villu_pattu.train_cnn", level="INFO")

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------

def _get_device():
    """Return the best available PyTorch device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cnn_model(n_classes: int):
    """Build the CNN model for mel spectrogram classification."""
    import torch
    import torch.nn as nn

    class TalaCNN(nn.Module):
        """3-layer CNN for Tala identification from mel spectrograms."""

        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1),

                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.15),

                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return TalaCNN(n_classes)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

def _load_mel_dataset(
    dataset_root: str | Path,
    target_shape: Tuple[int, int] = (128, 128),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all audio files as mel spectrograms and return arrays.

    Returns
    -------
    (X, y, class_names)
        X : (n_samples, 1, 128, 128) float32 array
        y : (n_samples,) int array of class indices
        class_names : sorted list of tala names
    """
    from feature_extraction.feature_extractor import FeatureExtractor

    dataset_root = Path(dataset_root)
    extractor = FeatureExtractor()
    SUPPORTED = {".wav", ".mp3", ".flac", ".ogg"}

    tala_dirs = sorted([
        d for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_", "splits"))
    ])
    class_names = [d.name for d in tala_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    X_list, y_list = [], []

    for tala_dir in tala_dirs:
        tala_name = tala_dir.name
        files = [f for f in sorted(tala_dir.rglob("*"))
                 if f.is_file() and f.suffix.lower() in SUPPORTED]
        logger.info(f"  {tala_name}: {len(files)} files")

        for f in files:
            mel = extractor.extract_mel_image(f, target_shape=target_shape)
            if mel is not None:
                X_list.append(mel)
                y_list.append(class_to_idx[tala_name])

    X = np.array(X_list, dtype=np.float32)[:, np.newaxis, :, :]  # Add channel dim
    y = np.array(y_list, dtype=np.int64)

    logger.info(f"Loaded {len(X)} mel spectrograms of shape {X.shape[1:]}")
    return X, y, class_names


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnn(
    dataset_root: str | Path = "dataset",
    output_dir: str | Path = "models/saved_models",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 10,
    test_size: float = 0.15,
) -> Dict[str, Any]:
    """Train the CNN model.

    Returns
    -------
    dict with training metadata.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    torch.manual_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    logger.info(f"Device: {device}")

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("CNN TRAINING — Mel Spectrogram → Tala")
    logger.info("=" * 60)

    # Load data
    X, y, class_names = _load_mel_dataset(dataset_root)
    n_classes = len(class_names)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {n_classes}")

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = build_cnn_model(n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_y)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "cnn_model.pt")
            tag = " ★ saved"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or tag:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}{tag}"
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # --- Final evaluation ---
    model.load_state_dict(torch.load(output_dir / "cnn_model.pt", weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    logger.info(f"\nFinal CNN Test Accuracy: {final_acc:.4f}")
    logger.info(f"\n{report}")

    # Save metadata
    metadata = {
        "model_type": "cnn",
        "architecture": "3xConv2D+BN+MaxPool+GAP+FC",
        "test_accuracy": float(final_acc),
        "best_val_accuracy": float(best_val_acc),
        "classes": class_names,
        "n_classes": n_classes,
        "epochs_trained": len(history["train_loss"]),
        "input_shape": [1, 128, 128],
        "training_time_seconds": float(time.time() - start_time),
    }

    with open(output_dir / "cnn_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "cnn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"CNN model saved to: {output_dir}/cnn_model.pt")
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Tala ID")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="models/saved_models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    train_cnn(
        dataset_root=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )
