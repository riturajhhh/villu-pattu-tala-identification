"""
training/train_crnn.py
=======================
CRNN (CNN + LSTM) model training on Mel Spectrograms for Tala identification.

Architecture
------------
Input:  1 × 128 × 128 (Mel spectrogram)
CNN blocks extract spatial-spectral features.
Then we reshape/flatten height and keep the time dimension.
BiLSTM processing along the time frames captures rhythm/temporal sequence patterns.
Dense/FC output layer classifies into Tala.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger("villu_pattu.train_crnn", level="INFO")

SEED = 42
np.random.seed(SEED)


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_crnn_model(n_classes: int, input_shape: Tuple[int, int, int] = (1, 128, 128)):
    """Build CRNN (CNN + BiLSTM) model."""
    import torch
    import torch.nn as nn

    class TalaCRNN(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            # CNN front-end
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 128x128 -> 64x64

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 64x64 -> 32x32

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 32x32 -> 16x16
            )
            
            # Recurrent back-end
            # Input dimension to LSTM = CNN channels (128) * reduced frequency height (16)
            self.lstm_input_dim = 128 * 16
            self.hidden_dim = 64
            self.lstm = nn.LSTM(
                input_size=self.lstm_input_dim,
                hidden_size=self.hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
            
            # Output classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(64, n_classes)
            )

        def forward(self, x):
            # x shape: (batch, 1, frequency_bins, time_frames) -> (B, 1, 128, 128)
            x = self.cnn(x)  # shape: (batch, 128, 16, 16)
            
            # Rearrange for sequence processing: sequence length is the time dimension (16 frames)
            # Permute to (batch, time_frames, channels, frequency_bins)
            # or permute to (batch, time_frames, channels * frequency_bins)
            # Here: (B, C, F, T) -> C=128, F=16, T=16. Let's make time_frames the sequence.
            # Original: (B, 128, 16, 16) where dim 3 is time_frames.
            # Let's permute to (B, T, C, F) -> (B, 16, 128, 16)
            x = x.permute(0, 3, 1, 2)
            batch_size, seq_len, channels, freq_bins = x.size()
            x = x.reshape(batch_size, seq_len, channels * freq_bins)  # (B, 16, 128 * 16)
            
            lstm_out, _ = self.lstm(x)  # (B, 16, hidden_dim * 2)
            
            # Use final time step output
            out = lstm_out[:, -1, :]  # (B, hidden_dim * 2)
            out = self.classifier(out)
            return out

    return TalaCRNN(n_classes)


def _load_mel_dataset(
    dataset_root: str | Path,
    target_shape: Tuple[int, int] = (128, 128),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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

    X = np.array(X_list, dtype=np.float32)[:, np.newaxis, :, :]
    y = np.array(y_list, dtype=np.int64)

    logger.info(f"Loaded {len(X)} mel spectrograms for CRNN of shape {X.shape[1:]}")
    return X, y, class_names


def train_crnn(
    dataset_root: str | Path = "dataset",
    output_dir: str | Path = "models/saved_models",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 10,
    test_size: float = 0.15,
) -> Dict[str, Any]:
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
    logger.info("CRNN TRAINING — CNN + LSTM → Tala")
    logger.info("=" * 60)

    X, y, class_names = _load_mel_dataset(dataset_root)
    n_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {n_classes}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_crnn_model(n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
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

        # Validate
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
            torch.save(model.state_dict(), output_dir / "crnn_model.pt")
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
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Final evaluation
    model.load_state_dict(torch.load(output_dir / "crnn_model.pt", weights_only=True))
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
    logger.info(f"\nFinal CRNN Test Accuracy: {final_acc:.4f}")
    logger.info(f"\n{report}")

    metadata = {
        "model_type": "crnn",
        "architecture": "3xConv2D+BiLSTM+FC",
        "test_accuracy": float(final_acc),
        "best_val_accuracy": float(best_val_acc),
        "classes": class_names,
        "n_classes": n_classes,
        "epochs_trained": len(history["train_loss"]),
        "training_time_seconds": float(time.time() - start_time),
    }

    with open(output_dir / "crnn_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "crnn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"CRNN model saved to: {output_dir}/crnn_model.pt")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN for Tala ID")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="models/saved_models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    train_crnn(
        dataset_root=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )
