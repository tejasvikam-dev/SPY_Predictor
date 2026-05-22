"""
Evaluation and visualization for the trained SPY LSTM model.

Metrics:  Accuracy, Precision, Recall, Confusion Matrix
Plots:    Training history (loss + val acc), Confusion matrix, Predicted vs actual
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # headless backend — saves to file without a display
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(
    model: nn.Module,
    test_dataset,
    model_path: str,
    device: torch.device = None,
    output_dir: str = "outputs",
) -> dict:
    """
    Load best checkpoint, run inference on the test set, print metrics, save plots.

    Returns:
        dict with keys: accuracy, precision, recall, confusion_matrix
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best saved weights
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 40)
    print("  TEST SET EVALUATION")
    print("=" * 40)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"    {cm}")
    print("=" * 40)

    os.makedirs(output_dir, exist_ok=True)
    _plot_confusion_matrix(cm, output_dir)
    _plot_predictions(all_labels, all_preds, output_dir)

    return {"accuracy": acc, "precision": prec, "recall": rec, "confusion_matrix": cm}


def plot_training_history(history: dict, output_dir: str = "outputs") -> None:
    """Save a two-panel plot of training/validation loss and validation accuracy."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="coral")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCEWithLogitsLoss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="seagreen")
    ax2.axhline(0.5, linestyle="--", color="gray", alpha=0.6, label="Random baseline")
    ax2.set_title("Validation Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[evaluate] Training history saved → {path}")


def _plot_confusion_matrix(cm: np.ndarray, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Down (0)", "Up (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("SPY Direction — Confusion Matrix (Test Set)")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {path}")


def _plot_predictions(
    labels: np.ndarray, preds: np.ndarray, output_dir: str, n: int = 150
) -> None:
    """Plot predicted vs actual direction for the first n test samples."""
    n = min(n, len(labels))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(labels[:n], label="Actual", linewidth=1.5, alpha=0.8)
    ax.plot(preds[:n], label="Predicted", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_title(f"Predicted vs Actual Direction (first {n} test samples)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Direction  (0 = Down, 1 = Up)")
    ax.set_yticks([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[evaluate] Prediction chart saved → {path}")
