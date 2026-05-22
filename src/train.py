"""
Training loop for the SPY LSTM model.

Key choices:
  - BCEWithLogitsLoss: numerically stable binary cross-entropy (model outputs raw logits)
  - Adam optimizer with ReduceLROnPlateau: halves LR if val loss stalls for 5 epochs
  - Gradient clipping (max norm 1.0): prevents exploding gradients in RNNs
  - Best model checkpoint: saved whenever val accuracy improves
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    model_save_path: str = "models/spy_lstm_model.pt",
    device: torch.device = None,
) -> dict:
    """
    Train the model, evaluate on validation set each epoch, and save the best checkpoint.

    Returns:
        history dict with lists: train_loss, val_loss, val_acc
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Weight the positive class inversely to its frequency so the model is
    # penalised equally for missing Down moves as Up moves.
    train_labels = train_dataset.y.numpy()
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"[train] Class balance — Up: {int(n_pos)}, Down: {int(n_neg)}, pos_weight: {pos_weight.item():.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Halve LR if validation loss doesn't improve for 5 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_losses.append(criterion(logits, y_batch).item())
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_acc = correct / total if total > 0 else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        # Checkpoint on best val loss (more stable than val acc for early stopping)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc  = max(best_val_acc, val_acc)
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
                + (" ← saved" if improved else f"  (no improve {epochs_no_improve}/{early_stopping_patience})")
            )

        if epochs_no_improve >= early_stopping_patience:
            print(f"\n[train] Early stopping at epoch {epoch} (no val loss improvement for {early_stopping_patience} epochs)")
            break

    print(f"\n[train] Best validation accuracy: {best_val_acc:.4f}")
    print(f"[train] Model saved to {model_save_path}")
    return history
