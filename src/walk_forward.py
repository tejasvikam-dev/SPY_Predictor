"""
Walk-forward cross-validation for the SPY LSTM model.

Expanding-window scheme:
  • The last 15 % of bars are held out as the test set (never seen during CV).
  • Within the remaining data, the training window starts at 40 % and grows
    by one validation-window per fold.
  • Each fold trains a fresh model with its own scaler fit on that fold's
    training slice — no look-ahead bias.
  • After all folds, the best fold's model is evaluated on the test set.

Fold layout (5 folds, 15 % test):

  |──── fold 1 train (40%) ────|─ val ─|                       | test |
  |──────── fold 2 train ───────────────|─ val ─|               | test |
  |──────────── fold 3 train ────────────────────|─ val ─|      | test |
  ...
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.dataset import walk_forward_splits, make_fold_datasets, make_test_dataset
from src.model import SPYLSTMModel
from src.train import train_model
from src.evaluate import evaluate_model


def run_walk_forward(
    df,
    feature_cols: list,
    seq_len: int = 30,
    n_folds: int = 5,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    model_dir: str = "models",
    output_dir: str = "outputs",
) -> list[dict]:
    """
    Run expanding-window walk-forward CV and evaluate on the held-out test set.

    Returns:
        fold_results — list of per-fold metric dicts
    """
    features = df[feature_cols].values
    targets  = df["target"].values
    n        = len(features)

    splits, test_start = walk_forward_splits(n, n_folds=n_folds)
    print(f"\n[walk_forward] {len(splits)} folds | test set starts at index {test_start} ({n - test_start} bars)")

    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    fold_results   = []
    best_val_acc   = -1.0
    best_fold_idx  = -1
    last_scaler    = None

    for fold_idx, (train_end, val_end) in enumerate(splits, 1):
        print(f"\n{'─'*52}")
        print(f"  Fold {fold_idx}/{len(splits)}  |  train: 0→{train_end}  val: {train_end}→{val_end}")
        print(f"{'─'*52}")

        train_ds, val_ds, scaler = make_fold_datasets(
            features, targets, train_end, val_end, seq_len
        )
        print(f"  Sequences — Train: {len(train_ds):,}  Val: {len(val_ds):,}")

        fold_model_path = os.path.join(model_dir, f"spy_lstm_fold{fold_idx}.pt")
        model = SPYLSTMModel(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        history = train_model(
            model, train_ds, val_ds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            early_stopping_patience=early_stopping_patience,
            model_save_path=fold_model_path,
        )

        fold_best_val_acc = max(history["val_acc"])
        fold_results.append({
            "fold":           fold_idx,
            "train_size":     len(train_ds),
            "val_size":       len(val_ds),
            "best_val_acc":   fold_best_val_acc,
            "stopped_epoch":  len(history["train_loss"]),
            "model_path":     fold_model_path,
        })

        last_scaler = scaler
        if fold_best_val_acc > best_val_acc:
            best_val_acc  = fold_best_val_acc
            best_fold_idx = fold_idx

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print("  WALK-FORWARD SUMMARY")
    print(f"{'═'*52}")
    print(f"  {'Fold':<6} {'Train seqs':>11} {'Val seqs':>10} {'Val Acc':>9} {'Epochs':>7}")
    print(f"  {'─'*6} {'─'*11} {'─'*10} {'─'*9} {'─'*7}")
    for r in fold_results:
        print(
            f"  {r['fold']:<6} {r['train_size']:>11,} {r['val_size']:>10,} "
            f"{r['best_val_acc']:>9.4f} {r['stopped_epoch']:>7}"
        )
    accs = [r["best_val_acc"] for r in fold_results]
    print(f"  {'─'*6} {'─'*11} {'─'*10} {'─'*9} {'─'*7}")
    print(f"  {'Mean':<6} {'':>11} {'':>10} {np.mean(accs):>9.4f}")
    print(f"  {'Std':<6} {'':>11} {'':>10} {np.std(accs):>9.4f}")
    print(f"\n  Best fold: {best_fold_idx}  (val acc {best_val_acc:.4f})")
    print(f"{'═'*52}")

    # ── Test set evaluation with last fold's scaler ────────────────────────────
    print(f"\n[walk_forward] Evaluating best fold ({best_fold_idx}) on held-out test set...")
    test_ds = make_test_dataset(features, targets, test_start, seq_len, last_scaler)

    best_model = SPYLSTMModel(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    best_model_path = fold_results[best_fold_idx - 1]["model_path"]
    evaluate_model(best_model, test_ds, model_path=best_model_path, output_dir=output_dir)

    # ── Plot fold val accuracies ───────────────────────────────────────────────
    _plot_fold_accuracies(fold_results, output_dir)

    return fold_results


def _plot_fold_accuracies(fold_results: list[dict], output_dir: str) -> None:
    folds = [r["fold"] for r in fold_results]
    accs  = [r["best_val_acc"] for r in fold_results]
    mean_acc = np.mean(accs)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(folds, accs, color="steelblue", alpha=0.8, zorder=2)
    ax.axhline(mean_acc, linestyle="--", color="coral",  linewidth=1.5, label=f"Mean {mean_acc:.3f}", zorder=3)
    ax.axhline(0.5,      linestyle=":",  color="gray",   linewidth=1.2, label="Random baseline (0.50)", zorder=3)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Walk-Forward CV — Best Val Accuracy per Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation Accuracy")
    ax.set_xticks(folds)
    ax.set_ylim(0.40, max(accs) + 0.06)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=1)
    plt.tight_layout()

    path = os.path.join(output_dir, "walk_forward_accuracies.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[walk_forward] Fold accuracy chart saved → {path}")
