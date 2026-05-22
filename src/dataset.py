"""
Sequence dataset preparation for the LSTM model.

Steps:
  1. Chronological train/val/test split (no shuffling — time series!)
  2. Fit StandardScaler on training features only to prevent look-ahead bias
  3. Transform val and test features with the same scaler
  4. Slide a window of length `seq_len` over each split to produce (X, y) pairs
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class SPYSequenceDataset(Dataset):
    """PyTorch Dataset wrapping (sequence, label) pairs."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _make_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int):
    """
    Slide a window of length `seq_len` over the feature array.

    For index i (starting from seq_len):
      X[i] = features[i-seq_len : i]   shape: (seq_len, n_features)
      y[i] = targets[i]                the label at the step we're predicting
    """
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def walk_forward_splits(n: int, n_folds: int = 5, test_ratio: float = 0.15, min_train_ratio: float = 0.40):
    """
    Generate (train_end, val_end) index pairs for expanding-window walk-forward CV.

    The final `test_ratio` of the data is reserved as a held-out test set and
    never included in any fold.  Within the remaining data the training window
    starts at `min_train_ratio` × n and expands by one val-window per fold.

    Returns:
        splits    — list of (train_end, val_end) tuples
        test_start — index where the held-out test set begins
    """
    test_start  = int(n * (1 - test_ratio))
    min_train   = int(n * min_train_ratio)
    available   = test_start - min_train           # rows available for folding
    val_size    = max(1, available // n_folds)

    splits = []
    for i in range(n_folds):
        train_end = min_train + i * val_size
        val_end   = min(train_end + val_size, test_start)
        if train_end >= test_start:
            break
        splits.append((train_end, val_end))

    return splits, test_start


def make_fold_datasets(
    features: np.ndarray,
    targets: np.ndarray,
    train_end: int,
    val_end: int,
    seq_len: int,
):
    """
    Scale and sequence one walk-forward fold.
    Scaler is fit on this fold's training slice only.

    Returns:
        train_ds, val_ds, scaler
    """
    scaler     = StandardScaler()
    train_feat = scaler.fit_transform(features[:train_end])
    val_feat   = scaler.transform(features[train_end:val_end])

    X_train, y_train = _make_sequences(train_feat, targets[:train_end],       seq_len)
    X_val,   y_val   = _make_sequences(val_feat,   targets[train_end:val_end], seq_len)

    return SPYSequenceDataset(X_train, y_train), SPYSequenceDataset(X_val, y_val), scaler


def make_test_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    test_start: int,
    seq_len: int,
    scaler: StandardScaler,
):
    """
    Scale and sequence the held-out test set using a pre-fitted scaler.
    Call this after all folds are done, passing the last fold's scaler.
    """
    test_feat = scaler.transform(features[test_start:])
    X_test, y_test = _make_sequences(test_feat, targets[test_start:], seq_len)
    return SPYSequenceDataset(X_test, y_test)


def prepare_data(
    df,
    feature_cols: list,
    seq_len: int = 30,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """
    Split, scale, and package data into three PyTorch Datasets.

    Returns:
        train_ds, val_ds, test_ds  — SPYSequenceDataset instances
        scaler                     — fitted StandardScaler (save for inference)
    """
    features = df[feature_cols].values
    targets = df["target"].values
    n = len(features)

    # Chronological boundaries
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Fit scaler on training data only to avoid look-ahead bias
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(features[:train_end])
    val_feat = scaler.transform(features[train_end:val_end])
    test_feat = scaler.transform(features[val_end:])

    train_tgt = targets[:train_end]
    val_tgt = targets[train_end:val_end]
    test_tgt = targets[val_end:]

    X_train, y_train = _make_sequences(train_feat, train_tgt, seq_len)
    X_val, y_val = _make_sequences(val_feat, val_tgt, seq_len)
    X_test, y_test = _make_sequences(test_feat, test_tgt, seq_len)

    print(
        f"[dataset] Sequences — Train: {len(X_train)}, "
        f"Val: {len(X_val)}, Test: {len(X_test)}"
    )

    return (
        SPYSequenceDataset(X_train, y_train),
        SPYSequenceDataset(X_val, y_val),
        SPYSequenceDataset(X_test, y_test),
        scaler,
    )
