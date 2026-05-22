# SPY Movement Predictor

> **This project is for educational and research purposes only. It is not financial advice. Past model performance does not predict future returns. Do not use this to make real trading decisions.**

---

## What the model does

Trains a two-layer LSTM to predict whether SPY (S&P 500 ETF) will close **higher or lower** on the next 1-hour candle — binary classification using multi-timeframe features and walk-forward cross-validation.

- `1` → price goes up next bar
- `0` → price goes down (or stays flat)

---

## Data sources (no API key required)

| Timeframe | Source | History | Bars |
|---|---|---|---|
| 1-hour (base) | yfinance | 730 days | ~3,200 RTH bars |
| Daily (context) | yfinance | 5 years | ~1,255 bars |

---

## Features (28 total)

| Group | Count | Description |
|---|---|---|
| 1h tech indicators | 12 | open/high/low/close ratios, returns, volume_rel, rolling mean/std, RSI, MACD, ATR — all relative (no absolute prices) |
| 1d tech indicators | 12 | Same 12 indicators computed on daily bars for long-term context |
| Time-of-day | 4 | `tod_sin`, `tod_cos` (cyclic encoding), `is_first_30m`, `is_last_30m` |

All price features are ratios/pct-changes — the model sees candle **shape**, not absolute price levels.

---

## Model architecture

```
Input (batch, 30, 28)
  → LSTM(hidden=64, layers=2, dropout=0.2)
  → last hidden state (batch, 64)
  → Dropout(0.2)
  → Linear(64 → 1)
  → raw logit  [BCEWithLogitsLoss during training]
  → sigmoid    [probability at inference]
```

---

## Validation

**Expanding-window walk-forward CV** (5 folds, 15% held-out test set):

```
|── fold 1 train (40%) ──|─ val ─|                    | test |
|──── fold 2 train ────────────|─ val ─|               | test |
...
```

Each fold trains a fresh model with its own scaler fit only on that fold's training slice — no look-ahead bias.

**Baseline results:** ~53.6% mean val accuracy, 52.3% test accuracy, balanced predictions.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to run

```bash
python3 main.py
```

This will:
1. Download SPY 1h (730d) and 1d (5y) data via yfinance
2. Compute 28 multi-timeframe features
3. Run 5-fold walk-forward CV
4. Evaluate the best fold's model on the held-out test set
5. Save plots to `outputs/`

### Output files

| File | Description |
|---|---|
| `outputs/walk_forward_accuracies.png` | Val accuracy per fold |
| `outputs/training_history.png` | Loss + val acc curves |
| `outputs/confusion_matrix.png` | Test set confusion matrix |
| `outputs/predicted_vs_actual.png` | Prediction chart |
| `models/spy_lstm_fold{N}.pt` | Per-fold model checkpoints |

---

## How to interpret results

| Metric | What it means |
|---|---|
| **Accuracy** | Overall % correct directional calls. Random ≈ 50%. |
| **Precision** | Of all "Up" predictions, how many were actually up. |
| **Recall** | Of all actual up moves, how many did the model catch. |
| **Confusion matrix** | Should be roughly balanced — all-Up predictions mean the model found no signal. |

---

## Project structure

```
SPY_Predictor/
├── src/
│   ├── data_loader.py   yfinance downloads (1h, 1d, 1m, 5m)
│   ├── features.py      RSI, MACD, ATR, time-of-day, multi-TF merge
│   ├── dataset.py       Sequences, walk-forward splits, scaling
│   ├── model.py         PyTorch LSTM
│   ├── train.py         Training loop, early stopping, checkpointing
│   ├── evaluate.py      Metrics and plots
│   └── walk_forward.py  Expanding-window CV orchestration
├── main.py              Pipeline entry point
├── requirements.txt
└── README.md
```

---

**Not financial advice.** This is a machine learning experiment on public market data.
