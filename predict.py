"""
predict.py — Live SPY direction prediction for the next 1-hour bar.

Downloads the latest 1h + 1d data, computes the same 28 multi-timeframe
features used during training, and runs the best saved model checkpoint.

Usage:
    python3 predict.py
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import download_spy_1h, download_spy_1d
from src.features import add_features_multitf, FEATURE_COLUMNS_MTF
from src.model import SPYLSTMModel

# ── Match training config exactly ─────────────────────────────────────────────
SEQ_LEN     = 30
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2

# Best fold from the last walk-forward run (fold 4 & 5 tied at 55.8% val acc)
MODEL_PATH  = "models/spy_lstm_fold4.pt"
# ─────────────────────────────────────────────────────────────────────────────


def predict():
    # 1. Download latest data
    print("[predict] Downloading latest SPY data...")
    df_1h = download_spy_1h()
    df_1d = download_spy_1d(period="5y")

    # 2. Compute 28 multi-timeframe features
    print("[predict] Computing features...")
    df = add_features_multitf(df_1h, df_1d, context_shift="1D")

    if len(df) < SEQ_LEN + 1:
        raise RuntimeError(f"Not enough bars after feature computation: need {SEQ_LEN + 1}, got {len(df)}")

    features = df[FEATURE_COLUMNS_MTF].values   # shape: (N, 28)

    # 3. Scale using all historical bars except the most recent one
    #    (mirrors training: scaler never sees the bar being predicted)
    scaler = StandardScaler()
    scaler.fit(features[:-1])
    scaled = scaler.transform(features)

    # 4. Build input sequence: last SEQ_LEN bars → predict the next bar
    sequence = scaled[-SEQ_LEN:]                                     # (30, 28)
    X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)    # (1, 30, 28)

    # 5. Load best model and run inference
    model = SPYLSTMModel(
        input_size=len(FEATURE_COLUMNS_MTF),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run main.py first.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()

    direction   = "UP   ▲" if prob >= 0.5 else "DOWN ▼"
    edge        = abs(prob - 0.5) * 100          # pp away from 50/50
    confidence  = "High" if edge > 10 else "Moderate" if edge > 5 else "Low"

    last_ts     = df.index[-1]
    last_close  = df_1h["close"].reindex(df.index).iloc[-1]

    print(f"\n{'═'*48}")
    print(f"  SPY — Next 1-Hour Bar Prediction")
    print(f"{'═'*48}")
    print(f"  Last bar (UTC) : {last_ts}")
    print(f"  Last close     : ${last_close:.2f}")
    print(f"  Prediction     : {direction}")
    print(f"  P(UP)          : {prob:.1%}")
    print(f"  Edge vs random : {edge:+.1f} pp  ({confidence} confidence)")
    print(f"{'═'*48}")
    print("  ⚠  Not financial advice.")
    print(f"{'═'*48}\n")

    return {"direction": "UP" if prob >= 0.5 else "DOWN", "probability": prob}


if __name__ == "__main__":
    predict()
