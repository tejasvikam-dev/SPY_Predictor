"""
SPY Direction Predictor — LSTM with multi-timeframe features + walk-forward CV.

Data (no API key required):
  Base    : 1-hour bars,  730 days  (~3,200 RTH bars, 2 years of regimes)
  Context : daily bars,   5 years   (~1,255 bars, long-term trend context)
  Features: 28 total  (12×1h tech  +  12×1d tech  +  4×time-of-day)

Target: will the next 1-hour bar close higher than the current one?

Run:
    python3 main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import download_spy_1h, download_spy_1d
from src.features import add_features_multitf, FEATURE_COLUMNS_MTF
from src.walk_forward import run_walk_forward

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEQ_LEN                 = 30    # past bars per sequence (30h ≈ 4 trading days)
HIDDEN_SIZE             = 64
NUM_LAYERS              = 2
DROPOUT                 = 0.2
EPOCHS                  = 50
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE              = 64
LEARNING_RATE           = 1e-3
N_FOLDS                 = 5
MODEL_DIR               = "models"
OUTPUT_DIR              = "outputs"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 55)
    print("  SPY Direction Predictor — Multi-TF LSTM")
    print("  Base: 1h/730d   Context: 1d/5y")
    print("  Walk-Forward Cross-Validation")
    print("=" * 55)

    os.makedirs("data",     exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Download both timeframes
    print("\n[main] Downloading data...")
    df_1h = download_spy_1h()
    df_1d = download_spy_1d(period="5y")
    df_1h.to_csv("data/spy_1h_raw.csv")
    df_1d.to_csv("data/spy_1d_raw.csv")

    # 2. Merge 1h + 1d features; daily bar is shifted 1 day forward so only
    #    completed daily bars are visible to each hourly bar (no look-ahead).
    print("\n[main] Computing multi-timeframe features...")
    df = add_features_multitf(df_1h, df_1d, context_shift="1D")
    df.to_csv("data/spy_multitf_features.csv")
    target_dist = df["target"].value_counts().to_dict()
    print(f"[main] {len(df):,} bars | {len(FEATURE_COLUMNS_MTF)} features | target: {target_dist}")

    # 3. Walk-forward CV + test set evaluation
    run_walk_forward(
        df,
        feature_cols=FEATURE_COLUMNS_MTF,
        seq_len=SEQ_LEN,
        n_folds=N_FOLDS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR,
    )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()
