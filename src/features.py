"""
Technical indicator feature engineering — single and multi-timeframe.

Feature groups
──────────────
TECH_COLUMNS  (12)  — price/volume indicators, all expressed as ratios so the
                       model sees candle shape rather than absolute price level.
TOD_COLUMNS   (4)   — time-of-day signals, appended once (time is time regardless
                       of bar size, so they are not duplicated in multi-TF mode).

  tod_sin / tod_cos  — sine & cosine of the bar's position within the 390-minute
                       trading day (09:30–16:00 ET).  Cyclic encoding lets the
                       model learn smooth intraday patterns without a hard boundary
                       between e.g. 09:59 and 10:00.
  is_first_30m       — 1 during 09:30–09:59 ET (open volatility / gap-fill window)
  is_last_30m        — 1 during 15:30–16:00 ET (close auction / position-squaring)

Feature totals
──────────────
  Single-TF  :  16  (12 tech  +  4 TOD)
  Multi-TF   :  28  (12 1m tech  +  12 5m tech  +  4 TOD)

Target (both modes):
  1  if 1m close[t+1] > 1m close[t]
  0  otherwise
"""

import numpy as np
import pandas as pd

# ── Column name lists ─────────────────────────────────────────────────────────

TECH_COLUMNS = [
    "open_rel", "high_rel", "low_rel", "returns",
    "volume_rel", "rolling_mean_rel", "rolling_std_rel",
    "rsi", "macd_rel", "macd_signal_rel", "macd_hist_rel", "atr_rel",
]

TOD_COLUMNS = ["tod_sin", "tod_cos", "is_first_30m", "is_last_30m"]

FEATURE_COLUMNS     = TECH_COLUMNS + TOD_COLUMNS                                   # 16
FEATURE_COLUMNS_MTF = TECH_COLUMNS + [f"{c}_5m" for c in TECH_COLUMNS] + TOD_COLUMNS  # 28


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _tod_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute four time-of-day features from a UTC-aware DatetimeIndex.

    The 390-minute regular session (09:30–16:00 ET) is mapped to [0, 1]
    and encoded with sin/cos so adjacent periods (e.g. 09:59 and 10:00)
    are close in feature space.
    """
    if index.tz is None:
        index = index.tz_localize("UTC")
    et = index.tz_convert("America/New_York")

    # Minutes since open, clamped to [0, 390]
    minutes = np.clip((et.hour - 9) * 60 + et.minute - 30, 0, 390)
    phase   = 2 * np.pi * minutes / 390

    out = pd.DataFrame(index=index)
    out["tod_sin"]      = np.sin(phase)
    out["tod_cos"]      = np.cos(phase)
    out["is_first_30m"] = ((et.hour == 9) & (et.minute >= 30)).astype(float)
    out["is_last_30m"]  = (((et.hour == 15) & (et.minute >= 30)) |
                           ((et.hour == 16) & (et.minute == 0))).astype(float)
    return out


# ── Technical feature computation (no TOD, no target) ────────────────────────

def _compute_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 12 TECH_COLUMNS on an OHLCV DataFrame.
    Returns only those columns; NaN rows are dropped.
    """
    macd, sig, hist = _macd(df["close"])
    atr          = _atr(df)
    rolling_mean = df["close"].rolling(10).mean()
    rolling_std  = df["close"].rolling(10).std()
    volume_avg   = df["volume"].rolling(10).mean()

    out = pd.DataFrame(index=df.index)
    out["open_rel"]         = df["open"]  / df["close"] - 1
    out["high_rel"]         = df["high"]  / df["close"] - 1
    out["low_rel"]          = df["low"]   / df["close"] - 1
    out["returns"]          = df["close"].pct_change()
    out["volume_rel"]       = df["volume"] / volume_avg
    out["rolling_mean_rel"] = rolling_mean / df["close"] - 1
    out["rolling_std_rel"]  = rolling_std  / df["close"]
    out["rsi"]              = _rsi(df["close"])
    out["macd_rel"]         = macd / df["close"]
    out["macd_signal_rel"]  = sig  / df["close"]
    out["macd_hist_rel"]    = hist / df["close"]
    out["atr_rel"]          = atr  / df["close"]

    return out.dropna()


# ── Public API ────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single-timeframe: 12 tech + 4 TOD features + target (16 features total)."""
    feat = _compute_tech_features(df)

    tod  = _tod_features(feat.index)
    feat = feat.join(tod)

    close         = df["close"].reindex(feat.index)
    feat["target"] = (close.shift(-1) > close).astype(int)
    return feat.dropna()


def add_features_multitf(
    df_base: pd.DataFrame,
    df_context: pd.DataFrame,
    context_shift: str = "5min",
) -> pd.DataFrame:
    """
    Multi-timeframe: 12 base-TF tech + 12 context-TF tech + 4 TOD = 28 features.

    The context bar is shifted forward by `context_shift` before the merge so
    each base bar only sees data from fully completed context bars (no look-ahead).

      1h base + 1d context  →  context_shift="1D"
      1m base + 5m context  →  context_shift="5min"

    For daily context bars the index may be timezone-naive; it is coerced to UTC
    before the merge so timestamps align with intraday base bars.
    """
    feat_base    = _compute_tech_features(df_base)
    feat_context = _compute_tech_features(df_context)

    feat_context = feat_context.rename(columns={c: f"{c}_5m" for c in TECH_COLUMNS})

    # Coerce context index to UTC if it is timezone-naive (daily bars from yfinance)
    if feat_context.index.tz is None:
        feat_context.index = feat_context.index.tz_localize("UTC")

    # Shift context forward so bar[T] only becomes visible after it closes
    feat_context.index = feat_context.index + pd.Timedelta(context_shift)

    merged = pd.merge_asof(
        feat_base.sort_index(),
        feat_context.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    tod    = _tod_features(merged.index)
    merged = merged.join(tod)

    close_base     = df_base["close"].reindex(merged.index)
    merged["target"] = (close_base.shift(-1) > close_base).astype(int)

    merged = merged.dropna()
    print(
        f"[features] Multi-TF merge: {len(merged):,} bars | "
        f"{len(FEATURE_COLUMNS_MTF)} features (12×base + 12×context + 4×TOD)"
    )
    return merged
