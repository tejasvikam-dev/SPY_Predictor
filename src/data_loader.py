"""
Downloads historical SPY OHLCV data — no API key required.

1h:  last 730 days (2 years), single request  → ~3,200 RTH bars
1d:  last 5 years,  single request            → ~1,255 bars
1m:  last 28 days,  batched in 7-day chunks   → ~7,800 bars  (kept for reference)
5m:  last 60 days,  single request            → ~3,100 bars  (kept for reference)
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def download_spy_1m(days_back: int = 28) -> pd.DataFrame:
    """
    Download SPY 1-minute OHLCV bars using yfinance, stitched from
    multiple 7-day windows.

    Yahoo Finance limits:
      - 1m bars only available for the last 30 days
      - Each request window must be ≤ 7 days

    Args:
        days_back: How many calendar days to look back. Max ~29.

    Returns:
        DataFrame indexed by UTC timestamp with lowercase columns:
        open, high, low, close, volume — regular trading hours only.
    """
    days_back = min(days_back, 29)   # hard cap at Yahoo's 30-day limit
    now = datetime.utcnow()
    chunks = []

    # Step backward in 7-day windows
    window = 7
    cursor = now
    while (now - cursor).days < days_back:
        chunk_end   = cursor
        chunk_start = cursor - timedelta(days=window)

        df_chunk = yf.download(
            "SPY",
            start=chunk_start.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval="1m",
            auto_adjust=True,
            progress=False,
        )

        if not df_chunk.empty:
            chunks.append(df_chunk)
            total = sum(len(c) for c in chunks)
            print(
                f"[data_loader] Chunk {chunk_start.date()} → {chunk_end.date()}: "
                f"{len(df_chunk)} bars  (total so far: {total})"
            )
        else:
            print(f"[data_loader] Chunk {chunk_start.date()} → {chunk_end.date()}: empty (skipped)")

        cursor = chunk_start
        time.sleep(0.4)   # avoid hammering Yahoo

    if not chunks:
        raise RuntimeError("yfinance returned no 1-minute data.")

    df = pd.concat(chunks)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Flatten MultiIndex columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.dropna()

    # Keep only regular trading hours (09:30–16:00 ET)
    df = _filter_rth(df)

    print(f"[data_loader] {len(df):,} 1-min RTH bars | {df.index[0]} → {df.index[-1]}")
    return df


def download_spy_1h() -> pd.DataFrame:
    """
    Download SPY 1-hour OHLCV bars for the last 730 days (2 years).
    yfinance supports 1h data up to 730 days in a single request.
    Gives ~3,200 regular-trading-hours bars covering multiple market regimes.
    """
    print("[data_loader] Downloading SPY 1h / 730d")
    df = yf.download("SPY", interval="1h", period="730d", auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError("yfinance returned no 1-hour data.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.dropna()
    df = _filter_rth(df)
    print(f"[data_loader] {len(df):,} 1h RTH bars | {df.index[0]} → {df.index[-1]}")
    return df


def download_spy_1d(period: str = "5y") -> pd.DataFrame:
    """
    Download SPY daily OHLCV bars (last 5 years by default).
    Used as the higher-timeframe context layer in multi-TF models.
    Index is timezone-naive dates; add_features_multitf handles alignment.
    """
    print(f"[data_loader] Downloading SPY 1d / {period}")
    df = yf.download("SPY", interval="1d", period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError("yfinance returned no daily data.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.dropna()
    print(f"[data_loader] {len(df):,} daily bars | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def download_spy_5m() -> pd.DataFrame:
    """
    Download SPY 5-minute OHLCV bars for the last 60 days.
    yfinance supports 5m data up to 60 days in a single request.
    """
    print("[data_loader] Downloading SPY 5m / 60d")
    df = yf.download("SPY", interval="5m", period="60d", auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError("yfinance returned no 5-minute data.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.dropna()
    df = _filter_rth(df)
    print(f"[data_loader] {len(df):,} 5-min RTH bars | {df.index[0]} → {df.index[-1]}")
    return df


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-trading-hours bars: 09:30–16:00 ET."""
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    et = idx.tz_convert("America/New_York")
    mask = (
        (et.hour > 9)  | ((et.hour == 9)  & (et.minute >= 30))
    ) & (
        (et.hour < 16) | ((et.hour == 16) & (et.minute == 0))
    )
    return df[mask]
