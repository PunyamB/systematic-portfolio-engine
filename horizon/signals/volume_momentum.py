# signals/volume_momentum.py
# Volume Momentum signal — ratio of 5-day to 60-day average volume.
# Rising volume relative to norm = increasing interest = buy signal.
# Sentiment/attention signal: stocks seeing volume surges tend to continue.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_prices

SIGNAL_NAME       = "volume_momentum"
SHORT_WINDOW      = 5    # 5-day average volume
LONG_WINDOW       = 60   # 60-day average volume


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes volume momentum (5d avg volume / 60d avg volume) for all tickers.
    Higher ratio = recent volume surge = higher score.
    Requires at least 60 days of volume history per ticker.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    prices = load_prices()
    if prices.empty:
        print(f"[{SIGNAL_NAME}] No price data available")
        return pd.DataFrame()

    if "volume" not in prices.columns:
        print(f"[{SIGNAL_NAME}] volume column missing from prices")
        return pd.DataFrame()

    cutoff = pd.Timestamp(as_of_date)
    prices = prices[prices["date"] <= cutoff].sort_values(["ticker", "date"])

    results = []
    tickers = prices["ticker"].unique()

    for ticker in tickers:
        px = prices[prices["ticker"] == ticker].sort_values("date")

        if len(px) < LONG_WINDOW + 1:
            continue

        volumes = px["volume"].iloc[-(LONG_WINDOW):].values

        if np.any(np.isnan(volumes)) or np.any(volumes <= 0):
            # Filter valid volumes
            volumes = volumes[np.isfinite(volumes) & (volumes > 0)]
            if len(volumes) < LONG_WINDOW * 0.8:
                continue

        avg_short = float(np.mean(volumes[-SHORT_WINDOW:]))
        avg_long  = float(np.mean(volumes))

        if avg_long <= 0:
            continue

        vol_ratio = avg_short / avg_long

        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   vol_ratio,
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df
