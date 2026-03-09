# signals/rsi_extremes.py
# RSI Extremes signal.
# RSI-14 computed from daily closes.
# Oversold (RSI < 30) = buy signal (high score).
# Overbought (RSI > 70) = sell signal (low score).
# Score is a continuous transformation of RSI centered around neutral 50.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_prices

SIGNAL_NAME  = "rsi_extremes"
RSI_PERIOD   = 14
MIN_PERIODS  = RSI_PERIOD + 1


def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    """
    Computes RSI for a price series.
    Returns float 0-100. Returns None if insufficient data.
    """
    if len(closes) < period + 1:
        return None

    deltas = closes.diff().dropna()
    gains  = deltas.clip(lower=0)
    losses = (-deltas).clip(lower=0)

    # Wilder smoothing (exponential)
    avg_gain = gains.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1/period, adjust=False).mean().iloc[-1]

    if avg_loss == 0:
        return 100.0

    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes RSI-14 extremes signal for all tickers as of as_of_date.
    Score = (50 - RSI) / 50 — ranges roughly -1 to +1.
    Oversold (low RSI) = positive score = buy.
    Overbought (high RSI) = negative score = sell.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    prices = load_prices()
    if prices.empty:
        print(f"[{SIGNAL_NAME}] No price data available")
        return pd.DataFrame()

    cutoff = pd.Timestamp(as_of_date)
    prices = prices[prices["date"] <= cutoff].sort_values(["ticker", "date"])

    results = []
    tickers = prices["ticker"].unique()

    for ticker in tickers:
        px = prices[prices["ticker"] == ticker].sort_values("date")

        if len(px) < MIN_PERIODS:
            continue

        rsi = _compute_rsi(px["close"], RSI_PERIOD)
        if rsi is None:
            continue

        # Transform: oversold = positive score, overbought = negative
        score = (50.0 - rsi) / 50.0

        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   score,
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df