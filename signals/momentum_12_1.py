# signals/momentum_12_1.py
# 12-1 Month Price Momentum signal.
# Classic momentum: 12-month cumulative return excluding the most recent month.
# Skips the last month to avoid short-term reversal contamination.
# Higher score = stronger momentum = buy signal.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_prices

SIGNAL_NAME = "momentum_12_1"


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes 12-1 month momentum for all tickers as of as_of_date.
    Requires at least 252 trading days of price history per ticker.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    prices = load_prices()
    if prices.empty:
        print(f"[{SIGNAL_NAME}] No price data available")
        return pd.DataFrame()

    prices = prices.sort_values(["ticker", "date"])
    cutoff = pd.Timestamp(as_of_date)
    prices = prices[prices["date"] <= cutoff]

    results = []
    tickers = prices["ticker"].unique()

    for ticker in tickers:
        px = prices[prices["ticker"] == ticker].sort_values("date")

        # Need at least 252 days (12 months) + buffer
        if len(px) < 252:
            continue

        # 12-month return = price 252 days ago to price 21 days ago (skip last month)
        price_t0  = float(px["close"].iloc[-21])   # 1 month ago
        price_t12 = float(px["close"].iloc[-252])  # 12 months ago

        if price_t12 == 0:
            continue

        momentum = (price_t0 / price_t12) - 1.0

        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   momentum,
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df