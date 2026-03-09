# signals/roe_stability.py
# ROE Stability signal.
# Measures consistency of Return on Equity over the last 8 quarters.
# Lower volatility of ROE = more stable earnings quality = buy signal.
# Score is inverted std — stable companies score higher.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_financials

SIGNAL_NAME  = "roe_stability"
MIN_QUARTERS = 8  # minimum quarters needed for meaningful std


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes ROE stability for all tickers as of as_of_date.
    Uses rolling std of quarterly ROE over last 8 quarters.
    Lower std = more stable = higher score (inverted).
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    financials = load_financials()
    if financials.empty:
        print(f"[{SIGNAL_NAME}] No financials data available")
        return pd.DataFrame()

    if "roe" not in financials.columns:
        print(f"[{SIGNAL_NAME}] roe column missing from financials")
        return pd.DataFrame()

    cutoff     = pd.Timestamp(as_of_date)
    financials = financials[financials["date"] <= cutoff]
    financials = financials.sort_values(["ticker", "date"])

    results = []
    tickers = financials["ticker"].unique()

    for ticker in tickers:
        fin = financials[financials["ticker"] == ticker].sort_values("date")

        if len(fin) < MIN_QUARTERS:
            continue

        roe_series = fin["roe"].tail(8).dropna()

        if len(roe_series) < MIN_QUARTERS:
            continue

        roe_std = float(roe_series.std())

        # Invert — lower std = higher score
        # Add small epsilon to avoid division by zero
        score = 1.0 / (roe_std + 1e-6)

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