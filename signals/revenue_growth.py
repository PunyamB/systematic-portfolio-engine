# signals/revenue_growth.py
# Revenue Growth signal — QoQ revenue acceleration.
# Measures whether revenue growth is speeding up or slowing down.
# Acceleration = current QoQ growth > prior QoQ growth = buy signal.
# Computed from FMP quarterly financials.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_financials

SIGNAL_NAME  = "revenue_growth"
MIN_QUARTERS = 3  # need 3 quarters to compute two QoQ growth figures


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes QoQ revenue growth acceleration for all tickers as of as_of_date.
    Requires at least 3 quarters of revenue history per ticker.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    financials = load_financials()
    if financials.empty:
        print(f"[{SIGNAL_NAME}] No financials data available")
        return pd.DataFrame()

    financials = financials.sort_values(["ticker", "date"])
    cutoff     = pd.Timestamp(as_of_date)
    financials = financials[financials["date"] <= cutoff]

    if "revenue" not in financials.columns:
        print(f"[{SIGNAL_NAME}] revenue column missing from financials")
        return pd.DataFrame()

    results = []
    tickers = financials["ticker"].unique()

    for ticker in tickers:
        fin = financials[financials["ticker"] == ticker].sort_values("date")

        if len(fin) < MIN_QUARTERS:
            continue

        rev = fin["revenue"].values

        rev_q0 = float(rev[-1])
        rev_q1 = float(rev[-2])
        rev_q2 = float(rev[-3])

        # Skip if any base quarter is zero or negative
        if rev_q1 <= 0 or rev_q2 <= 0:
            continue

        growth_recent = (rev_q0 - rev_q1) / abs(rev_q1)
        growth_prior  = (rev_q1 - rev_q2) / abs(rev_q2)

        # Acceleration = recent growth minus prior growth
        acceleration = growth_recent - growth_prior

        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   acceleration,
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df
