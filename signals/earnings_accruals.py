# signals/earnings_accruals.py
# Earnings Accruals signal.
# Measures earnings quality: cash-based earnings vs accrual-based earnings.
# Low accruals = earnings backed by real cash flow = higher quality = buy signal.
# Formula: (net income - operating cash flow) / total assets
# Lower accrual ratio = better quality = score is inverted.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_financials

SIGNAL_NAME  = "earnings_accruals"
MIN_QUARTERS = 4  # need stable asset base for meaningful ratio


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes earnings accruals ratio for all tickers as of as_of_date.
    Accrual ratio = (net income - operating cash flow) / total assets.
    Lower ratio = higher earnings quality = higher score (inverted).
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    financials = load_financials()
    if financials.empty:
        print(f"[{SIGNAL_NAME}] No financials data available")
        return pd.DataFrame()

    required = ["netIncome", "operatingCashFlow", "totalAssets"]
    missing  = [c for c in required if c not in financials.columns]
    if missing:
        print(f"[{SIGNAL_NAME}] Missing columns: {missing}")
        return pd.DataFrame()

    cutoff     = pd.Timestamp(as_of_date)
    financials = financials[financials["date"] <= cutoff]
    financials = financials.sort_values(["ticker", "date"])

    # Filter to tickers with minimum history before taking latest row
    ticker_counts = financials.groupby("ticker")["date"].count()
    eligible      = ticker_counts[ticker_counts >= MIN_QUARTERS].index
    financials    = financials[financials["ticker"].isin(eligible)]

    # Use most recent quarter per ticker
    latest = (
        financials.groupby("ticker")
        .last()
        .reset_index()
    )

    latest = latest[latest["totalAssets"] > 0]

    if latest.empty:
        print(f"[{SIGNAL_NAME}] No valid tickers after filtering")
        return pd.DataFrame()

    latest["accrual_ratio"] = (
        (latest["netIncome"] - latest["operatingCashFlow"]) /
        latest["totalAssets"]
    )

    results = []
    for _, row in latest.iterrows():
        results.append({
            "ticker":      row["ticker"],
            "date":        as_of_date,
            "raw_score":   -row["accrual_ratio"],  # inverted: lower accruals = higher score
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df