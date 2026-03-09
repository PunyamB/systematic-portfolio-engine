# signals/gross_margin_trend.py
# Gross Margin Trend signal.
# Measures the slope of gross margin over the last 4 quarters.
# Rising gross margin = improving pricing power/efficiency = buy signal.
# Uses linear regression slope — positive slope scores higher.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_financials

SIGNAL_NAME  = "gross_margin_trend"
MIN_QUARTERS = 4


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes gross margin trend for all tickers as of as_of_date.
    Fits OLS slope over last 4 quarters of gross margin.
    Positive slope = improving margins = higher score.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    financials = load_financials()
    if financials.empty:
        print(f"[{SIGNAL_NAME}] No financials data available")
        return pd.DataFrame()

    if "grossMargin" not in financials.columns:
        print(f"[{SIGNAL_NAME}] grossMargin column missing from financials")
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

        margins = fin["grossMargin"].tail(MIN_QUARTERS).dropna().values

        if len(margins) < MIN_QUARTERS:
            continue

        # OLS slope over equally spaced quarters
        x     = np.arange(len(margins), dtype=float)
        slope = np.polyfit(x, margins, 1)[0]

        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   slope,
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df