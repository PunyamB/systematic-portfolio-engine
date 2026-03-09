# signals/ev_ebitda_zscore.py
# EV/EBITDA Z-Score signal.
# Lower EV/EBITDA relative to sector = cheaper = buy signal (inverted).
# EV/EBITDA from key_metrics parquet — already computed by FMP.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_key_metrics, load_constituents

SIGNAL_NAME  = "ev_ebitda_zscore"
MIN_QUARTERS = 4  # minimum quarters of history before using a ticker's valuation

SECTOR_MAP = {
    "Consumer Cyclical":  "Consumer Discretionary",
    "Basic Materials":    "Materials",
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
}


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes EV/EBITDA Z-Score vs sector for all tickers as of as_of_date.
    Lower EV/EBITDA relative to sector peers = higher score (inverted).
    Requires at least 4 quarters of key metrics history per ticker.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    metrics      = load_key_metrics()
    constituents = load_constituents()

    if metrics.empty:
        print(f"[{SIGNAL_NAME}] No key metrics data available")
        return pd.DataFrame()

    if constituents.empty:
        print(f"[{SIGNAL_NAME}] No constituents data available")
        return pd.DataFrame()

    cutoff  = pd.Timestamp(as_of_date)
    metrics = metrics[metrics["date"] <= cutoff]

    if "evToEbitda" not in metrics.columns:
        print(f"[{SIGNAL_NAME}] evToEbitda column missing")
        return pd.DataFrame()

    # Filter to tickers with minimum history before taking latest row
    ticker_counts = metrics.groupby("ticker")["date"].count()
    eligible      = ticker_counts[ticker_counts >= MIN_QUARTERS].index
    metrics       = metrics[metrics["ticker"].isin(eligible)]

    # Latest EV/EBITDA per ticker
    latest = (
        metrics.groupby("ticker")
        .last()
        .reset_index()[["ticker", "evToEbitda"]]
        .dropna(subset=["evToEbitda"])
    )

    # Remove negatives and outliers
    latest = latest[(latest["evToEbitda"] > 0) & (latest["evToEbitda"] < 200)]

    # Merge sector and apply mapping
    constituents["sector_mapped"] = constituents["sector"].replace(SECTOR_MAP)
    merged = latest.merge(constituents[["ticker", "sector_mapped"]], on="ticker", how="left")
    merged = merged.dropna(subset=["sector_mapped"])

    if merged.empty:
        print(f"[{SIGNAL_NAME}] No tickers with both EV/EBITDA and sector data")
        return pd.DataFrame()

    # Z-Score within sector, inverted
    results = []
    for sector, group in merged.groupby("sector_mapped"):
        if len(group) < 2:
            continue

        mean_ev = group["evToEbitda"].mean()
        std_ev  = group["evToEbitda"].std()

        if std_ev == 0:
            continue

        for _, row in group.iterrows():
            zscore = (row["evToEbitda"] - mean_ev) / std_ev
            results.append({
                "ticker":      row["ticker"],
                "date":        as_of_date,
                "raw_score":   -zscore,
                "signal_name": SIGNAL_NAME
            })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df