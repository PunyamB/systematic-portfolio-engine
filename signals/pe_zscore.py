# signals/pe_zscore.py
# P/E Z-Score vs Sector signal.
# Measures how cheap/expensive a stock is relative to its sector peers.
# Low P/E relative to sector = undervalued = buy signal (score is inverted).
# Uses FMP sector names — remapped to GICS at load time.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_key_metrics, load_constituents

SIGNAL_NAME  = "pe_zscore"
MIN_QUARTERS = 4  # minimum quarters of history before using a ticker's valuation

# FMP -> GICS sector name mapping
SECTOR_MAP = {
    "Consumer Cyclical":  "Consumer Discretionary",
    "Basic Materials":    "Materials",
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
}


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes P/E Z-Score vs sector for all tickers as of as_of_date.
    Lower P/E relative to sector peers = higher score (inverted).
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

    metrics = metrics.sort_values(["ticker", "date"])
    cutoff  = pd.Timestamp(as_of_date)
    metrics = metrics[metrics["date"] <= cutoff]

    if "peRatio" not in metrics.columns:
        print(f"[{SIGNAL_NAME}] peRatio column missing from key metrics")
        return pd.DataFrame()

    # Filter to tickers with minimum history before taking latest row
    ticker_counts = metrics.groupby("ticker")["date"].count()
    eligible      = ticker_counts[ticker_counts >= MIN_QUARTERS].index
    metrics       = metrics[metrics["ticker"].isin(eligible)]

    # Get most recent P/E per ticker
    latest = (
        metrics.groupby("ticker")
        .last()
        .reset_index()[["ticker", "peRatio"]]
        .dropna(subset=["peRatio"])
    )

    # Remove negative P/E (loss-making companies — not meaningful for value)
    latest = latest[latest["peRatio"] > 0]

    # Merge sector info and apply FMP -> GICS mapping
    constituents["sector_mapped"] = constituents["sector"].replace(SECTOR_MAP)
    merged = latest.merge(constituents[["ticker", "sector_mapped"]], on="ticker", how="left")
    merged = merged.dropna(subset=["sector_mapped"])

    if merged.empty:
        print(f"[{SIGNAL_NAME}] No tickers with both P/E and sector data")
        return pd.DataFrame()

    # Compute Z-Score within each sector
    results = []
    for sector, group in merged.groupby("sector_mapped"):
        if len(group) < 2:
            continue

        mean_pe = group["peRatio"].mean()
        std_pe  = group["peRatio"].std()

        if std_pe == 0:
            continue

        for _, row in group.iterrows():
            zscore = (row["peRatio"] - mean_pe) / std_pe
            # Invert: low P/E = high score (value signal)
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