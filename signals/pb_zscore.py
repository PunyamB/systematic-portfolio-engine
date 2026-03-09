# signals/pb_zscore.py
# P/B Z-Score signal.
# Price-to-Book computed from market cap / total stockholders equity.
# Low P/B relative to sector = undervalued = buy signal (score inverted).
# P/B not directly in key_metrics — computed from prices + financials.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_prices, load_financials, load_constituents

SIGNAL_NAME  = "pb_zscore"
MIN_QUARTERS = 4  # minimum quarters of financial history before using a ticker

SECTOR_MAP = {
    "Consumer Cyclical":  "Consumer Discretionary",
    "Basic Materials":    "Materials",
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
}


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes P/B Z-Score vs sector for all tickers as of as_of_date.
    P/B = (shares outstanding * close price) / total stockholders equity.
    Lower P/B relative to sector peers = higher score (inverted).
    Requires at least 4 quarters of financial history per ticker.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    prices       = load_prices()
    financials   = load_financials()
    constituents = load_constituents()

    if prices.empty or financials.empty or constituents.empty:
        print(f"[{SIGNAL_NAME}] Missing required data")
        return pd.DataFrame()

    cutoff = pd.Timestamp(as_of_date)

    # Filter financials to cutoff and apply minimum history filter
    fin_cutoff    = financials[financials["date"] <= cutoff]
    ticker_counts = fin_cutoff.groupby("ticker")["date"].count()
    eligible      = ticker_counts[ticker_counts >= MIN_QUARTERS].index
    fin_cutoff    = fin_cutoff[fin_cutoff["ticker"].isin(eligible)]

    # Latest close price per ticker
    latest_prices = (
        prices[prices["date"] <= cutoff]
        .sort_values("date")
        .groupby("ticker")
        .last()
        .reset_index()[["ticker", "close"]]
    )

    # Latest financials per ticker
    latest_fin = (
        fin_cutoff.sort_values("date")
        .groupby("ticker")
        .last()
        .reset_index()
    )

    required = ["ticker", "totalStockholdersEquity", "weightedAverageShsOut"]
    missing  = [c for c in required if c not in latest_fin.columns]
    if missing:
        print(f"[{SIGNAL_NAME}] Missing columns: {missing}")
        return pd.DataFrame()

    # Compute market cap and book value
    merged = latest_prices.merge(latest_fin[required], on="ticker", how="inner")
    merged = merged[
        (merged["totalStockholdersEquity"] > 0) &
        (merged["weightedAverageShsOut"] > 0)
    ]

    if merged.empty:
        print(f"[{SIGNAL_NAME}] No valid tickers after filtering")
        return pd.DataFrame()

    merged["market_cap"] = merged["close"] * merged["weightedAverageShsOut"]
    merged["pb_ratio"]   = merged["market_cap"] / merged["totalStockholdersEquity"]

    # Remove outliers — P/B > 50 likely data error
    merged = merged[merged["pb_ratio"] < 50]

    # Merge sector and apply mapping
    constituents["sector_mapped"] = constituents["sector"].replace(SECTOR_MAP)
    merged = merged.merge(constituents[["ticker", "sector_mapped"]], on="ticker", how="left")
    merged = merged.dropna(subset=["sector_mapped"])

    # Z-Score within sector, inverted
    results = []
    for sector, group in merged.groupby("sector_mapped"):
        if len(group) < 2:
            continue

        mean_pb = group["pb_ratio"].mean()
        std_pb  = group["pb_ratio"].std()

        if std_pb == 0:
            continue

        for _, row in group.iterrows():
            zscore = (row["pb_ratio"] - mean_pb) / std_pb
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