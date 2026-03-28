# signals/low_volatility.py
# Low Volatility signal — inverse 252-day realized volatility.
# Lower volatility = higher score (defensive signal).
# Stocks with stable prices tend to outperform on risk-adjusted basis,
# especially in bear/crisis regimes.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_prices

SIGNAL_NAME   = "low_volatility"
LOOKBACK_DAYS = 252  # 1 year of trading days


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes inverse 252-day realized volatility for all tickers.
    Lower vol = higher score.
    Requires at least 252 days of price history per ticker.
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

        if len(px) < LOOKBACK_DAYS + 1:
            continue

        # Use last 252 trading days of close prices
        closes = px["close"].iloc[-(LOOKBACK_DAYS + 1):].values
        returns = np.diff(closes) / closes[:-1]

        # Filter out any zero/nan returns
        returns = returns[np.isfinite(returns)]
        if len(returns) < LOOKBACK_DAYS * 0.8:  # require 80% coverage
            continue

        # Annualized volatility
        vol = float(np.std(returns, ddof=1)) * np.sqrt(252)

        if vol <= 0:
            continue

        # Inverse: lower vol = higher score
        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   -vol,  # inverted so low vol scores high
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df
