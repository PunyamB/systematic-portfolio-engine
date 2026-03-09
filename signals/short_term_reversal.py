# signals/short_term_reversal.py
# Short-Term Reversal signal.
# 5-day return, inverted — recent losers tend to bounce, recent winners tend to fade.
# Mean reversion signal: negative recent return = buy signal.
# Score is inverted 5-day return.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_prices

SIGNAL_NAME   = "short_term_reversal"
LOOKBACK_DAYS = 5


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes 5-day return reversal for all tickers as of as_of_date.
    Inverted: stocks that fell recently score higher.
    Requires at least 6 days of price history per ticker.
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

        price_now  = float(px["close"].iloc[-1])
        price_past = float(px["close"].iloc[-(LOOKBACK_DAYS + 1)])

        if price_past == 0:
            continue

        ret_5d = (price_now / price_past) - 1.0

        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   -ret_5d,  # inverted: recent losers score higher
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df