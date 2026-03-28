# signals/fcf_yield.py
# FCF Yield signal — annualized free cash flow / market cap.
# Higher FCF yield = more cash generation per dollar of market value = buy signal.
# Uses trailing 4-quarter FCF sum for annualization.
# Market cap = latest close price * most recent weighted avg shares outstanding.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_prices, load_financials

SIGNAL_NAME  = "fcf_yield"
MIN_QUARTERS = 4  # need 4 quarters for trailing annual FCF


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes FCF yield (trailing 4Q FCF / market cap) for all tickers.
    Higher yield = higher score.
    Requires at least 4 quarters of FCF data and current price.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    prices     = load_prices()
    financials = load_financials()

    if prices.empty or financials.empty:
        print(f"[{SIGNAL_NAME}] Missing required data")
        return pd.DataFrame()

    if "freeCashFlow" not in financials.columns:
        print(f"[{SIGNAL_NAME}] freeCashFlow column missing from financials")
        return pd.DataFrame()

    if "weightedAverageShsOut" not in financials.columns:
        print(f"[{SIGNAL_NAME}] weightedAverageShsOut column missing from financials")
        return pd.DataFrame()

    financials = financials.sort_values(["ticker", "date"])
    cutoff     = pd.Timestamp(as_of_date)
    financials = financials[financials["date"] <= cutoff]

    prices = prices.sort_values(["ticker", "date"])
    prices = prices[prices["date"] <= cutoff]

    results = []
    tickers = financials["ticker"].unique()

    for ticker in tickers:
        fin = financials[financials["ticker"] == ticker].sort_values("date")

        if len(fin) < MIN_QUARTERS:
            continue

        # Trailing 4-quarter FCF sum
        fcf_4q = fin["freeCashFlow"].iloc[-4:].values
        if np.any(np.isnan(fcf_4q)):
            continue
        annual_fcf = float(np.sum(fcf_4q))

        # Most recent shares outstanding
        shares = fin["weightedAverageShsOut"].dropna()
        if shares.empty:
            continue
        shares_out = float(shares.iloc[-1])
        if shares_out <= 0:
            continue

        # Latest close price
        px = prices[prices["ticker"] == ticker]
        if px.empty:
            continue
        latest_price = float(px["close"].iloc[-1])
        if latest_price <= 0:
            continue

        market_cap = latest_price * shares_out
        if market_cap <= 0:
            continue

        fcf_yield = annual_fcf / market_cap

        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   fcf_yield,
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.4f} std={df['raw_score'].std():.4f}")
    return df
