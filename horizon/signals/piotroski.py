# signals/piotroski.py
# Piotroski F-Score signal.
# 9 binary criteria across profitability, leverage, and efficiency.
# Score ranges 0-9. Higher = stronger financial health = buy signal.
# All criteria computed from quarterly financials.

import pandas as pd
import numpy as np
from datetime import date
from data.storage import load_financials

SIGNAL_NAME  = "piotroski"
MIN_QUARTERS = 8  # need sufficient history for YoY comparisons across all 9 criteria


def _compute_fscore(curr: pd.Series, prior: pd.Series) -> int:
    """
    Computes Piotroski F-Score from current and prior quarter financials.
    Returns integer 0-9.
    """
    score = 0

    # ----------------------------------------------------------
    # PROFITABILITY (4 criteria)
    # ----------------------------------------------------------

    # F1: ROA positive (net income / total assets)
    if curr.get("totalAssets", 0) > 0:
        roa = curr.get("netIncome", 0) / curr["totalAssets"]
        if roa > 0:
            score += 1

    # F2: Operating cash flow positive
    if curr.get("operatingCashFlow", 0) > 0:
        score += 1

    # F3: ROA improving (current vs prior)
    if curr.get("totalAssets", 0) > 0 and prior.get("totalAssets", 0) > 0:
        roa_curr  = curr.get("netIncome", 0) / curr["totalAssets"]
        roa_prior = prior.get("netIncome", 0) / prior["totalAssets"]
        if roa_curr > roa_prior:
            score += 1

    # F4: Accruals — operating cash flow > net income (earnings quality)
    if curr.get("operatingCashFlow", 0) > curr.get("netIncome", 0):
        score += 1

    # ----------------------------------------------------------
    # LEVERAGE / LIQUIDITY (3 criteria)
    # ----------------------------------------------------------

    # F5: Long-term debt ratio decreasing
    if curr.get("totalAssets", 0) > 0 and prior.get("totalAssets", 0) > 0:
        lev_curr  = curr.get("longTermDebt", 0) / curr["totalAssets"]
        lev_prior = prior.get("longTermDebt", 0) / prior["totalAssets"]
        if lev_curr < lev_prior:
            score += 1

    # F6: Current ratio improving
    if (curr.get("totalCurrentLiabilities", 0) > 0 and
            prior.get("totalCurrentLiabilities", 0) > 0):
        cr_curr  = curr.get("totalCurrentAssets", 0) / curr["totalCurrentLiabilities"]
        cr_prior = prior.get("totalCurrentAssets", 0) / prior["totalCurrentLiabilities"]
        if cr_curr > cr_prior:
            score += 1

    # F7: No new shares issued (dilution check)
    shares_curr  = curr.get("weightedAverageShsOut", 0)
    shares_prior = prior.get("weightedAverageShsOut", 0)
    if shares_prior > 0 and shares_curr <= shares_prior * 1.02:
        score += 1

    # ----------------------------------------------------------
    # EFFICIENCY (2 criteria)
    # ----------------------------------------------------------

    # F8: Gross margin improving
    gm_curr  = curr.get("grossMargin", None)
    gm_prior = prior.get("grossMargin", None)
    if gm_curr is not None and gm_prior is not None:
        if gm_curr > gm_prior:
            score += 1

    # F9: Asset turnover improving (revenue / total assets)
    if (curr.get("totalAssets", 0) > 0 and
            prior.get("totalAssets", 0) > 0):
        at_curr  = curr.get("revenue", 0) / curr["totalAssets"]
        at_prior = prior.get("revenue", 0) / prior["totalAssets"]
        if at_curr > at_prior:
            score += 1

    return score


def compute(as_of_date: date = None) -> pd.DataFrame:
    """
    Computes Piotroski F-Score for all tickers as of as_of_date.
    Uses most recent two quarters of financials.
    Requires at least 8 quarters of history for reliable YoY comparisons.
    Returns DataFrame with columns: ticker, date, raw_score, signal_name.
    """
    if as_of_date is None:
        as_of_date = date.today()

    financials = load_financials()
    if financials.empty:
        print(f"[{SIGNAL_NAME}] No financials data available")
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

        curr  = fin.iloc[-1].to_dict()
        prior = fin.iloc[-2].to_dict()

        fscore = _compute_fscore(curr, prior)

        results.append({
            "ticker":      ticker,
            "date":        as_of_date,
            "raw_score":   float(fscore),
            "signal_name": SIGNAL_NAME
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"[{SIGNAL_NAME}] No valid scores computed")
        return df

    print(f"[{SIGNAL_NAME}] Computed {len(df)} scores | mean={df['raw_score'].mean():.2f} std={df['raw_score'].std():.2f}")
    return df