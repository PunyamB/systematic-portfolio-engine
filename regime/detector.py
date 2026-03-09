# regime/detector.py
# Two-layer, 4-state regime detection.
# L1 Market Stress (daily): low_stress, elevated, crisis
# L2 Economic Cycle (weekly): expansion, contraction
# Composite states: bull, recovery, bear, crisis
# All inputs read from storage — regime parquets must exist before running.

import pandas as pd
import numpy as np
from datetime import date, timedelta
from data.storage import load_regime_data, load_prices
from utils.config_loader import get_config
from utils.notifications import notify

cfg = get_config()
R = cfg["regime"]


# ------------------------------------------------------------
# BREADTH COMPUTATION
# ------------------------------------------------------------

def compute_breadth(as_of_date: date, lookback: int = 200) -> float:
    """
    Computes market breadth: % of S&P 500 stocks trading above their
    200-day moving average. Returns float between 0 and 1.
    Falls back to 0.5 (neutral) if insufficient price data.
    """
    prices = load_prices()
    if prices.empty:
        print("[regime] No price data for breadth — using neutral 0.5")
        return 0.5

    prices = prices.sort_values("date")
    cutoff = pd.Timestamp(as_of_date)

    # Need at least lookback days of history
    tickers = prices["ticker"].unique()
    above_ma = 0
    valid    = 0

    for ticker in tickers:
        ticker_px = prices[prices["ticker"] == ticker].sort_values("date")
        ticker_px = ticker_px[ticker_px["date"] <= cutoff]

        if len(ticker_px) < lookback:
            continue

        ma_200     = ticker_px["close"].tail(lookback).mean()
        last_close = ticker_px["close"].iloc[-1]

        valid += 1
        if last_close > ma_200:
            above_ma += 1

    if valid == 0:
        print("[regime] Insufficient tickers for breadth — using neutral 0.5")
        return 0.5

    breadth = above_ma / valid
    print(f"[regime] Breadth: {above_ma}/{valid} stocks above 200d MA = {breadth:.2%}")
    return breadth


# ------------------------------------------------------------
# L1 — MARKET STRESS (DAILY)
# ------------------------------------------------------------

def compute_market_stress(as_of_date: date) -> dict:
    """
    Classifies current market stress level using VIX and breadth.
    - low_stress:  VIX < 80% of 252d avg AND breadth > 60%
    - crisis:      VIX > 120% of 252d avg AND breadth < 40%
    - elevated:    everything else

    Returns dict with stress_state and component readings.
    """
    vix_df = load_regime_data("vix")

    result = {
        "stress_state": "elevated",
        "vix":          None,
        "vix_avg":      None,
        "vix_ratio":    None,
        "breadth":      None,
    }

    if vix_df.empty:
        print("[regime] No VIX data — defaulting to elevated")
        return result

    vix_df = vix_df.sort_values("date")
    vix_df = vix_df[vix_df["date"] <= pd.Timestamp(as_of_date)]

    if len(vix_df) < R["vix_lookback"]:
        print(f"[regime] Insufficient VIX history ({len(vix_df)} rows) — defaulting to elevated")
        return result

    latest_vix = float(vix_df["vix"].iloc[-1])
    avg_vix    = float(vix_df["vix"].tail(R["vix_lookback"]).mean())
    vix_ratio  = latest_vix / avg_vix if avg_vix > 0 else 1.0
    breadth    = compute_breadth(as_of_date)

    result["vix"]       = latest_vix
    result["vix_avg"]   = round(avg_vix, 2)
    result["vix_ratio"] = round(vix_ratio, 4)
    result["breadth"]   = round(breadth, 4)

    if vix_ratio < R["vix_elevated_threshold"] and breadth > R["breadth_high"]:
        result["stress_state"] = "low_stress"
    elif vix_ratio > R["vix_crisis_threshold"] and breadth < R["breadth_low"]:
        result["stress_state"] = "crisis"
    else:
        result["stress_state"] = "elevated"

    print(f"[regime] L1 stress: {result['stress_state']} | VIX={latest_vix:.1f} ratio={vix_ratio:.2f} breadth={breadth:.2%}")
    return result


# ------------------------------------------------------------
# L2 — ECONOMIC CYCLE (WEEKLY)
# ------------------------------------------------------------

def compute_economic_cycle(as_of_date: date) -> dict:
    """
    Classifies economic cycle using yield curve and credit spreads.
    - expansion:   yield curve not inverted AND credit spreads not widening
    - contraction: curve inverted AND credit spreads widening

    PMI not available from FRED fetcher — yield curve + credit spreads
    used as proxies per design spec.

    Returns dict with cycle_state and component readings.
    """
    yc_df = load_regime_data("yield_curve")
    cs_df = load_regime_data("credit_spreads")

    result = {
        "cycle_state":         "expansion",
        "yield_spread":        None,
        "curve_inverted":      False,
        "credit_spread_trend": "stable",
    }

    # Yield curve
    if not yc_df.empty:
        yc_df = yc_df.sort_values("date")
        yc_df = yc_df[yc_df["date"] <= pd.Timestamp(as_of_date)]

        if len(yc_df) >= 1 and "spread_10y2y" in yc_df.columns:
            latest_spread          = float(yc_df["spread_10y2y"].iloc[-1])
            result["yield_spread"] = round(latest_spread, 4)
            result["curve_inverted"] = latest_spread < 0

    # Credit spread trend — direction over last N days
    if not cs_df.empty:
        cs_df = cs_df.sort_values("date")
        cs_df = cs_df[cs_df["date"] <= pd.Timestamp(as_of_date)]
        lookback = R.get("economic_cycle_lookback", 5)

        if len(cs_df) >= lookback and "hy_spread" in cs_df.columns:
            recent = cs_df["hy_spread"].tail(lookback)
            trend  = float(recent.iloc[-1]) - float(recent.iloc[0])
            result["credit_spread_trend"] = "widening" if trend > 0 else "tightening"

    # Contraction: inverted curve AND spreads widening
    if result["curve_inverted"] and result["credit_spread_trend"] == "widening":
        result["cycle_state"] = "contraction"
    else:
        result["cycle_state"] = "expansion"

    print(f"[regime] L2 cycle: {result['cycle_state']} | spread={result['yield_spread']} inverted={result['curve_inverted']} credit={result['credit_spread_trend']}")
    return result


# ------------------------------------------------------------
# COMPOSITE REGIME
# ------------------------------------------------------------

def compute_composite_regime(stress_state: str, cycle_state: str) -> str:
    """
    Maps L1 + L2 to one of 4 composite states.

    crisis:   L1 = crisis (regardless of L2)
    bear:     L1 = elevated AND L2 = contraction
    recovery: L1 = elevated AND L2 = expansion
    bull:     L1 = low_stress AND L2 = expansion
    recovery: L1 = low_stress AND L2 = contraction (uncommon but handled)
    """
    if stress_state == "crisis":
        return "crisis"
    elif stress_state == "elevated" and cycle_state == "contraction":
        return "bear"
    elif stress_state == "low_stress" and cycle_state == "expansion":
        return "bull"
    else:
        return "recovery"


# ------------------------------------------------------------
# FULL REGIME DETECTION RUN
# ------------------------------------------------------------

def detect_regime(run_date: date = None) -> dict:
    """
    Full regime detection for the day.
    Returns dict with all regime readings and composite state.
    Called daily by pipeline runner — result used by signal combiner
    and optimizer for regime-conditional weights.
    """
    if run_date is None:
        run_date = date.today()

    print(f"[regime] Running regime detection for {run_date}")

    l1 = compute_market_stress(run_date)
    l2 = compute_economic_cycle(run_date)

    composite = compute_composite_regime(l1["stress_state"], l2["cycle_state"])

    result = {
        "date":          run_date,
        "composite":     composite,
        "stress_state":  l1["stress_state"],
        "cycle_state":   l2["cycle_state"],
        "vix":           l1["vix"],
        "vix_ratio":     l1["vix_ratio"],
        "breadth":       l1["breadth"],
        "yield_spread":  l2["yield_spread"],
        "curve_inverted": l2["curve_inverted"],
        "credit_trend":  l2["credit_spread_trend"],
    }

    print(f"[regime] Composite regime: {composite.upper()}")

    notify(
        f"Regime detection for {run_date}\n"
        f"Composite: {composite.upper()}\n"
        f"Stress: {l1['stress_state']} | Cycle: {l2['cycle_state']}\n"
        f"VIX: {l1['vix']} | Yield spread: {l2['yield_spread']}",
        level="info"
    )

    return result