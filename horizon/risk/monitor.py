# risk/monitor.py
# Horizon V4 daily risk monitoring module.
# Handles: 6-tier V4 circuit breaker, trailing stops, drift detection,
# beta computation, and liquidity scoring.
# Runs EOD after NAV computation.

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from scipy import stats
from data.storage import load_portfolio, save_portfolio, load_prices
from fund_accounting.nav import load_nav_history, load_cash
from utils.config_loader import get_config
from utils.notifications import notify

cfg = get_config()
CB = cfg["circuit_breaker"]

# Horizon V4 6-tier thresholds
CB_T1 = CB["t1_pct"]  # 0.020
CB_T2 = CB["t2_pct"]  # 0.030
CB_T3 = CB["t3_pct"]  # 0.0685
CB_T4 = CB["t4_pct"]  # 0.145
CB_T5 = CB["t5_pct"]  # 0.215

# Trailing stop settings
STOP_MULTIPLIER   = cfg["stop_loss"]["vol_multiplier"]
STOP_VOL_LOOKBACK = cfg["stop_loss"]["vol_lookback"]
STOP_FLOOR        = cfg["stop_loss"]["floor"]
STOP_CAP          = cfg["stop_loss"]["cap"]
STOP_SEASONING    = cfg["stop_loss"].get("seasoning_days", 1)

# Drift thresholds
DRIFT_POSITION  = cfg["drift_rebalance"]["max_position_drift"]
DRIFT_PORTFOLIO = cfg["drift_rebalance"]["max_portfolio_drift"]
DRIFT_SECTOR    = cfg["drift_rebalance"]["max_sector_drift"]

# Drift detection anchor -- only updated by execute.py and stop auto-execution
EXECUTED_WEIGHTS_PATH = Path("data/processed/executed_weights.parquet")


# ------------------------------------------------------------
# V4 SIX-TIER CIRCUIT BREAKER
# ------------------------------------------------------------

def compute_cb_tier(drawdown: float) -> int:
    """
    Maps drawdown (positive float, e.g. 0.07 = 7%) to V4 tier 0-5.
    Mirrors Horizon spec exactly. Thresholds are inclusive lower bound.
    """
    if drawdown >= CB_T5:
        return 5
    elif drawdown >= CB_T4:
        return 4
    elif drawdown >= CB_T3:
        return 3
    elif drawdown >= CB_T2:
        return 2
    elif drawdown >= CB_T1:
        return 1
    else:
        return 0


def _tier_actions(tier: int) -> list:
    """Returns the operational action list for the given V4 tier."""
    if tier == 5:
        return [
            "Catastrophic regime",
            "Invested target 40%, max weight 2.5%, sum floor 0.40",
            "Daily rebalance, NO NEW POSITIONS",
        ]
    elif tier == 4:
        return [
            "Crisis entry",
            "Invested target 60%, max weight 2.5%, sum floor 0.40",
            "Daily rebalance, NO NEW POSITIONS",
        ]
    elif tier == 3:
        return [
            "Deep correction",
            "Invested target 75%, max weight 3.0%, sum floor 0.55",
            "Rebalance every 3 trading days",
        ]
    elif tier == 2:
        return [
            "Notable correction",
            "Invested target 95%, max weight 3.5%, sum floor 0.70",
            "Rebalance every 5 trading days",
        ]
    elif tier == 1:
        return [
            "Mild stress (info)",
            "Rebalance interval forced to 21 days",
        ]
    else:
        return []


def check_circuit_breaker(nav: float) -> dict:
    """
    Computes current drawdown from peak NAV and returns V4 CB state.
    Returns dict with tier (0-5), drawdown, peak_nav, actions.
    """
    nav_history = load_nav_history()
    if nav_history.empty or len(nav_history) < 2:
        return {"tier": 0, "drawdown": 0.0, "peak_nav": nav, "actions": []}

    peak_nav = float(nav_history["nav"].max())
    drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0
    tier     = compute_cb_tier(drawdown)
    actions  = _tier_actions(tier)

    result = {
        "tier":     tier,
        "drawdown": drawdown,
        "peak_nav": peak_nav,
        "actions":  actions,
    }

    if tier >= 1:
        print(f"[risk] CB T{tier} | drawdown {drawdown:.2%} from peak ${peak_nav:,.2f}")
        level = "critical" if tier >= 4 else ("warning" if tier >= 2 else "info")
        notify(
            f"Circuit breaker T{tier} active\n"
            f"Drawdown: {drawdown:.2%} from peak ${peak_nav:,.2f}\n"
            f"Actions: {' | '.join(actions)}",
            level=level
        )

    return result


# ------------------------------------------------------------
# TRAILING STOPS (unchanged from Meridian)
# ------------------------------------------------------------

def compute_trailing_stops(prices: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Computes vol-adjusted trailing stop price for each held position.
    Stop distance = max(FLOOR, min(CAP, MULTIPLIER * 25d EWM vol))
    Reference price ratchets up (highest close since entry).
    Stop price = reference_price * (1 - stop_distance)
    """
    if portfolio.empty or prices.empty:
        return portfolio

    portfolio = portfolio.copy()

    if "stop_reference_price" not in portfolio.columns:
        portfolio["stop_reference_price"] = np.nan
    if "stop_price" not in portfolio.columns:
        portfolio["stop_price"] = np.nan

    for idx, row in portfolio.iterrows():
        ticker = row["ticker"]
        ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")

        if len(ticker_prices) < STOP_VOL_LOOKBACK:
            continue

        returns = ticker_prices["close"].pct_change().dropna()
        vol_25d = float(
            returns.tail(STOP_VOL_LOOKBACK).ewm(span=STOP_VOL_LOOKBACK, adjust=False).std().iloc[-1]
        )
        stop_distance = max(STOP_FLOOR, min(STOP_CAP, STOP_MULTIPLIER * vol_25d))

        entry_date = row.get("entry_date", None)
        if pd.notna(entry_date):
            prices_since_entry = ticker_prices[
                ticker_prices["date"] >= pd.Timestamp(entry_date)
            ]
        else:
            prices_since_entry = ticker_prices

        true_high = float(prices_since_entry["close"].max()) \
                    if not prices_since_entry.empty \
                    else float(ticker_prices["close"].iloc[-1])

        stored_ref = row["stop_reference_price"]
        new_reference = max(true_high, float(stored_ref)) if pd.notna(stored_ref) else true_high

        portfolio.at[idx, "stop_reference_price"] = new_reference
        portfolio.at[idx, "stop_price"]           = new_reference * (1 - stop_distance)

    return portfolio


def check_trailing_stops(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates trailing stops EOD. Skips positions opened today (seasoning).
    Returns DataFrame of triggered tickers for P2 execution next open.
    """
    from datetime import date as date_type

    portfolio = load_portfolio()
    if portfolio.empty:
        return pd.DataFrame()

    portfolio = compute_trailing_stops(prices, portfolio)
    save_portfolio(portfolio)

    if "stop_price" not in portfolio.columns:
        return pd.DataFrame()

    today = date_type.today()
    if "entry_date" in portfolio.columns:
        portfolio_eval = portfolio[
            portfolio["entry_date"].apply(
                lambda x: (pd.Timestamp(x).date() < today) if pd.notna(x) else True
            )
        ]
    else:
        portfolio_eval = portfolio

    if portfolio_eval.empty:
        return pd.DataFrame()

    latest_prices = (
        prices.sort_values("date")
        .groupby("ticker").last()
        .reset_index()[["ticker", "close"]]
    )

    portfolio_eval = portfolio_eval.merge(
        latest_prices, on="ticker", how="left", suffixes=("", "_latest")
    )
    close_col = "close_latest" if "close_latest" in portfolio_eval.columns else "close"

    triggered = portfolio_eval[
        portfolio_eval[close_col] < portfolio_eval["stop_price"]
    ][["ticker", close_col, "stop_price"]].copy()

    if not triggered.empty:
        print(f"[risk] Trailing stops triggered: {triggered['ticker'].tolist()}")
        notify(
            f"Trailing stops triggered: {triggered['ticker'].tolist()}\n"
            f"Scheduled for next open execution",
            level="warning"
        )

    return triggered


# ------------------------------------------------------------
# DRIFT DETECTION (unchanged from Meridian)
# ------------------------------------------------------------

def check_drift(executed_weights: pd.DataFrame) -> dict:
    """
    Compares current portfolio weights against last executed weights.
    Flags positions, portfolio, and sectors that have drifted beyond thresholds.
    """
    portfolio = load_portfolio()
    nav_history = load_nav_history()

    if portfolio.empty or nav_history.empty or executed_weights.empty:
        return {"position_drift": [], "portfolio_drift": False, "sector_drift": []}

    nav = float(nav_history.iloc[-1]["nav"])
    portfolio["current_weight"] = portfolio["market_value"] / nav

    merged = portfolio.merge(executed_weights, on="ticker", how="outer").fillna(0)
    merged["drift"] = (merged["current_weight"] - merged["target_weight"]).abs()

    position_drift = merged[merged["drift"] > DRIFT_POSITION]["ticker"].tolist()
    total_drift     = float(merged["drift"].sum())
    portfolio_drift = total_drift > DRIFT_PORTFOLIO

    sector_drift = []
    if "sector" in merged.columns:
        sector_current = merged.groupby("sector")["current_weight"].sum()
        sector_target  = merged.groupby("sector")["target_weight"].sum()
        sector_diff    = (sector_current - sector_target).abs()
        sector_drift   = sector_diff[sector_diff > DRIFT_SECTOR].index.tolist()

    result = {
        "position_drift":  position_drift,
        "portfolio_drift": portfolio_drift,
        "sector_drift":    sector_drift,
    }

    if position_drift or portfolio_drift or sector_drift:
        print(f"[risk] Drift: positions={position_drift}, portfolio={portfolio_drift}, sectors={sector_drift}")

    return result


# ------------------------------------------------------------
# BETA (unchanged from Meridian)
# ------------------------------------------------------------

def compute_beta(prices: pd.DataFrame, spy_prices: pd.DataFrame, lookback: int = 252) -> float:
    portfolio = load_portfolio()
    if portfolio.empty or prices.empty or spy_prices.empty:
        return 1.0

    spy_returns = (
        spy_prices.sort_values("date")
        .tail(lookback)
        .set_index("date")["close"]
        .pct_change()
        .dropna()
    )

    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else 1.0

    weighted_beta = 0.0

    for _, row in portfolio.iterrows():
        ticker = row["ticker"]
        weight = float(row["market_value"]) / nav

        ticker_prices = (
            prices[prices["ticker"] == ticker]
            .sort_values("date")
            .tail(lookback)
            .set_index("date")["close"]
            .pct_change()
            .dropna()
        )

        aligned = pd.concat([ticker_prices, spy_returns], axis=1).dropna()
        if len(aligned) < 30:
            continue

        aligned.columns = ["stock", "spy"]
        slope, _, _, _, _ = stats.linregress(aligned["spy"], aligned["stock"])
        weighted_beta += weight * slope

    return weighted_beta


# ------------------------------------------------------------
# LIQUIDITY SCORING (unchanged from Meridian)
# ------------------------------------------------------------

def compute_liquidity_scores(prices: pd.DataFrame) -> pd.DataFrame:
    portfolio = load_portfolio()
    if portfolio.empty or prices.empty:
        return pd.DataFrame()

    adv = (
        prices.sort_values("date")
        .groupby("ticker")
        .tail(20)
        .groupby("ticker")
        .apply(lambda x: (x["close"] * x["volume"]).mean())
        .reset_index()
        .rename(columns={0: "adv_value"})
    )

    scored = portfolio.merge(adv, on="ticker", how="left")
    scored["liquidity_ratio"] = scored["market_value"] / scored["adv_value"]
    scored["liquidity_flag"]  = scored["liquidity_ratio"] > cfg["risk"]["liquidity_flag_pct"]

    flagged = scored[scored["liquidity_flag"]]["ticker"].tolist()
    if flagged:
        print(f"[risk] Liquidity flags: {flagged}")

    return scored[["ticker", "market_value", "adv_value", "liquidity_ratio", "liquidity_flag"]]


# ------------------------------------------------------------
# FULL DAILY RISK RUN
# ------------------------------------------------------------

def run_risk_monitor(run_date: date = None) -> dict:
    """
    Full EOD risk monitoring run.
    Returns dict with: nav, circuit_breaker (V4 tier 0-5), stop_exits,
    stop_triggers, drift, liquidity, beta, drawdown.
    """
    if run_date is None:
        run_date = date.today()

    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty \
          else cfg["portfolio"]["initial_capital"]

    print(f"[risk] Running EOD risk monitor for {run_date}")

    prices = load_prices()
    from data.storage import load_spy_prices
    spy_prices = load_spy_prices()

    circuit_breaker  = check_circuit_breaker(nav)
    stop_triggers_df = check_trailing_stops(prices)
    liquidity_scores = compute_liquidity_scores(prices)

    stop_exits = []
    if not stop_triggers_df.empty:
        stop_exits = stop_triggers_df["ticker"].tolist()

    drift_result = {"triggered": False, "position_drift": [], "portfolio_drift": False, "sector_drift": []}
    if EXECUTED_WEIGHTS_PATH.exists():
        executed_weights = pd.read_parquet(EXECUTED_WEIGHTS_PATH)
        drift_result = check_drift(executed_weights)
        drift_result["triggered"] = bool(
            drift_result["position_drift"]
            or drift_result["portfolio_drift"]
            or drift_result["sector_drift"]
        )

    beta = 1.0
    if not spy_prices.empty:
        beta = compute_beta(prices, spy_prices)

    drawdown = circuit_breaker.get("drawdown", 0.0)

    print(f"[risk] Beta: {beta:.3f} | CB tier: T{circuit_breaker['tier']} | "
          f"Stop exits: {len(stop_exits)} | Drift: {drift_result['triggered']}")

    return {
        "nav":             nav,
        "circuit_breaker": circuit_breaker,
        "stop_exits":      stop_exits,
        "stop_triggers":   stop_triggers_df,
        "drift":           drift_result,
        "liquidity":       liquidity_scores,
        "beta":            beta,
        "drawdown":        drawdown,
    }
