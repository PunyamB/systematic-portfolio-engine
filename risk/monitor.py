# risk/monitor.py
# Daily risk monitoring module.
# Handles: circuit breakers, trailing stops, drift detection,
# beta/tracking error computation, and liquidity scoring.
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

# Circuit breaker thresholds
CB_T1 = cfg["circuit_breaker"]["t1_pct"]  # 0.05
CB_T2 = cfg["circuit_breaker"]["t2_pct"]  # 0.10
CB_T3 = cfg["circuit_breaker"]["t3_pct"]  # 0.15
CB_T4 = cfg["circuit_breaker"]["t4_pct"]  # 0.20

# Trailing stop settings
STOP_MULTIPLIER  = cfg["stop_loss"]["vol_multiplier"]    # 2.0
STOP_VOL_LOOKBACK = cfg["stop_loss"]["vol_lookback"] # 25
STOP_FLOOR       = cfg["stop_loss"]["floor"]         # 0.05
STOP_CAP         = cfg["stop_loss"]["cap"]           # 0.20

# Drift thresholds
DRIFT_POSITION  = cfg["drift_rebalance"]["max_position_drift"]  # 0.03
DRIFT_PORTFOLIO = cfg["drift_rebalance"]["max_portfolio_drift"] # 0.05
DRIFT_SECTOR    = cfg["drift_rebalance"]["max_sector_drift"]    # 0.05


# ------------------------------------------------------------
# CIRCUIT BREAKER
# ------------------------------------------------------------

def check_circuit_breaker(nav: float) -> dict:
    """
    Computes current drawdown from peak NAV.
    Returns circuit breaker tier (0-4) and required actions.
    T1 (5%)  — info alert only
    T2 (10%) — max position 3.5%, rebalance weekly, no new low-liquidity positions
    T3 (15%) — gross exposure 70%, positions capped 2.5%, rebalance daily, no new positions 3 days
    T4 (20%) — reduce to 40% invested, trading paused 5 days, manual review required
    """
    nav_history = load_nav_history()
    if nav_history.empty or len(nav_history) < 2:
        return {"tier": 0, "drawdown": 0.0, "peak_nav": nav, "actions": []}

    peak_nav = float(nav_history["nav"].max())
    drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0

    if drawdown >= CB_T4:
        tier = 4
        actions = [
            "Reduce to 40% invested",
            "Trading paused 5 days",
            "Manual review required"
        ]
    elif drawdown >= CB_T3:
        tier = 3
        actions = [
            "Gross exposure capped at 70%",
            "Max position size 2.5%",
            "Rebalance frequency daily",
            "No new positions for 3 days"
        ]
    elif drawdown >= CB_T2:
        tier = 2
        actions = [
            "Max position size 3.5%",
            "Rebalance frequency weekly",
            "No new low-liquidity positions"
        ]
    elif drawdown >= CB_T1:
        tier = 1
        actions = ["Info alert only"]
    else:
        tier = 0
        actions = []

    result = {
        "tier":     tier,
        "drawdown": drawdown,
        "peak_nav": peak_nav,
        "actions":  actions
    }

    if tier >= 1:
        print(f"[risk] Circuit breaker T{tier} triggered: drawdown {drawdown:.2%}")
        notify(
            f"Circuit breaker T{tier} triggered\n"
            f"Drawdown: {drawdown:.2%} from peak ${peak_nav:,.2f}\n"
            f"Actions: {', '.join(actions)}",
            level="critical" if tier >= 3 else "warning"
        )

    return result


# ------------------------------------------------------------
# TRAILING STOPS
# ------------------------------------------------------------

def compute_trailing_stops(prices: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Computes vol-adjusted trailing stop price for each held position.
    Stop distance = max(FLOOR, min(CAP, MULTIPLIER * 25d rolling vol))
    Stop price    = recent_high * (1 - stop_distance)
    Updates portfolio with stop_price and stop_reference_price columns.
    """
    if portfolio.empty or prices.empty:
        return portfolio

    portfolio = portfolio.copy()

    if "stop_reference_price" not in portfolio.columns:
        portfolio["stop_reference_price"] = None
    if "stop_price" not in portfolio.columns:
        portfolio["stop_price"] = None

    for idx, row in portfolio.iterrows():
        ticker = row["ticker"]
        ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")

        if len(ticker_prices) < STOP_VOL_LOOKBACK:
            continue

        # Compute 25-day rolling vol (annualized daily returns std)
        returns = ticker_prices["close"].pct_change().dropna()
        vol_25d = float(returns.tail(STOP_VOL_LOOKBACK).std())

        # Stop distance: multiplier * vol, floored and capped
        stop_distance = STOP_MULTIPLIER * vol_25d
        stop_distance = max(STOP_FLOOR, min(STOP_CAP, stop_distance))

        # Recent high — use entry price if no reference stored
        recent_high = float(ticker_prices["close"].tail(STOP_VOL_LOOKBACK).max())
        if pd.notna(row["stop_reference_price"]):
            recent_high = max(recent_high, float(row["stop_reference_price"]))

        stop_price = recent_high * (1 - stop_distance)

        portfolio.at[idx, "stop_reference_price"] = recent_high
        portfolio.at[idx, "stop_price"]           = stop_price

    return portfolio


def check_trailing_stops(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates trailing stops EOD against current close prices.
    Returns DataFrame of tickers that have breached their stop price.
    These are P2 priority trades — executed next market open.
    """
    portfolio = load_portfolio()
    if portfolio.empty:
        return pd.DataFrame()

    portfolio = compute_trailing_stops(prices, portfolio)
    save_portfolio(portfolio)

    if "stop_price" not in portfolio.columns:
        return pd.DataFrame()

    # Get latest close for each held ticker
    latest_prices = (
        prices.sort_values("date")
        .groupby("ticker")
        .last()
        .reset_index()[["ticker", "close"]]
    )

    portfolio = portfolio.merge(latest_prices, on="ticker", how="left", suffixes=("", "_latest"))
    close_col = "close_latest" if "close_latest" in portfolio.columns else "close"

    triggered = portfolio[
        portfolio[close_col] < portfolio["stop_price"]
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
# DRIFT DETECTION
# ------------------------------------------------------------

def check_drift(target_weights: pd.DataFrame) -> dict:
    """
    Compares current portfolio weights against target weights.
    Flags positions, portfolio, and sectors that have drifted beyond thresholds.
    target_weights: DataFrame with columns ticker, target_weight, sector
    Returns dict with position_drift, portfolio_drift, sector_drift flags.
    """
    portfolio = load_portfolio()
    nav_history = load_nav_history()

    if portfolio.empty or nav_history.empty or target_weights.empty:
        return {"position_drift": [], "portfolio_drift": False, "sector_drift": []}

    nav = float(nav_history.iloc[-1]["nav"])
    portfolio["current_weight"] = portfolio["market_value"] / nav

    merged = portfolio.merge(target_weights, on="ticker", how="outer").fillna(0)
    merged["drift"] = (merged["current_weight"] - merged["target_weight"]).abs()

    # Position-level drift
    position_drift = merged[merged["drift"] > DRIFT_POSITION]["ticker"].tolist()

    # Portfolio-level drift (sum of absolute drifts)
    total_drift = float(merged["drift"].sum())
    portfolio_drift = total_drift > DRIFT_PORTFOLIO

    # Sector-level drift
    sector_drift = []
    if "sector" in merged.columns:
        sector_current = merged.groupby("sector")["current_weight"].sum()
        sector_target  = merged.groupby("sector")["target_weight"].sum()
        sector_diff    = (sector_current - sector_target).abs()
        sector_drift   = sector_diff[sector_diff > DRIFT_SECTOR].index.tolist()

    result = {
        "position_drift":  position_drift,
        "portfolio_drift": portfolio_drift,
        "sector_drift":    sector_drift
    }

    if position_drift or portfolio_drift or sector_drift:
        print(f"[risk] Drift detected: positions={position_drift}, portfolio={portfolio_drift}, sectors={sector_drift}")

    return result


# ------------------------------------------------------------
# BETA AND TRACKING ERROR
# ------------------------------------------------------------

def compute_beta(prices: pd.DataFrame, spy_prices: pd.DataFrame, lookback: int = 252) -> float:
    """
    Computes portfolio beta vs SPY using OLS regression.
    Uses weighted average of position betas.
    """
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


def compute_tracking_error(nav_history: pd.DataFrame, spy_prices: pd.DataFrame, lookback: int = 63) -> float:
    """
    Computes annualized tracking error of portfolio vs SPY.
    Uses last 63 trading days (~1 quarter).
    """
    if nav_history.empty or spy_prices.empty or len(nav_history) < 10:
        return 0.0

    portfolio_returns = (
        nav_history.sort_values("date")
        .tail(lookback)
        .set_index("date")["nav"]
        .pct_change()
        .dropna()
    )

    spy_returns = (
        spy_prices.sort_values("date")
        .tail(lookback)
        .set_index("date")["close"]
        .pct_change()
        .dropna()
    )

    aligned = pd.concat([portfolio_returns, spy_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0

    aligned.columns = ["portfolio", "spy"]
    active_returns = aligned["portfolio"] - aligned["spy"]
    tracking_error = float(active_returns.std() * np.sqrt(252))

    return tracking_error


# ------------------------------------------------------------
# LIQUIDITY SCORING
# ------------------------------------------------------------

def compute_liquidity_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes position_value / ADV ratio for each held position.
    Flags positions where ratio > 10% (liquidity_flag_pct).
    """
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
    Loads prices and SPY prices from storage (cached).
    Returns dict with all risk metrics and flags.
    Keys: nav, circuit_breaker, stop_exits, stop_triggers, drift,
          liquidity, beta, tracking_error, te_breach, drawdown
    """
    if run_date is None:
        run_date = date.today()

    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else cfg["portfolio"]["initial_capital"]

    print(f"[risk] Running EOD risk monitor for {run_date}")

    # Load prices from cache
    prices = load_prices()
    from data.storage import load_spy_prices
    spy_prices = load_spy_prices()

    circuit_breaker  = check_circuit_breaker(nav)
    stop_triggers_df = check_trailing_stops(prices)
    liquidity_scores = compute_liquidity_scores(prices)

    # Build stop_exits list for runner (P2 priority trades)
    stop_exits = []
    if not stop_triggers_df.empty:
        stop_exits = stop_triggers_df["ticker"].tolist()

    # Drift detection — compare current vs last target weights
    drift_result = {"triggered": False, "position_drift": [], "portfolio_drift": False, "sector_drift": []}
    target_path = Path("data/processed/target_weights.parquet")
    if target_path.exists():
        target_weights = pd.read_parquet(target_path)
        drift_result = check_drift(target_weights)
        drift_result["triggered"] = bool(
            drift_result["position_drift"]
            or drift_result["portfolio_drift"]
            or drift_result["sector_drift"]
        )

    tracking_error = 0.0
    beta           = 1.0
    if not spy_prices.empty:
        tracking_error = compute_tracking_error(nav_history, spy_prices)
        beta           = compute_beta(prices, spy_prices)

    te_breach = tracking_error > cfg["optimizer"]["tracking_error_cap"]

    drawdown = circuit_breaker.get("drawdown", 0.0)

    print(f"[risk] Beta: {beta:.3f} | Tracking Error: {tracking_error:.2%} | TE breach: {te_breach}")
    print(f"[risk] Stop exits: {len(stop_exits)} | CB tier: {circuit_breaker['tier']} | Drift triggered: {drift_result['triggered']}")

    if te_breach:
        notify(
            f"Tracking error breach: {tracking_error:.2%} exceeds cap {cfg['optimizer']['tracking_error_cap']:.0%}",
            level="warning"
        )

    return {
        "nav":             nav,
        "circuit_breaker": circuit_breaker,
        "stop_exits":      stop_exits,
        "stop_triggers":   stop_triggers_df,
        "drift":           drift_result,
        "liquidity":       liquidity_scores,
        "beta":            beta,
        "tracking_error":  tracking_error,
        "te_breach":       te_breach,
        "drawdown":        drawdown,
    }