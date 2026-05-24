# optimizer/portfolio_optimizer.py
# Horizon mean-variance portfolio optimizer.
# Implements V4 six-tier circuit breaker via cvxpy constraints:
#   - per-tier max_weight (upper bound on each w_i)
#   - per-tier invested target (sum(w) == cb_invested[tier])
#   - per-tier sum_floor (sum(w) >= cb_sum_floor[tier])
#   - no-new-positions block at tiers >= NO_NEW_POSITIONS_TIER
#   - turnover cost with quadratic penalty (lambda * ||w - w_prev||^2)
#
# Reads CB params from settings.yaml. Falls back gracefully if cvxpy fails.

import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import date
from pathlib import Path
from data.storage import load_portfolio, load_signals, load_prices, save_portfolio
from data.storage import load_constituents
from fund_accounting.nav import load_nav_history
from utils.config_loader import get_config
from utils.notifications import notify

cfg = get_config()
OPT = cfg["optimizer"]
CB  = cfg["circuit_breaker"]
PORT = cfg["portfolio"]
TURN = cfg["turnover"]

LAMBDA          = float(OPT["turnover_lambda"])
RISK_AVERSION   = float(OPT["risk_aversion"])
COV_LOOKBACK    = int(OPT["cov_lookback"])
MAX_WEIGHT_BASE = float(PORT["max_position_weight"])
MIN_POSITIONS   = int(PORT.get("min_positions", 20))
MAX_TURNOVER    = float(TURN["max_turnover_per_rebalance"])
COST_BPS        = float(TURN["round_trip_cost_bps"]) / 10000.0  # decimal

NO_NEW_POS_TIER = int(CB["no_new_positions_tier"])

EXECUTED_WEIGHTS_PATH = Path("data/processed/executed_weights.parquet")


def _cb_param(name: str, tier: int, default):
    """Resolve a per-tier CB parameter from YAML (keys may be int or str)."""
    d = CB.get(name, {})
    if tier in d:
        return d[tier]
    if str(tier) in d:
        return d[str(tier)]
    return default


def _ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimator."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_


def _build_returns_panel(tickers: list, prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Returns a wide DataFrame (date x ticker) of daily returns, last `lookback` rows."""
    sub = prices[prices["ticker"].isin(tickers)].copy()
    pivot = (
        sub.pivot_table(index="date", columns="ticker", values="close")
        .sort_index()
        .pct_change()
        .dropna(how="all")
        .tail(lookback)
    )
    return pivot.dropna(axis=1, how="any")


def _expected_returns(signal_scores: pd.Series) -> np.ndarray:
    """Convert composite signal scores to expected return estimates."""
    return signal_scores.values.astype(float)


def _previous_weights(tickers: list, nav: float) -> dict:
    """Return current portfolio weights keyed by ticker (0.0 if not held)."""
    portfolio = load_portfolio()
    if portfolio.empty or nav <= 0:
        return {t: 0.0 for t in tickers}
    weights = {}
    for t in tickers:
        row = portfolio[portfolio["ticker"] == t]
        if row.empty:
            weights[t] = 0.0
        else:
            weights[t] = float(row["market_value"].iloc[0]) / nav
    return weights


def _currently_held_tickers() -> set:
    portfolio = load_portfolio()
    if portfolio.empty:
        return set()
    return set(portfolio["ticker"].tolist())


# ------------------------------------------------------------
# CORE OPTIMIZATION
# ------------------------------------------------------------

def optimize_portfolio(
    signals_df: pd.DataFrame,
    regime: str,
    cb_tier: int,
    run_date: date = None
) -> pd.DataFrame:
    """
    Solves the mean-variance optimization with V4 CB constraints.

    Inputs:
        signals_df: DataFrame with columns ['ticker', 'composite_score', ...]
        regime: composite regime string (bull/recovery/bear/crisis)
        cb_tier: integer 0-5 from risk monitor
        run_date: date for logging

    Returns:
        DataFrame with columns ['ticker', 'target_weight'] for non-zero positions.
    """
    if run_date is None:
        run_date = date.today()

    if signals_df.empty:
        print("[optimizer] No signals — returning empty allocation")
        return pd.DataFrame(columns=["ticker", "target_weight"])

    # ----------------------------------------------------------
    # CB tier params
    # ----------------------------------------------------------
    cb_max_wt   = float(_cb_param("cb_max_weights", cb_tier, MAX_WEIGHT_BASE))
    cb_invested = float(_cb_param("cb_invested",    cb_tier, 0.98))
    cb_floor    = float(_cb_param("cb_sum_floor",   cb_tier, 0.85))
    block_new   = cb_tier >= NO_NEW_POS_TIER
    effective_max_wt = min(MAX_WEIGHT_BASE, cb_max_wt)

    print(f"[optimizer] V4 CB tier T{cb_tier} | inv={cb_invested:.2f} "
          f"max_wt={effective_max_wt:.3f} floor={cb_floor:.2f} "
          f"block_new={block_new}")

    # ----------------------------------------------------------
    # Universe construction
    # ----------------------------------------------------------
    candidates = signals_df.dropna(subset=["composite_score"]).copy()
    if candidates.empty:
        print("[optimizer] No candidates after signal coverage filter")
        return pd.DataFrame(columns=["ticker", "target_weight"])

    if block_new:
        held = _currently_held_tickers()
        before = len(candidates)
        candidates = candidates[candidates["ticker"].isin(held)]
        print(f"[optimizer] NO_NEW_POSITIONS active (T{cb_tier}): "
              f"{before} -> {len(candidates)} candidates (held only)")
        if candidates.empty:
            print("[optimizer] No held tickers eligible — emergency exit, holding cash")
            return pd.DataFrame(columns=["ticker", "target_weight"])

    # Score-rank universe, take top N most likely to fit MIN_POSITIONS feasibility
    # At max_wt=2.5% with MIN_POSITIONS=20, sum floor needs >= 20 positions.
    # Cap at 60 candidates to keep cvxpy fast.
    candidates = candidates.sort_values("composite_score", ascending=False).head(60)
    tickers = candidates["ticker"].tolist()

    # ----------------------------------------------------------
    # Build inputs
    # ----------------------------------------------------------
    prices = load_prices()
    returns_panel = _build_returns_panel(tickers, prices, COV_LOOKBACK)
    valid_tickers = list(returns_panel.columns)

    if len(valid_tickers) < MIN_POSITIONS:
        msg = (f"Only {len(valid_tickers)} valid tickers (need >= {MIN_POSITIONS}). "
               f"Insufficient universe for optimization.")
        print(f"[optimizer] {msg}")
        notify(msg, level="warning")
        if not block_new:
            return pd.DataFrame(columns=["ticker", "target_weight"])
        # Under block_new, try anyway with reduced floor
        if len(valid_tickers) == 0:
            return pd.DataFrame(columns=["ticker", "target_weight"])

    signals_aligned = candidates.set_index("ticker").loc[valid_tickers]
    mu = _expected_returns(signals_aligned["composite_score"])
    cov = _ledoit_wolf_cov(returns_panel)

    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty \
          else float(PORT["initial_capital"])
    prev_w_dict = _previous_weights(valid_tickers, nav)
    w_prev = np.array([prev_w_dict[t] for t in valid_tickers])

    # ----------------------------------------------------------
    # cvxpy problem
    # ----------------------------------------------------------
    n = len(valid_tickers)
    w = cp.Variable(n, nonneg=True)

    # Objective: maximize expected return - risk_aversion * variance
    #            - lambda * turnover penalty - transaction cost
    expected_ret = mu @ w
    risk         = cp.quad_form(w, cov)
    turnover     = cp.norm1(w - w_prev)
    tx_cost      = COST_BPS * turnover

    objective = cp.Maximize(
        expected_ret
        - RISK_AVERSION * risk
        - LAMBDA * cp.sum_squares(w - w_prev)
        - tx_cost
    )

    # Constraints
    feasible_floor = min(cb_floor, len(valid_tickers) * effective_max_wt * 0.99)
    constraints = [
        cp.sum(w) <= cb_invested,
        cp.sum(w) >= feasible_floor,
        w <= effective_max_wt,
        turnover <= MAX_TURNOVER,
    ]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        print(f"[optimizer] ECOS failed ({e}) — trying SCS")
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception as e2:
            print(f"[optimizer] SCS also failed: {e2}")
            notify(f"Optimizer failed: {e2}", level="critical")
            return pd.DataFrame(columns=["ticker", "target_weight"])

    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"[optimizer] Solver status: {prob.status} — returning empty allocation")
        notify(f"Optimizer status: {prob.status}", level="critical")
        return pd.DataFrame(columns=["ticker", "target_weight"])

    # ----------------------------------------------------------
    # Post-process: clip tiny weights, renormalize
    # ----------------------------------------------------------
    weights = np.array(w.value).flatten()
    weights = np.clip(weights, 0.0, effective_max_wt)

    # Drop weights below 0.1% (rounding noise)
    weights[weights < 0.001] = 0.0

    if weights.sum() == 0:
        print("[optimizer] All weights zero after clipping")
        return pd.DataFrame(columns=["ticker", "target_weight"])

    # Renormalize to invested target if needed (only down-scale; never inflate above target)
    current_sum = weights.sum()
    if current_sum > cb_invested:
        weights = weights * (cb_invested / current_sum)
    # Cap any weight that floats over after renorm
    weights = np.minimum(weights, effective_max_wt)

    result = pd.DataFrame({
        "ticker":        valid_tickers,
        "target_weight": weights,
    })
    result = result[result["target_weight"] > 0].sort_values("target_weight", ascending=False)

    print(f"[optimizer] Solved: {len(result)} positions | "
          f"sum_wt={result['target_weight'].sum():.4f} | "
          f"max_wt={result['target_weight'].max():.4f}")

    return result.reset_index(drop=True)


# ------------------------------------------------------------
# DIAGNOSTIC: PROPOSED VS CURRENT
# ------------------------------------------------------------

def diff_against_current(target_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Compares optimizer output to current portfolio weights.
    Returns DataFrame with columns: ticker, current_weight, target_weight, delta.
    Used by approve.py for human review.
    """
    portfolio = load_portfolio()
    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty \
          else float(PORT["initial_capital"])

    current = pd.DataFrame({"ticker": [], "current_weight": []})
    if not portfolio.empty:
        current = portfolio[["ticker", "market_value"]].copy()
        current["current_weight"] = current["market_value"] / nav
        current = current[["ticker", "current_weight"]]

    merged = current.merge(target_weights, on="ticker", how="outer").fillna(0.0)
    merged["delta"] = merged["target_weight"] - merged["current_weight"]
    return merged.sort_values("delta", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
