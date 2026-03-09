# optimizer/portfolio_optimizer.py
# Mean-variance portfolio optimizer using cvxpy.
# Objective: maximize Sharpe ratio (maximize returns - risk penalty)
# Constraints: 5% max weight, 2% cash buffer, tracking error cap 6%,
#              turnover penalty lambda, sector neutralization.
# Covariance: Ledoit-Wolf shrinkage on 252-day returns.
# Sector neutralization: FMP names remapped to GICS, pooling for small sectors.

import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import date
from sklearn.covariance import LedoitWolf
from data.storage import load_prices, load_signals, load_portfolio, load_constituents
from fund_accounting.nav import load_nav_history
from utils.config_loader import get_config
from utils.notifications import notify

cfg = get_config()
OPT = cfg["optimizer"]
PF  = cfg["portfolio"]
TO  = cfg["turnover"]

MAX_WEIGHT      = PF["max_position_weight"]        # 0.05
CASH_BUFFER     = PF["cash_buffer"]                # 0.02
TE_TARGET       = OPT["tracking_error_target"]     # 0.04
TE_CAP          = OPT["tracking_error_cap"]        # 0.06
TURNOVER_LAMBDA = OPT["turnover_lambda"]           # 0.005
COV_LOOKBACK    = OPT["cov_lookback"]              # 252
ROUND_TRIP_BPS  = TO["round_trip_cost_bps"]        # 10
MAX_TURNOVER    = TO["max_turnover_per_rebalance"]  # 0.30

# FMP -> GICS sector mapping
SECTOR_MAP = {
    "Consumer Cyclical":  "Consumer Discretionary",
    "Basic Materials":    "Materials",
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
}

# Sector pooling — sectors with < 30 stocks are pooled
SECTOR_POOLS = {
    "Materials":   "Materials_Industrials",
    "Industrials": "Materials_Industrials",
    "Real Estate": "RealEstate_Financials",
    "Financials":  "RealEstate_Financials",
    "Utilities":   "Utilities_Energy",
    "Energy":      "Utilities_Energy",
}

# Standalone sectors (>= 30 stocks, no pooling)
STANDALONE_SECTORS = {
    "Technology", "Healthcare", "Consumer Discretionary",
    "Consumer Staples", "Communication Services"
}


# ------------------------------------------------------------
# COVARIANCE ESTIMATION
# ------------------------------------------------------------

def compute_covariance(prices: pd.DataFrame, tickers: list) -> np.ndarray:
    """
    Computes Ledoit-Wolf shrinkage covariance matrix for given tickers.
    Uses COV_LOOKBACK days of daily returns.
    Returns annualized covariance matrix (N x N).
    """
    price_pivot = (
        prices[prices["ticker"].isin(tickers)]
        .sort_values("date")
        .pivot(index="date", columns="ticker", values="close")
        .tail(COV_LOOKBACK)
    )

    # Align to requested ticker order, drop missing
    price_pivot = price_pivot.reindex(columns=tickers)
    price_pivot = price_pivot.dropna(axis=1, how="all")

    returns = price_pivot.pct_change().dropna()

    if returns.shape[0] < 60 or returns.shape[1] < 2:
        print("[optimizer] Insufficient return history — using diagonal covariance")
        n = len(tickers)
        return np.eye(n) * (0.20 ** 2 / 252)

    lw = LedoitWolf()
    lw.fit(returns.values)

    # Annualize
    cov_daily      = lw.covariance_
    cov_annualized = cov_daily * 252

    # Map back to full ticker list (some may have been dropped)
    valid_tickers = price_pivot.dropna(axis=1, how="all").columns.tolist()
    n             = len(tickers)
    full_cov      = np.eye(n) * (0.20 ** 2)  # default for missing tickers

    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if ti in valid_tickers and tj in valid_tickers:
                ii = valid_tickers.index(ti)
                jj = valid_tickers.index(tj)
                full_cov[i, j] = cov_annualized[ii, jj]

    return full_cov


# ------------------------------------------------------------
# SECTOR NEUTRALIZATION
# ------------------------------------------------------------

def build_sector_constraints(tickers: list, constituents: pd.DataFrame) -> dict:
    """
    Builds sector group index mapping for given tickers.
    Uses pooling for small sectors per design spec.
    Returns dict of sector_name -> list of ticker indices.
    """
    constituents = constituents.copy()
    constituents["sector_mapped"] = constituents["sector"].replace(SECTOR_MAP)
    constituents["sector_final"]  = constituents["sector_mapped"].apply(
        lambda s: SECTOR_POOLS.get(s, s)
    )

    ticker_to_sector = dict(zip(constituents["ticker"], constituents["sector_final"]))
    sector_groups    = {}

    for i, ticker in enumerate(tickers):
        sector = ticker_to_sector.get(ticker, "Unknown")
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(i)

    return sector_groups


def compute_spy_sector_weights(constituents: pd.DataFrame) -> dict:
    """
    Computes equal-weight SPY sector weights as benchmark reference.
    In production this would use actual SPY sector weights from data provider.
    """
    constituents = constituents.copy()
    constituents["sector_mapped"] = constituents["sector"].replace(SECTOR_MAP)
    constituents["sector_final"]  = constituents["sector_mapped"].apply(
        lambda s: SECTOR_POOLS.get(s, s)
    )

    counts      = constituents["sector_final"].value_counts()
    total       = counts.sum()
    spy_weights = (counts / total).to_dict()
    return spy_weights


# ------------------------------------------------------------
# CURRENT WEIGHTS
# ------------------------------------------------------------

def get_current_weights(tickers: list, nav: float) -> np.ndarray:
    """
    Returns current portfolio weights for given tickers.
    Zero weight for tickers not currently held.
    """
    portfolio = load_portfolio()
    weights   = np.zeros(len(tickers))

    if portfolio.empty or nav <= 0:
        return weights

    for i, ticker in enumerate(tickers):
        if ticker in portfolio["ticker"].values:
            mkt_val    = float(portfolio.loc[portfolio["ticker"] == ticker, "market_value"].values[0])
            weights[i] = mkt_val / nav

    return weights


# ------------------------------------------------------------
# MAIN OPTIMIZER
# ------------------------------------------------------------

def run_optimizer(run_date: date = None, regime: str = "recovery") -> pd.DataFrame:
    """
    Full portfolio optimization run.
    1. Load signals and select investable universe (top 50% by composite rank)
    2. Compute Ledoit-Wolf covariance
    3. Run cvxpy optimization with all constraints
    4. Return target weights DataFrame

    Returns DataFrame with columns: ticker, target_weight, sector
    """
    if run_date is None:
        run_date = date.today()

    print(f"[optimizer] Running optimization for {run_date} | regime={regime}")

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    signals      = load_signals()
    prices       = load_prices()
    constituents = load_constituents()
    nav_history  = load_nav_history()

    if signals.empty:
        print("[optimizer] No signals — aborting")
        notify("Optimizer failed — no signals available", level="critical")
        return pd.DataFrame()

    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else PF["initial_capital"]

    # ----------------------------------------------------------
    # UNIVERSE SELECTION — top 50% by composite rank
    # ----------------------------------------------------------
    signals = signals[signals["composite_score"] != 0.0]  # exclude signal floor failures
    signals = signals.sort_values("composite_rank", ascending=False)

    # Take top 50% — natural position count ~30-80 after optimization
    top_n    = max(30, len(signals) // 2)
    universe = signals.head(top_n)
    tickers  = universe["ticker"].tolist()

    print(f"[optimizer] Universe: {len(tickers)} tickers after signal filtering")

    if len(tickers) < 10:
        print("[optimizer] Universe too small — aborting")
        notify("Optimizer failed — universe too small", level="critical")
        return pd.DataFrame()

    # ----------------------------------------------------------
    # EXPECTED RETURNS — scaled composite score
    # ----------------------------------------------------------
    mu = universe.set_index("ticker")["composite_score"].reindex(tickers).fillna(0).values
    mu = mu / (np.std(mu) + 1e-8)  # normalize to unit std

    # ----------------------------------------------------------
    # COVARIANCE
    # ----------------------------------------------------------
    sigma = compute_covariance(prices, tickers)

    # ----------------------------------------------------------
    # CURRENT WEIGHTS (for turnover penalty)
    # ----------------------------------------------------------
    w_current = get_current_weights(tickers, nav)

    # ----------------------------------------------------------
    # SPY WEIGHTS (for tracking error)
    # ----------------------------------------------------------
    n = len(tickers)
    # SPY proxy: equal weight across full 503 constituents, subset to universe.
    # Using total_constituents as denominator correctly represents active deviation
    # from the full benchmark, not just the investable universe.
    total_constituents = len(constituents)
    w_spy = np.ones(n) / total_constituents  # 1/503 per ticker

    # ----------------------------------------------------------
    # CVXPY OPTIMIZATION
    # ----------------------------------------------------------
    w = cp.Variable(n)

    # Turnover cost (round-trip bps converted to decimal)
    round_trip_cost = ROUND_TRIP_BPS / 10000.0
    turnover        = cp.norm1(w - w_current)

    # Active weights vs SPY benchmark
    w_active = w - w_spy

    # Objective: maximize alpha - risk penalty - turnover penalty
    risk_aversion = OPT.get("risk_aversion", 1.5)
    objective = cp.Maximize(
        mu @ w
        - (risk_aversion / 2) * cp.quad_form(w, cp.psd_wrap(sigma))
        - TURNOVER_LAMBDA * turnover
    )

    constraints = [
        # Long only
        w >= 0,
        # Max position size — 5% cap
        w <= MAX_WEIGHT,
        # Stay invested — leave cash buffer
        cp.sum(w) <= 1.0 - CASH_BUFFER,
        cp.sum(w) >= 0.85,
        # Max turnover per rebalance — skipped on first run (no existing portfolio)
        *([] if w_current.sum() == 0 else [turnover <= MAX_TURNOVER * 2]),
        # Tracking error cap vs SPY
        cp.quad_form(w_active, cp.psd_wrap(sigma)) <= TE_CAP ** 2,
    ]

    # Sector neutralization constraints
    # Lower bound skipped if sector has too few tickers to meet it (feasibility guard)
    sector_groups  = build_sector_constraints(tickers, constituents)
    spy_sector_wts = compute_spy_sector_weights(constituents)

    for sector, indices in sector_groups.items():
        if len(indices) == 0:
            continue

        spy_wt        = spy_sector_wts.get(sector, 0.05)
        sector_weight = cp.sum(w[indices])
        max_capacity  = len(indices) * MAX_WEIGHT
        lower_bound   = max(0, spy_wt - 0.05)

        # Only apply constraints if sector has enough tickers to be feasible
        if max_capacity >= lower_bound:
            constraints.append(sector_weight >= lower_bound)
            constraints.append(sector_weight <= spy_wt + 0.05)

    problem = cp.Problem(objective, constraints)

    # Try CLARABEL first, fall back to SCS
    for solver in [cp.CLARABEL, cp.SCS]:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ["optimal", "optimal_inaccurate"]:
                break
        except Exception as e:
            print(f"[optimizer] Solver {solver} failed: {e}")

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[optimizer] Optimization failed: {problem.status}")
        notify(f"Optimizer failed: {problem.status} — holding current portfolio", level="critical")
        return pd.DataFrame()

    if problem.status == "optimal_inaccurate":
        print("[optimizer] Warning: optimal_inaccurate — using result with caution")

    # ----------------------------------------------------------
    # PROCESS RESULTS
    # ----------------------------------------------------------
    weights = w.value
    if weights is None:
        print("[optimizer] No weights returned — aborting")
        return pd.DataFrame()

    # Zero out tiny weights below 0.1%
    weights[weights < 0.001] = 0.0

    # Renormalize to sum to invested amount
    total = weights.sum()
    if total > 0:
        weights = weights / total * (1.0 - CASH_BUFFER)

    # Build result DataFrame
    constituents_map = constituents.copy()
    constituents_map["sector_mapped"] = constituents_map["sector"].replace(SECTOR_MAP)

    result = pd.DataFrame({
        "ticker":        tickers,
        "target_weight": weights,
    })

    result = result[result["target_weight"] > 0].reset_index(drop=True)
    result = result.merge(
        constituents_map[["ticker", "sector_mapped"]].rename(columns={"sector_mapped": "sector"}),
        on="ticker", how="left"
    )

    # Diagnostics
    n_positions  = len(result)
    max_wt       = result["target_weight"].max()
    active_wts   = weights - w_spy
    te_realized  = float(np.sqrt(active_wts @ sigma @ active_wts))
    turnover_val = float(np.sum(np.abs(weights - w_current)))

    print(f"[optimizer] Positions: {n_positions} | Max weight: {max_wt:.2%} | "
          f"TE: {te_realized:.2%} | Turnover: {turnover_val:.2%}")

    notify(
        f"Optimization complete for {run_date}\n"
        f"Positions: {n_positions} | Max weight: {max_wt:.2%}\n"
        f"Tracking error: {te_realized:.2%} | Turnover: {turnover_val:.2%}",
        level="info"
    )

    return result