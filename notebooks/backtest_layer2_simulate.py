# notebooks/backtest_layer2_simulate.py
# LAYER 2 — Portfolio Simulation (re-runnable, ~60-90 mins for 4 strategies)
#
# Simulates 4 strategies using precomputed Layer 1 data:
#   1. Mean-Variance Monthly    (mv_monthly)
#   2. Mean-Variance Quarterly  (mv_quarterly)
#   3. Black-Litterman Monthly  (bl_monthly)
#   4. Black-Litterman Quarterly(bl_quarterly)
#
# Black-Litterman blends market equilibrium returns with signal-based views.
# Both use the same optimizer and constraints — only mu differs.
#
# Run: python notebooks/backtest_layer2_simulate.py

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import get_config

cfg = get_config()

# ============================================================
# CONFIG
# ============================================================

BACKTEST_START     = pd.Timestamp("2009-01-01")
BACKTEST_END       = pd.Timestamp("2026-01-01")
INITIAL_CAPITAL    = 1_000_000.0
COST_BPS_ONE_WAY   = 5
MAX_WEIGHT         = 0.05
TURNOVER_LAMBDA    = cfg["optimizer"]["turnover_lambda"]
TE_CAP             = cfg["optimizer"]["tracking_error_cap"] ** 2
RISK_AVERSION      = 1.0

# Black-Litterman parameters
BL_TAU             = 0.05    # Uncertainty scaling on equilibrium returns
BL_VIEW_CONFIDENCE = 0.25    # 0 = ignore signals, 1 = full trust in signals

# Trailing stop
VOL_LOOKBACK       = 25
VOL_MULTIPLIER     = 2.0
STOP_FLOOR         = 0.05
STOP_CAP           = 0.20

BACKTEST_DIR = Path("data/backtest")
PRECOMP_DIR  = Path("data/backtest/precomputed")
RESULTS_DIR  = Path("data/backtest/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("[layer2] ================================================")
print(f"[layer2] Simulation: {BACKTEST_START.date()} to {BACKTEST_END.date()}")
print(f"[layer2] Strategies: MV-Monthly | MV-Quarterly | BL-Monthly | BL-Quarterly")
print("[layer2] ================================================\n")


# ============================================================
# LOAD PRECOMPUTED DATA
# ============================================================

print("[layer2] Loading precomputed data...")

signals_history = pd.read_parquet(PRECOMP_DIR / "signals_history.parquet")
signals_history["date"] = pd.to_datetime(signals_history["date"])

with open(PRECOMP_DIR / "covariance_matrices.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

with open(PRECOMP_DIR / "universe_history.pkl", "rb") as f:
    universe_history = pickle.load(f)

prices_all = pd.read_parquet(BACKTEST_DIR / "prices.parquet")
prices_all["date"] = pd.to_datetime(prices_all["date"])
prices_all = prices_all.sort_values(["ticker", "date"])

print(f"[layer2] Signals history: {len(signals_history)} rows | {signals_history['date'].nunique()} dates")
print(f"[layer2] Covariance matrices: {len(cov_matrices)} dates")
print(f"[layer2] Universe snapshots: {len(universe_history)} dates\n")


# ============================================================
# TRADING CALENDAR
# ============================================================

aapl         = prices_all[prices_all["ticker"] == "AAPL"].sort_values("date")
trading_days = aapl[
    (aapl["date"] >= BACKTEST_START) & (aapl["date"] < BACKTEST_END)
]["date"].tolist()


def get_rebalance_dates(trading_days: list, frequency: str) -> set:
    seen, dates = set(), set()
    for day in trading_days:
        period = (day.year, day.month) if frequency == "monthly" \
                 else (day.year, (day.month - 1) // 3 + 1)
        if period not in seen:
            seen.add(period)
            dates.add(day)
    return dates


# ============================================================
# EXPECTED RETURN FUNCTIONS
# ============================================================

def compute_mu_mv(signals_subset: pd.DataFrame, tickers: list) -> np.ndarray:
    """
    Mean-Variance: composite signal scores shifted to positive range.
    """
    score_map = signals_subset.set_index("ticker")["composite_score"]
    mu        = np.array([score_map.get(t, 0.0) for t in tickers])
    mu        = mu - mu.min() + 0.01
    return mu


def compute_mu_bl(signals_subset: pd.DataFrame, tickers: list,
                  sigma: np.ndarray) -> np.ndarray:
    """
    Black-Litterman posterior expected returns.

    Pi   = risk_aversion * sigma @ w_eq  (market equilibrium, equal-weight proxy)
    Q    = z-scored signal views (one per asset, P = identity)
    Omega= diagonal uncertainty matrix (scaled by BL_VIEW_CONFIDENCE)
    mu_bl= posterior mean blending Pi and Q weighted by confidence
    """
    n    = len(tickers)
    w_eq = np.ones(n) / n
    pi   = RISK_AVERSION * sigma @ w_eq

    score_map = signals_subset.set_index("ticker")["composite_score"]
    raw       = np.array([score_map.get(t, 0.0) for t in tickers])
    Q         = (raw - raw.mean()) / (raw.std() + 1e-8)

    P     = np.eye(n)
    tau_s = BL_TAU * sigma

    view_var = (1.0 - BL_VIEW_CONFIDENCE) / (BL_VIEW_CONFIDENCE + 1e-8)
    omega    = np.diag(np.diag(P @ tau_s @ P.T) * view_var)

    try:
        reg       = np.eye(n) * 1e-6
        tau_s_inv = np.linalg.inv(tau_s + reg)
        omega_inv = np.linalg.inv(omega + np.eye(n) * 1e-8)
        M         = tau_s_inv + P.T @ omega_inv @ P
        M_inv     = np.linalg.inv(M + reg)
        mu_bl     = M_inv @ (tau_s_inv @ pi + P.T @ omega_inv @ Q)
    except np.linalg.LinAlgError:
        mu_bl = pi  # fallback to equilibrium

    mu_bl = mu_bl - mu_bl.min() + 0.01
    return mu_bl


# ============================================================
# OPTIMIZER
# ============================================================

def run_optimizer(mu: np.ndarray, tickers: list, sigma: np.ndarray,
                  current_weights: dict, universe: list) -> dict:
    """
    Mean-variance optimizer with turnover penalty and TE constraint.
    Identical for MV and BL — only mu differs.
    """
    if len(tickers) < 5:
        return {}

    n      = len(tickers)
    w      = cp.Variable(n)
    w_curr = np.array([current_weights.get(t, 0.0) for t in tickers])
    w_eq   = np.ones(n) / n

    risk    = cp.quad_form(w, cp.psd_wrap(sigma))
    ret_    = mu @ w
    penalty = TURNOVER_LAMBDA * cp.norm1(w - w_curr)
    te      = cp.quad_form(w - w_eq, cp.psd_wrap(sigma))

    objective   = cp.Maximize(ret_ - RISK_AVERSION * risk - penalty)
    constraints = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT, te <= TE_CAP]
    prob        = cp.Problem(objective, constraints)

    for solver in [cp.CLARABEL, cp.SCS]:
        try:
            prob.solve(solver=solver, warm_start=True)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                break
        except Exception:
            continue

    if w.value is None:
        return {}

    weights = {tickers[i]: float(max(0, w.value[i])) for i in range(n)}
    total   = sum(weights.values())
    if total == 0:
        return {}
    return {t: v / total for t, v in weights.items() if v > 1e-4}


def prepare_inputs(signals_today: pd.DataFrame,
                   cov: pd.DataFrame) -> tuple:
    """Selects top-half tickers by rank, filters to tickers in cov matrix."""
    if signals_today.empty or cov.empty:
        return [], np.array([]), pd.DataFrame()

    n_total  = len(signals_today)
    top_half = signals_today.nsmallest(max(10, n_total // 2), "composite_rank")
    tickers  = [t for t in top_half["ticker"].tolist() if t in cov.columns]

    if len(tickers) < 5:
        return [], np.array([]), pd.DataFrame()

    sigma = cov.loc[tickers, tickers].values
    return tickers, sigma, top_half


# ============================================================
# SIMULATION
# ============================================================

def get_px(prices_all, day, ticker, col):
    try:
        row = prices_all[(prices_all["ticker"] == ticker) & (prices_all["date"] == day)]
        return float(row[col].iloc[0]) if not row.empty else None
    except Exception:
        return None


def run_simulation(frequency: str, strategy: str) -> tuple:
    label = f"{strategy.upper()}-{frequency}"
    print(f"\n[layer2] {'='*52}")
    print(f"[layer2] Simulating: {label}")
    print(f"[layer2] {'='*52}")

    rebalance_dates = get_rebalance_dates(trading_days, frequency)
    print(f"[layer2] Trading days: {len(trading_days)} | Rebalance dates: {len(rebalance_dates)}")

    cash             = INITIAL_CAPITAL
    positions        = {}
    stop_levels      = {}
    recent_highs     = {}
    entry_prices     = {}
    pending_buys     = {}
    pending_sells    = set()
    sl_replace_dates = set()
    nav_history      = []
    trade_log        = []

    for day_idx, today in enumerate(trading_days):

        # 1. Trailing stop breach check
        sl_today = []
        for ticker in list(positions.keys()):
            if ticker not in stop_levels:
                continue
            low = get_px(prices_all, today, ticker, "low")
            if low is not None and low <= stop_levels[ticker]:
                sl_today.append(ticker)
        for ticker in sl_today:
            pending_sells.add(ticker)
        if sl_today:
            t2_idx = day_idx + 2
            if t2_idx < len(trading_days):
                sl_replace_dates.add(trading_days[t2_idx])

        # 2. Execute pending sells at open
        for ticker in list(pending_sells):
            open_px = get_px(prices_all, today, ticker, "open")
            if open_px is None or open_px <= 0:
                pending_sells.discard(ticker)
                continue
            shares = positions.pop(ticker, 0)
            if shares > 0:
                cash += shares * open_px * (1 - COST_BPS_ONE_WAY / 10000)
                trade_log.append({"date": today, "ticker": ticker, "action": "sell",
                                   "shares": shares, "price": open_px, "strategy": label})
            stop_levels.pop(ticker, None)
            recent_highs.pop(ticker, None)
            entry_prices.pop(ticker, None)
            pending_sells.discard(ticker)

        # 3. Execute pending buys at open
        for ticker, dollar_amt in list(pending_buys.items()):
            open_px = get_px(prices_all, today, ticker, "open")
            if open_px is None or open_px <= 0:
                del pending_buys[ticker]
                continue
            spend  = min(dollar_amt, cash * 0.99)
            shares = spend * (1 - COST_BPS_ONE_WAY / 10000) / open_px
            if shares > 0:
                cash                -= spend
                positions[ticker]    = positions.get(ticker, 0) + shares
                entry_prices[ticker] = open_px
                recent_highs[ticker] = max(recent_highs.get(ticker, open_px), open_px)
                trade_log.append({"date": today, "ticker": ticker, "action": "buy",
                                   "shares": shares, "price": open_px, "strategy": label})
            del pending_buys[ticker]

        # 4. Valuation at close
        equity = sum(
            shares * (get_px(prices_all, today, t, "close") or entry_prices.get(t, 0))
            for t, shares in positions.items()
        )
        nav = cash + equity
        nav_history.append({"date": today, "nav": nav, "cash": cash, "equity": equity})

        # 5. Update trailing stops
        for ticker in list(positions.keys()):
            close = get_px(prices_all, today, ticker, "close")
            if close is None:
                continue
            recent_highs[ticker] = max(recent_highs.get(ticker, close), close)
            px_hist = (prices_all[
                (prices_all["ticker"] == ticker) &
                (prices_all["date"] <= today)
            ]["close"].tail(VOL_LOOKBACK + 1))
            if len(px_hist) >= VOL_LOOKBACK:
                daily_vol = px_hist.pct_change().dropna().std()
                dist      = float(np.clip(VOL_MULTIPLIER * daily_vol * np.sqrt(VOL_LOOKBACK),
                                          STOP_FLOOR, STOP_CAP))
            else:
                dist = STOP_FLOOR
            stop_levels[ticker] = recent_highs[ticker] * (1 - dist)

        # 6. Optimizer on rebalance and SL replacement days
        if today in rebalance_dates or today in sl_replace_dates:
            event = "rebalance" if today in rebalance_dates else "sl_replace"

            sig_today = signals_history[signals_history["date"] == today]
            if sig_today.empty:
                prior = signals_history[signals_history["date"] < today]
                if not prior.empty:
                    sig_today = prior[prior["date"] == prior["date"].max()]

            cov      = cov_matrices.get(today, pd.DataFrame())
            universe = universe_history.get(today, [])

            total_nav = nav if nav > 0 else 1.0
            curr_w    = {
                t: (shares * (get_px(prices_all, today, t, "close") or 0)) / total_nav
                for t, shares in positions.items()
            }

            tickers, sigma, sig_subset = prepare_inputs(sig_today, cov)

            if len(tickers) >= 5:
                mu = compute_mu_mv(sig_subset, tickers) if strategy == "mv" \
                     else compute_mu_bl(sig_subset, tickers, sigma)

                target_w = run_optimizer(mu, tickers, sigma, curr_w, universe)

                if target_w:
                    for ticker in list(positions.keys()):
                        if ticker not in target_w:
                            pending_sells.add(ticker)
                    for ticker, tw in target_w.items():
                        diff = tw * nav - curr_w.get(ticker, 0.0) * nav
                        if diff > 500:
                            pending_buys[ticker] = diff

                    print(f"[layer2] {today.date()} {event} | NAV=${nav:,.0f} | "
                          f"Pos={len(positions)} | Target={len(target_w)}")

        if (day_idx + 1) % 1000 == 0:
            print(f"[layer2] Progress: {day_idx+1}/{len(trading_days)} | "
                  f"NAV=${nav:,.0f} | Positions={len(positions)}")

    nav_df   = pd.DataFrame(nav_history)
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df   = nav_df.set_index("date")
    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    print(f"\n[layer2] {label} done | Final NAV: ${nav_df['nav'].iloc[-1]:,.0f} | "
          f"Trades: {len(trade_df)}")

    return nav_df, trade_df


# ============================================================
# RUN ALL 4 STRATEGIES
# ============================================================

STRATEGIES = [
    ("monthly",   "mv"),
    ("quarterly", "mv"),
    ("monthly",   "bl"),
    ("quarterly", "bl"),
]

for frequency, strategy in STRATEGIES:
    file_label   = f"{strategy}_{frequency}"
    nav_df, trade_df = run_simulation(frequency, strategy)

    nav_df.to_parquet(RESULTS_DIR / f"nav_{file_label}.parquet")
    if not trade_df.empty:
        trade_df.to_parquet(RESULTS_DIR / f"trades_{file_label}.parquet")
    print(f"[layer2] Saved: nav_{file_label}.parquet | trades_{file_label}.parquet")

print("\n[layer2] ================================================")
print("[layer2] All 4 strategies complete. Ready for Layer 3.")
print("[layer2] ================================================")