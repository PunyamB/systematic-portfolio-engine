# notebooks/wf_layer2_simulate.py
# WALK-FORWARD LAYER 2 — Test Window Simulation
#
# Reads walk_forward_log.json to find the current window and its
# calibrated parameters (set by Claude after Layer 0 output).
# Simulates ALL 4 strategies on the test window only using
# parameters calibrated on the training window.
# Saves test-window NAV to data/backtest/wf_results/
#
# Run: python notebooks/wf_layer2_simulate.py

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRECOMP_DIR = Path("data/backtest/precomputed")
BACKTEST_DIR = Path("data/backtest")
WF_RESULTS  = Path("data/backtest/wf_results")
WF_RESULTS.mkdir(parents=True, exist_ok=True)
LOG_FILE    = Path("walk_forward_log.json")

SIGNAL_COLS = [
    "momentum_12_1", "earnings_momentum", "pe_zscore", "pb_zscore",
    "ev_ebitda_zscore", "roe_stability", "gross_margin_trend",
    "piotroski", "earnings_accruals", "short_term_reversal", "rsi_extremes",
]

# ============================================================
# LOAD LOG — FIND CURRENT WINDOW
# ============================================================

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

current_window = None
for w in log["windows"]:
    if w["status"] == "calibrated":
        current_window = w
        break
if current_window is None and log["final_holdout"]["status"] == "calibrated":
    current_window = log["final_holdout"]

if current_window is None:
    print("[wf_layer2] All windows complete.")
    sys.exit(0)

if current_window["calibrated_params"] is None:
    print("[wf_layer2] ERROR: No calibrated params found for this window.")
    print("[wf_layer2] Run wf_layer0_calibrate.py first, then have Claude update walk_forward_log.json.")
    sys.exit(1)

WINDOW_ID      = current_window["window_id"]
TRAIN_START    = pd.Timestamp(current_window["train_start"])
TRAIN_END      = pd.Timestamp(current_window["train_end"])
TEST_START     = pd.Timestamp(current_window["test_start"])
TEST_END       = pd.Timestamp(current_window["test_end"])
params         = current_window["calibrated_params"]
TURNOVER_LAMBDA = params["lambda"]
RISK_AVERSION   = params["risk_aversion"]
IC_WEIGHTED     = current_window.get("ic_weighted_signals", False)
SIGNAL_WEIGHTS  = params.get("signal_weights", None)  # None = equal weight

# BL parameters (fixed, not swept)
BL_TAU             = 0.05
BL_VIEW_CONFIDENCE = 0.25

# Execution
MAX_WEIGHT       = 0.05
COST_BPS_ONE_WAY = 5
TE_CAP           = 0.06 ** 2
INITIAL_CAPITAL  = 1_000_000.0

# Trailing stop
VOL_LOOKBACK  = 25
VOL_MULTIPLIER = 2.0
STOP_FLOOR    = 0.05
STOP_CAP      = 0.20

print("[wf_layer2] ================================================")
print(f"[wf_layer2] Window {WINDOW_ID} | Test: {TEST_START.date()} to {TEST_END.date()}")
print(f"[wf_layer2] Params: lambda={TURNOVER_LAMBDA} | risk_aversion={RISK_AVERSION}")
print(f"[wf_layer2] IC-weighted signals: {IC_WEIGHTED}")
print(f"[wf_layer2] Strategies: MV-Monthly | MV-Quarterly | BL-Monthly | BL-Quarterly")
print("[wf_layer2] ================================================\n")


# ============================================================
# LOAD DATA
# ============================================================

print("[wf_layer2] Loading precomputed data...")

signals_history = pd.read_parquet(PRECOMP_DIR / "signals_history.parquet")
signals_history["date"] = pd.to_datetime(signals_history["date"])

with open(PRECOMP_DIR / "covariance_matrices.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

# Load prices — filter to window + VOL_LOOKBACK buffer for vol computation
prices_raw = pd.read_parquet(BACKTEST_DIR / "prices.parquet")
prices_raw["date"] = pd.to_datetime(prices_raw["date"])
prices_raw = prices_raw.sort_values(["ticker", "date"])

# Trading calendar for test window
aapl         = prices_raw[prices_raw["ticker"] == "AAPL"].sort_values("date")
trading_days = aapl[
    (aapl["date"] >= TEST_START) & (aapl["date"] < TEST_END)
]["date"].tolist()

print(f"[wf_layer2] Test window trading days: {len(trading_days)}")

# --- Price lookup index: (date, ticker) -> {open, high, low, close} ---
# Built once, O(1) access during simulation. Covers test window only.
print("[wf_layer2] Building price lookup index...")
prices_test = prices_raw[
    (prices_raw["date"] >= TEST_START) & (prices_raw["date"] < TEST_END)
].copy()

price_index = {}
for row in prices_test[["date", "ticker", "open", "high", "low", "close"]].itertuples(index=False):
    price_index[(row.date, row.ticker)] = {
        "open": row.open, "high": row.high, "low": row.low, "close": row.close
    }
print(f"[wf_layer2] Price index: {len(price_index):,} entries")

# --- Precompute trailing stop volatility for all tickers across test window ---
# Uses prices from VOL_LOOKBACK days before test start through test end.
# Avoids per-ticker per-day full DataFrame scans inside the simulation loop.
print("[wf_layer2] Precomputing trailing stop volatility...")
vol_start    = prices_raw["date"].unique()
vol_start    = sorted(vol_start[vol_start < TEST_START])
vol_start    = vol_start[-(VOL_LOOKBACK + 5):][0] if len(vol_start) >= VOL_LOOKBACK + 5 \
               else TEST_START

prices_vol   = prices_raw[
    (prices_raw["date"] >= vol_start) & (prices_raw["date"] < TEST_END)
][["date", "ticker", "close"]].copy()

# For each ticker, compute daily rolling vol (std of pct_change over VOL_LOOKBACK days)
# Result: dict {(date, ticker): vol_distance}  where vol_distance = multiplier * vol * sqrt(lookback)
vol_index = {}
for ticker, grp in prices_vol.groupby("ticker"):
    grp   = grp.sort_values("date")
    pct   = grp["close"].pct_change()
    roll  = pct.rolling(VOL_LOOKBACK).std() * np.sqrt(VOL_LOOKBACK) * VOL_MULTIPLIER
    roll  = roll.clip(STOP_FLOOR, STOP_CAP)
    for dt, dist in zip(grp["date"], roll):
        if not np.isnan(dist):
            vol_index[(dt, ticker)] = float(dist)

del prices_raw, prices_vol, prices_test  # free memory
print(f"[wf_layer2] Vol index: {len(vol_index):,} entries\n")


# ============================================================
# SIGNAL WEIGHT VECTOR
# ============================================================

def get_signal_weights() -> dict:
    if SIGNAL_WEIGHTS and IC_WEIGHTED:
        return SIGNAL_WEIGHTS
    return {s: 1.0 / len(SIGNAL_COLS) for s in SIGNAL_COLS}


def apply_signal_weights(sig_df: pd.DataFrame, sw: dict) -> pd.DataFrame:
    """Recompute composite_score using calibrated signal weights."""
    sig_df = sig_df.copy()
    cols_present = [c for c in SIGNAL_COLS if c in sig_df.columns]
    weights = np.array([sw.get(c, 0.0) for c in cols_present])
    total   = weights.sum()
    if total > 0:
        weights = weights / total
    sig_df["composite_score"] = sig_df[cols_present].fillna(0).values @ weights
    sig_df["composite_rank"]  = sig_df["composite_score"].rank(ascending=False)
    return sig_df


# ============================================================
# EXPECTED RETURN FUNCTIONS
# ============================================================

def compute_mu_mv(sig_subset: pd.DataFrame, tickers: list) -> np.ndarray:
    score_map = sig_subset.set_index("ticker")["composite_score"]
    mu        = np.array([score_map.get(t, 0.0) for t in tickers])
    return mu - mu.min() + 0.01


def compute_mu_bl(sig_subset: pd.DataFrame, tickers: list,
                  sigma: np.ndarray) -> np.ndarray:
    n    = len(tickers)
    w_eq = np.ones(n) / n
    pi   = RISK_AVERSION * sigma @ w_eq

    score_map = sig_subset.set_index("ticker")["composite_score"]
    raw = np.array([score_map.get(t, 0.0) for t in tickers])
    Q   = (raw - raw.mean()) / (raw.std() + 1e-8)

    P     = np.eye(n)
    tau_s = BL_TAU * sigma
    view_var = (1.0 - BL_VIEW_CONFIDENCE) / (BL_VIEW_CONFIDENCE + 1e-8)
    omega    = np.diag(np.diag(P @ tau_s @ P.T) * view_var)

    try:
        reg       = np.eye(n) * 1e-6
        tau_s_inv = np.linalg.inv(tau_s + reg)
        omega_inv = np.linalg.inv(omega + np.eye(n) * 1e-8)
        M         = tau_s_inv + P.T @ omega_inv @ P
        mu_bl     = np.linalg.inv(M + reg) @ (tau_s_inv @ pi + P.T @ omega_inv @ Q)
    except np.linalg.LinAlgError:
        mu_bl = pi

    return mu_bl - mu_bl.min() + 0.01


# ============================================================
# OPTIMIZER
# ============================================================

def run_optimizer(mu: np.ndarray, tickers: list, sigma: np.ndarray,
                  current_weights: dict) -> dict:
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

    prob = cp.Problem(
        cp.Maximize(ret_ - RISK_AVERSION * risk - penalty),
        [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT, te <= TE_CAP]
    )
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


def prepare_inputs(sig_today: pd.DataFrame, cov: pd.DataFrame) -> tuple:
    if sig_today.empty or cov.empty:
        return [], np.array([]), pd.DataFrame()
    n_total  = len(sig_today)
    top_half = sig_today.nsmallest(max(10, n_total // 2), "composite_rank")
    tickers  = [t for t in top_half["ticker"].tolist() if t in cov.columns]
    if len(tickers) < 5:
        return [], np.array([]), pd.DataFrame()
    return tickers, cov.loc[tickers, tickers].values, top_half


# ============================================================
# REBALANCE CALENDAR
# ============================================================

def get_rebalance_dates(tdays: list, frequency: str) -> set:
    seen, dates = set(), set()
    for day in tdays:
        period = (day.year, day.month) if frequency == "monthly" \
                 else (day.year, (day.month - 1) // 3 + 1)
        if period not in seen:
            seen.add(period)
            dates.add(day)
    return dates


# ============================================================
# SIMULATION
# ============================================================

def get_px(day, ticker, col):
    px = price_index.get((day, ticker))
    return px[col] if px and px[col] and px[col] == px[col] else None  # nan check


def run_simulation(frequency: str, strategy: str) -> tuple:
    label = f"{strategy.upper()}-{frequency}"
    print(f"\n[wf_layer2] Simulating: {label}")

    sw              = get_signal_weights()
    rebalance_dates = get_rebalance_dates(trading_days, frequency)
    print(f"[wf_layer2] Test trading days: {len(trading_days)} | "
          f"Rebalance dates in test: {len(rebalance_dates)}")

    # Warm-start positions from end of training window
    # (start each test window fully invested based on last training rebalance)
    cash                = INITIAL_CAPITAL
    positions           = {}
    stop_levels         = {}
    recent_highs        = {}
    entry_prices        = {}
    pending_buys        = {}
    pending_sells       = set()
    sl_replace_dates    = set()
    nav_history         = []
    trade_log           = []
    portfolio_snapshots = []  # full portfolio state at every rebalance date

    for day_idx, today in enumerate(trading_days):

        # 1. Stop checks
        sl_today = []
        for ticker in list(positions.keys()):
            if ticker not in stop_levels:
                continue
            low = get_px(today, ticker, "low")
            if low is not None and low <= stop_levels[ticker]:
                sl_today.append(ticker)
        for ticker in sl_today:
            pending_sells.add(ticker)
        if sl_today:
            t2_idx = day_idx + 2
            if t2_idx < len(trading_days):
                sl_replace_dates.add(trading_days[t2_idx])

        # 2. Sells
        for ticker in list(pending_sells):
            open_px = get_px(today, ticker, "open")
            if open_px is None or open_px <= 0:
                pending_sells.discard(ticker)
                continue
            shares = positions.pop(ticker, 0)
            if shares > 0:
                cash += shares * open_px * (1 - COST_BPS_ONE_WAY / 10000)
                trade_log.append({"date": today, "ticker": ticker, "action": "sell",
                                   "shares": shares, "price": open_px})
            stop_levels.pop(ticker, None)
            recent_highs.pop(ticker, None)
            entry_prices.pop(ticker, None)
            pending_sells.discard(ticker)

        # 3. Buys
        for ticker, dollar_amt in list(pending_buys.items()):
            open_px = get_px(today, ticker, "open")
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
                                   "shares": shares, "price": open_px})
            del pending_buys[ticker]

        # 4. Valuation
        equity = sum(
            shares * (get_px(today, t, "close") or entry_prices.get(t, 0))
            for t, shares in positions.items()
        )
        nav = cash + equity
        nav_history.append({"date": today, "nav": nav})

        # 5. Trailing stops update — O(1) lookup from precomputed vol_index
        for ticker in list(positions.keys()):
            close = get_px(today, ticker, "close")
            if close is None:
                continue
            recent_highs[ticker] = max(recent_highs.get(ticker, close), close)
            dist = vol_index.get((today, ticker), STOP_FLOOR)
            stop_levels[ticker] = recent_highs[ticker] * (1 - dist)

        # 6. Optimizer
        if today in rebalance_dates or today in sl_replace_dates:
            event     = "rebalance" if today in rebalance_dates else "sl_replace"
            sig_today = signals_history[signals_history["date"] == today]
            if sig_today.empty:
                prior = signals_history[signals_history["date"] < today]
                if not prior.empty:
                    sig_today = prior[prior["date"] == prior["date"].max()]

            sig_today = apply_signal_weights(sig_today, sw)
            cov       = cov_matrices.get(today, pd.DataFrame())

            total_nav = nav if nav > 0 else 1.0
            curr_w    = {
                t: (s * (get_px(today, t, "close") or 0)) / total_nav
                for t, s in positions.items()
            }

            tickers, sigma, sig_subset = prepare_inputs(sig_today, cov)

            if len(tickers) >= 5:
                mu = compute_mu_mv(sig_subset, tickers) if strategy == "mv" \
                     else compute_mu_bl(sig_subset, tickers, sigma)

                target_w = run_optimizer(mu, tickers, sigma, curr_w)

                if target_w:
                    for ticker in list(positions.keys()):
                        if ticker not in target_w:
                            pending_sells.add(ticker)
                    for ticker, tw in target_w.items():
                        diff = tw * nav - curr_w.get(ticker, 0.0) * nav
                        if diff > 500:
                            pending_buys[ticker] = diff

                    # Portfolio snapshot — full state at this rebalance date
                    score_map = sig_subset.set_index("ticker")["composite_score"] \
                                if not sig_subset.empty else {}
                    for ticker, tw in target_w.items():
                        portfolio_snapshots.append({
                            "date":            today,
                            "ticker":          ticker,
                            "target_weight":   round(tw, 6),
                            "current_weight":  round(curr_w.get(ticker, 0.0), 6),
                            "weight_change":   round(tw - curr_w.get(ticker, 0.0), 6),
                            "composite_score": round(float(score_map.get(ticker, 0.0)), 4)
                                               if hasattr(score_map, "get") else 0.0,
                            "nav":             round(nav, 2),
                            "event":           event,
                            "strategy":        label,
                            "window_id":       WINDOW_ID,
                        })

                    print(f"[wf_layer2] {today.date()} {event} | "
                          f"NAV=${nav:,.0f} | Pos={len(positions)} | Target={len(target_w)}")

    nav_df   = pd.DataFrame(nav_history)
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df   = nav_df.set_index("date")
    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    snap_df  = pd.DataFrame(portfolio_snapshots) if portfolio_snapshots else pd.DataFrame()

    print(f"[wf_layer2] {label} done | Final NAV: ${nav_df['nav'].iloc[-1]:,.0f} | "
          f"Trades: {len(trade_df)} | Snapshots: {len(snap_df)}")

    return nav_df, trade_df, snap_df


# ============================================================
# RUN ALL 4 STRATEGIES ON TEST WINDOW
# ============================================================

STRATEGIES = [
    ("monthly",   "mv"),
    ("quarterly", "mv"),
    ("monthly",   "bl"),
    ("quarterly", "bl"),
]

for frequency, strategy in STRATEGIES:
    file_key                  = f"window_{WINDOW_ID}_{strategy}_{frequency}"
    nav_df, trade_df, snap_df = run_simulation(frequency, strategy)

    nav_df.to_parquet(WF_RESULTS / f"nav_{file_key}.parquet")
    if not trade_df.empty:
        trade_df.to_parquet(WF_RESULTS / f"trades_{file_key}.parquet")
    if not snap_df.empty:
        snap_df.to_parquet(WF_RESULTS / f"portfolios_{file_key}.parquet")
    print(f"[wf_layer2] Saved: nav | trades | portfolios for {file_key}")

print(f"\n[wf_layer2] ================================================")
print(f"[wf_layer2] Window {WINDOW_ID} test simulation complete.")
print(f"[wf_layer2] Files saved to: {WF_RESULTS}")
print(f"[wf_layer2] Paste this output to Claude to proceed to next window.")
print(f"[wf_layer2] ================================================")