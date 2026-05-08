# notebooks/wf_layer0_calibrate.py
# WALK-FORWARD LAYER 0 — Calibration (optimized)
#
# Key optimization: all price lookups, signal prep, tickers, sigma, and mu
# are precomputed ONCE before the sweep. Each combo only runs the optimizer.
# This reduces runtime from ~2hrs to ~5-10 mins per window.
#
# Run: python notebooks/wf_layer0_calibrate.py

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path
from itertools import product

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRECOMP_DIR  = Path("data/backtest/precomputed")
BACKTEST_DIR = Path("data/backtest")
LOG_FILE     = Path("walk_forward_log.json")

# ============================================================
# LOAD LOG — FIND CURRENT WINDOW
# ============================================================

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

current_window = None
for w in log["windows"]:
    if w["status"] == "pending":
        current_window = w
        break

if current_window is None:
    if log["final_holdout"]["status"] == "pending":
        current_window = log["final_holdout"]
    else:
        print("[wf_layer0] All windows complete. Nothing to calibrate.")
        sys.exit(0)

TRAIN_START  = pd.Timestamp(current_window["train_start"])
TRAIN_END    = pd.Timestamp(current_window["train_end"])
WINDOW_ID    = current_window["window_id"]
IC_THRESHOLD = log["ic_threshold_rebalances"]
LAMBDA_SWEEP = log["lambda_sweep_values"]
RA_SWEEP     = log["risk_aversion_sweep_values"]
MAX_WEIGHT   = 0.05
TE_CAP       = 0.06 ** 2
COST_BPS     = 5

print("[wf_layer0] ================================================")
print(f"[wf_layer0] Window {WINDOW_ID} | Train: {TRAIN_START.date()} to {TRAIN_END.date()}")
print("[wf_layer0] ================================================\n")

SIGNAL_COLS = [
    "momentum_12_1", "earnings_momentum", "pe_zscore", "pb_zscore",
    "ev_ebitda_zscore", "roe_stability", "gross_margin_trend",
    "piotroski", "earnings_accruals", "short_term_reversal", "rsi_extremes",
]

# ============================================================
# LOAD + FILTER DATA
# ============================================================

print("[wf_layer0] Loading precomputed data...")

signals_history = pd.read_parquet(PRECOMP_DIR / "signals_history.parquet")
signals_history["date"] = pd.to_datetime(signals_history["date"])

fwd_returns = pd.read_parquet(PRECOMP_DIR / "forward_returns.parquet")
fwd_returns["date"] = pd.to_datetime(fwd_returns["date"])

with open(PRECOMP_DIR / "covariance_matrices.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

# Load only training window prices — much smaller than full 6M row file
prices_raw = pd.read_parquet(BACKTEST_DIR / "prices.parquet")
prices_raw["date"] = pd.to_datetime(prices_raw["date"])
prices_train = prices_raw[
    (prices_raw["date"] >= TRAIN_START) &
    (prices_raw["date"] <= TRAIN_END)
].copy()
del prices_raw  # free memory immediately

sig_train = signals_history[
    (signals_history["date"] >= TRAIN_START) &
    (signals_history["date"] <= TRAIN_END)
].copy()

fwd_train = fwd_returns[
    (fwd_returns["date"] >= TRAIN_START) &
    (fwd_returns["date"] <= TRAIN_END)
].copy()

train_dates  = sorted(sig_train["date"].unique())
n_rebalances = len(train_dates)

print(f"[wf_layer0] Training rebalances: {n_rebalances}")
print(f"[wf_layer0] IC-weighted signals eligible: {n_rebalances >= IC_THRESHOLD} "
      f"(threshold={IC_THRESHOLD})")

# ============================================================
# PRICE LOOKUP DICT — built once, O(1) access forever
# ============================================================

print("[wf_layer0] Building price lookup index...")
price_index = {}
for row in prices_train[["date", "ticker", "close"]].itertuples(index=False):
    price_index[(row.date, row.ticker)] = row.close

print(f"[wf_layer0] Price index: {len(price_index):,} entries\n")


# ============================================================
# SIGNAL IC COMPUTATION
# ============================================================

print("[wf_layer0] Computing per-signal IC...")

# Pre-index signals and forward returns by date for fast lookup
sig_by_date = {dt: grp for dt, grp in sig_train.groupby("date")}
fwd_by_date = {dt: grp[["ticker", "fwd_1m"]].dropna() for dt, grp in fwd_train.groupby("date")}

ic_results = {}
for sig in SIGNAL_COLS:
    if sig not in sig_train.columns:
        continue
    ics = []
    for dt in train_dates:
        sig_dt = sig_by_date.get(dt, pd.DataFrame())
        fwd_dt = fwd_by_date.get(dt, pd.DataFrame())
        if sig_dt.empty or fwd_dt.empty:
            continue
        merged = sig_dt[["ticker", sig]].dropna().merge(fwd_dt, on="ticker")
        if len(merged) < 20:
            continue
        ic = merged[sig].corr(merged["fwd_1m"], method="spearman")
        if not np.isnan(ic):
            ics.append(ic)
    if ics:
        ic_results[sig] = {
            "mean_ic":      round(np.mean(ics), 4),
            "ic_ir":        round(np.mean(ics) / (np.std(ics) + 1e-8), 3),
            "n_obs":        len(ics),
            "pct_positive": round((np.array(ics) > 0).mean() * 100, 1),
        }

print("\n[wf_layer0] SIGNAL IC TABLE")
print("-" * 65)
print(f"{'Signal':<25} {'Mean IC':>10} {'IC-IR':>10} {'N Obs':>8} {'% Positive':>12}")
print("-" * 65)
for sig, vals in sorted(ic_results.items(), key=lambda x: abs(x[1]["mean_ic"]), reverse=True):
    print(f"{sig:<25} {vals['mean_ic']:>10} {vals['ic_ir']:>10} "
          f"{vals['n_obs']:>8} {vals['pct_positive']:>11}%")
print("-" * 65)

if n_rebalances >= IC_THRESHOLD:
    ic_weights = {s: max(0.0, v["mean_ic"]) for s, v in ic_results.items()}
    total_w    = sum(ic_weights.values())
    ic_weights = {s: round(v / total_w, 4) for s, v in ic_weights.items()} if total_w > 0 else ic_weights
    print(f"\n[wf_layer0] IC-WEIGHTED SIGNAL WEIGHTS (n={n_rebalances} >= {IC_THRESHOLD})")
    for sig, w in sorted(ic_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sig:<25} {w:.4f}")
else:
    print(f"\n[wf_layer0] Equal-weight signals (n={n_rebalances} < {IC_THRESHOLD} threshold)")
    ic_weights = {sig: round(1.0 / len(SIGNAL_COLS), 4) for sig in SIGNAL_COLS}


# ============================================================
# PRECOMPUTE OPTIMIZER INPUTS FOR ALL TRAINING DATES
# — done ONCE, shared across all 24 sweep combos
# ============================================================

print("\n[wf_layer0] Precomputing optimizer inputs for all training dates...")

precomputed = []  # list of dicts, one per rebalance date

sig_cols_present = [c for c in SIGNAL_COLS if c in sig_train.columns]
weights_arr = np.array([ic_weights.get(c, 1.0 / len(sig_cols_present)) for c in sig_cols_present])
weights_arr = weights_arr / weights_arr.sum()

for dt in train_dates:
    sig_today = sig_by_date.get(dt, pd.DataFrame())
    cov       = cov_matrices.get(dt, pd.DataFrame())

    if sig_today.empty or cov.empty:
        continue

    # Weighted composite score
    sig_today = sig_today.copy()
    sig_today["weighted_score"] = sig_today[sig_cols_present].fillna(0).values @ weights_arr

    n_total  = len(sig_today)
    top_half = sig_today.nsmallest(max(10, n_total // 2), "composite_rank")
    tickers  = [t for t in top_half["ticker"].tolist() if t in cov.columns]

    if len(tickers) < 5:
        continue

    sigma     = cov.loc[tickers, tickers].values
    score_map = top_half.set_index("ticker")["weighted_score"]
    mu        = np.array([score_map.get(t, 0.0) for t in tickers])
    mu        = mu - mu.min() + 0.01

    # Precompute close prices for this date (dict lookup, instant)
    closes = {t: price_index.get((dt, t)) for t in tickers}

    precomputed.append({
        "date":    dt,
        "tickers": tickers,
        "sigma":   sigma,
        "mu":      mu,
        "closes":  closes,
        "n":       len(tickers),
    })

print(f"[wf_layer0] Precomputed {len(precomputed)} valid rebalance dates\n")


# ============================================================
# PARAMETER SWEEP — optimizer only, no data loading inside loop
# ============================================================

print(f"[wf_layer0] Running sweep ({len(LAMBDA_SWEEP)} lambdas x "
      f"{len(RA_SWEEP)} risk_aversion = {len(LAMBDA_SWEEP) * len(RA_SWEEP)} combos)...")
print(f"[wf_layer0] Each combo runs optimizer on {len(precomputed)} dates only.\n")


def run_sweep_combo(lam: float, risk_aversion: float) -> dict | None:
    """
    Runs one parameter combo using precomputed inputs.
    No data loading, no DataFrame filtering — pure optimization.
    """
    cash      = 1_000_000.0
    positions = {}  # {ticker: shares}
    nav_list  = []

    for pc in precomputed:
        tickers = pc["tickers"]
        sigma   = pc["sigma"]
        mu      = pc["mu"]
        closes  = pc["closes"]
        n       = pc["n"]

        # Current weights from positions
        equity  = sum(shares * (closes.get(t) or 0) for t, shares in positions.items())
        nav     = cash + equity
        if nav <= 0:
            nav = cash

        w_curr = np.array([
            (positions.get(t, 0) * (closes.get(t) or 0)) / nav
            for t in tickers
        ])
        w_eq = np.ones(n) / n

        # Optimizer
        w       = cp.Variable(n)
        risk    = cp.quad_form(w, cp.psd_wrap(sigma))
        ret_    = mu @ w
        penalty = lam * cp.norm1(w - w_curr)
        te      = cp.quad_form(w - w_eq, cp.psd_wrap(sigma))

        prob = cp.Problem(
            cp.Maximize(ret_ - risk_aversion * risk - penalty),
            [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT, te <= TE_CAP]
        )
        solved = False
        for solver in [cp.CLARABEL, cp.SCS]:
            try:
                prob.solve(solver=solver, warm_start=True)
                if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                    solved = True
                    break
            except Exception:
                continue

        if not solved:
            continue

        target_w = {tickers[i]: float(max(0, w.value[i])) for i in range(n)}
        total    = sum(target_w.values())
        if total == 0:
            continue
        target_w = {t: v / total for t, v in target_w.items() if v > 1e-4}

        # Simplified execution at close prices
        for ticker in list(positions.keys()):
            if ticker not in target_w:
                close = closes.get(ticker)
                if close:
                    cash += positions.pop(ticker) * close * (1 - COST_BPS / 10000)

        for ticker, tw in target_w.items():
            target_val = tw * nav
            curr_val   = positions.get(ticker, 0) * (closes.get(ticker) or 0)
            diff       = target_val - curr_val
            if diff > 500:
                close = closes.get(ticker)
                if close and close > 0:
                    spend             = min(diff, cash * 0.99)
                    positions[ticker] = positions.get(ticker, 0) + \
                                        spend * (1 - COST_BPS / 10000) / close
                    cash             -= spend

        equity = sum(shares * (closes.get(t) or 0) for t, shares in positions.items())
        nav_list.append(cash + equity)

    if len(nav_list) < 6:
        return None

    navs   = pd.Series(nav_list)
    rets   = navs.pct_change().dropna()
    sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(12)
    dd     = (navs / navs.cummax()) - 1
    max_dd = dd.min()
    calmar = (navs.iloc[-1] / navs.iloc[0] - 1) / abs(max_dd) if max_dd != 0 else 0

    return {
        "sharpe":    round(float(sharpe), 3),
        "max_dd":    round(float(max_dd) * 100, 2),
        "calmar":    round(float(calmar), 3),
        "final_nav": round(float(navs.iloc[-1]), 0),
    }


sweep_results = []
total_combos  = len(LAMBDA_SWEEP) * len(RA_SWEEP)
done          = 0

for lam, ra in product(LAMBDA_SWEEP, RA_SWEEP):
    done += 1
    print(f"[wf_layer0] Sweep {done}/{total_combos} | lambda={lam} | risk_aversion={ra} ... ",
          end="", flush=True)
    result = run_sweep_combo(lam, ra)
    if result:
        sweep_results.append({"lambda": lam, "risk_aversion": ra, **result})
        print(f"Sharpe={result['sharpe']} | MaxDD={result['max_dd']}% | Calmar={result['calmar']}")
    else:
        print("insufficient data")


# ============================================================
# RESULTS
# ============================================================

if not sweep_results:
    print("[wf_layer0] No sweep results. Check data.")
    sys.exit(1)

sweep_df = pd.DataFrame(sweep_results)

print("\n[wf_layer0] PARAMETER SWEEP RESULTS (sorted by Sharpe)")
print("=" * 70)
print(f"{'Lambda':>10} {'RiskAversion':>14} {'Sharpe':>10} {'MaxDD%':>10} {'Calmar':>10}")
print("-" * 70)
for _, row in sweep_df.sort_values("sharpe", ascending=False).iterrows():
    print(f"{row['lambda']:>10} {row['risk_aversion']:>14} "
          f"{row['sharpe']:>10} {row['max_dd']:>10} {row['calmar']:>10}")
print("=" * 70)

best_sharpe = sweep_df.loc[sweep_df["sharpe"].idxmax()]
best_calmar = sweep_df.loc[sweep_df["calmar"].idxmax()]

print(f"\n[wf_layer0] Best by Sharpe: lambda={best_sharpe['lambda']} | "
      f"risk_aversion={best_sharpe['risk_aversion']} | Sharpe={best_sharpe['sharpe']}")
print(f"[wf_layer0] Best by Calmar: lambda={best_calmar['lambda']} | "
      f"risk_aversion={best_calmar['risk_aversion']} | Calmar={best_calmar['calmar']}")

print(f"\n[wf_layer0] ================================================")
print(f"[wf_layer0] Window {WINDOW_ID} calibration complete.")
print(f"[wf_layer0] Rebalances: {n_rebalances} | IC-weighted: {n_rebalances >= IC_THRESHOLD}")
print(f"[wf_layer0] Paste this output to Claude to get calibrated parameters.")
print(f"[wf_layer0] ================================================")