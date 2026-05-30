# experiments/exp008_sweep/run_exp008.py
# ---------------------------------------------------------------
# EXP008 Orchestrator — Automated Walk-Forward
#
# Combines SPE's Layer 0 (calibrate) + Layer 2 (simulate) into one
# automated pipeline. No manual JSON editing between windows.
#
# For each window:
#   1. Sweep lambda × risk_aversion on training data
#   2. Auto-pick best params (best Sharpe)
#   3. Simulate test window with daily execution + trailing stops
#   4. Save NAV, trades, portfolio snapshots
#   5. Move to next window
#
# After all windows: stitch + analytics (Layer 3).
#
# Changes from SPE:
#   - 15 signals (11 original + 4 new)
#   - Regime multipliers applied to signal weights
#   - VIX threshold 0.95 (SPE: 0.80)
#   - MV-monthly only (no BL, no quarterly)
#
# Run: python experiments/exp008_sweep/run_exp008.py
#
# Prerequisites: layer1_precompute.py must be run first.
# ---------------------------------------------------------------

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from itertools import product
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ============================================================
# PATHS
# ============================================================

EXP_DIR     = Path("experiments/exp008_sweep")
PRECOMP_DIR = Path("experiments/exp006_extended_wf/data/precomputed")
WF_RESULTS  = EXP_DIR / "data" / "wf_results" / "hl49_ms13"
RESULTS_DIR = EXP_DIR / "data" / "results" / "hl49_ms13"
LOG_FILE    = EXP_DIR / "walk_forward_log.json"

# SPE raw data for prices (trailing stop vol computation needs OHLC)
SPE_DIR      = Path("D:/Projects/SystematicPortfolioEngine")
BACKTEST_DIR = SPE_DIR / "data" / "backtest"

for d in [WF_RESULTS, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SIGNAL_COLS = [
    "momentum_12_1", "earnings_momentum", "pe_zscore", "pb_zscore",
    "ev_ebitda_zscore", "roe_stability", "gross_margin_trend",
    "piotroski", "earnings_accruals", "short_term_reversal", "rsi_extremes",
    "revenue_growth", "low_volatility", "fcf_yield", "volume_momentum",
]

# ============================================================
# REGIME DETECTION (SPE's rule-based, VIX threshold 0.95)
# ============================================================

VIX_ELEVATED_THRESHOLD = 0.95

# Sweep globals — set per combo
CURRENT_HALFLIFE = 49
CURRENT_MIN_SIGNALS = 13   # EXP008: was 0.80 in SPE
VIX_CRISIS_THRESHOLD   = 1.20
BREADTH_HIGH = 0.60
BREADTH_LOW  = 0.40

REGIME_MULTIPLIERS = {
    "bull": {
        "momentum_12_1": 1.3, "earnings_momentum": 1.2,
        "pe_zscore": 1.0, "pb_zscore": 1.0, "ev_ebitda_zscore": 1.0,
        "roe_stability": 1.0, "gross_margin_trend": 1.0,
        "piotroski": 1.0, "earnings_accruals": 1.0,
        "short_term_reversal": 0.7, "rsi_extremes": 0.7,
        "revenue_growth": 1.3, "low_volatility": 0.7,
        "fcf_yield": 1.0, "volume_momentum": 1.2,
    },
    "recovery": {
        "momentum_12_1": 1.1, "earnings_momentum": 1.1,
        "pe_zscore": 1.2, "pb_zscore": 1.2, "ev_ebitda_zscore": 1.2,
        "roe_stability": 1.1, "gross_margin_trend": 1.1,
        "piotroski": 1.1, "earnings_accruals": 1.1,
        "short_term_reversal": 1.0, "rsi_extremes": 1.0,
        "revenue_growth": 1.1, "low_volatility": 1.0,
        "fcf_yield": 1.1, "volume_momentum": 1.0,
    },
    "bear": {
        "momentum_12_1": 0.7, "earnings_momentum": 0.8,
        "pe_zscore": 1.2, "pb_zscore": 1.2, "ev_ebitda_zscore": 1.2,
        "roe_stability": 1.3, "gross_margin_trend": 1.1,
        "piotroski": 1.3, "earnings_accruals": 1.3,
        "short_term_reversal": 1.2, "rsi_extremes": 1.2,
        "revenue_growth": 0.8, "low_volatility": 1.3,
        "fcf_yield": 1.2, "volume_momentum": 0.8,
    },
    "crisis": {
        "momentum_12_1": 0.5, "earnings_momentum": 0.5,
        "pe_zscore": 1.0, "pb_zscore": 1.0, "ev_ebitda_zscore": 1.0,
        "roe_stability": 1.5, "gross_margin_trend": 1.0,
        "piotroski": 1.5, "earnings_accruals": 1.5,
        "short_term_reversal": 1.3, "rsi_extremes": 1.3,
        "revenue_growth": 0.5, "low_volatility": 1.5,
        "fcf_yield": 1.3, "volume_momentum": 0.5,
    },
}


def detect_regime(macro_features, breadth_data, as_of):
    """SPE's rule-based regime detector with VIX threshold 0.95."""
    macro_asof = macro_features[macro_features.index <= as_of]
    if macro_asof.empty:
        return "recovery"

    row = macro_asof.iloc[-1]

    # L1: Market Stress
    vix = row.get("vix_level")
    stress = "elevated"
    if vix is not None and not pd.isna(vix) and len(macro_asof) >= 252:
        avg_vix = float(macro_asof["vix_level"].tail(252).mean())
        vix_ratio = float(vix) / avg_vix if avg_vix > 0 else 1.0

        breadth_asof = breadth_data[breadth_data.index <= as_of]
        breadth_val = float(breadth_asof.iloc[-1].get("breadth_pct", 0.5)) if not breadth_asof.empty else 0.5

        if vix_ratio < VIX_ELEVATED_THRESHOLD and breadth_val > BREADTH_HIGH:
            stress = "low_stress"
        elif vix_ratio > VIX_CRISIS_THRESHOLD and breadth_val < BREADTH_LOW:
            stress = "crisis"

    # L2: Economic Cycle
    yield_spread = row.get("yield_curve_slope")
    curve_inverted = yield_spread is not None and not pd.isna(yield_spread) and yield_spread < 0

    credit_history = macro_asof["credit_spread"].dropna() if "credit_spread" in macro_asof.columns else pd.Series(dtype=float)
    credit_trend = "stable"
    if len(credit_history) >= 5:
        trend = float(credit_history.iloc[-1]) - float(credit_history.iloc[-5])
        credit_trend = "widening" if trend > 0 else "tightening"

    cycle = "contraction" if curve_inverted and credit_trend == "widening" else "expansion"

    # Composite
    if stress == "crisis":
        return "crisis"
    elif stress == "elevated" and cycle == "contraction":
        return "bear"
    elif stress == "low_stress" and cycle == "expansion":
        return "bull"
    else:
        return "recovery"


# ============================================================
# EXECUTION PARAMS (identical to SPE)
# ============================================================

MAX_WEIGHT       = 0.05
COST_BPS_ONE_WAY = 5
TE_CAP           = 0.06 ** 2
INITIAL_CAPITAL  = 1_000_000.0

# Trailing stop
VOL_LOOKBACK   = 25
VOL_MULTIPLIER = 2.0
STOP_FLOOR     = 0.05
STOP_CAP       = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("[exp008] Loading data...")

signals_history = pd.read_parquet(PRECOMP_DIR / "signals_history.parquet")
signals_history["date"] = pd.to_datetime(signals_history["date"])

fwd_returns = pd.read_parquet(PRECOMP_DIR / "forward_returns.parquet")
fwd_returns["date"] = pd.to_datetime(fwd_returns["date"])

with open(PRECOMP_DIR / "covariance_matrices.pkl", "rb") as f:
    cov_matrices = pickle.load(f)

prices_raw = pd.read_parquet(BACKTEST_DIR / "prices.parquet")
prices_raw["date"] = pd.to_datetime(prices_raw["date"])
prices_raw = prices_raw.sort_values(["ticker", "date"])

# Macro data for regime detection
macro_file = Path("data/raw/macro_features.parquet")
breadth_file = Path("data/raw/breadth.parquet")
macro_features = pd.read_parquet(macro_file)
macro_features.index = pd.to_datetime(macro_features.index)
macro_features = macro_features.sort_index()
breadth_data = pd.read_parquet(breadth_file)
breadth_data.index = pd.to_datetime(breadth_data.index)
breadth_data = breadth_data.sort_index()

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

print(f"[exp008] Signals: {signals_history.shape} | Fwd: {fwd_returns.shape}")
print(f"[exp008] Prices: {prices_raw.shape} | Cov matrices: {len(cov_matrices)}")
print(f"[exp008] Windows: {len(log['windows'])} + holdout\n")


# ============================================================
# IC COMPUTATION + SIGNAL WEIGHTS
# ============================================================

def compute_ic_weights(sig_train, fwd_train, train_dates, ewm_halflife=60):
    """IC-IR weights with EWM decay. Halflife in months."""
    sig_by_date = {dt: grp for dt, grp in sig_train.groupby("date")}
    fwd_by_date = {dt: grp[["ticker", "fwd_1m"]].dropna() for dt, grp in fwd_train.groupby("date")}

    ic_results = {}
    for sig in SIGNAL_COLS:
        if sig not in sig_train.columns:
            continue
        ic_series = []
        ic_idx = []
        for dt in train_dates:
            sig_dt = sig_by_date.get(dt, pd.DataFrame())
            fwd_dt = fwd_by_date.get(dt, pd.DataFrame())
            if sig_dt.empty or fwd_dt.empty: continue
            merged = sig_dt[["ticker", sig]].dropna().merge(fwd_dt, on="ticker")
            if len(merged) < 20: continue
            ic = merged[sig].corr(merged["fwd_1m"], method="spearman")
            if not np.isnan(ic):
                ic_series.append(ic)
                ic_idx.append(dt)
        if ic_series:
            ic_s = pd.Series(ic_series, index=ic_idx)
            ewm_mean = float(ic_s.ewm(halflife=ewm_halflife).mean().iloc[-1])
            ewm_std = float(ic_s.ewm(halflife=ewm_halflife).std().iloc[-1])
            ic_results[sig] = {"mean_ic": ewm_mean, "ic_ir": ewm_mean / (ewm_std + 1e-8)}

    weights = {s: max(0.0, v["mean_ic"]) for s, v in ic_results.items()}
    total = sum(weights.values())
    if total > 0:
        weights = {s: v / total for s, v in weights.items()}
    else:
        weights = {s: 1.0 / len(SIGNAL_COLS) for s in SIGNAL_COLS}

    return weights


def apply_regime_multipliers(weights, regime):
    """Apply regime multipliers and renormalize."""
    multipliers = REGIME_MULTIPLIERS.get(regime, {})
    for sig, mult in multipliers.items():
        if sig in weights:
            weights[sig] *= mult
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def apply_signal_weights(sig_df, sw):
    """Recompute composite_score using calibrated signal weights."""
    sig_df = sig_df.copy()
    cols_present = [c for c in SIGNAL_COLS if c in sig_df.columns]
    w_arr = np.array([sw.get(c, 0.0) for c in cols_present])
    total = w_arr.sum()
    if total > 0:
        w_arr = w_arr / total
    sig_df["composite_score"] = sig_df[cols_present].fillna(0).values @ w_arr
    sig_df["composite_rank"]  = sig_df["composite_score"].rank(ascending=False)
    return sig_df


# ============================================================
# OPTIMIZER (identical to SPE)
# ============================================================

def run_optimizer(mu, tickers, sigma, current_weights, turnover_lambda, risk_aversion):
    if len(tickers) < 5:
        return {}
    n      = len(tickers)
    w      = cp.Variable(n)
    w_curr = np.array([current_weights.get(t, 0.0) for t in tickers])
    w_eq   = np.ones(n) / n

    prob = cp.Problem(
        cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma))
                    - turnover_lambda * cp.norm1(w - w_curr)),
        [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT,
         cp.quad_form(w - w_eq, cp.psd_wrap(sigma)) <= TE_CAP]
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


def prepare_inputs(sig_today, cov, min_signals=13):
    if sig_today.empty or cov.empty:
        return [], np.array([]), pd.DataFrame()
    sig_cols_check = [c for c in SIGNAL_COLS if c in sig_today.columns]
    sig_today = sig_today.copy()
    sig_today["_n_valid"] = sig_today[sig_cols_check].notna().sum(axis=1)
    sig_today = sig_today[sig_today["_n_valid"] >= min_signals].drop(columns=["_n_valid"])
    if sig_today.empty:
        return [], np.array([]), pd.DataFrame()
    n_total  = len(sig_today)
    top_half = sig_today.nsmallest(max(10, n_total // 2), "composite_rank")
    tickers  = [t for t in top_half["ticker"].tolist() if t in cov.columns]
    if len(tickers) < 5:
        return [], np.array([]), pd.DataFrame()
    return tickers, cov.loc[tickers, tickers].values, top_half


def get_rebalance_dates(tdays, frequency="monthly"):
    seen, dates = set(), set()
    for day in tdays:
        period = (day.year, day.month)
        if period not in seen:
            seen.add(period)
            dates.add(day)
    return dates


# ============================================================
# LAYER 0: CALIBRATION (per window)
# ============================================================

def calibrate_window(window):
    """Sweep lambda × risk_aversion, return best params."""
    train_start = pd.Timestamp(window["train_start"])
    train_end   = pd.Timestamp(window["train_end"])
    wid         = window["window_id"]

    sig_train = signals_history[
        (signals_history["date"] >= train_start) & (signals_history["date"] <= train_end)
    ].copy()
    fwd_train = fwd_returns[
        (fwd_returns["date"] >= train_start) & (fwd_returns["date"] <= train_end)
    ].copy()
    train_dates = sorted(sig_train["date"].unique())

    ic_weighted = window.get("ic_weighted_signals", True) and len(train_dates) >= log["ic_threshold_rebalances"]
    if ic_weighted:
        base_weights = compute_ic_weights(sig_train, fwd_train, train_dates, ewm_halflife=CURRENT_HALFLIFE)
    else:
        base_weights = {s: 1.0 / len(SIGNAL_COLS) for s in SIGNAL_COLS}

    # Precompute optimizer inputs
    sig_cols_present = [c for c in SIGNAL_COLS if c in sig_train.columns]
    w_arr = np.array([base_weights.get(c, 1.0/len(sig_cols_present)) for c in sig_cols_present])
    w_arr = w_arr / w_arr.sum()

    # Load training prices for simplified P&L
    prices_train = prices_raw[
        (prices_raw["date"] >= train_start) & (prices_raw["date"] <= train_end)
    ]
    price_idx = {}
    for row in prices_train[["date", "ticker", "close"]].itertuples(index=False):
        price_idx[(row.date, row.ticker)] = row.close

    precomputed = []
    for dt in train_dates:
        sig_today = sig_train[sig_train["date"] == dt].copy()
        cov = cov_matrices.get(dt, pd.DataFrame())
        if sig_today.empty or cov.empty:
            continue

        # Detect regime and apply multipliers
        regime = detect_regime(macro_features, breadth_data, dt)
        regime_weights = apply_regime_multipliers(dict(base_weights), regime)
        rw_arr = np.array([regime_weights.get(c, 1.0/len(sig_cols_present)) for c in sig_cols_present])
        rw_arr = rw_arr / rw_arr.sum()

        sig_today["weighted_score"] = sig_today[sig_cols_present].fillna(0).values @ rw_arr
        n_total  = len(sig_today)
        top_half = sig_today.nsmallest(max(10, n_total // 2), "composite_rank")
        tickers  = [t for t in top_half["ticker"].tolist() if t in cov.columns]
        if len(tickers) < 5:
            continue

        sigma     = cov.loc[tickers, tickers].values
        score_map = top_half.set_index("ticker")["weighted_score"]
        mu        = np.array([score_map.get(t, 0.0) for t in tickers])
        mu        = mu - mu.min() + 0.01
        closes    = {t: price_idx.get((dt, t)) for t in tickers}

        precomputed.append({"date": dt, "tickers": tickers, "sigma": sigma,
                            "mu": mu, "closes": closes, "n": len(tickers), "regime": regime})

    print(f"[exp008] Window {wid}: {len(precomputed)} valid training dates | IC-weighted: {ic_weighted}")

    # Sweep
    best_result = None
    best_sharpe = -999
    LAMBDA_SWEEP = log["lambda_sweep_values"]
    RA_SWEEP     = log["risk_aversion_sweep_values"]

    for lam, ra in product(LAMBDA_SWEEP, RA_SWEEP):
        cash, positions, nav_list = 1_000_000.0, {}, []
        for pc in precomputed:
            tickers, sigma, mu, closes = pc["tickers"], pc["sigma"], pc["mu"], pc["closes"]
            n = pc["n"]
            equity = sum(shares * (closes.get(t) or 0) for t, shares in positions.items())
            nav = cash + equity
            if nav <= 0: nav = cash
            w_curr = np.array([(positions.get(t, 0) * (closes.get(t) or 0)) / nav for t in tickers])

            w = cp.Variable(n)
            w_eq = np.ones(n) / n
            prob = cp.Problem(
                cp.Maximize(mu @ w - ra * cp.quad_form(w, cp.psd_wrap(sigma)) - lam * cp.norm1(w - w_curr)),
                [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT, cp.quad_form(w - w_eq, cp.psd_wrap(sigma)) <= TE_CAP]
            )
            solved = False
            for solver in [cp.CLARABEL, cp.SCS]:
                try:
                    prob.solve(solver=solver, warm_start=True)
                    if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                        solved = True; break
                except Exception:
                    continue
            if not solved: continue

            target_w = {tickers[i]: float(max(0, w.value[i])) for i in range(n)}
            total = sum(target_w.values())
            if total == 0: continue
            target_w = {t: v/total for t, v in target_w.items() if v > 1e-4}

            for ticker in list(positions.keys()):
                if ticker not in target_w:
                    close = closes.get(ticker)
                    if close: cash += positions.pop(ticker) * close * (1 - COST_BPS_ONE_WAY/10000)
            for ticker, tw in target_w.items():
                target_val = tw * nav
                curr_val = positions.get(ticker, 0) * (closes.get(ticker) or 0)
                diff = target_val - curr_val
                if diff > 500:
                    close = closes.get(ticker)
                    if close and close > 0:
                        spend = min(diff, cash * 0.99)
                        positions[ticker] = positions.get(ticker, 0) + spend * (1-COST_BPS_ONE_WAY/10000) / close
                        cash -= spend

            equity = sum(shares * (closes.get(t) or 0) for t, shares in positions.items())
            nav_list.append(cash + equity)

        if len(nav_list) < 6: continue
        navs = pd.Series(nav_list)
        rets = navs.pct_change().dropna()
        sharpe = float((rets.mean() / (rets.std() + 1e-8)) * np.sqrt(12))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            dd = (navs / navs.cummax()) - 1
            best_result = {
                "lambda": lam, "risk_aversion": ra, "sharpe": round(sharpe, 3),
                "max_dd": round(float(dd.min()) * 100, 2),
                "signal_weights": {s: round(v, 4) for s, v in base_weights.items()} if ic_weighted else None,
            }

    if best_result:
        print(f"[exp008] Window {wid}: Best lambda={best_result['lambda']} ra={best_result['risk_aversion']} "
              f"Sharpe={best_result['sharpe']} MaxDD={best_result['max_dd']}%")
    return best_result, ic_weighted


# ============================================================
# LAYER 2: SIMULATION (per window, with trailing stops)
# ============================================================

def simulate_window(window, turnover_lambda, risk_aversion, signal_weights, ic_weighted):
    """Daily simulation with trailing stops — identical to SPE Layer 2."""
    test_start = pd.Timestamp(window["test_start"])
    test_end   = pd.Timestamp(window["test_end"])
    wid        = window["window_id"]

    # Trading calendar
    aapl = prices_raw[prices_raw["ticker"] == "AAPL"].sort_values("date")
    trading_days = aapl[(aapl["date"] >= test_start) & (aapl["date"] < test_end)]["date"].tolist()

    if not trading_days:
        print(f"[exp008] Window {wid}: no trading days!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build signal weight dict
    if signal_weights and ic_weighted:
        sw = signal_weights
    else:
        sw = {s: 1.0 / len(SIGNAL_COLS) for s in SIGNAL_COLS}

    # Price index for test window
    prices_test = prices_raw[(prices_raw["date"] >= test_start) & (prices_raw["date"] < test_end)]
    price_index = {}
    for row in prices_test[["date", "ticker", "open", "high", "low", "close"]].itertuples(index=False):
        price_index[(row.date, row.ticker)] = {"open": row.open, "high": row.high, "low": row.low, "close": row.close}

    # Trailing stop vol index
    vol_start = sorted(prices_raw["date"].unique())
    vol_start = [d for d in vol_start if d < test_start]
    vol_start = vol_start[-(VOL_LOOKBACK + 5)] if len(vol_start) >= VOL_LOOKBACK + 5 else test_start
    prices_vol = prices_raw[(prices_raw["date"] >= vol_start) & (prices_raw["date"] < test_end)][["date", "ticker", "close"]]

    vol_index = {}
    for ticker, grp in prices_vol.groupby("ticker"):
        grp  = grp.sort_values("date")
        pct  = grp["close"].pct_change()
        roll = pct.rolling(VOL_LOOKBACK).std() * np.sqrt(VOL_LOOKBACK) * VOL_MULTIPLIER
        roll = roll.clip(STOP_FLOOR, STOP_CAP)
        for dt, dist in zip(grp["date"], roll):
            if not np.isnan(dist):
                vol_index[(dt, ticker)] = float(dist)

    rebalance_dates = get_rebalance_dates(trading_days)

    def get_px(day, ticker, col):
        px = price_index.get((day, ticker))
        return px[col] if px and px[col] and px[col] == px[col] else None

    cash, positions, stop_levels, recent_highs, entry_prices = INITIAL_CAPITAL, {}, {}, {}, {}
    pending_buys, pending_sells, sl_replace_dates = {}, set(), set()
    nav_history, trade_log, portfolio_snapshots = [], [], []

    for day_idx, today in enumerate(trading_days):
        # Detect regime for today
        regime = detect_regime(macro_features, breadth_data, today)

        # 1. Stop checks
        sl_today = []
        for ticker in list(positions.keys()):
            if ticker not in stop_levels: continue
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
                pending_sells.discard(ticker); continue
            shares = positions.pop(ticker, 0)
            if shares > 0:
                cash += shares * open_px * (1 - COST_BPS_ONE_WAY / 10000)
                trade_log.append({"date": today, "ticker": ticker, "action": "sell",
                                   "shares": shares, "price": open_px, "window": wid})
            stop_levels.pop(ticker, None)
            recent_highs.pop(ticker, None)
            entry_prices.pop(ticker, None)
            pending_sells.discard(ticker)

        # 3. Buys
        for ticker, dollar_amt in list(pending_buys.items()):
            open_px = get_px(today, ticker, "open")
            if open_px is None or open_px <= 0:
                del pending_buys[ticker]; continue
            spend  = min(dollar_amt, cash * 0.99)
            shares = spend * (1 - COST_BPS_ONE_WAY / 10000) / open_px
            if shares > 0:
                cash -= spend
                positions[ticker] = positions.get(ticker, 0) + shares
                entry_prices[ticker] = open_px
                recent_highs[ticker] = max(recent_highs.get(ticker, open_px), open_px)
                trade_log.append({"date": today, "ticker": ticker, "action": "buy",
                                   "shares": shares, "price": open_px, "window": wid})
            del pending_buys[ticker]

        # 4. Valuation
        equity = sum(shares * (get_px(today, t, "close") or entry_prices.get(t, 0))
                     for t, shares in positions.items())
        nav = cash + equity
        nav_history.append({"date": today, "nav": nav, "cash": cash, "equity": equity})

        # 5. Trailing stop update
        for ticker in list(positions.keys()):
            close = get_px(today, ticker, "close")
            if close is None: continue
            recent_highs[ticker] = max(recent_highs.get(ticker, close), close)
            dist = vol_index.get((today, ticker), STOP_FLOOR)
            stop_levels[ticker] = recent_highs[ticker] * (1 - dist)

        # 6. Optimizer on rebalance or stop-replacement dates
        if today in rebalance_dates or today in sl_replace_dates:
            event = "rebalance" if today in rebalance_dates else "sl_replace"
            sig_today = signals_history[signals_history["date"] == today]
            if sig_today.empty:
                prior = signals_history[signals_history["date"] < today]
                if not prior.empty:
                    sig_today = prior[prior["date"] == prior["date"].max()]

            # Apply regime-adjusted signal weights
            regime_sw = apply_regime_multipliers(dict(sw), regime)
            sig_today = apply_signal_weights(sig_today, regime_sw)
            cov = cov_matrices.get(today, pd.DataFrame())

            total_nav = nav if nav > 0 else 1.0
            curr_w = {t: (s * (get_px(today, t, "close") or 0)) / total_nav
                      for t, s in positions.items()}

            tickers, sigma, sig_subset = prepare_inputs(sig_today, cov)
            if len(tickers) >= 5:
                score_map = sig_subset.set_index("ticker")["composite_score"]
                mu = np.array([score_map.get(t, 0.0) for t in tickers])
                mu = mu - mu.min() + 0.01

                target_w = run_optimizer(mu, tickers, sigma, curr_w, turnover_lambda, risk_aversion)
                if target_w:
                    for ticker in list(positions.keys()):
                        if ticker not in target_w:
                            pending_sells.add(ticker)
                    for ticker, tw in target_w.items():
                        diff = tw * nav - curr_w.get(ticker, 0.0) * nav
                        if diff > 500:
                            pending_buys[ticker] = diff

                    # Portfolio snapshot
                    for ticker, tw in target_w.items():
                        portfolio_snapshots.append({
                            "date": today, "ticker": ticker,
                            "target_weight": round(tw, 6),
                            "current_weight": round(curr_w.get(ticker, 0.0), 6),
                            "composite_score": round(float(score_map.get(ticker, 0.0)), 4),
                            "nav": round(nav, 2), "event": event, "regime": regime,
                            "window_id": wid,
                        })

                    if event == "rebalance":
                        print(f"[exp008]   {today.date()} | NAV=${nav:,.0f} | Pos={len(positions)} | "
                              f"Target={len(target_w)} | Regime={regime}")

    nav_df = pd.DataFrame(nav_history).set_index(pd.to_datetime(pd.DataFrame(nav_history)["date"]))
    nav_df = nav_df.drop(columns=["date"], errors="ignore")
    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    snap_df = pd.DataFrame(portfolio_snapshots) if portfolio_snapshots else pd.DataFrame()

    print(f"[exp008] Window {wid} done | Final NAV: ${nav_df['nav'].iloc[-1]:,.0f} | "
          f"Trades: {len(trade_df)}")

    return nav_df, trade_df, snap_df


# ============================================================
# MAIN: RUN ALL WINDOWS + HOLDOUT
# ============================================================

def run_single():
    print("[exp008] ================================================")
    print("[exp008] EXP008: EWM Halflife Sweep (Fixed Params)")
    print("[exp008] Fixed Params, Sweep Signal Layer")
    print("[exp008] ================================================\n")

    all_windows = log["windows"] + [log["final_holdout"]]

    for window in all_windows:
        wid = window["window_id"]
        if window["status"] == "complete":
            print(f"[exp008] Window {wid}: already complete, skipping")
            continue

        print(f"\n[exp008] ========== WINDOW {wid} ==========")

        # LAYER 0: Calibrate
        if wid == "holdout":
            # Use median params from completed windows
            completed = [w for w in log["windows"] if w["status"] == "complete"]
            if not completed:
                print("[exp008] No completed windows for holdout — using defaults")
                best_lambda, best_ra = 0.005, 1.5
            else:
                lambdas = [w["calibrated_params"]["lambda"] for w in completed]
                ras = [w["calibrated_params"]["risk_aversion"] for w in completed]
                best_lambda = float(np.median(lambdas))
                best_ra = float(np.median(ras))
                print(f"[exp008] Holdout: median lambda={best_lambda} ra={best_ra}")

            ic_weighted = True
            # Compute fresh IC weights from full training
            train_end = pd.Timestamp(window["train_end"])
            sig_train = signals_history[signals_history["date"] <= train_end]
            fwd_train = fwd_returns[fwd_returns["date"] <= train_end]
            train_dates = sorted(sig_train["date"].unique())
            signal_weights = compute_ic_weights(sig_train, fwd_train, train_dates, ewm_halflife=CURRENT_HALFLIFE)
        else:
            # EXP008: fixed params, no calibration sweep
            fixed = log["fixed_params"]
            result = {"lambda": fixed["lambda"], "risk_aversion": fixed["risk_aversion"],
                      "best_sharpe": 0, "best_max_dd": 0}
            ic_weighted = True
            if result is None:
                print(f"[exp008] Window {wid}: calibration failed, skipping")
                continue
            best_lambda = result["lambda"]
            best_ra = result["risk_aversion"]
            signal_weights = result.get("signal_weights")

        # Update log
        window["calibrated_params"] = {
            "lambda": best_lambda, "risk_aversion": best_ra,
            "signal_weights": signal_weights,
        }
        window["ic_weighted_signals"] = ic_weighted
        window["status"] = "calibrated"

        # LAYER 2: Simulate
        nav_df, trade_df, snap_df = simulate_window(
            window, best_lambda, best_ra, signal_weights, ic_weighted
        )

        # Save results
        file_key = f"window_{wid}_mv_monthly"
        nav_df.to_parquet(WF_RESULTS / f"nav_{file_key}.parquet")
        if not trade_df.empty:
            trade_df.to_parquet(WF_RESULTS / f"trades_{file_key}.parquet")
        if not snap_df.empty:
            snap_df.to_parquet(WF_RESULTS / f"portfolios_{file_key}.parquet")

        window["status"] = "complete"
        window["test_nav_file"] = str(WF_RESULTS / f"nav_{file_key}.parquet")

        # Save log after each window
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

        print(f"[exp008] Window {wid} saved.")

    # ============================================================
    # LAYER 3: STITCH + ANALYTICS
    # ============================================================

    print("\n[exp008] ========== STITCH + ANALYTICS ==========")

    completed = [w for w in log["windows"] if w["status"] == "complete"]
    if not completed:
        print("[exp008] No completed windows!")
        return

    # Stitch NAVs
    all_pieces = []
    running_nav = INITIAL_CAPITAL
    for w in sorted(completed, key=lambda x: x["window_id"]):
        path = WF_RESULTS / f"nav_window_{w['window_id']}_mv_monthly.parquet"
        if not path.exists(): continue
        nav = pd.read_parquet(path)["nav"]
        nav.index = pd.to_datetime(nav.index)
        scale = running_nav / nav.iloc[0]
        nav = nav * scale
        running_nav = nav.iloc[-1]
        all_pieces.append(nav)

    # Append holdout
    holdout_path = WF_RESULTS / "nav_window_holdout_mv_monthly.parquet"
    if holdout_path.exists():
        nav = pd.read_parquet(holdout_path)["nav"]
        nav.index = pd.to_datetime(nav.index)
        scale = running_nav / nav.iloc[0]
        nav = nav * scale
        all_pieces.append(nav)

    if not all_pieces:
        print("[exp008] No NAV data to stitch!")
        return

    stitched = pd.concat(all_pieces).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="last")]

    # SPY benchmark
    try:
        spy_path = SPE_DIR / "data" / "raw" / "prices.parquet"
        spy_df = pd.read_parquet(spy_path)
        spy_df["date"] = pd.to_datetime(spy_df["date"])
        spy = spy_df[spy_df["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    except Exception:
        spy = pd.Series(dtype=float)

    spy_window = spy[(spy.index >= stitched.index[0]) & (spy.index <= stitched.index[-1])].dropna()

    # Metrics
    common = stitched.index.intersection(spy_window.index)
    nav_c = stitched.reindex(common).dropna()
    spy_c = spy_window.reindex(common).dropna()
    if spy_c.empty or nav_c.empty:
        print("[exp008] WARNING: No SPY overlap")
        nav_c = stitched
        spy_nav = pd.Series(1.0, index=nav_c.index)
        spy_c = spy_nav
    else:
        spy_nav = (spy_c / spy_c.iloc[0]) * nav_c.iloc[0]

    pr = nav_c.pct_change().dropna()
    br = spy_nav.pct_change().dropna()
    n_y = len(pr) / 252

    total_ret = (nav_c.iloc[-1] / nav_c.iloc[0]) - 1
    cagr = (1 + total_ret) ** (1/n_y) - 1
    ann_vol = pr.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0
    max_dd = float(((nav_c / nav_c.cummax()) - 1).min())

    bench_total = (spy_nav.iloc[-1] / spy_nav.iloc[0]) - 1
    bench_cagr = (1 + bench_total) ** (1/n_y) - 1
    alpha = cagr - bench_cagr

    ex = pr.reindex(br.index) - br
    te = ex.std() * np.sqrt(252)
    ir = (ex.mean() * 252) / te if te > 0 else 0

    print(f"\n[exp008] ============ FINAL RESULTS ============")
    print(f"[exp008] Period: {stitched.index[0].date()} to {stitched.index[-1].date()}")
    print(f"[exp008] CAGR:     {cagr:.2%}")
    print(f"[exp008] Sharpe:   {sharpe:.3f}")
    print(f"[exp008] MaxDD:    {max_dd:.2%}")
    print(f"[exp008] Alpha:    {alpha:+.2%}")
    print(f"[exp008] IR:       {ir:.3f}")
    print(f"[exp008] TE:       {te:.2%}")
    print(f"[exp008] SPY CAGR: {bench_cagr:.2%}")
    print(f"[exp008] ==========================================")

    # Save stitched NAV
    stitched.to_frame("nav").to_parquet(RESULTS_DIR / "nav_mv_monthly.parquet")

    # Stitch all trade logs
    all_trades = []
    for w in completed + [log["final_holdout"]]:
        path = WF_RESULTS / f"trades_window_{w['window_id']}_mv_monthly.parquet"
        if path.exists():
            all_trades.append(pd.read_parquet(path))
    if all_trades:
        try:
            trades_combined = pd.concat(all_trades, ignore_index=True)
            trades_combined["window"] = trades_combined["window"].astype(str)
            trades_combined.to_parquet(RESULTS_DIR / "all_trades.parquet")
        except Exception as e:
            print(f"[exp008] WARNING: Could not save all_trades: {e}")

    # Save metrics
    metrics = {
        "experiment": "EXP008", "strategy": "mv_monthly",
        "cagr": round(cagr * 100, 2), "bench_cagr": round(bench_cagr * 100, 2),
        "sharpe": round(sharpe, 3), "max_drawdown": round(max_dd * 100, 2),
        "alpha": round(alpha * 100, 2), "info_ratio": round(ir, 3),
        "tracking_error": round(te * 100, 2),
        "total_return": round(total_ret * 100, 2),
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[exp008] Results saved to {RESULTS_DIR}")


def main():
    import time
    combos = log["combos"]
    results_summary = []

    for combo in combos:
        global CURRENT_HALFLIFE, CURRENT_MIN_SIGNALS
        CURRENT_HALFLIFE = combo["ewm_halflife"]
        CURRENT_MIN_SIGNALS = combo["min_signals"]
        cid = combo["combo_id"]
        label = f"hl{CURRENT_HALFLIFE}_ms{CURRENT_MIN_SIGNALS}"

        print(f"\n{'='*60}")
        print(f"COMBO {cid}/4: halflife={CURRENT_HALFLIFE}, min_signals={CURRENT_MIN_SIGNALS}")
        print(f"{'='*60}")

        # Set output paths for this combo
        combo_wf = WF_RESULTS / label
        combo_wf.mkdir(parents=True, exist_ok=True)
        combo_res = RESULTS_DIR / label
        combo_res.mkdir(parents=True, exist_ok=True)

        # Temporarily redirect WF_RESULTS and RESULTS_DIR
        import experiments.exp008_sweep.run_exp008 as mod
        orig_wf = mod.WF_RESULTS
        orig_res = mod.RESULTS_DIR
        mod.WF_RESULTS = combo_wf
        mod.RESULTS_DIR = combo_res

        # Reset window statuses
        for w in log["windows"]:
            w["status"] = "pending"
            w["calibrated_params"] = None
        log["final_holdout"]["status"] = "pending"
        log["final_holdout"]["calibrated_params"] = None

        t0 = time.time()
        run_single()
        elapsed = time.time() - t0

        # Read back the stitched metrics
        mf = combo_res / "metrics.json"
        if mf.exists():
            import json as jj
            with open(mf) as f:
                m = jj.load(f)
            m["combo_id"] = cid
            m["halflife"] = CURRENT_HALFLIFE
            m["min_signals"] = CURRENT_MIN_SIGNALS
            m["elapsed_min"] = elapsed / 60
            results_summary.append(m)

        # Restore paths
        mod.WF_RESULTS = orig_wf
        mod.RESULTS_DIR = orig_res

    # Print comparison
    print(f"\n{'='*80}")
    print("EXP008 SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'Combo':<8} {'Halflife':>10} {'MinSig':>8} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10} {'Time':>8}")
    print("-" * 66)
    for r in results_summary:
        print(f"C{r['combo_id']:<7} {r['halflife']:>10} {r['min_signals']:>8} {r.get('cagr','?'):>10} {r.get('sharpe','?'):>10} {r.get('max_dd','?'):>10} {r['elapsed_min']:>7.1f}m")


if __name__ == "__main__":
    run_single()
