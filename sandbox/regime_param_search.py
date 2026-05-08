"""
regime_param_search.py
======================
Discovers the best (risk_aversion, turnover_lambda) for each market regime
by running a full grid search over the walk-forward OOS windows.

Usage (from Jupyter notebook):
    from sandbox.regime_param_search import run_regime_param_search
    best_params = run_regime_param_search("sandbox/02_adaptive_settings.yaml")
    # Returns dict like: {'bull': {'risk_aversion': 1.0, 'turnover_lambda': 0.007}, ...}
"""

import sys
import json
import pickle
import itertools
import warnings
import importlib
import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRECOMP_DIR  = _PROJECT_ROOT / "data/backtest/precomputed"
BACKTEST_DIR = _PROJECT_ROOT / "data/backtest"
RESULTS_DIR  = _PROJECT_ROOT / "sandbox/results"
LOG_FILE     = _PROJECT_ROOT / "walk_forward_log.json"

SIGNAL_COLS = [
    "momentum_12_1", "earnings_momentum", "pe_zscore", "pb_zscore",
    "ev_ebitda_zscore", "roe_stability", "gross_margin_trend",
    "piotroski", "earnings_accruals", "short_term_reversal", "rsi_extremes",
]


# ---------------------------------------------------------------------------
# INNER SIMULATION (returns daily returns tagged with regime, no file I/O)
# ---------------------------------------------------------------------------

def _simulate(windows, risk_aversion, turnover_lambda,
              config, signals_history, cov_matrices, prices_raw,
              vol_index, regime_index):
    """
    Runs the core backtest loop for a single (risk_aversion, turnover_lambda) combo.
    Returns a DataFrame with columns [date, daily_return, regime].
    """
    MAX_WEIGHT      = config["portfolio"]["max_position_weight"]
    INITIAL_CAPITAL = config["portfolio"]["initial_capital"]
    COST_BPS        = config["turnover"]["round_trip_cost_bps"] / 2 / 10000
    STOP_FLOOR      = config["stop_loss"]["floor"]
    STOP_CAP        = config["stop_loss"]["cap"]
    VOL_LOOKBACK    = config["stop_loss"]["vol_lookback"]

    def get_rebalance_dates(tdays, freq):
        seen, dates = set(), set()
        for day in tdays:
            period = (day.year, day.month) if freq == "monthly" else \
                     (day.year, (day.month - 1) // 3 + 1)
            if period not in seen:
                seen.add(period)
                dates.add(day)
        return dates

    all_rows = []

    for w in windows:
        TEST_START = pd.Timestamp(w["test_start"])
        TEST_END   = pd.Timestamp(w["test_end"])

        sw = w.get("calibrated_params", {}).get("signal_weights") or \
             {s: 1.0 / len(SIGNAL_COLS) for s in SIGNAL_COLS}

        aapl = prices_raw[prices_raw["ticker"] == "AAPL"].sort_values("date")
        trading_days = aapl[(aapl["date"] >= TEST_START) & (aapl["date"] < TEST_END)]["date"].tolist()

        prices_test = prices_raw[(prices_raw["date"] >= TEST_START) & (prices_raw["date"] < TEST_END)]
        price_index = {}
        for row in prices_test[["date", "ticker", "open", "high", "low", "close"]].itertuples(index=False):
            price_index[(row.date, row.ticker)] = {
                "open": row.open, "high": row.high, "low": row.low, "close": row.close
            }

        def get_px(day, ticker, col):
            px = price_index.get((day, ticker))
            return px[col] if px and px[col] == px[col] else None

        def apply_signal_weights(sig_df):
            sig_df = sig_df.copy()
            cols_present = [c for c in SIGNAL_COLS if c in sig_df.columns]
            weights = np.array([sw.get(c, 0.0) for c in cols_present])
            total = weights.sum()
            if total > 0:
                weights /= total
            sig_df["composite_score"] = sig_df[cols_present].fillna(0).values @ weights
            sig_df["composite_rank"] = sig_df["composite_score"].rank(ascending=False)
            return sig_df

        rebalance_dates = get_rebalance_dates(trading_days, config["rebalance"]["frequency"])

        cash = INITIAL_CAPITAL
        positions = {}
        stop_levels = {}
        recent_highs = {}
        entry_prices = {}
        pending_buys = {}
        pending_sells = set()
        sl_replace_dates = set()
        prev_nav = INITIAL_CAPITAL

        for day_idx, today in enumerate(trading_days):
            # Stop checks
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

            # Sells
            for ticker in list(pending_sells):
                open_px = get_px(today, ticker, "open")
                if open_px is None or open_px <= 0:
                    pending_sells.discard(ticker)
                    continue
                shares = positions.pop(ticker, 0)
                if shares > 0:
                    cash += shares * open_px * (1 - COST_BPS)
                stop_levels.pop(ticker, None)
                recent_highs.pop(ticker, None)
                entry_prices.pop(ticker, None)
                pending_sells.discard(ticker)

            # Buys
            for ticker, dollar_amt in list(pending_buys.items()):
                open_px = get_px(today, ticker, "open")
                if open_px is None or open_px <= 0:
                    del pending_buys[ticker]
                    continue
                spend = min(dollar_amt, cash * 0.99)
                shares = spend * (1 - COST_BPS) / open_px
                if shares > 0:
                    cash -= spend
                    positions[ticker] = positions.get(ticker, 0) + shares
                    entry_prices[ticker] = open_px
                    recent_highs[ticker] = max(recent_highs.get(ticker, open_px), open_px)
                del pending_buys[ticker]

            # Valuation
            equity = sum(
                sh * (get_px(today, t, "close") or entry_prices.get(t, 0))
                for t, sh in positions.items()
            )
            nav = cash + equity
            daily_ret = (nav / prev_nav - 1) if prev_nav > 0 else 0.0
            prev_nav = nav

            reg = regime_index.get(today, "recovery")
            all_rows.append({"date": today, "daily_return": daily_ret, "regime": reg})

            # Trailing stops
            for ticker in list(positions.keys()):
                close = get_px(today, ticker, "close")
                if close is None:
                    continue
                recent_highs[ticker] = max(recent_highs.get(ticker, close), close)
                dist = vol_index.get((today, ticker), STOP_FLOOR)
                stop_levels[ticker] = recent_highs[ticker] * (1 - dist)

            # Optimizer
            if today in rebalance_dates or today in sl_replace_dates:
                sig_today = signals_history[signals_history["date"] == today]
                if sig_today.empty:
                    prior = signals_history[signals_history["date"] < today]
                    if not prior.empty:
                        sig_today = prior[prior["date"] == prior["date"].max()]

                sig_today = apply_signal_weights(sig_today)
                cov = cov_matrices.get(today, pd.DataFrame())

                total_nav = nav if nav > 0 else 1.0
                curr_w = {t: (s * (get_px(today, t, "close") or 0)) / total_nav
                          for t, s in positions.items()}

                if sig_today.empty or cov.empty:
                    continue

                top_half = sig_today.nsmallest(max(10, len(sig_today) // 2), "composite_rank")
                tickers = [t for t in top_half["ticker"].tolist() if t in cov.columns]
                if len(tickers) < 5:
                    continue

                sigma     = cov.loc[tickers, tickers].values
                score_map = top_half.set_index("ticker")["composite_score"]
                mu = np.array([score_map.get(t, 0.0) for t in tickers])
                mu = mu - mu.min() + 0.01

                n = len(tickers)
                w_var  = cp.Variable(n)
                w_curr = np.array([curr_w.get(t, 0.0) for t in tickers])

                risk    = cp.quad_form(w_var, cp.psd_wrap(sigma))
                ret_    = mu @ w_var
                penalty = turnover_lambda * cp.norm1(w_var - w_curr)

                prob = cp.Problem(
                    cp.Maximize(ret_ - risk_aversion * risk - penalty),
                    [cp.sum(w_var) == 1, w_var >= 0, w_var <= MAX_WEIGHT],
                )

                target_w = {}
                for solver in [cp.CLARABEL, cp.SCS]:
                    try:
                        prob.solve(solver=solver, warm_start=True)
                        if prob.status in ["optimal", "optimal_inaccurate"] and w_var.value is not None:
                            weights   = {tickers[i]: float(max(0, w_var.value[i])) for i in range(n)}
                            total_wt  = sum(weights.values())
                            if total_wt > 0:
                                target_w = {t: v / total_wt for t, v in weights.items() if v > 1e-4}
                            break
                    except Exception:
                        continue

                if target_w:
                    for ticker in list(positions.keys()):
                        if ticker not in target_w:
                            pending_sells.add(ticker)
                    for ticker, tw in target_w.items():
                        diff = tw * nav - curr_w.get(ticker, 0.0) * nav
                        if diff > 500:
                            pending_buys[ticker] = diff

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# SHARPE HELPER
# ---------------------------------------------------------------------------

def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 20:
        return np.nan
    mu  = returns.mean() * 252
    sig = returns.std()  * np.sqrt(252)
    return mu / sig if sig > 0 else 0.0


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def run_regime_param_search(battery_path: str) -> dict:
    """
    Runs a grid search over (risk_aversion x turnover_lambda) using the
    walk-forward OOS windows, tags each day with its regime, and computes
    per-regime Sharpe for every combo.

    Returns
    -------
    best_params : dict
        { regime_name: { 'risk_aversion': float, 'turnover_lambda': float } }
        These are ready to be patched into the adaptive_optimizer config.
    """
    import yaml

    target_path = Path(battery_path)
    if not target_path.exists():
        target_path = _PROJECT_ROOT / battery_path
    if not target_path.exists():
        raise FileNotFoundError(f"Battery not found: {battery_path}")

    with open(target_path) as f:
        config = yaml.safe_load(f)

    # ---- Grid -----------------------------------------------------------
    grid_cfg = config.get("adaptive_optimizer", {}).get("search_grid", {})
    ra_grid  = grid_cfg.get("risk_aversion",   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    lam_grid = grid_cfg.get("turnover_lambda", [0.002, 0.005, 0.007, 0.010])
    combos   = list(itertools.product(ra_grid, lam_grid))
    print(f"🔍 Grid search: {len(ra_grid)} risk_aversion × {len(lam_grid)} lambda = {len(combos)} combos")

    # ---- Walk-forward windows -------------------------------------------
    with open(LOG_FILE) as f:
        wf_data = json.load(f)
    windows = wf_data.get("windows", [])
    holdout = wf_data.get("final_holdout")
    if holdout:
        windows.append(holdout)

    # ---- Load shared data (once) ----------------------------------------
    print("📦 Loading precomputed data...")
    signals_history = pd.read_parquet(PRECOMP_DIR / "signals_history.parquet")
    signals_history["date"] = pd.to_datetime(signals_history["date"])

    with open(PRECOMP_DIR / "covariance_matrices.pkl", "rb") as f:
        cov_matrices = pickle.load(f)

    prices_raw = pd.read_parquet(BACKTEST_DIR / "prices.parquet")
    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    prices_raw = prices_raw.sort_values(["ticker", "date"])

    regime_history = pd.read_parquet(RESULTS_DIR / "regime_history.parquet")
    regime_history["date"] = pd.to_datetime(regime_history["date"])
    regime_index = dict(zip(regime_history["date"], regime_history["composite"]))
    known_regimes = sorted(regime_history["composite"].unique())

    # ---- Volatility index (trailing stop distances) ---------------------
    VOL_LOOKBACK  = config["stop_loss"]["vol_lookback"]
    VOL_MULT      = config["stop_loss"]["vol_multiplier"]
    STOP_FLOOR    = config["stop_loss"]["floor"]
    STOP_CAP      = config["stop_loss"]["cap"]

    all_dates = sorted(prices_raw["date"].unique())
    prices_vol = prices_raw[["date", "ticker", "close"]].copy()
    vol_index = {}
    for ticker, grp in prices_vol.groupby("ticker"):
        grp = grp.sort_values("date")
        pct  = grp["close"].pct_change()
        roll = pct.rolling(VOL_LOOKBACK).std() * np.sqrt(VOL_LOOKBACK) * VOL_MULT
        roll = roll.clip(STOP_FLOOR, STOP_CAP)
        for dt, dist in zip(grp["date"], roll):
            if not np.isnan(dist):
                vol_index[(dt, ticker)] = float(dist)

    # ---- Grid search loop -----------------------------------------------
    records = []
    total   = len(combos)

    print(f"\n🏃 Running {total} simulations...\n")
    for idx, (ra, lam) in enumerate(combos, 1):
        print(f"  [{idx:>3}/{total}]  risk_aversion={ra:.1f}  lambda={lam:.3f}", end="  ... ", flush=True)
        df = _simulate(
            windows, ra, lam,
            config, signals_history, cov_matrices, prices_raw,
            vol_index, regime_index,
        )
        for regime in known_regimes:
            sub = df[df["regime"] == regime]["daily_return"]
            sh  = _sharpe(sub)
            records.append({
                "risk_aversion":    ra,
                "turnover_lambda":  lam,
                "regime":           regime,
                "sharpe":           sh,
                "n_days":           len(sub),
            })
        print("✓")

    results = pd.DataFrame(records)

    # ---- Find best combo per regime -------------------------------------
    best_params = {}
    print("\n" + "=" * 65)
    print(f"{'REGIME':<12} {'Best Risk Avers.':<20} {'Best Lambda':<14} {'Sharpe':>8}  {'Days':>6}")
    print("=" * 65)

    for regime in known_regimes:
        sub = results[results["regime"] == regime].dropna(subset=["sharpe"])
        if sub.empty:
            continue
        best_row = sub.loc[sub["sharpe"].idxmax()]
        best_params[regime] = {
            "risk_aversion":   float(best_row["risk_aversion"]),
            "turnover_lambda": float(best_row["turnover_lambda"]),
        }
        print(f"{regime:<12} {best_row['risk_aversion']:<20.1f} "
              f"{best_row['turnover_lambda']:<14.3f} "
              f"{best_row['sharpe']:>8.3f}  {int(best_row['n_days']):>6}")

    print("=" * 65)
    print("\n✅ Best parameters discovered per regime.")

    # ---- Pretty heatmaps (one per regime) in Plotly ---------------------
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_reg = len(known_regimes)
        fig   = make_subplots(
            rows=1, cols=n_reg,
            subplot_titles=[r.upper() for r in known_regimes],
        )

        for col_i, regime in enumerate(known_regimes, 1):
            sub = results[results["regime"] == regime].pivot(
                index="risk_aversion", columns="turnover_lambda", values="sharpe"
            )
            fig.add_trace(
                go.Heatmap(
                    z=sub.values,
                    x=[f"λ={v:.3f}" for v in sub.columns],
                    y=[f"RA={v}" for v in sub.index],
                    colorscale="RdYlGn",
                    showscale=(col_i == n_reg),
                    text=np.round(sub.values, 2),
                    texttemplate="%{text}",
                ),
                row=1, col=col_i,
            )

        fig.update_layout(
            title="Per-Regime Sharpe Ratio Heatmap  (risk_aversion × turnover_lambda)",
            template="plotly_dark",
            height=420,
        )
        fig.show()
    except Exception as e:
        print(f"[viz skipped: {e}]")

    return best_params
