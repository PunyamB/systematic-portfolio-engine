import sys
import json
import pickle
import warnings
import importlib
import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path

# Ensure project root is in sys.path when running from sandbox directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

# Project root = the folder that CONTAINS the sandbox/ directory
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRECOMP_DIR = _PROJECT_ROOT / "data/backtest/precomputed"
BACKTEST_DIR = _PROJECT_ROOT / "data/backtest"
RESULTS_DIR = _PROJECT_ROOT / "sandbox/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = _PROJECT_ROOT / "walk_forward_log.json"

def load_backtest_battery(yaml_path: str):
    import yaml
    target_path = Path(yaml_path)
    # If the path doesn't exist relative to cwd, try resolving from project root
    if not target_path.exists():
        target_path = _PROJECT_ROOT / yaml_path
    if not target_path.exists():
        raise FileNotFoundError(f"Battery not found: {yaml_path} (tried cwd and project root {_PROJECT_ROOT})")

    with open(target_path, "r") as f:
        custom_config = yaml.safe_load(f)

    import utils.config_loader
    utils.config_loader._config = custom_config
    importlib.reload(utils.config_loader)
    utils.config_loader._config = custom_config

    print(f"🔋 Plug-and-Play Battery Loaded: {target_path.name}")
    return custom_config


def execute_backtest(battery_path: str):
    config = load_backtest_battery(battery_path)

    print("\n🚀 Initiating walk-forward backtest simulator...")

    # Load the official walk-forward schedule
    if not LOG_FILE.exists():
        raise FileNotFoundError("walk_forward_log.json not found.")
        
    with open(LOG_FILE, "r") as f:
        wf_data = json.load(f)
        
    windows = wf_data.get("windows", [])
    holdout = wf_data.get("final_holdout")
    if holdout:
        windows.append(holdout)

    # Global config extraction
    TURNOVER_LAMBDA = config["optimizer"]["turnover_lambda"]
    RISK_AVERSION = config["optimizer"]["risk_aversion"]
    MAX_WEIGHT = config["portfolio"]["max_position_weight"]
    INITIAL_CAPITAL = config["portfolio"]["initial_capital"]
    COST_BPS_ONE_WAY = config["turnover"]["round_trip_cost_bps"] / 2

    # Trailing stop
    VOL_LOOKBACK = config["stop_loss"]["vol_lookback"]
    VOL_MULTIPLIER = config["stop_loss"]["vol_multiplier"]
    STOP_FLOOR = config["stop_loss"]["floor"]
    STOP_CAP = config["stop_loss"]["cap"]

    SIGNAL_COLS = [
        "momentum_12_1", "earnings_momentum", "pe_zscore", "pb_zscore",
        "ev_ebitda_zscore", "roe_stability", "gross_margin_trend",
        "piotroski", "earnings_accruals", "short_term_reversal", "rsi_extremes",
    ]

    print("[sandbox] Loading precomputed data globally...")
    
    # Load daily regimes if adaptive mode is enabled
    adaptive_mode = config.get("adaptive_optimizer", {}).get("enabled", False)
    if adaptive_mode:
        print("🌍 [Adaptive Mode: ON] Loading precomputed regimes...")
        regime_history = pd.read_parquet(RESULTS_DIR / "regime_history.parquet")
        regime_history["date"] = pd.to_datetime(regime_history["date"])
        regime_index = dict(zip(regime_history["date"], regime_history["composite"]))
    else:
        regime_index = {}
        
    signals_history = pd.read_parquet(PRECOMP_DIR / "signals_history.parquet")
    signals_history["date"] = pd.to_datetime(signals_history["date"])

    with open(PRECOMP_DIR / "covariance_matrices.pkl", "rb") as f:
        cov_matrices = pickle.load(f)

    prices_raw = pd.read_parquet(BACKTEST_DIR / "prices.parquet")
    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    prices_raw = prices_raw.sort_values(["ticker", "date"])

    # Loop through each window to ensure strict OOS validation matches production
    for w in windows:
        WINDOW_ID = w["window_id"]
        TEST_START = pd.Timestamp(w["test_start"])
        TEST_END = pd.Timestamp(w["test_end"])
        
        # Pull signal weights from window calibration exactly as production did.
        # But we override Risk Aversion and Lambda with our battery config.
        sw = w.get("calibrated_params", {}).get("signal_weights", None)
        if not sw:
            sw = {s: 1.0 / len(SIGNAL_COLS) for s in SIGNAL_COLS}
            
        print(f"\n=======================================================")
        print(f"[{WINDOW_ID}] Testing OOS: {TEST_START.date()} to {TEST_END.date()}")
        print(f"Params (Battery): Lambda={TURNOVER_LAMBDA} | Risk Aversion={RISK_AVERSION} | MaxWt={MAX_WEIGHT}")
        print(f"=======================================================")

        # Trading calendar for this window
        aapl = prices_raw[prices_raw["ticker"] == "AAPL"].sort_values("date")
        trading_days = aapl[(aapl["date"] >= TEST_START) & (aapl["date"] < TEST_END)]["date"].tolist()

        prices_test = prices_raw[(prices_raw["date"] >= TEST_START) & (prices_raw["date"] < TEST_END)].copy()
        price_index = {}
        for row in prices_test[["date", "ticker", "open", "high", "low", "close"]].itertuples(index=False):
            price_index[(row.date, row.ticker)] = {
                "open": row.open, "high": row.high, "low": row.low, "close": row.close
            }

        vol_start = prices_raw["date"].unique()
        vol_start = sorted(vol_start[vol_start < TEST_START])
        vol_start = vol_start[-(VOL_LOOKBACK + 5):][0] if len(vol_start) >= VOL_LOOKBACK + 5 else TEST_START
        prices_vol = prices_raw[(prices_raw["date"] >= vol_start) & (prices_raw["date"] < TEST_END)][["date", "ticker", "close"]].copy()

        vol_index = {}
        for ticker, grp in prices_vol.groupby("ticker"):
            grp = grp.sort_values("date")
            pct = grp["close"].pct_change()
            roll = pct.rolling(VOL_LOOKBACK).std() * np.sqrt(VOL_LOOKBACK) * VOL_MULTIPLIER
            roll = roll.clip(STOP_FLOOR, STOP_CAP)
            for dt, dist in zip(grp["date"], roll):
                if not np.isnan(dist):
                    vol_index[(dt, ticker)] = float(dist)

        def get_px(day, ticker, col):
            px = price_index.get((day, ticker))
            return px[col] if px and px[col] and px[col] == px[col] else None

        def apply_signal_weights(sig_df: pd.DataFrame) -> pd.DataFrame:
            sig_df = sig_df.copy()
            cols_present = [c for c in SIGNAL_COLS if c in sig_df.columns]
            weights = np.array([sw.get(c, 0.0) for c in cols_present])
            total = weights.sum()
            if total > 0:
                weights = weights / total
            sig_df["composite_score"] = sig_df[cols_present].fillna(0).values @ weights
            sig_df["composite_rank"] = sig_df["composite_score"].rank(ascending=False)
            return sig_df

        def get_rebalance_dates(tdays: list, freq: str) -> set:
            seen, dates = set(), set()
            for day in tdays:
                period = (day.year, day.month) if freq == "monthly" else (day.year, (day.month - 1) // 3 + 1)
                if period not in seen:
                    seen.add(period)
                    dates.add(day)
            return dates

        rebalance_dates = get_rebalance_dates(trading_days, config["rebalance"]["frequency"])

        cash = INITIAL_CAPITAL
        positions = {}
        stop_levels = {}
        recent_highs = {}
        entry_prices = {}
        pending_buys = {}
        pending_sells = set()
        sl_replace_dates = set()
        nav_history = []
        
        for day_idx, today in enumerate(trading_days):
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
                    pending_sells.discard(ticker)
                    continue
                shares = positions.pop(ticker, 0)
                if shares > 0:
                    cash += shares * open_px * (1 - COST_BPS_ONE_WAY / 10000)
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
                spend = min(dollar_amt, cash * 0.99)
                shares = spend * (1 - COST_BPS_ONE_WAY / 10000) / open_px
                if shares > 0:
                    cash -= spend
                    positions[ticker] = positions.get(ticker, 0) + shares
                    entry_prices[ticker] = open_px
                    recent_highs[ticker] = max(recent_highs.get(ticker, open_px), open_px)
                del pending_buys[ticker]

            # 4. Valuation
            equity = sum(shares * (get_px(today, t, "close") or entry_prices.get(t, 0)) for t, shares in positions.items())
            nav = cash + equity
            nav_history.append({"date": today, "nav": nav})

            # 5. Trailing stops update
            for ticker in list(positions.keys()):
                close = get_px(today, ticker, "close")
                if close is None: continue
                recent_highs[ticker] = max(recent_highs.get(ticker, close), close)
                dist = vol_index.get((today, ticker), STOP_FLOOR)
                stop_levels[ticker] = recent_highs[ticker] * (1 - dist)

            # 6. Optimizer
            if today in rebalance_dates or today in sl_replace_dates:
                sig_today = signals_history[signals_history["date"] == today]
                if sig_today.empty:
                    prior = signals_history[signals_history["date"] < today]
                    if not prior.empty: sig_today = prior[prior["date"] == prior["date"].max()]

                sig_today = apply_signal_weights(sig_today)
                cov = cov_matrices.get(today, pd.DataFrame())

                total_nav = nav if nav > 0 else 1.0
                curr_w = {t: (s * (get_px(today, t, "close") or 0)) / total_nav for t, s in positions.items()}

                # Setup Baseline Optimizer params
                current_lambda = TURNOVER_LAMBDA
                current_risk_av = RISK_AVERSION
                current_regime = "baseline"
                
                # ADAPTIVE REGIME OVERRIDES
                if adaptive_mode and today in regime_index:
                    current_regime = regime_index[today]
                    regimes_config = config["adaptive_optimizer"].get("regimes", {})
                    if current_regime in regimes_config:
                        r_config = regimes_config[current_regime]
                        if "risk_aversion" in r_config:
                            current_risk_av = r_config["risk_aversion"]
                        if "turnover_lambda" in r_config:
                            current_lambda = r_config["turnover_lambda"]
                            
                    print(f"   -> Optimizer triggering in [{current_regime.upper()}] regime. "
                          f"RiskAversion={current_risk_av} | Lambda={current_lambda}")

                # Prepare Inputs
                if sig_today.empty or cov.empty: continue
                top_half = sig_today.nsmallest(max(10, len(sig_today) // 2), "composite_rank")
                tickers = [t for t in top_half["ticker"].tolist() if t in cov.columns]
                if len(tickers) < 5: continue
                sigma = cov.loc[tickers, tickers].values
                score_map = top_half.set_index("ticker")["composite_score"]
                mu = np.array([score_map.get(t, 0.0) for t in tickers])
                mu = mu - mu.min() + 0.01

                # MVO
                n = len(tickers)
                w_var = cp.Variable(n)
                w_curr = np.array([curr_w.get(t, 0.0) for t in tickers])

                risk = cp.quad_form(w_var, cp.psd_wrap(sigma))
                ret_ = mu @ w_var
                penalty = current_lambda * cp.norm1(w_var - w_curr)

                prob = cp.Problem(
                    cp.Maximize(ret_ - current_risk_av * risk - penalty),
                    [cp.sum(w_var) == 1, w_var >= 0, w_var <= MAX_WEIGHT]
                )
                
                target_w = {}
                for solver in [cp.CLARABEL, cp.SCS]:
                    try:
                        prob.solve(solver=solver, warm_start=True)
                        if prob.status in ["optimal", "optimal_inaccurate"] and w_var.value is not None:
                            weights = {tickers[i]: float(max(0, w_var.value[i])) for i in range(n)}
                            total_wt = sum(weights.values())
                            if total_wt > 0:
                                target_w = {t: v / total_wt for t, v in weights.items() if v > 1e-4}
                            break
                    except Exception:
                        continue

                if target_w:
                    for ticker in list(positions.keys()):
                        if ticker not in target_w: pending_sells.add(ticker)
                    for ticker, tw in target_w.items():
                        diff = tw * nav - curr_w.get(ticker, 0.0) * nav
                        if diff > 500: pending_buys[ticker] = diff

        # Save result for window
        nav_df = pd.DataFrame(nav_history)
        nav_df.to_parquet(RESULTS_DIR / f"sim_nav_window_{WINDOW_ID}.parquet")
        print(f"[{WINDOW_ID}] Complete. Final NAV: ${nav_df['nav'].iloc[-1]:,.2f}")

    print("\n🏁 Full historical sequence simulated successfully.")


if __name__ == "__main__":
    battery_file = "sandbox/backtest_settings.yaml"
    try:
        execute_backtest(battery_file)
    except Exception as e:
        print(f"Error executing backtest sequence: {e}")
        import traceback
        traceback.print_exc()
