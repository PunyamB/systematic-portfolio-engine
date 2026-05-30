# pipeline/runner.py - Horizon streaming pretty output (module stdout suppressed)
import json, time, traceback, io, sys, contextlib
from datetime import date, datetime
from pathlib import Path
import pandas as pd

from utils.config_loader import get_config
from utils.notifications import notify
from utils.broker_health import check_broker_health
from utils.rebalance_calendar import is_rebalance_day, get_next_rebalance_date
from utils.pretty_output import PipelineReport, is_market_holiday
from data.pipeline_data import run_data_refresh
from corporate_actions.processor import run_corporate_actions
from fund_accounting.nav import run_nav
from regime.detector import detect_regime
from risk.monitor import run_risk_monitor
from signals.combiner import run_combiner
from signals.decay_tracker import run_decay_tracker
from optimizer.portfolio_optimizer import run_optimizer, run_stop_replacement_optimizer, get_cash_requirement
from execution.order_manager import submit_orders, confirm_fills, update_portfolio_from_fills, reconcile_with_alpaca
from data.storage import (
    clear_cache, save_snapshot, append_decision_log,
    append_portfolio_history, append_pipeline_history,
    load_portfolio, load_prices, save_stop_exits,
)
from fund_accounting.nav import load_nav_history

cfg = get_config()
HEALTH_FILE  = Path("logs/pipeline_health.json")
PROPOSED_DIR = Path("data/proposed")
APPROVED_DIR = Path("data/approved")
FLAG_FILE    = Path("pipeline_running.flag")
COOLDOWN_FILE = Path("data/stops/stop_cooldown.json")
COOLDOWN_TRADING_DAYS = 4
REBAL_STATE_FILE = Path("data/processed/rebalance_state.json")
REPL_COOLDOWN_FILE = Path("data/stops/last_replacement.json")
REPL_COOLDOWN_DAYS = cfg.get("stop_replacement", {}).get("cooldown_days", 5)
REPL_CASH_THRESHOLD = cfg.get("stop_replacement", {}).get("cash_threshold", 100000)

# Captured logs from suppressed modules (saved to file, not printed)
_STAGE_LOGS = Path("logs/stage_output.log")


def _acquire_lock():
    if FLAG_FILE.exists():
        return False
    FLAG_FILE.touch()
    return True

def _release_lock():
    FLAG_FILE.unlink(missing_ok=True)


@contextlib.contextmanager
def _capture_stdout(stage_name):
    """Redirect module stdout to file so it doesn't break pretty output boxes."""
    _STAGE_LOGS.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    orig = sys.stdout
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = orig
        captured = buf.getvalue()
        if captured.strip():
            with open(_STAGE_LOGS, "a", encoding="utf-8") as f:
                f.write(f"\n=== {stage_name} @ {datetime.now().isoformat()} ===\n")
                f.write(captured)


class PipelineHealth:
    def __init__(self, run_date):
        self.run_date = run_date
        self.stages = {}
        self.start_time = time.time()

    def record(self, stage, status, duration_sec, detail=""):
        self.stages[stage] = {"status": status, "duration_sec": round(duration_sec, 2), "detail": detail}

    def save(self):
        HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        total_sec = round(time.time() - self.start_time, 2)
        with open(HEALTH_FILE, "w") as f:
            json.dump({
                "run_date": str(self.run_date), "run_at": datetime.now().isoformat(),
                "total_sec": total_sec, "stages": self.stages,
            }, f, indent=2)
        append_pipeline_history({
            "run_date": str(self.run_date), "run_at": datetime.now().isoformat(),
            "total_sec": total_sec,
            "status": "success" if not any(v["status"] == "failed" for v in self.stages.values()) else "failed",
            "stages_json": json.dumps(self.stages),
        })


def _run_stage(health, report, stage, fn, *args, **kwargs):
    """Run stage with stdout captured to file."""
    t0 = time.time()
    try:
        with _capture_stdout(stage):
            result = fn(*args, **kwargs)
        dur = time.time() - t0
        health.record(stage, "success", dur)
        report.log_stage(stage, "ok", "", dur)
        return result
    except Exception as e:
        dur = time.time() - t0
        health.record(stage, "failed", dur, detail=str(e))
        report.log_stage(stage, "fail", str(e)[:25], dur)
        with open(_STAGE_LOGS, "a", encoding="utf-8") as f:
            f.write(f"\n=== EXCEPTION in {stage} ===\n{traceback.format_exc()}\n")
        return None


def _record_skip(health, report, stage, detail=""):
    health.record(stage, "skipped", 0, detail)
    report.log_stage(stage, "skip", detail, 0)


def _emit_broker(report):
    try:
        from utils.broker_health import _get_alpaca_client
        with _capture_stdout("broker_info"):
            acct = _get_alpaca_client().get_account()
        report.print_broker(acct.status, float(acct.portfolio_value), float(acct.cash), float(acct.equity))
    except Exception:
        pass


def _emit_data_refresh(report):
    try:
        with _capture_stdout("data_summary"):
            from data.storage import load_constituents, load_prices, load_regime_data, get_last_fetch_date
            cons = load_constituents()
            prices = load_prices()
            vix = load_regime_data("vix")
            yc = load_regime_data("yield_curve")
            cs = load_regime_data("credit_spreads")
            fin_last = get_last_fetch_date("financials")
            km_last = get_last_fetch_date("key_metrics")
        prices_range = f"{prices['date'].min().date()} to {prices['date'].max().date()}" if not prices.empty else ""
        report.print_data_refresh(
            constituents=len(cons) if not cons.empty else 0,
            constituents_cached=True, constituents_age_days=0,
            prices_new_rows=len(prices), prices_backfill=False, prices_range=prices_range,
            financials_status=f"last fetched {fin_last}" if fin_last else "first run",
            keymetrics_status=f"last fetched {km_last}" if km_last else "first run",
            macro_status=f"VIX {len(vix)} * Yield {len(yc)} * Credit {len(cs)}",
        )
    except Exception:
        pass


def _emit_nav(report, nav_result):
    if not nav_result:
        return
    with _capture_stdout("nav_summary"):
        nav_hist = load_nav_history()
    peak = float(nav_hist["nav"].max()) if not nav_hist.empty else nav_result.get("nav", 0)
    report.print_nav(
        nav=nav_result.get("nav", 0), daily_return=nav_result.get("daily_return", 0),
        equity=nav_result.get("equity_value", 0), cash=nav_result.get("cash", 0),
        history_count=len(nav_hist), peak=peak,
    )


def _emit_regime(report, regime_result):
    if not regime_result:
        return
    changed, prev = False, ""
    if REBAL_STATE_FILE.exists():
        try:
            with open(REBAL_STATE_FILE) as f:
                state = json.load(f)
            prev = state.get("last_regime", "")
            if prev and prev != regime_result.get("composite", ""):
                changed = True
        except Exception:
            pass
    with _capture_stdout("regime_summary"):
        from data.storage import load_constituents
        cons = load_constituents()
    n_uni = len(cons) if not cons.empty else 503
    breadth = regime_result.get("breadth") or 0.0
    report.print_regime(
        composite=regime_result.get("composite", ""),
        l1_stress=regime_result.get("stress_state", ""),
        vix=regime_result.get("vix") or 0.0,
        vix_ratio=regime_result.get("vix_ratio") or 0.0,
        breadth=breadth, breadth_count=(int(breadth * n_uni), n_uni),
        l2_cycle=regime_result.get("cycle_state", ""),
        yield_spread=regime_result.get("yield_spread") or 0.0,
        curve_inverted=bool(regime_result.get("curve_inverted", False)),
        credit_trend=regime_result.get("credit_trend", ""),
        changed=changed, previous=prev,
    )


def _emit_risk(report, risk_result):
    if not risk_result:
        return
    cb = risk_result.get("circuit_breaker", {})
    cb_tier = cb.get("tier", 0)
    cb_tier_changed, cb_tier_prev = False, 0
    if REBAL_STATE_FILE.exists():
        try:
            with open(REBAL_STATE_FILE) as f:
                state = json.load(f)
            cb_tier_prev = int(state.get("last_cb_tier", 0))
            cb_tier_changed = cb_tier_prev != cb_tier
        except Exception:
            pass

    stop_list = []
    st = risk_result.get("stop_triggers")
    if st is not None and hasattr(st, "empty") and not st.empty:
        for _, row in st.iterrows():
            close_col = next((c for c in row.index if "close" in c.lower()), None)
            close_val = float(row[close_col]) if close_col else 0.0
            stop_val = float(row.get("stop_price", 0.0))
            dist = (close_val - stop_val) / stop_val if stop_val > 0 else 0.0
            stop_list.append((row["ticker"], close_val, stop_val, dist))

    drift = risk_result.get("drift", {})
    liq_flags = []
    liq = risk_result.get("liquidity")
    if liq is not None and hasattr(liq, "empty") and not liq.empty and "liquidity_flag" in liq.columns:
        for _, row in liq[liq["liquidity_flag"] == True].iterrows():
            liq_flags.append((row["ticker"], float(row.get("liquidity_ratio", 0.0))))

    report.print_risk(
        cb_tier=cb_tier, drawdown=risk_result.get("drawdown", 0.0),
        peak_nav=cb.get("peak_nav", 0.0), beta=risk_result.get("beta", 1.0),
        cb_actions=cb.get("actions", []), stop_exits=stop_list,
        drift_positions=[(t, 0.0) for t in drift.get("position_drift", [])],
        drift_portfolio=drift.get("portfolio_drift", False), drift_portfolio_total=0.0,
        drift_sectors=[(s, 0.0) for s in drift.get("sector_drift", [])],
        liquidity_flags=liq_flags,
        cb_tier_changed=cb_tier_changed, cb_tier_previous=cb_tier_prev,
    )


def _build_signals_data(signals_df):
    if signals_df is None or signals_df.empty:
        return None
    signal_cols = [c for c in signals_df.columns if c not in ("ticker", "date", "composite_score", "composite_rank")]
    table = []
    for sig in signal_cols:
        vals = signals_df[sig].dropna()
        if len(vals) == 0:
            continue
        table.append((sig, len(vals), float(vals.mean()), float(vals.std())))
    weighting = "equal weights"
    ic_path = Path("data/processed/ic_history.parquet")
    if ic_path.exists():
        try:
            with _capture_stdout("ic_check"):
                ic_hist = pd.read_parquet(ic_path)
            min_lb = cfg["signals"].get("ic_lookback_min", 12)
            weighting = "IC-IR weighted" if len(ic_hist) >= min_lb else "equal weights"
        except Exception:
            pass
    top5 = signals_df.nlargest(5, "composite_score")["ticker"].tolist()
    return (table, weighting, top5)


def _emit_rebalance(report, due, reason, regime, cb_tier, run_date):
    with _capture_stdout("rebal_check"):
        next_d = get_next_rebalance_date(run_date, regime, cb_tier)
    prev_date = prev_regime = ""
    prev_cb = 0
    if REBAL_STATE_FILE.exists():
        try:
            with open(REBAL_STATE_FILE) as f:
                state = json.load(f)
            prev_date = state.get("last_rebalance_date", "")
            prev_regime = state.get("last_regime", "")
            prev_cb = int(state.get("last_cb_tier", 0))
        except Exception:
            pass
    report.print_rebalance(due=due, reason=reason,
        next_date=str(next_d) if next_d else "",
        prev_date=prev_date, prev_regime=prev_regime, prev_cb=prev_cb)


def _reprice_portfolio(run_date):
    portfolio = load_portfolio()
    if portfolio.empty:
        return
    prices = load_prices()
    if prices.empty:
        return
    latest = prices.sort_values("date").groupby("ticker").last().reset_index()[["ticker", "close"]]
    portfolio = portfolio.merge(latest, on="ticker", how="left", suffixes=("", "_latest"))
    close_col = "close_latest" if "close_latest" in portfolio.columns else "close"
    mask = portfolio[close_col].notna()
    portfolio.loc[mask, "market_value"] = portfolio.loc[mask, "shares"] * portfolio.loc[mask, close_col]
    if "cost_basis" in portfolio.columns:
        portfolio.loc[mask, "unrealized_pnl"] = (
            portfolio.loc[mask, "market_value"] - portfolio.loc[mask, "shares"] * portfolio.loc[mask, "cost_basis"])
    portfolio = portfolio.drop(columns=[c for c in portfolio.columns if c.endswith("_latest")], errors="ignore")
    from data.storage import save_portfolio
    save_portfolio(portfolio)


def _get_cooled_down_tickers():
    if not COOLDOWN_FILE.exists():
        return set()
    try:
        with open(COOLDOWN_FILE) as f:
            cooldown = json.load(f)
    except Exception:
        return set()
    if not cooldown:
        return set()
    prices = load_prices()
    if prices.empty:
        return set()
    trading_dates = sorted(prices["date"].unique())
    blocked, expired = set(), []
    for ticker, sd in cooldown.items():
        days_after = [d for d in trading_dates if d > pd.Timestamp(sd)]
        if len(days_after) < COOLDOWN_TRADING_DAYS:
            blocked.add(ticker)
        else:
            expired.append(ticker)
    if expired:
        for t in expired:
            del cooldown[t]
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(cooldown, f, indent=2)
    return blocked


def _check_repl_cooldown(run_date):
    if not REPL_COOLDOWN_FILE.exists():
        return True
    try:
        with open(REPL_COOLDOWN_FILE) as f:
            data = json.load(f)
        last = date.fromisoformat(data["last_replacement_date"])
        return (run_date - last).days * 5 / 7 >= REPL_COOLDOWN_DAYS
    except Exception:
        return True


def _run_stop_replacement(run_date, regime, cb_tier, nav, cash, cooled_down, health, report):
    required_cash = get_cash_requirement(cb_tier) * nav
    excess_cash = cash - required_cash
    if excess_cash < REPL_CASH_THRESHOLD:
        return
    if not _check_repl_cooldown(run_date):
        return

    t0 = time.time()
    with _capture_stdout("stop_replacement"):
        target_weights = run_stop_replacement_optimizer(
            run_date=run_date, regime=regime, cb_tier=cb_tier, excess_cash=excess_cash)
    if target_weights is None or target_weights.empty:
        health.record("stop_replacement", "skipped", time.time() - t0, "optimizer empty")
        return

    nav_history = load_nav_history()
    nav_val = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else nav
    portfolio = load_portfolio()
    current_weights, current_shares = {}, {}
    if not portfolio.empty:
        for _, row in portfolio.iterrows():
            current_weights[row["ticker"]] = row["market_value"] / nav_val if nav_val > 0 else 0
            current_shares[row["ticker"]] = int(row["shares"])

    from data.storage import load_prices as _lp
    prices = _lp()
    trades, report_trades = [], []

    for _, row in target_weights.iterrows():
        ticker = row["ticker"]
        target_wt = row["target_weight"]
        current_wt = current_weights.get(ticker, 0.0)
        delta_wt = target_wt - current_wt
        if abs(delta_wt) < 0.001:
            continue
        if delta_wt > 0 and ticker in cooled_down:
            continue
        trade_value = delta_wt * nav_val
        direction = "BUY" if delta_wt > 0 else "SELL"
        latest = prices[prices["ticker"] == ticker].sort_values("date")
        if latest.empty:
            continue
        price = float(latest["close"].iloc[-1])
        if price <= 0:
            continue
        shares = int(abs(trade_value) / price)
        if shares <= 0:
            continue
        if delta_wt < 0:
            shares = min(shares, current_shares.get(ticker, 0))
            if shares <= 0:
                continue
        trades.append({
            "ticker": ticker, "trade_type": "buy" if delta_wt > 0 else "sell",
            "direction": direction, "shares": shares,
            "current_weight": round(current_wt, 4), "target_weight": round(target_wt, 4),
            "trade_value_usd": round(trade_value, 2),
        })
        report_trades.append((ticker, direction, current_wt, target_wt, delta_wt, trade_value))

    if not trades:
        health.record("stop_replacement", "skipped", time.time() - t0, "no viable trades")
        return

    trades_df = pd.DataFrame(trades)
    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(PROPOSED_DIR / f"replacement_trades_{run_date}.csv", index=False)
    n_buys = len(trades_df[trades_df["direction"] == "BUY"])
    n_sells = len(trades_df[trades_df["direction"] == "SELL"])
    total = trades_df["trade_value_usd"].abs().sum()
    dur = time.time() - t0
    health.record("stop_replacement", "success", dur, f"{len(trades_df)} trades")
    report.log_stage("stop_replacement", "ok", f"${total:,.0f} {len(trades_df)} trades", dur)
    report.print_replacement(
        excess_cash=excess_cash, universe=len(target_weights), cb_tier=cb_tier,
        buys=n_buys, trims=n_sells, value=total, trades=report_trades)
    report.actions_list.append("python execute_replacement.py")


def _reconcile_stops_and_trades(stop_exits, cooled_down, target_weights):
    if target_weights is None or target_weights.empty:
        return target_weights
    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else 1.0
    portfolio = load_portfolio()
    current_weights = {}
    if not portfolio.empty:
        for _, row in portfolio.iterrows():
            current_weights[row["ticker"]] = row["market_value"] / nav if nav > 0 else 0
    all_blocked = set(stop_exits) | cooled_down
    if not all_blocked:
        return target_weights
    keep = []
    for _, row in target_weights.iterrows():
        delta = row["target_weight"] - current_weights.get(row["ticker"], 0.0)
        keep.append(not (row["ticker"] in all_blocked and delta > 0))
    return target_weights[keep].copy()


def write_proposed_trades(target_weights, run_date, regime):
    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else cfg["portfolio"]["initial_capital"]
    portfolio = load_portfolio()
    current_weights = {}
    if not portfolio.empty:
        for _, row in portfolio.iterrows():
            current_weights[row["ticker"]] = row["market_value"] / nav if nav > 0 else 0
    trades, report_trades = [], []
    for _, row in target_weights.iterrows():
        ticker = row["ticker"]; target_wt = row["target_weight"]
        current_wt = current_weights.get(ticker, 0.0); delta_wt = target_wt - current_wt
        if abs(delta_wt) < 0.001:
            continue
        trade_value = delta_wt * nav
        direction = "BUY" if delta_wt > 0 else "SELL"
        trades.append({"ticker": ticker, "direction": direction,
            "current_weight": round(current_wt, 4), "target_weight": round(target_wt, 4),
            "delta_weight": round(delta_wt, 4), "trade_value_usd": round(trade_value, 2),
            "sector": row.get("sector", ""), "regime": regime, "run_date": str(run_date)})
        report_trades.append((ticker, direction, current_wt, target_wt, delta_wt, trade_value))
    target_set = set(target_weights["ticker"].tolist())
    traded = set(t["ticker"] for t in trades)
    for ticker, cw in current_weights.items():
        if ticker not in target_set and ticker not in traded and cw > 0.001:
            trades.append({"ticker": ticker, "direction": "SELL",
                "current_weight": round(cw, 4), "target_weight": 0.0,
                "delta_weight": round(-cw, 4), "trade_value_usd": round(-cw * nav, 2),
                "sector": "", "regime": regime, "run_date": str(run_date)})
            report_trades.append((ticker, "SELL", cw, 0.0, -cw, -cw * nav))
    if not trades:
        return None, []
    df = pd.DataFrame(trades)
    out_path = PROPOSED_DIR / f"proposed_trades_{run_date}.csv"
    df.to_csv(out_path, index=False)
    return out_path, report_trades


def _notify_safe(msg, level="info"):
    """Wrap notify in stdout capture so Slack failures don't break boxes."""
    try:
        with _capture_stdout("notify"):
            notify(msg, level=level)
    except Exception:
        pass


def run_pipeline(run_date=None, force_rebalance=False):
    if run_date is None:
        run_date = date.today()
    if not _acquire_lock():
        return {"status": "aborted", "reason": "already_running"}
    try:
        return _run_pipeline_inner(run_date, force_rebalance)
    finally:
        _release_lock()


def _run_pipeline_inner(run_date, force_rebalance):
    # Clear stage log for this run
    _STAGE_LOGS.parent.mkdir(parents=True, exist_ok=True)
    with open(_STAGE_LOGS, "w", encoding="utf-8") as f:
        f.write(f"=== Pipeline run {run_date} @ {datetime.now().isoformat()} ===\n")

    with _capture_stdout("setup"):
        clear_cache()
    health = PipelineHealth(run_date)
    is_holiday, holiday_name = is_market_holiday(run_date)
    is_cold = not REBAL_STATE_FILE.exists()
    report = PipelineReport(
        run_date=run_date, started_at=datetime.now(),
        is_weekend=(run_date.weekday() >= 5),
        is_holiday=is_holiday, holiday_name=holiday_name, is_cold_start=is_cold)
    report.actions_list = []
    report.print_header()
    decision = {"date": str(run_date), "run_at": datetime.now().isoformat()}
    _notify_safe(f"Pipeline started for {run_date}", "info")

    # 1: Broker
    broker_ok = _run_stage(health, report, "broker_health", check_broker_health)
    if not broker_ok:
        report.print_halt("Broker health check failed")
        report.print_footer("halted")
        health.save()
        return {"status": "aborted", "reason": "broker_health"}
    _emit_broker(report)

    # 1.5: Reconcile
    _run_stage(health, report, "reconciliation", reconcile_with_alpaca)

    # 2: Corp actions
    _run_stage(health, report, "corporate_actions", run_corporate_actions, run_date)

    # 3: Data
    data_ok = _run_stage(health, report, "data_refresh", run_data_refresh)
    if data_ok is None:
        report.print_halt("Data refresh failed quality gate")
        report.print_footer("halted")
        health.save()
        return {"status": "aborted", "reason": "data_refresh"}
    _emit_data_refresh(report)

    # 4: NAV
    _run_stage(health, report, "reprice", _reprice_portfolio, run_date)
    nav_result = _run_stage(health, report, "nav", run_nav, run_date)
    nav_val = cfg["portfolio"]["initial_capital"]
    daily_return = 0.0
    if nav_result:
        nav_val = nav_result.get("nav", nav_val)
        daily_return = nav_result.get("daily_return", 0.0)
    decision["nav"] = nav_val
    decision["daily_return"] = daily_return
    decision["cash"] = nav_result.get("cash", 0.0) if nav_result else 0.0
    _emit_nav(report, nav_result)

    # 5: Regime
    regime_result = _run_stage(health, report, "regime", detect_regime, run_date)
    regime = regime_result.get("composite", "recovery") if regime_result else "recovery"
    decision["regime"] = regime
    if regime_result:
        with _capture_stdout("regime_snap"):
            save_snapshot(regime_result, "regime", run_date)
    _emit_regime(report, regime_result)

    with _capture_stdout("cooldown"):
        cooled_down = _get_cooled_down_tickers()

    # 6: Risk
    risk_result = _run_stage(health, report, "risk_monitor", run_risk_monitor, run_date)
    stop_exits = []
    rebalance = False
    rebalance_reason = "risk_monitor_failed"
    drift_trigger = False
    if risk_result:
        cb_level = risk_result.get("circuit_breaker", {}).get("tier", 0)
        stop_exits = risk_result.get("stop_exits", [])
        if risk_result.get("drift", {}).get("triggered"):
            drift_trigger = True
        decision["cb_tier"] = cb_level
        decision["drawdown"] = risk_result.get("drawdown", 0.0)
        decision["beta"] = risk_result.get("beta", 1.0)
        with _capture_stdout("rebal_check"):
            rebalance, rebalance_reason = is_rebalance_day(run_date, regime, cb_tier=cb_level, force=force_rebalance)
        decision["rebalance_day"] = rebalance
        decision["rebalance_reason"] = rebalance_reason
        with _capture_stdout("risk_snap"):
            save_snapshot({"date": str(run_date), "cb_tier": cb_level,
                "drawdown": risk_result.get("drawdown", 0.0), "beta": risk_result.get("beta", 1.0),
                "stop_exits": ",".join(stop_exits)}, "risk", run_date)
    _emit_risk(report, risk_result)

    if stop_exits:
        with _capture_stdout("stop_save"):
            save_stop_exits(stop_exits, run_date)
        decision["stop_exits"] = ",".join(stop_exits)
        report.print_stop_execution(market_open=False)
        report.actions_list.append("python execute_stops.py")

    # 7: Signals
    signals = _run_stage(health, report, "signals", run_combiner, run_date, regime)
    sig_data = None
    if signals is not None and not signals.empty:
        with _capture_stdout("signals_snap"):
            save_snapshot(signals, "signals", run_date)
        decision["n_signals_scored"] = len(signals)
        sig_data = _build_signals_data(signals)

    # 8: Decay
    decay_result = _run_stage(health, report, "decay_tracker", run_decay_tracker, run_date)
    decay_trigger = False
    decay_alerts = []
    decay_mean = 1.0
    if decay_result:
        if decay_result.get("triggered"):
            decay_trigger = True
        decay_mean = decay_result.get("mean_correlation", 1.0)
        for entry in decay_result.get("triggered_signals", []):
            if isinstance(entry, dict):
                decay_alerts.append((entry.get("signal", "?"),
                    entry.get("correlation", 0.0), entry.get("half_life", 0.0)))

    if sig_data is not None:
        table, weighting, top5 = sig_data
        report.print_signals(table, weighting, top5, decay_alerts, decay_mean)

    _emit_rebalance(report, rebalance, rebalance_reason, regime, decision.get("cb_tier", 0), run_date)

    # 9: Optimizer
    run_opt = rebalance or drift_trigger or decay_trigger
    decision["optimizer_ran"] = run_opt
    if run_opt:
        reason = "scheduled" if rebalance else ("drift" if drift_trigger else "decay")
        decision["optimizer_reason"] = reason
        cb_tier = decision.get("cb_tier", 0)
        target_weights = _run_stage(health, report, "optimizer", run_optimizer, run_date, regime, cb_tier)
        if target_weights is not None and not target_weights.empty:
            decision["n_positions"] = len(target_weights)
            decision["max_weight"] = round(float(target_weights["target_weight"].max()), 4)
            with _capture_stdout("opt_snap"):
                save_snapshot(target_weights, "optimizer", run_date)
            from optimizer.portfolio_optimizer import get_invested_target, get_max_weight, is_no_new_positions_active
            floors = cfg.get("circuit_breaker", {}).get("cb_sum_floor", {})
            cb_floor = float(floors.get(cb_tier, floors.get(str(cb_tier), 0.85)))
            filt = _reconcile_stops_and_trades(stop_exits, cooled_down, target_weights)
            with _capture_stdout("write_trades"):
                proposed_path, opt_trades = write_proposed_trades(filt, run_date, regime)
            trades_total = sum(abs(t[5]) for t in opt_trades)
            report.print_optimizer(
                params_lambda=float(cfg["optimizer"]["turnover_lambda"]),
                params_ra=float(cfg["optimizer"]["risk_aversion"]),
                cb_tier=cb_tier, cb_invested=get_invested_target(cb_tier),
                cb_maxwt=get_max_weight(cb_tier), cb_floor=cb_floor,
                no_new_pos=is_no_new_positions_active(cb_tier),
                universe_size=len(target_weights), universe_filtered_to_held=False,
                solver="ECOS", solver_status="optimal",
                n_positions=int((target_weights["target_weight"] > 0.001).sum()),
                sum_weights=float(target_weights["target_weight"].sum()),
                max_weight=float(target_weights["target_weight"].max()),
                turnover_pct=0.0, turnover_binding=False, sectors=[],
                trades=opt_trades, trades_total_value=trades_total)
            if proposed_path:
                decision["proposed_trades_file"] = str(proposed_path)
                report.actions_list.append("python approve.py")
                report.actions_list.append("python execute.py")
                health.record("proposed_trades", "success", 0, f"{len(opt_trades)} trades")
                report.log_stage("proposed_trades", "ok", f"{len(opt_trades)} trades", 0)
        else:
            _record_skip(health, report, "proposed_trades", "optimizer empty")
    else:
        with _capture_stdout("next_rebal"):
            next_r = get_next_rebalance_date(run_date, regime, decision.get("cb_tier", 0))
        _record_skip(health, report, "optimizer", f"next: {next_r}")
        _record_skip(health, report, "proposed_trades", "non-rebalance day")
        decision["next_rebalance"] = str(next_r) if next_r else ""

    # 11: Portfolio history
    with _capture_stdout("portfolio_hist"):
        portfolio = load_portfolio()
        nav_hist = load_nav_history()
        nav_for_wt = float(nav_hist.iloc[-1]["nav"]) if not nav_hist.empty else nav_val
        snap = portfolio.copy() if not portfolio.empty else portfolio
        if not snap.empty and nav_for_wt > 0:
            snap["weight"] = snap["market_value"] / nav_for_wt
        append_portfolio_history(snap, run_date)
    decision["n_held_positions"] = len(portfolio) if not portfolio.empty else 0

    # 12: Stop replacement
    with _capture_stdout("cash_check"):
        _nav_h = load_nav_history()
        _current_nav = float(_nav_h.iloc[-1]["nav"]) if not _nav_h.empty else nav_val
        _portfolio = load_portfolio()
        _current_cash = _current_nav - _portfolio["market_value"].sum() if not _portfolio.empty else _current_nav
    _cb_tier = decision.get("cb_tier", 0)
    _run_stop_replacement(run_date, regime, _cb_tier, _current_nav, _current_cash, cooled_down, health, report)

    # Final
    with _capture_stdout("save"):
        health.save()
        failed_stages = [s for s, v in health.stages.items() if v["status"] == "failed"]
        status = "failed" if failed_stages else "success"
        decision["status"] = status
        decision["failed_stages"] = ",".join(failed_stages) if failed_stages else ""
        decision["total_sec"] = round(time.time() - health.start_time, 2)
        append_decision_log(decision)
    _notify_safe(f"Pipeline complete for {run_date}\nStatus: {status} | Regime: {regime}\n"
                 f"NAV: ${nav_val:,.2f} | Return: {daily_return:.4%}",
                 "info" if status == "success" else "warning")
    report.print_actions(report.actions_list)
    report.print_footer(status if status == "success" else "fatal")
    return {"status": status, "regime": regime, "rebalance": rebalance, "failed": failed_stages}


if __name__ == "__main__":
    run_pipeline()
