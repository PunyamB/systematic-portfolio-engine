# pipeline/runner.py
# Master pipeline orchestrator.
#
# Sequence:
# 1. Broker health check
# 1.5. Reconcile internal portfolio with Alpaca (ground truth)
# 2. Corporate actions
# 3. Data refresh
# 4. Reprice portfolio + NAV
# 5. Regime detection
# 6. Risk monitoring (circuit breaker, trailing stops, drift)
# 6b. Save stop exits to file (execute via: python execute_stops.py)
# 7. Signal computation
# 8. Signal decay tracking
# 9. [REBALANCE/DRIFT/DECAY] Optimizer (cooled-down tickers excluded)
# 10. [REBALANCE/DRIFT/DECAY] Write proposed_trades.csv
# 11. Portfolio history snapshot
# 12. Check excess cash, write replacement proposal (cooled-down tickers excluded)
# 13. Decision log + pipeline health
# 14. Action summary

import json
import time
import traceback
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from utils.config_loader import get_config
from utils.notifications import notify
from utils.broker_health import check_broker_health
from utils.rebalance_calendar import is_rebalance_day, get_next_rebalance_date
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
    load_portfolio, load_prices, save_parquet,
    save_stop_exits,
)
from fund_accounting.nav import load_nav_history

cfg = get_config()

HEALTH_FILE  = Path("logs/pipeline_health.json")
PROPOSED_DIR = Path("data/proposed")
APPROVED_DIR = Path("data/approved")
STOPS_DIR    = Path("data/stops")
FLAG_FILE    = Path("pipeline_running.flag")
EXECUTED_WEIGHTS_PATH = Path("data/processed/executed_weights.parquet")
COOLDOWN_FILE = Path("data/stops/stop_cooldown.json")
COOLDOWN_TRADING_DAYS = 4


# ------------------------------------------------------------
# PIPELINE LOCK
# ------------------------------------------------------------

def _acquire_lock() -> bool:
    if FLAG_FILE.exists():
        print("[runner] Pipeline already running \u2014 flag file exists. Aborting.")
        return False
    FLAG_FILE.touch()
    return True

def _release_lock():
    FLAG_FILE.unlink(missing_ok=True)


# ------------------------------------------------------------
# PIPELINE HEALTH TRACKING
# ------------------------------------------------------------

class PipelineHealth:
    def __init__(self, run_date: date):
        self.run_date   = run_date
        self.stages     = {}
        self.start_time = time.time()

    def record(self, stage: str, status: str, duration_sec: float, detail: str = ""):
        self.stages[stage] = {
            "status":       status,
            "duration_sec": round(duration_sec, 2),
            "detail":       detail,
        }
        symbol = "OK" if status == "success" else ("SKIP" if status == "skipped" else "FAIL")
        print(f"[runner] [{symbol}] {stage} ({duration_sec:.1f}s) {detail}")

    def save(self):
        HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        total_sec = round(time.time() - self.start_time, 2)
        payload = {
            "run_date":  str(self.run_date),
            "run_at":    datetime.now().isoformat(),
            "total_sec": total_sec,
            "stages":    self.stages,
        }
        with open(HEALTH_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        append_pipeline_history({
            "run_date":    str(self.run_date),
            "run_at":      datetime.now().isoformat(),
            "total_sec":   total_sec,
            "status":      "success" if not any(
                v["status"] == "failed" for v in self.stages.values()
            ) else "failed",
            "stages_json": json.dumps(self.stages),
        })
        print(f"[runner] Pipeline health saved")


def _run_stage(health: PipelineHealth, stage: str, fn, *args, **kwargs):
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        health.record(stage, "success", time.time() - t0)
        return result
    except Exception as e:
        health.record(stage, "failed", time.time() - t0, detail=str(e))
        print(f"[runner] Stage {stage} failed: {e}")
        traceback.print_exc()
        return None


# ------------------------------------------------------------
# STOP COOLDOWN LEDGER
# ------------------------------------------------------------

def _get_cooled_down_tickers() -> set:
    """
    Returns set of tickers that were stopped out within the last
    COOLDOWN_TRADING_DAYS trading days. Uses actual market trading
    dates from prices.parquet so weekends/holidays don't count.
    """
    if not COOLDOWN_FILE.exists():
        return set()

    try:
        with open(COOLDOWN_FILE) as f:
            cooldown = json.load(f)
    except Exception:
        return set()

    if not cooldown:
        return set()

    # Get actual trading dates from prices
    prices = load_prices()
    if prices.empty:
        return set()

    trading_dates = sorted(prices["date"].unique())

    # Find the most recent trading date as reference
    today = pd.Timestamp(date.today())
    # Get the last trading date at or before today
    past_dates = [d for d in trading_dates if d <= today]
    if not past_dates:
        return set()

    blocked = set()
    expired = []

    for ticker, stop_date_str in cooldown.items():
        stop_ts = pd.Timestamp(stop_date_str)

        # Count trading days strictly after the stop date
        days_after_stop = [d for d in trading_dates if d > stop_ts]
        trading_days_elapsed = len(days_after_stop)

        if trading_days_elapsed < COOLDOWN_TRADING_DAYS:
            blocked.add(ticker)
        else:
            expired.append(ticker)

    # Clean up expired entries
    if expired:
        for ticker in expired:
            del cooldown[ticker]
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(cooldown, f, indent=2)
        print(f"[runner] Cooldown expired: {expired}")

    if blocked:
        print(f"[runner] Cooldown active ({COOLDOWN_TRADING_DAYS}d): {sorted(blocked)}")

    return blocked


# ------------------------------------------------------------
# PORTFOLIO REPRICING
# ------------------------------------------------------------

def _reprice_portfolio(run_date: date) -> None:
    portfolio = load_portfolio()
    if portfolio.empty:
        return
    prices = load_prices()
    if prices.empty:
        return

    latest = (
        prices.sort_values("date")
        .groupby("ticker").last()
        .reset_index()[["ticker", "close"]]
    )
    portfolio = portfolio.merge(latest, on="ticker", how="left", suffixes=("", "_latest"))
    close_col = "close_latest" if "close_latest" in portfolio.columns else "close"

    mask = portfolio[close_col].notna()
    portfolio.loc[mask, "market_value"] = portfolio.loc[mask, "shares"] * portfolio.loc[mask, close_col]

    if "cost_basis" in portfolio.columns:
        portfolio.loc[mask, "unrealized_pnl"] = (
            portfolio.loc[mask, "market_value"]
            - portfolio.loc[mask, "shares"] * portfolio.loc[mask, "cost_basis"]
        )

    portfolio = portfolio.drop(columns=[c for c in portfolio.columns if c.endswith("_latest")], errors="ignore")
    from data.storage import save_portfolio
    save_portfolio(portfolio)


# ------------------------------------------------------------
# STOP REPLACEMENT \u2014 PROPOSAL ONLY (execution via execute_replacement.py)
# ------------------------------------------------------------

REPLACEMENT_COOLDOWN_FILE = Path("data/stops/last_replacement.json")
REPLACEMENT_COOLDOWN_DAYS = cfg.get("stop_replacement", {}).get("cooldown_days", 5)
REPLACEMENT_CASH_THRESHOLD = cfg.get("stop_replacement", {}).get("cash_threshold", 100000)


def _check_replacement_cooldown(run_date: date) -> bool:
    if not REPLACEMENT_COOLDOWN_FILE.exists():
        return True
    try:
        with open(REPLACEMENT_COOLDOWN_FILE) as f:
            data = json.load(f)
        last_date = date.fromisoformat(data["last_replacement_date"])
        calendar_days = (run_date - last_date).days
        trading_days_approx = int(calendar_days * 5 / 7)
        return trading_days_approx >= REPLACEMENT_COOLDOWN_DAYS
    except Exception:
        return True


def _run_stop_replacement(run_date: date, regime: str, cb_tier: int,
                          nav: float, cash: float, cooled_down: set,
                          health: PipelineHealth) -> None:
    """
    Checks if excess cash warrants a replacement. If triggered, runs
    the cash-deployment optimizer and writes a proposal file.
    Cooled-down tickers are excluded from the trade list.
    Execution is manual via: python execute_replacement.py
    """
    required_cash = get_cash_requirement(cb_tier) * nav
    excess_cash = cash - required_cash

    if excess_cash < REPLACEMENT_CASH_THRESHOLD:
        print(f"[runner] Stop replacement: excess cash ${excess_cash:,.0f} "
              f"< ${REPLACEMENT_CASH_THRESHOLD:,.0f} threshold -- skipping")
        return

    if not _check_replacement_cooldown(run_date):
        print(f"[runner] Stop replacement: cooldown active -- skipping")
        return

    print(f"[runner] Stop replacement TRIGGERED: excess cash ${excess_cash:,.0f}")

    t0 = time.time()

    target_weights = run_stop_replacement_optimizer(
        run_date=run_date,
        regime=regime,
        cb_tier=cb_tier,
        excess_cash=excess_cash,
    )

    if target_weights is None or target_weights.empty:
        health.record("stop_replacement", "skipped", time.time() - t0, "optimizer returned empty")
        return

    # Generate trade list with full detail for display
    nav_history = load_nav_history()
    nav_val = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else nav
    portfolio = load_portfolio()

    current_weights = {}
    current_shares = {}
    if not portfolio.empty:
        for _, row in portfolio.iterrows():
            current_weights[row["ticker"]] = row["market_value"] / nav_val if nav_val > 0 else 0
            current_shares[row["ticker"]] = int(row["shares"])

    from data.storage import load_prices as _lp
    prices = _lp()

    trades = []
    blocked_tickers = []
    for _, row in target_weights.iterrows():
        ticker = row["ticker"]
        target_wt = row["target_weight"]
        current_wt = current_weights.get(ticker, 0.0)
        delta_wt = target_wt - current_wt

        if abs(delta_wt) < 0.001:
            continue

        # Block cooled-down tickers from being bought back
        if delta_wt > 0 and ticker in cooled_down:
            blocked_tickers.append(ticker)
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
            held = current_shares.get(ticker, 0)
            shares = min(shares, held)
            if shares <= 0:
                continue

        trades.append({
            "ticker":          ticker,
            "trade_type":      "buy" if delta_wt > 0 else "sell",
            "direction":       direction,
            "shares":          shares,
            "current_weight":  round(current_wt, 4),
            "target_weight":   round(target_wt, 4),
            "trade_value_usd": round(trade_value, 2),
        })

    if blocked_tickers:
        print(f"[runner] Replacement blocked (cooldown): {blocked_tickers}")

    if not trades:
        health.record("stop_replacement", "skipped", time.time() - t0, "no viable trades after cooldown filter")
        return

    # Write proposal file
    trades_df = pd.DataFrame(trades)
    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROPOSED_DIR / f"replacement_trades_{run_date}.csv"
    trades_df.to_csv(out_path, index=False)

    n_buys = len(trades_df[trades_df["direction"] == "BUY"])
    n_sells = len(trades_df[trades_df["direction"] == "SELL"])
    total_turnover = trades_df["trade_value_usd"].abs().sum()

    health.record("stop_replacement", "success", time.time() - t0,
                   f"{len(trades_df)} trades proposed ({n_buys} buys, {n_sells} trims)")

    print(f"[runner] Replacement trades proposed: {len(trades_df)} ({n_buys} buys, {n_sells} trims)")
    print(f"[runner] \u2192 Execute with: python execute_replacement.py")

    notify(
        f"Stop replacement proposed for {run_date}\n"
        f"Excess cash: ${excess_cash:,.0f}\n"
        f"Trades: {len(trades_df)} ({n_buys} buys, {n_sells} trims)\n"
        f"Turnover: ${total_turnover:,.0f}\n"
        f"Blocked by cooldown: {blocked_tickers if blocked_tickers else 'none'}\n"
        f"To execute: python execute_replacement.py",
        level="info"
    )


# ------------------------------------------------------------
# STOP / TRADE RECONCILIATION (includes cooldown check)
# ------------------------------------------------------------

def _reconcile_stops_and_trades(stop_exits: list, cooled_down: set, target_weights) -> object:
    """
    Removes from proposed trades:
    1. Tickers stopped in this pipeline run (same-day rebuy prevention)
    2. Tickers in cooldown from recent stop exits (4 trading day block)
    Only blocks BUYS for cooled-down tickers. Sells are always allowed.
    """
    if target_weights is None or target_weights.empty:
        return target_weights

    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else 1.0
    portfolio = load_portfolio()
    current_weights = {}
    if not portfolio.empty:
        for _, row in portfolio.iterrows():
            current_weights[row["ticker"]] = row["market_value"] / nav if nav > 0 else 0

    # Combine same-run stops + cooldown tickers
    all_blocked = set(stop_exits) | cooled_down

    if not all_blocked:
        return target_weights

    blocked_buys = []
    keep_mask = []
    for _, row in target_weights.iterrows():
        ticker = row["ticker"]
        target_wt = row["target_weight"]
        current_wt = current_weights.get(ticker, 0.0)
        delta_wt = target_wt - current_wt

        # Block buys for cooled-down tickers, allow sells
        if ticker in all_blocked and delta_wt > 0:
            blocked_buys.append(ticker)
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    filtered = target_weights[keep_mask].copy()

    if blocked_buys:
        print(f"[runner] Removed from proposed trades (stop cooldown): {blocked_buys}")

    return filtered


# ------------------------------------------------------------
# PROPOSED TRADES WRITER
# ------------------------------------------------------------

def write_proposed_trades(target_weights, run_date: date, regime: str) -> Path:
    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)

    nav_history = load_nav_history()
    nav         = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else cfg["portfolio"]["initial_capital"]
    portfolio   = load_portfolio()

    current_weights = {}
    if not portfolio.empty:
        for _, row in portfolio.iterrows():
            current_weights[row["ticker"]] = row["market_value"] / nav if nav > 0 else 0

    trades = []
    for _, row in target_weights.iterrows():
        ticker      = row["ticker"]
        target_wt   = row["target_weight"]
        current_wt  = current_weights.get(ticker, 0.0)
        delta_wt    = target_wt - current_wt
        trade_value = delta_wt * nav
        direction   = "BUY" if delta_wt > 0 else "SELL"

        if abs(delta_wt) < 0.001:
            continue

        trades.append({
            "ticker":          ticker,
            "direction":       direction,
            "current_weight":  round(current_wt, 4),
            "target_weight":   round(target_wt, 4),
            "delta_weight":    round(delta_wt, 4),
            "trade_value_usd": round(trade_value, 2),
            "sector":          row.get("sector", ""),
            "regime":          regime,
            "run_date":        str(run_date),
        })

    target_tickers = set(target_weights["ticker"].tolist())
    already_traded = set(t["ticker"] for t in trades)
    for ticker, current_wt in current_weights.items():
        if ticker not in target_tickers and ticker not in already_traded and current_wt > 0.001:
            trades.append({
                "ticker":          ticker,
                "direction":       "SELL",
                "current_weight":  round(current_wt, 4),
                "target_weight":   0.0,
                "delta_weight":    round(-current_wt, 4),
                "trade_value_usd": round(-current_wt * nav, 2),
                "sector":          "",
                "regime":          regime,
                "run_date":        str(run_date),
            })

    if not trades:
        print("[runner] No trades above threshold \u2014 skipping proposed trades file")
        return None

    df       = pd.DataFrame(trades)
    out_path = PROPOSED_DIR / f"proposed_trades_{run_date}.csv"
    df.to_csv(out_path, index=False)

    n_buys         = len(df[df["direction"] == "BUY"])
    n_sells        = len(df[df["direction"] == "SELL"])
    total_turnover = df["trade_value_usd"].abs().sum()

    print(f"[runner] Proposed trades written: {len(df)} trades | "
          f"buys={n_buys} sells={n_sells} | turnover=${total_turnover:,.0f}")

    notify(
        f"Proposed trades ready for {run_date}\n"
        f"Regime: {regime}\n"
        f"Trades: {len(df)} | Buys: {n_buys} | Sells: {n_sells}\n"
        f"Estimated turnover: ${total_turnover:,.0f}\n"
        f"File: {out_path}\n"
        f"To approve: python approve.py",
        level="info"
    )

    return out_path


# ------------------------------------------------------------
# ACTION SUMMARY
# ------------------------------------------------------------

def _print_action_summary(run_date: date, stop_exits: list = None, cooled_down: set = None) -> None:
    portfolio = load_portfolio()
    prices    = load_prices()

    print("\n" + "=" * 62)
    print("  ACTION SUMMARY")
    print("=" * 62)

    # Stop losses
    stops = stop_exits if stop_exits else []
    if stops and not portfolio.empty and not prices.empty:
        latest = (
            prices.sort_values("date")
            .groupby("ticker").last()
            .reset_index()[["ticker", "close"]]
        )
        pf        = portfolio.merge(latest, on="ticker", how="left", suffixes=("", "_latest"))
        close_col = "close_latest" if "close_latest" in pf.columns else "close"

        print(f"\n  STOP LOSSES \u2014 {len(stops)} positions triggered")
        print(f"  {'Ticker':<8} {'Close':>8} {'Stop':>8} {'% Below':>9}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
        for ticker in stops:
            row = pf[pf["ticker"] == ticker]
            if row.empty:
                continue
            close     = float(row[close_col].values[0])
            stop      = float(row["stop_price"].values[0])
            pct_below = (stop - close) / stop * 100
            print(f"  {ticker:<8} {close:>8.2f} {stop:>8.2f} {pct_below:>8.1f}%")

        print(f"\n  \u2192 Run: python execute_stops.py")
    elif stops:
        print(f"\n  STOP LOSSES: {stops}")
        print(f"  \u2192 Run: python execute_stops.py")
    else:
        print("\n  STOP LOSSES:     none triggered")

    # Cooldown status
    if cooled_down:
        print(f"\n  COOLDOWN \u2014 {len(cooled_down)} tickers blocked from rebuy ({COOLDOWN_TRADING_DAYS} trading days)")
        print(f"  {', '.join(sorted(cooled_down))}")

    # Proposed rebalance trades
    proposed_file = PROPOSED_DIR / f"proposed_trades_{run_date}.csv"
    if proposed_file.exists():
        trades   = pd.read_csv(proposed_file)
        n_buys   = len(trades[trades["direction"] == "BUY"])
        n_sells  = len(trades[trades["direction"] == "SELL"])
        turnover = trades["trade_value_usd"].abs().sum()

        print(f"\n  PROPOSED TRADES \u2014 {len(trades)} trades "
              f"({n_buys} buys, {n_sells} sells) | turnover ${turnover:,.0f}")
        print(f"  {'Ticker':<8} {'Dir':<5} {'Cur Wt':>8} {'Tgt Wt':>8} {'Value ($)':>12}")
        print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*12}")
        for _, row in trades.iterrows():
            print(f"  {row['ticker']:<8} {row['direction']:<5} "
                  f"{row['current_weight']:>7.2%} {row['target_weight']:>7.2%} "
                  f"{row['trade_value_usd']:>12,.0f}")
        print(f"\n  \u2192 Run: python approve.py   then   python execute.py")
    else:
        print("\n  PROPOSED TRADES: none")

    # Cash deployment (replacement trades)
    repl_file = PROPOSED_DIR / f"replacement_trades_{run_date}.csv"
    if repl_file.exists():
        repl = pd.read_csv(repl_file)
        n_buys  = len(repl[repl["direction"] == "BUY"])
        n_sells = len(repl[repl["direction"] == "SELL"])
        turnover = repl["trade_value_usd"].abs().sum()

        print(f"\n  CASH DEPLOYMENT \u2014 {len(repl)} trades "
              f"({n_buys} buys, {n_sells} trims) | turnover ${turnover:,.0f}")
        print(f"  {'Ticker':<8} {'Dir':<5} {'Cur Wt':>8} {'Tgt Wt':>8} {'Value ($)':>12}")
        print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*12}")
        for _, row in repl.iterrows():
            print(f"  {row['ticker']:<8} {row['direction']:<5} "
                  f"{row['current_weight']:>7.2%} {row['target_weight']:>7.2%} "
                  f"{row['trade_value_usd']:>12,.0f}")
        print(f"\n  \u2192 Run: python execute_replacement.py")

    print("\n" + "=" * 62 + "\n")


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def run_pipeline(run_date: date = None, force_rebalance: bool = False) -> dict:
    if run_date is None:
        run_date = date.today()
    if not _acquire_lock():
        return {"status": "aborted", "reason": "already_running"}
    try:
        return _run_pipeline_inner(run_date, force_rebalance)
    finally:
        _release_lock()


def _run_pipeline_inner(run_date: date, force_rebalance: bool) -> dict:

    clear_cache()

    health    = PipelineHealth(run_date)
    rebalance = is_rebalance_day(run_date) or force_rebalance

    decision = {
        "date":          str(run_date),
        "run_at":        datetime.now().isoformat(),
        "rebalance_day": rebalance,
    }

    print(f"\n[runner] ============================================================")
    print(f"[runner] Pipeline start: {run_date} | rebalance_day={rebalance}")
    print(f"[runner] ============================================================\n")

    notify(f"Pipeline started for {run_date} | rebalance={rebalance}", level="info")

    # ----------------------------------------------------------
    # STAGE 1 \u2014 Broker health check
    # ----------------------------------------------------------
    broker_ok = _run_stage(health, "broker_health", check_broker_health)
    if not broker_ok:
        notify(f"Pipeline aborted {run_date} \u2014 broker health check failed", level="critical")
        health.save()
        return {"status": "aborted", "reason": "broker_health"}

    # ----------------------------------------------------------
    # STAGE 1.5 \u2014 Reconcile internal portfolio with Alpaca (ground truth)
    # ----------------------------------------------------------
    _run_stage(health, "reconciliation", reconcile_with_alpaca)

    # ----------------------------------------------------------
    # STAGE 2 \u2014 Corporate actions
    # ----------------------------------------------------------
    _run_stage(health, "corporate_actions", run_corporate_actions, run_date)

    # ----------------------------------------------------------
    # STAGE 3 \u2014 Data refresh
    # ----------------------------------------------------------
    data_ok = _run_stage(health, "data_refresh", run_data_refresh)
    if data_ok is None:
        notify(f"Pipeline aborted {run_date} \u2014 data refresh failed", level="critical")
        health.save()
        return {"status": "aborted", "reason": "data_refresh"}

    # ----------------------------------------------------------
    # STAGE 4 \u2014 Reprice portfolio + NAV calculation
    # ----------------------------------------------------------
    _run_stage(health, "reprice", _reprice_portfolio, run_date)
    nav_result = _run_stage(health, "nav", run_nav, run_date)

    nav_val      = cfg["portfolio"]["initial_capital"]
    daily_return = 0.0
    if nav_result:
        nav_val      = nav_result.get("nav", nav_val)
        daily_return = nav_result.get("daily_return", 0.0)
    decision["nav"]          = nav_val
    decision["daily_return"] = daily_return
    decision["cash"]         = nav_result.get("cash", 0.0) if nav_result else 0.0

    # ----------------------------------------------------------
    # STAGE 5 \u2014 Regime detection
    # ----------------------------------------------------------
    regime_result = _run_stage(health, "regime", detect_regime, run_date)
    regime = regime_result.get("composite", "recovery") if regime_result else "recovery"
    print(f"[runner] Regime: {regime}")

    decision["regime"] = regime
    if regime_result:
        decision["stress_state"] = regime_result.get("stress_state", "")
        decision["cycle_state"]  = regime_result.get("cycle_state", "")
        decision["vix"]          = regime_result.get("vix")
        decision["vix_ratio"]    = regime_result.get("vix_ratio")
        decision["breadth"]      = regime_result.get("breadth")
        decision["yield_spread"] = regime_result.get("yield_spread")
        save_snapshot(regime_result, "regime", run_date)

    # ----------------------------------------------------------
    # STOP COOLDOWN CHECK (after data refresh so prices are current)
    # ----------------------------------------------------------
    cooled_down = _get_cooled_down_tickers()

    # ----------------------------------------------------------
    # STAGE 6 \u2014 Risk monitoring
    # ----------------------------------------------------------
    risk_result   = _run_stage(health, "risk_monitor", run_risk_monitor, run_date)
    stop_exits    = []
    drift_trigger = False

    if risk_result:
        circuit_breaker = risk_result.get("circuit_breaker", {})
        stop_exits      = risk_result.get("stop_exits", [])
        drift_result_d  = risk_result.get("drift", {})
        cb_level        = circuit_breaker.get("tier", 0)

        if cb_level >= 1:
            print(f"[runner] Circuit breaker T{cb_level} active")
        if stop_exits:
            print(f"[runner] Trailing stop exits: {stop_exits}")
        if drift_result_d.get("triggered"):
            drift_trigger = True
            print(f"[runner] Drift trigger detected \u2014 interim rebalance needed")

        decision["cb_tier"]        = cb_level
        decision["drawdown"]       = risk_result.get("drawdown", 0.0)
        decision["beta"]           = risk_result.get("beta", 1.0)
        decision["tracking_error"] = risk_result.get("tracking_error", 0.0)

        save_snapshot({
            "date":           str(run_date),
            "cb_tier":        cb_level,
            "drawdown":       risk_result.get("drawdown", 0.0),
            "beta":           risk_result.get("beta", 1.0),
            "tracking_error": risk_result.get("tracking_error", 0.0),
            "te_breach":      risk_result.get("te_breach", False),
            "stop_exits":     ",".join(stop_exits),
        }, "risk", run_date)

    # ----------------------------------------------------------
    # STAGE 6b \u2014 Save stop exits (execution via execute_stops.py)
    # ----------------------------------------------------------
    if stop_exits:
        save_stop_exits(stop_exits, run_date)
        decision["stop_exits"] = ",".join(stop_exits)
        print(f"[runner] Stop exits saved: {stop_exits}")
        print(f"[runner] \u2192 Execute with: python execute_stops.py")

    # ----------------------------------------------------------
    # STAGE 7 \u2014 Signal computation
    # ----------------------------------------------------------
    signals = _run_stage(health, "signals", run_combiner, run_date, regime)
    if signals is not None and not signals.empty:
        save_snapshot(signals, "signals", run_date)
        decision["n_signals_scored"] = len(signals)

    # ----------------------------------------------------------
    # STAGE 8 \u2014 Signal decay tracking
    # ----------------------------------------------------------
    decay_result  = _run_stage(health, "decay_tracker", run_decay_tracker, run_date)
    decay_trigger = False
    if decay_result and decay_result.get("triggered"):
        decay_trigger = True
        print(f"[runner] Signal decay trigger \u2014 interim rebalance needed")

    # ----------------------------------------------------------
    # STAGE 9 \u2014 Optimizer (cooled-down tickers excluded from buys)
    # ----------------------------------------------------------
    run_optimizer_flag        = rebalance or drift_trigger or decay_trigger
    decision["optimizer_ran"] = run_optimizer_flag

    if run_optimizer_flag:
        reason = "scheduled" if rebalance else ("drift" if drift_trigger else "decay")
        print(f"[runner] Running optimizer \u2014 reason: {reason}")
        decision["optimizer_reason"] = reason

        cb_tier = decision.get("cb_tier", 0)
        target_weights = _run_stage(
            health, "optimizer", run_optimizer, run_date, regime, cb_tier
        )

        if target_weights is not None and not target_weights.empty:
            decision["n_positions"] = len(target_weights)
            decision["max_weight"]  = round(float(target_weights["target_weight"].max()), 4)
            save_snapshot(target_weights, "optimizer", run_date)

            # Filter: same-run stops + cooldown tickers blocked from buys
            target_weights = _reconcile_stops_and_trades(stop_exits, cooled_down, target_weights)

            proposed_path = _run_stage(
                health, "proposed_trades",
                write_proposed_trades, target_weights, run_date, regime
            )
            decision["proposed_trades_file"] = str(proposed_path) if proposed_path else ""
        else:
            health.record("proposed_trades", "skipped", 0, "optimizer returned empty")

    else:
        next_rebalance = get_next_rebalance_date(run_date)
        print(f"[runner] Non-rebalance day \u2014 skipping optimizer | next rebalance: {next_rebalance}")
        health.record("optimizer",       "skipped", 0, f"next rebalance: {next_rebalance}")
        health.record("proposed_trades", "skipped", 0, "non-rebalance day")
        decision["next_rebalance"] = str(next_rebalance) if next_rebalance else ""

    # ----------------------------------------------------------
    # STAGE 11 \u2014 Portfolio history snapshot
    # ----------------------------------------------------------
    portfolio       = load_portfolio()
    nav_history     = load_nav_history()
    nav_for_weights = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else nav_val

    portfolio_snap = portfolio.copy() if not portfolio.empty else portfolio
    if not portfolio_snap.empty and nav_for_weights > 0:
        portfolio_snap["weight"] = portfolio_snap["market_value"] / nav_for_weights
    append_portfolio_history(portfolio_snap, run_date)
    decision["n_held_positions"] = len(portfolio) if not portfolio.empty else 0

    # ----------------------------------------------------------
    # STAGE 12 \u2014 Check excess cash, write replacement proposal
    # ----------------------------------------------------------
    _nav_h = load_nav_history()
    _current_nav = float(_nav_h.iloc[-1]["nav"]) if not _nav_h.empty else nav_val
    _portfolio = load_portfolio()
    if not _portfolio.empty:
        _invested = _portfolio["market_value"].sum()
        _current_cash = _current_nav - _invested
    else:
        _current_cash = _current_nav

    _cb_tier = decision.get("cb_tier", 0)

    _run_stop_replacement(
        run_date=run_date,
        regime=regime,
        cb_tier=_cb_tier,
        nav=_current_nav,
        cash=_current_cash,
        cooled_down=cooled_down,
        health=health,
    )

    # ----------------------------------------------------------
    # FINAL \u2014 Health + decision log + Slack summary
    # ----------------------------------------------------------
    health.save()

    failed_stages             = [s for s, v in health.stages.items() if v["status"] == "failed"]
    status                    = "failed" if failed_stages else "success"
    decision["status"]        = status
    decision["failed_stages"] = ",".join(failed_stages) if failed_stages else ""
    decision["total_sec"]     = round(time.time() - health.start_time, 2)

    append_decision_log(decision)

    notify(
        f"Pipeline complete for {run_date}\n"
        f"Status: {status}\n"
        f"Regime: {regime}\n"
        f"NAV: ${nav_val:,.2f} | Return: {daily_return:.4%}\n"
        f"Rebalance: {rebalance}\n"
        f"Failed stages: {failed_stages if failed_stages else 'none'}\n"
        f"Total time: {round(time.time() - health.start_time, 1)}s",
        level="info" if status == "success" else "warning"
    )

    print(f"\n[runner] ============================================================")
    print(f"[runner] Pipeline complete: {status} | {round(time.time() - health.start_time, 1)}s")
    print(f"[runner] ============================================================")

    _print_action_summary(run_date, stop_exits, cooled_down)

    return {
        "status":    status,
        "regime":    regime,
        "rebalance": rebalance,
        "failed":    failed_stages,
    }


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------

if __name__ == "__main__":
    result = run_pipeline()
    print(result)
