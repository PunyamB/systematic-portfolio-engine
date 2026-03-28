# pipeline/runner.py
# Master pipeline orchestrator.
# Runs the full daily pipeline in the correct sequence.
# Daily stages run every day.
# Rebalance stages only run on rebalance days or when drift/risk triggers fire.
#
# Sequence:
# 1. Broker health check
# 2. Corporate actions
# 3. Data refresh
# 4. NAV calculation
# 5. Regime detection
# 6. Risk monitoring (circuit breaker, trailing stops, drift)
# 7. Signal computation (daily)
# 8. Signal decay tracking (between rebalances)
# 9. [REBALANCE ONLY] Optimizer + stop/trade deduplication
# 10. [REBALANCE ONLY] Write proposed_trades.csv + Slack notification
# 11. Portfolio history snapshot
# 12. Decision log + pipeline health
# 13. Action summary — stop losses + proposed trades printed to terminal

import json
import time
import traceback
from datetime import date, datetime
from pathlib import Path

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
from optimizer.portfolio_optimizer import run_optimizer
from data.storage import (
    clear_cache, save_snapshot, append_decision_log,
    append_portfolio_history, append_pipeline_history,
    load_portfolio, load_prices, save_parquet,
    save_stop_exits,
)
from execution.order_manager import submit_orders, confirm_fills, update_portfolio_from_fills
from fund_accounting.nav import load_nav_history

cfg = get_config()

HEALTH_FILE  = Path("logs/pipeline_health.json")
PROPOSED_DIR = Path("data/proposed")
APPROVED_DIR = Path("data/approved")
STOPS_DIR    = Path("data/stops")
FLAG_FILE    = Path("pipeline_running.flag")


# ------------------------------------------------------------
# PIPELINE LOCK
# ------------------------------------------------------------

def _acquire_lock() -> bool:
    """Creates pipeline_running.flag. Returns False if already running."""
    if FLAG_FILE.exists():
        print("[runner] Pipeline already running — flag file exists. Aborting.")
        return False
    FLAG_FILE.touch()
    return True


def _release_lock():
    """Removes pipeline_running.flag."""
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
    """Runs a pipeline stage with timing and error handling."""
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
# PORTFOLIO REPRICING
# ------------------------------------------------------------

def _reprice_portfolio(run_date: date) -> None:
    """
    Updates market_value and unrealized_pnl for all positions
    using latest close prices. Called daily so portfolio_history
    reflects actual market values, not stale execution prices.
    """
    portfolio = load_portfolio()
    if portfolio.empty:
        return

    prices = load_prices()
    if prices.empty:
        return

    latest = (
        prices.sort_values("date")
        .groupby("ticker")
        .last()
        .reset_index()[["ticker", "close"]]
    )

    portfolio = portfolio.merge(latest, on="ticker", how="left", suffixes=("", "_latest"))
    close_col = "close_latest" if "close_latest" in portfolio.columns else "close"

    mask = portfolio[close_col].notna()
    portfolio.loc[mask, "market_value"] = (
        portfolio.loc[mask, "shares"] * portfolio.loc[mask, close_col]
    )

    if "cost_basis" in portfolio.columns:
        portfolio.loc[mask, "unrealized_pnl"] = (
            portfolio.loc[mask, "market_value"]
            - portfolio.loc[mask, "shares"] * portfolio.loc[mask, "cost_basis"]
        )

    portfolio = portfolio.drop(
        columns=[c for c in portfolio.columns if c.endswith("_latest")],
        errors="ignore"
    )

    from data.storage import save_portfolio
    save_portfolio(portfolio)


# ------------------------------------------------------------
# AUTO-EXECUTE STOP LOSSES
# ------------------------------------------------------------

def _auto_execute_stops(stop_exits: list, run_date, health: PipelineHealth) -> list:
    """
    Auto-executes trailing stop exits immediately during pipeline run.
    Stops are P2 priority safety exits — they never wait for manual approval.
    Returns list of tickers that were successfully filled.
    """
    import pandas as pd

    if not stop_exits:
        return []

    t0 = time.time()

    portfolio = load_portfolio()
    if portfolio.empty:
        health.record("stop_execution", "skipped", time.time() - t0, "no portfolio")
        return []

    stop_positions = portfolio[portfolio["ticker"].isin(stop_exits)]
    if stop_positions.empty:
        health.record("stop_execution", "skipped", time.time() - t0, "no matching positions")
        return []

    trades = pd.DataFrame({
        "ticker":     stop_positions["ticker"].tolist(),
        "trade_type": ["sell"] * len(stop_positions),
        "shares":     stop_positions["shares"].astype(int).tolist(),
    })

    print(f"[runner] Auto-executing {len(trades)} stop exits: {stop_exits}")

    try:
        submitted = submit_orders(trades)
        if submitted.empty:
            health.record("stop_execution", "failed", time.time() - t0, "no orders submitted")
            notify(f"Stop auto-execution FAILED for {run_date} — no orders submitted: {stop_exits}", level="critical")
            return []

        # Wait for fills (market orders at open fill fast)
        print(f"[runner] Waiting 30s for stop exit fills...")
        time.sleep(30)

        fills = confirm_fills(submitted)
        update_portfolio_from_fills(fills)

        filled_count = int(fills["filled"].sum()) if not fills.empty and "filled" in fills.columns else 0
        filled_tickers = fills[fills["filled"] == True]["ticker"].tolist() if not fills.empty and "filled" in fills.columns else stop_exits

        health.record("stop_execution", "success", time.time() - t0,
                       f"{filled_count}/{len(submitted)} filled")

        notify(
            f"Stop exits AUTO-EXECUTED for {run_date}\n"
            f"Tickers: {stop_exits}\n"
            f"Filled: {filled_count}/{len(submitted)}",
            level="warning"
        )

        print(f"[runner] Stop exits executed: {filled_count}/{len(submitted)} filled")
        return filled_tickers

    except Exception as e:
        health.record("stop_execution", "failed", time.time() - t0, str(e))
        notify(f"Stop auto-execution FAILED for {run_date}: {e}\nTickers: {stop_exits}", level="critical")
        print(f"[runner] Stop execution failed: {e}")
        return []


# ------------------------------------------------------------
# STOP / TRADE RECONCILIATION
# ------------------------------------------------------------

def _reconcile_stops_and_trades(stop_exits: list, target_weights) -> object:
    """
    Removes stop-triggered tickers from proposed trades.
    Stop exits always execute in full via auto-execute in Stage 6b.
    Removing them from proposed trades prevents a same-day rebuy.
    Position will reappear as a fresh buy at the next rebalance if signal still holds.
    """
    if not stop_exits or target_weights is None or target_weights.empty:
        return target_weights

    filtered = target_weights[~target_weights["ticker"].isin(stop_exits)].copy()
    removed  = [t for t in stop_exits if t in target_weights["ticker"].values]

    if removed:
        print(f"[runner] Removed from proposed trades (stop triggered): {removed}")

    return filtered


# ------------------------------------------------------------
# PROPOSED TRADES WRITER
# ------------------------------------------------------------

def write_proposed_trades(target_weights, run_date: date, regime: str) -> Path:
    """
    Writes optimizer output to data/proposed/proposed_trades_YYYY-MM-DD.csv.
    Also saves target_weights to processed/ for drift detection.
    Sends Slack notification for manual review and approval.
    """
    import pandas as pd

    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)

    target_path = Path("data/processed/target_weights.parquet")
    target_weights.to_parquet(target_path, index=False)

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
        print("[runner] No trades above threshold — skipping proposed trades file")
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

def _print_action_summary(run_date: date) -> None:
    """
    Prints a clean action summary at the end of every pipeline run.
    Shows triggered stop losses and proposed trades with next steps.
    """
    import pandas as pd

    portfolio = load_portfolio()
    prices    = load_prices()

    print("\n" + "=" * 62)
    print("  ACTION SUMMARY")
    print("=" * 62)

    # ── Stop losses ──
    stop_file = STOPS_DIR / f"{run_date.strftime('%Y%m%d')}_stop_exits.csv"
    if stop_file.exists():
        stops = pd.read_csv(stop_file)["ticker"].tolist()
        if stops and not portfolio.empty and not prices.empty:
            latest = (
                prices.sort_values("date")
                .groupby("ticker").last()
                .reset_index()[["ticker", "close"]]
            )
            pf        = portfolio.merge(latest, on="ticker", how="left", suffixes=("", "_latest"))
            close_col = "close_latest" if "close_latest" in pf.columns else "close"

            print(f"\n  STOP LOSSES — {len(stops)} positions triggered")
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

            print(f"\n  Auto-executed during pipeline run")
        elif stops:
            print(f"\n  STOP LOSSES: {stops}")
            print(f"  Auto-executed during pipeline run")
        else:
            print("\n  STOP LOSSES:     none triggered")
    else:
        print("\n  STOP LOSSES:     none triggered")

    # ── Proposed trades ──
    proposed_file = PROPOSED_DIR / f"proposed_trades_{run_date}.csv"
    if proposed_file.exists():
        trades   = pd.read_csv(proposed_file)
        n_buys   = len(trades[trades["direction"] == "BUY"])
        n_sells  = len(trades[trades["direction"] == "SELL"])
        turnover = trades["trade_value_usd"].abs().sum()

        print(f"\n  PROPOSED TRADES — {len(trades)} trades "
              f"({n_buys} buys, {n_sells} sells) | turnover ${turnover:,.0f}")
        print(f"  {'Ticker':<8} {'Dir':<5} {'Cur Wt':>8} {'Tgt Wt':>8} {'Value ($)':>12}")
        print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*12}")
        for _, row in trades.iterrows():
            print(f"  {row['ticker']:<8} {row['direction']:<5} "
                  f"{row['current_weight']:>7.2%} {row['target_weight']:>7.2%} "
                  f"{row['trade_value_usd']:>12,.0f}")
        print(f"\n  → Run: python approve.py   then   python execute.py")
    else:
        print("\n  PROPOSED TRADES: none")

    print("\n" + "=" * 62 + "\n")


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def run_pipeline(run_date: date = None, force_rebalance: bool = False) -> dict:
    """
    Full daily pipeline run.
    run_date:        date to run for (default today)
    force_rebalance: override rebalance calendar and run optimizer regardless
    """
    if run_date is None:
        run_date = date.today()

    if not _acquire_lock():
        return {"status": "aborted", "reason": "already_running"}

    try:
        return _run_pipeline_inner(run_date, force_rebalance)
    finally:
        _release_lock()


def _run_pipeline_inner(run_date: date, force_rebalance: bool) -> dict:
    """Inner pipeline logic — lock is held by caller."""

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
    # STAGE 1 — Broker health check
    # ----------------------------------------------------------
    broker_ok = _run_stage(health, "broker_health", check_broker_health)
    if not broker_ok:
        notify(f"Pipeline aborted {run_date} — broker health check failed", level="critical")
        health.save()
        return {"status": "aborted", "reason": "broker_health"}

    # ----------------------------------------------------------
    # STAGE 2 — Corporate actions
    # ----------------------------------------------------------
    _run_stage(health, "corporate_actions", run_corporate_actions, run_date)

    # ----------------------------------------------------------
    # STAGE 3 — Data refresh
    # ----------------------------------------------------------
    data_ok = _run_stage(health, "data_refresh", run_data_refresh)
    if data_ok is None:
        notify(f"Pipeline aborted {run_date} — data refresh failed", level="critical")
        health.save()
        return {"status": "aborted", "reason": "data_refresh"}

    # ----------------------------------------------------------
    # STAGE 4 — Reprice portfolio + NAV calculation
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
    # STAGE 5 — Regime detection
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
    # STAGE 6 — Risk monitoring
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
            print(f"[runner] Drift trigger detected — interim rebalance needed")

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
    # STAGE 6b — Auto-execute stop losses (P2 — no manual approval)
    # ----------------------------------------------------------
    if stop_exits:
        save_stop_exits(stop_exits, run_date)
        filled_stops = _auto_execute_stops(stop_exits, run_date, health)
        decision["stop_exits"]     = ",".join(stop_exits)
        decision["stops_executed"] = ",".join(filled_stops)

    # ----------------------------------------------------------
    # STAGE 7 — Signal computation
    # ----------------------------------------------------------
    signals = _run_stage(health, "signals", run_combiner, run_date, regime)
    if signals is not None and not signals.empty:
        save_snapshot(signals, "signals", run_date)
        decision["n_signals_scored"] = len(signals)

    # ----------------------------------------------------------
    # STAGE 8 — Signal decay tracking (between rebalances)
    # ----------------------------------------------------------
    decay_result  = _run_stage(health, "decay_tracker", run_decay_tracker, run_date)
    decay_trigger = False
    if decay_result and decay_result.get("triggered"):
        decay_trigger = True
        print(f"[runner] Signal decay trigger — interim rebalance needed")

    # ----------------------------------------------------------
    # STAGE 9 — Optimizer
    # ----------------------------------------------------------
    run_optimizer_flag        = rebalance or drift_trigger or decay_trigger
    decision["optimizer_ran"] = run_optimizer_flag

    if run_optimizer_flag:
        reason = "scheduled" if rebalance else ("drift" if drift_trigger else "decay")
        print(f"[runner] Running optimizer — reason: {reason}")
        decision["optimizer_reason"] = reason

        target_weights = _run_stage(
            health, "optimizer", run_optimizer, run_date, regime
        )

        if target_weights is not None and not target_weights.empty:
            decision["n_positions"] = len(target_weights)
            decision["max_weight"]  = round(float(target_weights["target_weight"].max()), 4)
            save_snapshot(target_weights, "optimizer", run_date)

            # Remove stop-triggered tickers from proposed trades — no same-day rebuy.
            # Stop exits already auto-executed in Stage 6b.
            target_weights = _reconcile_stops_and_trades(stop_exits, target_weights)

            # Write all stop exits to file — always in full, no filtering
            if stop_exits:
                save_stop_exits(stop_exits, run_date)
                print(f"[runner] Stop exits written: {stop_exits}")

            decision["stop_exits"] = ",".join(stop_exits) if stop_exits else ""

            # Stage 10 — Write proposed trades
            proposed_path = _run_stage(
                health, "proposed_trades",
                write_proposed_trades, target_weights, run_date, regime
            )
            decision["proposed_trades_file"] = str(proposed_path) if proposed_path else ""

        else:
            health.record("proposed_trades", "skipped", 0, "optimizer returned empty")

    else:
        # Non-rebalance day — write stops directly
        if stop_exits:
            save_stop_exits(stop_exits, run_date)
            decision["stop_exits"] = ",".join(stop_exits)

        next_rebalance = get_next_rebalance_date(run_date)
        print(f"[runner] Non-rebalance day — skipping optimizer | next rebalance: {next_rebalance}")
        health.record("optimizer",       "skipped", 0, f"next rebalance: {next_rebalance}")
        health.record("proposed_trades", "skipped", 0, "non-rebalance day")
        decision["next_rebalance"] = str(next_rebalance) if next_rebalance else ""

    # ----------------------------------------------------------
    # STAGE 11 — Portfolio history snapshot
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
    # FINAL — Health + decision log + Slack summary
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

    _print_action_summary(run_date)

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