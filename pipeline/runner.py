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
# 9. [REBALANCE ONLY] Optimizer
# 10. [REBALANCE ONLY] Write proposed_trades.csv + Slack notification
# 11. Pipeline health report

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

cfg = get_config()

HEALTH_FILE       = Path("data/pipeline_health.json")
PROPOSED_DIR      = Path("data/proposed")
APPROVED_DIR      = Path("data/approved")


# ------------------------------------------------------------
# PIPELINE HEALTH TRACKING
# ------------------------------------------------------------

class PipelineHealth:
    def __init__(self, run_date: date):
        self.run_date  = run_date
        self.stages    = {}
        self.start_time = time.time()

    def record(self, stage: str, status: str, duration_sec: float, detail: str = ""):
        self.stages[stage] = {
            "status":       status,   # success / failed / skipped
            "duration_sec": round(duration_sec, 2),
            "detail":       detail,
        }
        symbol = "OK" if status == "success" else ("SKIP" if status == "skipped" else "FAIL")
        print(f"[runner] [{symbol}] {stage} ({duration_sec:.1f}s) {detail}")

    def save(self):
        HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_date":         str(self.run_date),
            "run_at":           datetime.now().isoformat(),
            "total_sec":        round(time.time() - self.start_time, 2),
            "stages":           self.stages,
        }
        with open(HEALTH_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[runner] Pipeline health saved to {HEALTH_FILE}")


def _run_stage(health: PipelineHealth, stage: str, fn, *args, **kwargs):
    """
    Runs a pipeline stage with timing and error handling.
    Returns result on success, None on failure.
    """
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
# PROPOSED TRADES WRITER
# ------------------------------------------------------------

def write_proposed_trades(target_weights, run_date: date, regime: str) -> Path:
    """
    Writes optimizer output to data/proposed/proposed_trades_YYYY-MM-DD.csv.
    Sends Slack notification for manual review and approval.
    """
    import pandas as pd
    from data.storage import load_portfolio
    from fund_accounting.nav import load_nav_history

    PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)

    nav_history = load_nav_history()
    nav         = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else cfg["portfolio"]["initial_capital"]
    portfolio   = load_portfolio()

    # Current weights
    current_weights = {}
    if not portfolio.empty:
        for _, row in portfolio.iterrows():
            current_weights[row["ticker"]] = row["market_value"] / nav

    # Build trade list
    trades = []
    for _, row in target_weights.iterrows():
        ticker         = row["ticker"]
        target_wt      = row["target_weight"]
        current_wt     = current_weights.get(ticker, 0.0)
        delta_wt       = target_wt - current_wt
        trade_value    = delta_wt * nav
        direction      = "BUY" if delta_wt > 0 else "SELL"

        if abs(delta_wt) < 0.001:  # skip tiny rebalances below 0.1%
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
            "run_date":        run_date,
        })

    # Add full exits for tickers no longer in target
    target_tickers = set(target_weights["ticker"].tolist())
    for ticker, current_wt in current_weights.items():
        if ticker not in target_tickers and current_wt > 0.001:
            trades.append({
                "ticker":          ticker,
                "direction":       "SELL",
                "current_weight":  round(current_wt, 4),
                "target_weight":   0.0,
                "delta_weight":    round(-current_wt, 4),
                "trade_value_usd": round(-current_wt * nav, 2),
                "sector":          "",
                "regime":          regime,
                "run_date":        run_date,
            })

    if not trades:
        print("[runner] No trades above threshold — skipping proposed trades file")
        return None

    df        = pd.DataFrame(trades)
    out_path  = PROPOSED_DIR / f"proposed_trades_{run_date}.csv"
    df.to_csv(out_path, index=False)

    n_buys  = len(df[df["direction"] == "BUY"])
    n_sells = len(df[df["direction"] == "SELL"])
    total_turnover = df["trade_value_usd"].abs().sum()

    print(f"[runner] Proposed trades written: {len(df)} trades | "
          f"buys={n_buys} sells={n_sells} | turnover=${total_turnover:,.0f}")

    notify(
        f"Proposed trades ready for {run_date}\n"
        f"Regime: {regime}\n"
        f"Trades: {len(df)} | Buys: {n_buys} | Sells: {n_sells}\n"
        f"Estimated turnover: ${total_turnover:,.0f}\n"
        f"File: {out_path}\n"
        f"To approve: copy file to data/approved/ with same filename",
        level="info"
    )

    return out_path


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

    health    = PipelineHealth(run_date)
    rebalance = is_rebalance_day(run_date) or force_rebalance

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
    # STAGE 4 — NAV calculation
    # ----------------------------------------------------------
    _run_stage(health, "nav", run_nav, run_date)

    # ----------------------------------------------------------
    # STAGE 5 — Regime detection
    # ----------------------------------------------------------
    regime_result = _run_stage(health, "regime", detect_regime, run_date)
    regime        = regime_result.get("composite_regime", "recovery") if regime_result else "recovery"
    print(f"[runner] Regime: {regime}")

    # ----------------------------------------------------------
    # STAGE 6 — Risk monitoring
    # ----------------------------------------------------------
    risk_result = _run_stage(health, "risk_monitor", run_risk_monitor, run_date)

    # Check for risk-triggered priority trades (P1, P2)
    priority_trades = []
    drift_trigger   = False

    if risk_result:
        circuit_breaker = risk_result.get("circuit_breaker", {})
        stop_exits      = risk_result.get("stop_exits", [])
        drift_result    = risk_result.get("drift", {})

        # P1 — Circuit breaker / delisted / index deletion exits
        cb_level = circuit_breaker.get("level", 0)
        if cb_level >= 1:
            print(f"[runner] Circuit breaker T{cb_level} active")

        # P2 — Trailing stop exits
        if stop_exits:
            print(f"[runner] Trailing stop exits: {stop_exits}")
            priority_trades.extend(stop_exits)

        # Drift trigger — forces interim rebalance
        if drift_result.get("triggered"):
            drift_trigger = True
            print(f"[runner] Drift trigger detected — interim rebalance needed")

    # ----------------------------------------------------------
    # STAGE 7 — Signal computation
    # ----------------------------------------------------------
    signals = _run_stage(health, "signals", run_combiner, run_date, regime)

    # ----------------------------------------------------------
    # STAGE 8 — Signal decay tracking (between rebalances)
    # ----------------------------------------------------------
    decay_result  = _run_stage(health, "decay_tracker", run_decay_tracker, run_date)
    decay_trigger = False

    if decay_result and decay_result.get("triggered"):
        decay_trigger = True
        print(f"[runner] Signal decay trigger — interim rebalance needed")

    # ----------------------------------------------------------
    # STAGE 9 — Optimizer (rebalance days + drift/decay triggers)
    # ----------------------------------------------------------
    run_optimizer_flag = rebalance or drift_trigger or decay_trigger

    if run_optimizer_flag:
        reason = "scheduled" if rebalance else ("drift" if drift_trigger else "decay")
        print(f"[runner] Running optimizer — reason: {reason}")

        target_weights = _run_stage(
            health, "optimizer", run_optimizer, run_date, regime
        )

        # ----------------------------------------------------------
        # STAGE 10 — Write proposed trades
        # ----------------------------------------------------------
        if target_weights is not None and not target_weights.empty:
            _run_stage(
                health, "proposed_trades",
                write_proposed_trades, target_weights, run_date, regime
            )
        else:
            health.record("proposed_trades", "skipped", 0, "optimizer returned empty")
    else:
        next_rebalance = get_next_rebalance_date(run_date)
        print(f"[runner] Non-rebalance day — skipping optimizer | next rebalance: {next_rebalance}")
        health.record("optimizer",       "skipped", 0, f"next rebalance: {next_rebalance}")
        health.record("proposed_trades", "skipped", 0, "non-rebalance day")

    # ----------------------------------------------------------
    # FINAL — Health report + summary Slack
    # ----------------------------------------------------------
    health.save()

    failed_stages = [s for s, v in health.stages.items() if v["status"] == "failed"]
    status        = "failed" if failed_stages else "success"

    notify(
        f"Pipeline complete for {run_date}\n"
        f"Status: {status}\n"
        f"Regime: {regime}\n"
        f"Rebalance: {rebalance}\n"
        f"Failed stages: {failed_stages if failed_stages else 'none'}\n"
        f"Total time: {round(time.time() - health.start_time, 1)}s",
        level="info" if status == "success" else "warning"
    )

    print(f"\n[runner] ============================================================")
    print(f"[runner] Pipeline complete: {status} | {round(time.time() - health.start_time, 1)}s")
    print(f"[runner] ============================================================\n")

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