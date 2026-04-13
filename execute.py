# execute.py
# Morning execution entry point.
# Reads approved trades, submits to Alpaca, confirms fills,
# updates portfolio, reconciles, logs everything.
#
# Usage:
#   python execute.py            # executes today's approved trades
#   python execute.py 2026-03-07 # executes a specific date's trades
#
# Prerequisites:
#   1. Pipeline must have run (python main.py)
#   2. Trades must be approved (python approve.py)

import sys
import pandas as pd
from datetime import date, datetime
from pathlib import Path

from execution.order_manager import (
    load_approved_trades,
    submit_orders,
    confirm_fills,
    update_portfolio_from_fills,
    reconcile_with_alpaca,
    FILL_WAIT_SECS,
)
from compliance.checker import run_pre_trade_checks, run_post_trade_checks
from data.storage import load_prices, load_portfolio, append_execution_log
from fund_accounting.nav import load_nav_history
from utils.notifications import notify
import time

EXECUTED_WEIGHTS_PATH = Path("data/processed/executed_weights.parquet")


def _save_executed_weights() -> None:
    """
    Snapshots current portfolio weights to executed_weights.parquet.
    This is the anchor for drift detection -- only updated after real
    trade execution (rebalance or stop exits), never by proposals.
    """
    portfolio = load_portfolio()
    if portfolio.empty:
        return

    nav_history = load_nav_history()
    if nav_history.empty:
        return

    nav = float(nav_history.iloc[-1]["nav"])
    if nav <= 0:
        return

    weights = portfolio[["ticker"]].copy()
    weights["target_weight"] = portfolio["market_value"] / nav

    # Include sector if available
    if "sector" in portfolio.columns:
        weights["sector"] = portfolio["sector"]

    weights.to_parquet(EXECUTED_WEIGHTS_PATH, index=False)
    print(f"[execute] Executed weights saved: {len(weights)} positions")


def run_execution(run_date: date = None) -> dict:
    """
    Full execution run with logging.
    1. Load approved trades
    2. Run pre-trade compliance checks
    3. Submit orders to Alpaca
    4. Wait for fills
    5. Confirm fills + compute slippage
    6. Update internal portfolio
    7. Save executed weights (drift detection anchor)
    8. Run post-trade compliance checks
    9. Reconcile with Alpaca
    10. Log everything to execution_log.parquet
    """
    if run_date is None:
        run_date = date.today()

    print(f"\n[execute] ============================================================")
    print(f"[execute] Execution start: {run_date}")
    print(f"[execute] ============================================================\n")

    # ----------------------------------------------------------
    # STEP 1 -- Load approved trades
    # ----------------------------------------------------------
    approved = load_approved_trades(run_date)
    if approved.empty:
        print("[execute] No approved trades found -- nothing to execute")
        print(f"[execute] Expected file: data/approved/{run_date.strftime('%Y%m%d')}_approved_trades.csv")
        print(f"[execute] Run 'python approve.py' first")
        return {"submitted": 0, "filled": 0, "skipped": True}

    print(f"[execute] Loaded {len(approved)} approved trades")

    # ----------------------------------------------------------
    # STEP 2 -- Pre-trade compliance
    # ----------------------------------------------------------
    print("\n[execute] Running pre-trade compliance checks...")
    compliance_results = run_pre_trade_checks(approved)
    failures = [r for r in compliance_results if not r.passed]

    if failures:
        print("[execute] Pre-trade compliance FAILED -- aborting execution")
        for f in failures:
            print(f"  {f}")
        notify(
            f"Execution aborted -- pre-trade compliance failed\n"
            + "\n".join(str(f) for f in failures),
            level="critical"
        )
        return {"submitted": 0, "filled": 0, "skipped": False, "reason": "compliance"}

    # ----------------------------------------------------------
    # STEP 3 -- Submit orders
    # ----------------------------------------------------------
    print("\n[execute] Submitting orders to Alpaca...")
    submitted = submit_orders(approved)

    if submitted.empty:
        print("[execute] No orders submitted")
        return {"submitted": 0, "filled": 0, "skipped": False}

    print(f"[execute] {len(submitted)} orders submitted")

    # ----------------------------------------------------------
    # STEP 4 -- Wait for fills
    # ----------------------------------------------------------
    print(f"[execute] Waiting {FILL_WAIT_SECS}s for fills...")
    time.sleep(FILL_WAIT_SECS)

    # ----------------------------------------------------------
    # STEP 5 -- Confirm fills + compute slippage
    # ----------------------------------------------------------
    fills = confirm_fills(submitted)

    # Compute slippage vs prior close
    if not fills.empty:
        prices = load_prices()
        if not prices.empty:
            latest_close = (
                prices.sort_values("date")
                .groupby("ticker")
                .last()
                .reset_index()[["ticker", "close"]]
                .rename(columns={"close": "prior_close"})
            )
            fills = fills.merge(latest_close, on="ticker", how="left")
            fills["slippage_bps"] = fills.apply(
                lambda r: round(
                    (r["fill_price"] / r["prior_close"] - 1) * 10000, 2
                ) if pd.notna(r.get("fill_price")) and pd.notna(r.get("prior_close"))
                    and r["prior_close"] > 0
                else 0.0,
                axis=1
            )
        else:
            fills["prior_close"]  = None
            fills["slippage_bps"] = 0.0

        fills["execution_date"] = run_date
        fills["executed_at"]    = datetime.now().isoformat()

    # ----------------------------------------------------------
    # STEP 6 -- Update portfolio
    # ----------------------------------------------------------
    update_portfolio_from_fills(fills)

    # ----------------------------------------------------------
    # STEP 7 -- Save executed weights (drift detection anchor)
    # ----------------------------------------------------------
    _save_executed_weights()

    # ----------------------------------------------------------
    # STEP 8 -- Post-trade compliance
    # ----------------------------------------------------------
    print("\n[execute] Running post-trade compliance checks...")
    post_results = run_post_trade_checks()
    post_failures = [r for r in post_results if not r.passed]
    if post_failures:
        print("[execute] Post-trade compliance issues detected:")
        for f in post_failures:
            print(f"  {f}")

    # ----------------------------------------------------------
    # STEP 9 -- Reconcile with Alpaca
    # ----------------------------------------------------------
    print("\n[execute] Reconciling with Alpaca...")
    reconcile_with_alpaca()

    # ----------------------------------------------------------
    # STEP 10 -- Log execution
    # ----------------------------------------------------------
    if not fills.empty:
        append_execution_log(fills)
        print(f"[execute] Execution log updated: {len(fills)} records")

    filled_count = int(fills["filled"].sum()) if not fills.empty else 0
    failed_count = len(fills) - filled_count if not fills.empty else 0

    print(f"\n[execute] ============================================================")
    print(f"[execute] Execution complete: {filled_count} filled, {failed_count} failed")
    print(f"[execute] ============================================================\n")

    notify(
        f"Execution complete for {run_date}\n"
        f"Submitted: {len(submitted)} | Filled: {filled_count} | Failed: {failed_count}",
        level="info"
    )

    return {
        "submitted": len(submitted),
        "filled":    filled_count,
        "failed":    failed_count,
        "skipped":   False,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_date = date.fromisoformat(sys.argv[1])
    else:
        target_date = date.today()

    result = run_execution(target_date)
    print(f"\nResult: {result}")
