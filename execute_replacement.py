# execute_replacement.py
# Executes stop replacement (cash deployment) trades proposed by the pipeline.
# Run after reviewing the replacement trades in the pipeline action summary.
#
# Usage:
#   python execute_replacement.py            # today's replacement
#   python execute_replacement.py 2026-04-13 # specific date
#
# Flow:
#   1. python main.py              -> detects excess cash, proposes trades, prints summary
#   2. You review the summary
#   3. python execute_replacement.py -> submits orders, updates portfolio

import sys
import time
import json
import pandas as pd
from datetime import date
from pathlib import Path

from data.storage import load_portfolio
from execution.order_manager import submit_orders, confirm_fills, update_portfolio_from_fills
from fund_accounting.nav import load_nav_history
from utils.notifications import notify

REPLACEMENT_DIR = Path("data/proposed")
EXECUTED_WEIGHTS_PATH = Path("data/processed/executed_weights.parquet")
COOLDOWN_FILE = Path("data/stops/last_replacement.json")
FILL_WAIT_SECS = 60


def _find_replacement_file(run_date: date) -> Path:
    return REPLACEMENT_DIR / f"replacement_trades_{run_date}.csv"


def _update_executed_weights() -> None:
    """Snapshots current portfolio weights after execution."""
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
    if "sector" in portfolio.columns:
        weights["sector"] = portfolio["sector"]
    weights.to_parquet(EXECUTED_WEIGHTS_PATH, index=False)
    print(f"[execute_replacement] Executed weights updated: {len(weights)} positions")


def _record_cooldown(run_date: date) -> None:
    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COOLDOWN_FILE, "w") as f:
        json.dump({"last_replacement_date": str(run_date)}, f)


def run_replacement_execution(run_date: date = None) -> dict:
    if run_date is None:
        run_date = date.today()

    print(f"\n[execute_replacement] ============================================================")
    print(f"[execute_replacement] Replacement execution: {run_date}")
    print(f"[execute_replacement] ============================================================\n")

    # ----------------------------------------------------------
    # STEP 1 -- Find and read replacement file
    # ----------------------------------------------------------
    repl_file = _find_replacement_file(run_date)
    if not repl_file.exists():
        print(f"[execute_replacement] No replacement file for {run_date}")
        print(f"[execute_replacement] Expected: {repl_file}")
        return {"submitted": 0, "filled": 0, "skipped": True}

    trades_df = pd.read_csv(repl_file)
    if trades_df.empty:
        print("[execute_replacement] Replacement file is empty")
        return {"submitted": 0, "filled": 0, "skipped": True}

    n_buys = len(trades_df[trades_df["trade_type"] == "buy"])
    n_sells = len(trades_df[trades_df["trade_type"] == "sell"])

    print(f"[execute_replacement] Trades to execute: {len(trades_df)} ({n_buys} buys, {n_sells} trims)")
    print()
    print(f"  {'Ticker':<8} {'Side':<6} {'Shares':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*8}")
    for _, row in trades_df.iterrows():
        print(f"  {row['ticker']:<8} {row['trade_type']:<6} {int(row['shares']):>8}")

    # ----------------------------------------------------------
    # STEP 2 -- Submit orders
    # ----------------------------------------------------------
    print(f"\n[execute_replacement] Submitting {len(trades_df)} orders...")
    submitted = submit_orders(trades_df)

    if submitted.empty:
        print("[execute_replacement] No orders submitted")
        return {"submitted": 0, "filled": 0, "skipped": False}

    # ----------------------------------------------------------
    # STEP 3 -- Wait for fills
    # ----------------------------------------------------------
    print(f"[execute_replacement] Waiting {FILL_WAIT_SECS}s for fills...")
    time.sleep(FILL_WAIT_SECS)

    # ----------------------------------------------------------
    # STEP 4 -- Confirm fills and update portfolio
    # ----------------------------------------------------------
    fills = confirm_fills(submitted)
    update_portfolio_from_fills(fills)

    filled_count = int(fills["filled"].sum()) if not fills.empty and "filled" in fills.columns else 0

    # ----------------------------------------------------------
    # STEP 5 -- Update executed weights and cooldown
    # ----------------------------------------------------------
    _update_executed_weights()
    _record_cooldown(run_date)

    # ----------------------------------------------------------
    # STEP 6 -- Clear the replacement file
    # ----------------------------------------------------------
    repl_file.unlink()
    print(f"[execute_replacement] Cleared: {repl_file.name}")

    # ----------------------------------------------------------
    # DONE
    # ----------------------------------------------------------
    print(f"\n[execute_replacement] ============================================================")
    print(f"[execute_replacement] Complete: {filled_count}/{len(submitted)} filled")
    print(f"[execute_replacement] ============================================================\n")

    notify(
        f"Replacement trades executed for {run_date}\n"
        f"Filled: {filled_count}/{len(submitted)}",
        level="info"
    )

    return {
        "submitted": len(submitted),
        "filled": filled_count,
        "skipped": False,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_date = date.fromisoformat(sys.argv[1])
    else:
        target_date = date.today()

    result = run_replacement_execution(target_date)
    print(f"\nResult: {result}")
