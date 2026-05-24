# execute_stops.py
# Executes trailing stop exits detected by the pipeline.
# Run after reviewing stops in the pipeline action summary.
#
# Usage:
#   python execute_stops.py            # today's stops
#   python execute_stops.py 2026-04-08 # specific date
#
# Flow:
#   1. python main.py          -> detects stops, writes file, prints summary
#   2. You review the summary
#   3. python execute_stops.py -> submits sell orders, updates portfolio
#
# After execution:
#   - Stopped tickers written to stop_cooldown.json (4 trading day rebuy block)
#   - executed_weights.parquet updated (drift anchor)
#   - Stop exits file cleared

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

STOPS_DIR = Path("data/stops")
EXECUTED_WEIGHTS_PATH = Path("data/processed/executed_weights.parquet")
COOLDOWN_FILE = Path("data/stops/stop_cooldown.json")
FILL_WAIT_SECS = 60


def _find_stop_exits_file(run_date: date) -> Path:
    return STOPS_DIR / f"{run_date.strftime('%Y%m%d')}_stop_exits.csv"


def _update_executed_weights(stop_tickers: list) -> None:
    """Zeros out stopped tickers in executed_weights.parquet."""
    if not stop_tickers or not EXECUTED_WEIGHTS_PATH.exists():
        return
    ew = pd.read_parquet(EXECUTED_WEIGHTS_PATH)
    if ew.empty:
        return
    mask = ew["ticker"].isin(stop_tickers)
    if mask.sum() > 0:
        ew.loc[mask, "target_weight"] = 0.0
        ew.to_parquet(EXECUTED_WEIGHTS_PATH, index=False)
        print(f"[execute_stops] Executed weights zeroed: {stop_tickers}")


def _record_stop_cooldown(tickers: list, run_date: date) -> None:
    """
    Writes stopped tickers to the cooldown ledger.
    These tickers cannot be bought back for 4 trading days.
    Format: {"TICKER": "2026-04-23", ...}
    """
    cooldown = {}
    if COOLDOWN_FILE.exists():
        try:
            with open(COOLDOWN_FILE) as f:
                cooldown = json.load(f)
        except Exception:
            cooldown = {}

    for ticker in tickers:
        cooldown[ticker] = str(run_date)

    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COOLDOWN_FILE, "w") as f:
        json.dump(cooldown, f, indent=2)

    print(f"[execute_stops] Cooldown recorded: {tickers} (4 trading day rebuy block)")


def _clear_stop_exits_file(filepath: Path) -> None:
    if filepath.exists():
        filepath.unlink()
        print(f"[execute_stops] Cleared: {filepath.name}")


def run_stop_execution(run_date: date = None) -> dict:
    if run_date is None:
        run_date = date.today()

    print(f"\n[execute_stops] ============================================================")
    print(f"[execute_stops] Stop execution: {run_date}")
    print(f"[execute_stops] ============================================================\n")

    # STEP 1 -- Find and read stop exits file
    stop_file = _find_stop_exits_file(run_date)
    if not stop_file.exists():
        print(f"[execute_stops] No stop exits file for {run_date}")
        print(f"[execute_stops] Expected: {stop_file}")
        print(f"[execute_stops] Run 'python main.py' first")
        return {"submitted": 0, "filled": 0, "skipped": True}

    stop_df = pd.read_csv(stop_file)
    if stop_df.empty or "ticker" not in stop_df.columns:
        print("[execute_stops] Stop exits file is empty")
        return {"submitted": 0, "filled": 0, "skipped": True}

    stop_tickers = stop_df["ticker"].tolist()
    print(f"[execute_stops] Stop exits to execute: {stop_tickers}")

    # STEP 2 -- Build sell orders from current portfolio
    portfolio = load_portfolio()
    if portfolio.empty:
        print("[execute_stops] No portfolio found")
        return {"submitted": 0, "filled": 0, "skipped": True}

    stop_positions = portfolio[portfolio["ticker"].isin(stop_tickers)]
    if stop_positions.empty:
        print("[execute_stops] None of the stopped tickers are in portfolio")
        print("[execute_stops] They may have already been executed")
        _clear_stop_exits_file(stop_file)
        return {"submitted": 0, "filled": 0, "skipped": True}

    print(f"\n  Stops to execute ({len(stop_positions)}):")
    print(f"  {'Ticker':<8} {'Shares':>8} {'Mkt Value':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*12}")
    for _, row in stop_positions.iterrows():
        print(f"  {row['ticker']:<8} {int(row['shares']):>8} {row['market_value']:>12,.2f}")

    trades = pd.DataFrame({
        "ticker":     stop_positions["ticker"].tolist(),
        "trade_type": ["sell"] * len(stop_positions),
        "shares":     stop_positions["shares"].astype(int).tolist(),
    })

    # STEP 3 -- Submit orders
    print(f"\n[execute_stops] Submitting {len(trades)} sell orders...")
    submitted = submit_orders(trades)

    if submitted.empty:
        print("[execute_stops] No orders submitted")
        return {"submitted": 0, "filled": 0, "skipped": False}

    # STEP 4 -- Wait for fills
    print(f"[execute_stops] Waiting {FILL_WAIT_SECS}s for fills...")
    time.sleep(FILL_WAIT_SECS)

    # STEP 5 -- Confirm fills and update portfolio
    fills = confirm_fills(submitted)
    update_portfolio_from_fills(fills)

    filled_count = int(fills["filled"].sum()) if not fills.empty and "filled" in fills.columns else 0

    # STEP 6 -- Update executed weights (drift anchor)
    _update_executed_weights(stop_tickers)

    # STEP 7 -- Record stop cooldown (4 trading day rebuy block)
    _record_stop_cooldown(stop_tickers, run_date)

    # STEP 8 -- Clear the stop exits file
    _clear_stop_exits_file(stop_file)

    print(f"\n[execute_stops] ============================================================")
    print(f"[execute_stops] Complete: {filled_count}/{len(submitted)} filled")
    print(f"[execute_stops] ============================================================\n")

    notify(
        f"Stop exits executed for {run_date}\n"
        f"Tickers: {stop_tickers}\n"
        f"Filled: {filled_count}/{len(submitted)}\n"
        f"Cooldown: 4 trading days",
        level="warning"
    )

    return {
        "submitted": len(submitted),
        "filled":    filled_count,
        "skipped":   False,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_date = date.fromisoformat(sys.argv[1])
    else:
        target_date = date.today()

    result = run_stop_execution(target_date)
    print(f"\nResult: {result}")
