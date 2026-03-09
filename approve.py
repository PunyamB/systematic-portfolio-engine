# approve.py
# Trade approval helper.
# Converts proposed trades (weights/dollars) to execution-ready format (shares).
# Fetches latest prices to compute accurate share counts.
#
# Usage:
#   python approve.py            # approves today's proposed trades
#   python approve.py 2026-03-07 # approves a specific date's trades

import sys
import pandas as pd
from pathlib import Path
from datetime import date

from data.storage import load_prices
from fund_accounting.nav import load_nav_history
from utils.config_loader import get_config

cfg = get_config()

PROPOSED_DIR = Path("data/proposed")
APPROVED_DIR = Path("data/approved")


def approve_trades(run_date: date = None) -> Path | None:
    """
    Loads proposed trades for run_date, converts to execution format,
    and writes to data/approved/.

    Conversion:
    - Fetches latest close prices for all tickers
    - Computes share counts: shares = abs(trade_value_usd) / close_price
    - Rounds down to whole shares (no fractional)
    - Maps 'direction' -> 'trade_type' (BUY->buy, SELL->sell)
    - Drops trades with 0 shares (too small to execute)

    Returns path to approved file, or None if no proposed trades found.
    """
    if run_date is None:
        run_date = date.today()

    proposed_file = PROPOSED_DIR / f"proposed_trades_{run_date}.csv"

    if not proposed_file.exists():
        print(f"[approve] No proposed trades found: {proposed_file}")
        print(f"[approve] Available files:")
        for f in sorted(PROPOSED_DIR.glob("proposed_trades_*.csv")):
            print(f"  {f.name}")
        return None

    proposed = pd.read_csv(proposed_file)
    print(f"[approve] Loaded {len(proposed)} proposed trades for {run_date}")

    # Get latest close prices
    prices = load_prices()
    if prices.empty:
        print("[approve] ERROR: No price data available — cannot compute share counts")
        return None

    latest_prices = (
        prices.sort_values("date")
        .groupby("ticker")
        .last()
        .reset_index()[["ticker", "close"]]
    )

    # Merge prices
    proposed = proposed.merge(latest_prices, on="ticker", how="left")

    missing_prices = proposed[proposed["close"].isna()]["ticker"].tolist()
    if missing_prices:
        print(f"[approve] WARNING: No price data for {len(missing_prices)} tickers: {missing_prices[:5]}")
        proposed = proposed.dropna(subset=["close"])

    # Compute share counts
    proposed["shares"] = (proposed["trade_value_usd"].abs() / proposed["close"]).apply(
        lambda x: int(x)  # round down to whole shares
    )

    # Map direction to trade_type
    proposed["trade_type"] = proposed["direction"].str.lower()

    # Drop zero-share trades
    proposed = proposed[proposed["shares"] > 0].reset_index(drop=True)

    if proposed.empty:
        print("[approve] All trades rounded to 0 shares — nothing to approve")
        return None

    # Select columns for execution
    approved = proposed[[
        "ticker", "trade_type", "shares", "direction",
        "current_weight", "target_weight", "delta_weight",
        "trade_value_usd", "close", "sector", "regime", "run_date"
    ]].copy()

    # Write approved file
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    approved_filename = f"{run_date.strftime('%Y%m%d')}_approved_trades.csv"
    approved_path = APPROVED_DIR / approved_filename
    approved.to_csv(approved_path, index=False)

    n_buys  = len(approved[approved["trade_type"] == "buy"])
    n_sells = len(approved[approved["trade_type"] == "sell"])
    total_shares = int(approved["shares"].sum())
    total_value  = approved["trade_value_usd"].abs().sum()

    print(f"\n[approve] ========================================")
    print(f"[approve] TRADES APPROVED for {run_date}")
    print(f"[approve] Buys: {n_buys} | Sells: {n_sells}")
    print(f"[approve] Total shares: {total_shares:,}")
    print(f"[approve] Estimated value: ${total_value:,.0f}")
    print(f"[approve] File: {approved_path}")
    print(f"[approve] ========================================")
    print(f"\n[approve] Next step: python execute.py")

    return approved_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_date = date.fromisoformat(sys.argv[1])
    else:
        target_date = date.today()

    result = approve_trades(target_date)
    if result is None:
        sys.exit(1)
