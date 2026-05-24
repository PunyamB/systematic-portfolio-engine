# fund_accounting/nav.py
# Computes daily NAV, tracks cash, dividend income, and NAV per share.
# No fee accrual — this is a paper trading system with no investors.
# NAV = market value of all positions + cash balance.

import pandas as pd
import json
from pathlib import Path
from datetime import date
from data.storage import load_portfolio, save_portfolio, load_parquet, save_parquet
from utils.config_loader import get_config
from utils.notifications import notify

cfg = get_config()

NAV_FILE       = Path("data/processed/nav_history.parquet")
CASH_FILE      = Path("data/processed/cash.json")
INITIAL_CAPITAL = cfg["portfolio"]["initial_capital"]


# ------------------------------------------------------------
# CASH MANAGEMENT
# ------------------------------------------------------------

def load_cash() -> float:
    """
    Loads current cash balance from disk.
    Returns initial capital if no cash file exists yet.
    """
    if not CASH_FILE.exists():
        return float(INITIAL_CAPITAL)
    with open(CASH_FILE, "r") as f:
        return float(json.load(f)["cash"])


def save_cash(cash: float) -> None:
    CASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CASH_FILE, "w") as f:
        json.dump({"cash": cash}, f)
    print(f"[fund_accounting] Cash saved: ${cash:,.2f}")


def adjust_cash(amount: float, reason: str = "") -> float:
    """
    Adds or subtracts from cash balance.
    Positive amount = inflow (dividend, sale proceeds).
    Negative amount = outflow (purchase cost).
    Returns new cash balance.
    """
    cash = load_cash()
    new_cash = cash + amount
    save_cash(new_cash)
    if reason:
        print(f"[fund_accounting] Cash adjustment: {reason} ${amount:,.2f} -> balance ${new_cash:,.2f}")
    return new_cash


# ------------------------------------------------------------
# NAV COMPUTATION
# ------------------------------------------------------------

def compute_nav(run_date: date = None) -> dict:
    """
    Computes NAV for the day.
    NAV = sum(shares * current_price) + cash
    Returns a dict with nav, cash, equity_value, and daily_return.
    """
    if run_date is None:
        run_date = date.today()

    portfolio = load_portfolio()
    cash      = load_cash()

    if portfolio.empty:
        equity_value = 0.0
    else:
        if "market_value" not in portfolio.columns:
            print("[fund_accounting] market_value column missing from portfolio")
            equity_value = 0.0
        else:
            equity_value = float(portfolio["market_value"].sum())

    nav = equity_value + cash

    # Load previous NAV for daily return calculation
    nav_history = load_nav_history()
    if nav_history.empty:
        daily_return = 0.0
    else:
        prev_nav = float(nav_history.iloc[-1]["nav"])
        daily_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0

    result = {
        "date":         run_date,
        "nav":          nav,
        "equity_value": equity_value,
        "cash":         cash,
        "daily_return": daily_return
    }

    print(f"[fund_accounting] NAV: ${nav:,.2f} | Equity: ${equity_value:,.2f} | Cash: ${cash:,.2f} | Return: {daily_return:.4%}")
    return result


# ------------------------------------------------------------
# NAV HISTORY
# ------------------------------------------------------------

def load_nav_history() -> pd.DataFrame:
    if not NAV_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(NAV_FILE)


def save_nav_entry(nav_record: dict) -> None:
    """
    Appends a daily NAV record to the history file.
    """
    new_row = pd.DataFrame([nav_record])
    new_row["date"] = pd.to_datetime(new_row["date"])

    history = load_nav_history()

    if not history.empty:
        # Remove existing entry for same date if re-running
        history = history[history["date"].dt.date != nav_record["date"]]
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history = history.sort_values("date").reset_index(drop=True)
    history.to_parquet(NAV_FILE, index=False)
    print(f"[fund_accounting] NAV history updated: {len(history)} records")


def record_dividend_income(dividends_df: pd.DataFrame) -> None:
    """
    Credits dividend income to cash balance.
    Called by corporate_actions after processing dividends.
    """
    if dividends_df.empty:
        return

    total = float(dividends_df["total_dividend"].sum())
    adjust_cash(total, reason="dividend income")
    print(f"[fund_accounting] Dividend income credited: ${total:,.2f}")


def run_nav(run_date: date = None, dividends_df: pd.DataFrame = None) -> dict:
    """
    Full daily NAV run.
    1. Credits any dividend income
    2. Computes NAV
    3. Saves to history
    4. Sends Slack summary
    """
    if run_date is None:
        run_date = date.today()

    if dividends_df is not None and not dividends_df.empty:
        record_dividend_income(dividends_df)

    nav_record = compute_nav(run_date)
    save_nav_entry(nav_record)

    notify(
        f"NAV update for {run_date}\n"
        f"NAV: ${nav_record['nav']:,.2f}\n"
        f"Daily Return: {nav_record['daily_return']:.4%}\n"
        f"Cash: ${nav_record['cash']:,.2f}",
        level="info"
    )

    return nav_record