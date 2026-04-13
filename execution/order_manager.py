# execution/order_manager.py
# Handles all order submission and fill confirmation with Alpaca.
# Reads approved trades from data/approved/, submits market orders at 9:35 AM,
# confirms fills at 9:45 AM, updates portfolio and cash in fund_accounting.
# Never submits unapproved trades.

import pandas as pd
import time
from pathlib import Path
from datetime import date, datetime
from utils.config_loader import get_config
from utils.broker_health import _get_alpaca_client
from utils.notifications import notify
from data.storage import load_portfolio, save_portfolio
from fund_accounting.nav import load_cash, adjust_cash

cfg = get_config()

APPROVED_DIR    = Path("data/approved")
FILLS_DIR       = Path("data/processed/fills")
FILL_WAIT_SECS  = 60  # wait time before checking fills


# ------------------------------------------------------------
# APPROVED TRADE LOADING
# ------------------------------------------------------------

def load_approved_trades(run_date: date = None) -> pd.DataFrame:
    """
    Loads the approved trades file for today.
    File must be placed at data/approved/YYYYMMDD_approved_trades.csv
    Returns empty DataFrame if no approved file found.
    """
    if run_date is None:
        run_date = date.today()

    filename = APPROVED_DIR / f"{run_date.strftime('%Y%m%d')}_approved_trades.csv"

    if not filename.exists():
        print(f"[execution] No approved trades file found: {filename}")
        return pd.DataFrame()

    df = pd.read_csv(filename)
    print(f"[execution] Loaded {len(df)} approved trades from {filename}")
    return df


# ------------------------------------------------------------
# ORDER SUBMISSION
# ------------------------------------------------------------

def submit_orders(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Submits market orders to Alpaca for all approved trades.
    trade_type: 'buy' or 'sell'
    shares: number of shares (positive integer)
    Returns DataFrame with order_id added.
    """
    if trades.empty:
        print("[execution] No trades to submit")
        return pd.DataFrame()

    client = _get_alpaca_client()
    submitted = []

    for _, trade in trades.iterrows():
        ticker     = trade["ticker"]
        side       = str(trade["trade_type"]).lower()
        shares     = int(abs(trade["shares"]))

        if shares == 0:
            print(f"[execution] Skipping {ticker} — 0 shares")
            continue

        try:
            order = client.submit_order(
                symbol     = ticker,
                qty        = shares,
                side       = side,
                type       = "market",
                time_in_force = "day"
            )

            submitted.append({
                "ticker":    ticker,
                "side":      side,
                "shares":    shares,
                "order_id":  order.id,
                "status":    order.status,
                "submitted_at": datetime.now()
            })

            print(f"[execution] Order submitted: {side.upper()} {shares} {ticker} | id={order.id}")

        except Exception as e:
            print(f"[execution] Order failed for {ticker}: {e}")
            notify(f"Order submission failed: {ticker} {side} {shares} shares\n{e}", level="critical")

    result = pd.DataFrame(submitted)
    if not result.empty:
        notify(
            f"Orders submitted: {len(result)} orders\n" +
            "\n".join(f"{r['side'].upper()} {r['shares']} {r['ticker']}" for _, r in result.iterrows()),
            level="info"
        )

    return result


# ------------------------------------------------------------
# FILL CONFIRMATION
# ------------------------------------------------------------

def confirm_fills(submitted_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Checks fill status for all submitted orders.
    Called ~10 minutes after submission (9:45 AM).
    Returns DataFrame of confirmed fills with fill price and value.
    """
    if submitted_orders.empty:
        return pd.DataFrame()

    client = _get_alpaca_client()
    fills  = []

    for _, order in submitted_orders.iterrows():
        try:
            result = client.get_order(order["order_id"])

            fill_price = float(result.filled_avg_price) if result.filled_avg_price else None
            fill_qty   = int(result.filled_qty) if result.filled_qty else 0

            fills.append({
                "ticker":      order["ticker"],
                "side":        order["side"],
                "shares":      fill_qty,
                "fill_price":  fill_price,
                "fill_value":  fill_qty * fill_price if fill_price else 0,
                "order_id":    order["order_id"],
                "status":      result.status,
                "filled":      result.status == "filled"
            })

            print(f"[execution] Fill check: {order['ticker']} status={result.status} fill_price={fill_price}")

        except Exception as e:
            print(f"[execution] Fill check failed for {order['ticker']}: {e}")

    return pd.DataFrame(fills)


# ------------------------------------------------------------
# PORTFOLIO UPDATE
# ------------------------------------------------------------

def update_portfolio_from_fills(fills: pd.DataFrame) -> None:
    """
    Updates internal portfolio ledger from confirmed fills.
    Preserves stop_reference_price on all position updates — never resets it.
    """
    if fills.empty:
        return

    portfolio = load_portfolio()
    confirmed = fills[fills["filled"] == True]

    if confirmed.empty:
        print("[execution] No confirmed fills to process")
        return

    if portfolio.empty:
        portfolio = pd.DataFrame(columns=[
            "ticker", "shares", "cost_basis", "market_value",
            "entry_date", "entry_price", "unrealized_pnl",
            "stop_reference_price", "stop_price"
        ])

    for _, fill in confirmed.iterrows():
        ticker     = fill["ticker"]
        side       = fill["side"]
        shares     = int(fill["shares"])
        fill_price = float(fill["fill_price"])
        fill_value = float(fill["fill_value"])

        if side == "buy":
            if ticker in portfolio["ticker"].values:
                idx = portfolio[portfolio["ticker"] == ticker].index[0]
                existing_shares = float(portfolio.at[idx, "shares"])
                existing_cost   = float(portfolio.at[idx, "cost_basis"])
                new_shares      = existing_shares + shares
                new_cost        = (existing_cost * existing_shares + fill_value) / new_shares

                portfolio.at[idx, "shares"]       = new_shares
                portfolio.at[idx, "cost_basis"]   = new_cost
                portfolio.at[idx, "market_value"] = new_shares * fill_price
                # stop_reference_price intentionally NOT touched —
                # compute_trailing_stops will ratchet it up if price has risen
            else:
                new_row = {
                    "ticker":               ticker,
                    "shares":               shares,
                    "cost_basis":           fill_price,
                    "market_value":         fill_value,
                    "entry_date":           date.today(),
                    "entry_price":          fill_price,
                    "unrealized_pnl":       0.0,
                    "stop_reference_price": fill_price,  # entry price is correct initial anchor
                    "stop_price":           None
                }
                portfolio = pd.concat([portfolio, pd.DataFrame([new_row])], ignore_index=True)

            adjust_cash(-fill_value, reason=f"buy {ticker}")

        elif side == "sell":
            if ticker not in portfolio["ticker"].values:
                print(f"[execution] Warning: sell fill for {ticker} but no position found")
                continue

            idx = portfolio[portfolio["ticker"] == ticker].index[0]
            existing_shares = float(portfolio.at[idx, "shares"])
            new_shares      = existing_shares - shares

            if new_shares <= 0.5:  # full exit with float tolerance
                portfolio = portfolio[portfolio["ticker"] != ticker].reset_index(drop=True)
                print(f"[execution] Position closed: {ticker}")
            else:
                portfolio.at[idx, "shares"]       = new_shares
                portfolio.at[idx, "market_value"] = new_shares * fill_price

            adjust_cash(fill_value, reason=f"sell {ticker}")

    save_portfolio(portfolio)
    print(f"[execution] Portfolio updated: {len(portfolio)} positions")

# ------------------------------------------------------------
# RECONCILIATION
# ------------------------------------------------------------

def reconcile_with_alpaca() -> pd.DataFrame:
    """
    Single-tier reconciliation: Alpaca is ground truth for shares and market value.
    NEVER resets stop_reference_price, entry_date, or entry_price for existing positions.
    For positions found only in Alpaca (not internal), reconstructs stop anchor from prices.
    """
    from data.storage import load_prices

    client    = _get_alpaca_client()
    portfolio = load_portfolio()

    try:
        alpaca_positions = client.list_positions()
    except Exception as e:
        print(f"[execution] Reconciliation failed — could not fetch Alpaca positions: {e}")
        return pd.DataFrame()

    alpaca_df = pd.DataFrame([{
        "ticker":        p.symbol,
        "shares":        float(p.qty),
        "market_value":  float(p.market_value),
        "current_price": float(p.current_price)
    } for p in alpaca_positions])

    if alpaca_df.empty and (portfolio.empty or len(portfolio) == 0):
        print("[execution] Reconciliation clean: no positions on either side")
        return pd.DataFrame()

    if not portfolio.empty and not alpaca_df.empty:
        merged = portfolio.merge(
            alpaca_df, on="ticker", how="outer", suffixes=("_internal", "_alpaca")
        )
        discrepancies = merged[
            (merged["shares_internal"] - merged["shares_alpaca"]).abs() > 0.5
        ]
        if not discrepancies.empty:
            print(f"[execution] Reconciliation discrepancies: {discrepancies['ticker'].tolist()}")
            notify(
                f"Reconciliation discrepancies: {discrepancies['ticker'].tolist()}",
                level="warning"
            )

    # Remove positions in internal portfolio but not in Alpaca (sold externally or missed)
    if not portfolio.empty and not alpaca_df.empty:
        alpaca_tickers = set(alpaca_df['ticker'].tolist())
        stale = portfolio[~portfolio['ticker'].isin(alpaca_tickers)]
        if not stale.empty:
            stale_list = stale['ticker'].tolist()
            portfolio = portfolio[portfolio['ticker'].isin(alpaca_tickers)].reset_index(drop=True)
            print(f'[execution] Reconciliation removed stale positions: {stale_list}')
    elif portfolio.empty and not alpaca_df.empty:
        pass  # all positions are new, handled below
    elif not portfolio.empty and alpaca_df.empty:
        stale_list = portfolio['ticker'].tolist()
        portfolio = portfolio.iloc[0:0]  # clear all
        print(f'[execution] Reconciliation: Alpaca has no positions, cleared internal: {stale_list}')

    if not alpaca_df.empty:
        prices = load_prices()

        for _, pos in alpaca_df.iterrows():
            ticker = pos["ticker"]

            if not portfolio.empty and ticker in portfolio["ticker"].values:
                # Existing position — update ONLY shares and market_value
                # Never touch stop_reference_price, entry_date, entry_price
                idx = portfolio[portfolio["ticker"] == ticker].index[0]
                portfolio.at[idx, "shares"]       = pos["shares"]
                portfolio.at[idx, "market_value"] = pos["market_value"]

            else:
                # Position exists in Alpaca but not internally (manual trade or missed fill)
                # Reconstruct stop anchor from price history rather than using today's price
                entry_date = date.today()
                stop_ref   = pos["current_price"]  # fallback

                if not prices.empty and ticker in prices["ticker"].values:
                    ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")
                    if not ticker_prices.empty:
                        # Use max close in the last 252 days as best available anchor
                        stop_ref = float(ticker_prices["close"].tail(252).max())

                new_row = {
                    "ticker":               ticker,
                    "shares":               pos["shares"],
                    "cost_basis":           pos["current_price"],
                    "market_value":         pos["market_value"],
                    "entry_date":           entry_date,
                    "entry_price":          pos["current_price"],
                    "unrealized_pnl":       0.0,
                    "stop_reference_price": stop_ref,
                    "stop_price":           None
                }
                portfolio = pd.concat([portfolio, pd.DataFrame([new_row])], ignore_index=True)
                print(f"[execution] Reconciliation added missing position: {ticker} stop_ref={stop_ref:.2f}")

        save_portfolio(portfolio)

    print(f"[execution] Reconciliation complete: {len(alpaca_df)} Alpaca positions")
    return alpaca_df


# ------------------------------------------------------------
# FULL EXECUTION RUN
# ------------------------------------------------------------

def run_execution(run_date: date = None) -> dict:
    """
    Full execution run for the day.
    1. Load approved trades
    2. Submit orders
    3. Wait for fills
    4. Confirm fills
    5. Update portfolio
    6. Reconcile with Alpaca
    """
    if run_date is None:
        run_date = date.today()

    print(f"[execution] Starting execution run for {run_date}")

    approved = load_approved_trades(run_date)
    if approved.empty:
        print("[execution] No approved trades — skipping execution")
        return {"submitted": 0, "filled": 0, "skipped": True}

    submitted = submit_orders(approved)
    if submitted.empty:
        return {"submitted": 0, "filled": 0, "skipped": False}

    print(f"[execution] Waiting {FILL_WAIT_SECS}s for fills")
    time.sleep(FILL_WAIT_SECS)

    fills = confirm_fills(submitted)
    update_portfolio_from_fills(fills)
    reconcile_with_alpaca()

    filled_count = int(fills["filled"].sum()) if not fills.empty else 0

    notify(
        f"Execution complete for {run_date}\n"
        f"Submitted: {len(submitted)} | Filled: {filled_count}",
        level="info"
    )

    return {
        "submitted": len(submitted),
        "filled":    filled_count,
        "skipped":   False
    }