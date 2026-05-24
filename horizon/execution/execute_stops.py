# execute_stops.py
# Manually execute today's trailing stop exits.
# Run after reviewing the stop list in the pipeline log.
# Usage: python execute_stops.py

import pandas as pd
import time
from datetime import date
from data.storage import load_portfolio, load_prices
from execution.order_manager import submit_orders, confirm_fills, update_portfolio_from_fills
from utils.notifications import notify

FILL_WAIT_SECS = 60
run_date = date.today()

# Load triggered stops from portfolio — positions below stop_price
portfolio = load_portfolio()
prices    = load_prices()

if portfolio.empty:
    print("No portfolio found.")
    exit()

latest_prices = (
    prices.sort_values("date")
    .groupby("ticker").last()
    .reset_index()[["ticker", "close"]]
)

portfolio = portfolio.merge(latest_prices, on="ticker", how="left", suffixes=("", "_latest"))
close_col = "close_latest" if "close_latest" in portfolio.columns else "close"

triggered = portfolio[
    portfolio[close_col].notna() &
    portfolio["stop_price"].notna() &
    (portfolio[close_col] < portfolio["stop_price"])
]

if triggered.empty:
    print("No stop losses triggered.")
    exit()

print(f"\nTriggered stops ({len(triggered)}):")
for _, row in triggered.iterrows():
    print(f"  {row['ticker']}  close={row[close_col]:.2f}  stop={row['stop_price']:.2f}")

confirm = input("\nExecute all? (yes/no): ").strip().lower()
if confirm != "yes":
    print("Aborted.")
    exit()

trades = pd.DataFrame({
    "ticker":     triggered["ticker"].tolist(),
    "trade_type": ["sell"] * len(triggered),
    "shares":     triggered["shares"].astype(int).tolist(),
})

submitted = submit_orders(trades)
if submitted.empty:
    print("No orders submitted.")
    exit()

print(f"Waiting {FILL_WAIT_SECS}s for fills...")
time.sleep(FILL_WAIT_SECS)

fills = confirm_fills(submitted)
update_portfolio_from_fills(fills)

filled = int(fills["filled"].sum()) if not fills.empty else 0
print(f"\nDone. {filled}/{len(submitted)} filled.")
notify(f"Stop exits executed manually: {filled} fills | {triggered['ticker'].tolist()}", level="info")