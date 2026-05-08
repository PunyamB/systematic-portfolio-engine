import pandas as pd
import numpy as np
from data.storage import load_portfolio, load_prices, save_portfolio

portfolio = load_portfolio()
prices    = load_prices()

print(f"Resetting stop_reference_price for {len(portfolio)} positions...")

for idx, row in portfolio.iterrows():
    ticker        = row["ticker"]
    entry_date    = row.get("entry_date", None)
    ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")

    if ticker_prices.empty:
        continue

    if pd.notna(entry_date):
        prices_since_entry = ticker_prices[ticker_prices["date"] >= pd.Timestamp(entry_date)]
    else:
        prices_since_entry = ticker_prices.tail(252)

    true_high = float(prices_since_entry["close"].max()) if not prices_since_entry.empty \
                else float(ticker_prices["close"].iloc[-1])

    print(f"  {ticker}: {row.get('stop_reference_price', 'None')} → {true_high:.2f}")
    portfolio.at[idx, "stop_reference_price"] = true_high
    portfolio.at[idx, "stop_price"]           = None

save_portfolio(portfolio)
print("Done. Run main.py again.")