# build_universe_history.py
# Builds per-date S&P 500 membership lists from constituent change log.
# Replays additions/removals chronologically to get point-in-time membership.
# Output: data/backtest/universe_history.pkl

import pandas as pd
import pickle
from pathlib import Path

df = pd.read_parquet("data/backtest/constituent_history.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Build membership by replaying all changes
members = set()
universe_history = {}

for _, row in df.iterrows():
    # Add new ticker
    if pd.notna(row["symbol"]) and str(row["symbol"]).strip():
        members.add(str(row["symbol"]).strip())
    # Remove old ticker
    if pd.notna(row["removedTicker"]) and str(row["removedTicker"]).strip():
        members.discard(str(row["removedTicker"]).strip())
    # Snapshot membership on this date
    universe_history[row["date"]] = set(members)

# Forward-fill to all trading dates in price data
prices = pd.read_parquet("data/backtest/prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
all_trading_dates = sorted(prices["date"].unique())

change_dates = sorted(universe_history.keys())
filled = {}
current_members = set()

for td in all_trading_dates:
    # Apply all changes up to and including this date
    while change_dates and change_dates[0] <= td:
        current_members = universe_history[change_dates.pop(0)]
    filled[td] = set(current_members)

# Save
out = Path("data/backtest/universe_history.pkl")
with open(out, "wb") as f:
    pickle.dump(filled, f)

# Verify
sample_dates = [d for d in all_trading_dates if pd.Timestamp("2009-01-01") <= d <= pd.Timestamp("2026-01-01")]
sample_dates = sample_dates[::252]  # one per year
print(f"Saved {len(filled)} dated membership snapshots → {out}")
print("\nMembers per year (sample):")
for d in sample_dates:
    print(f"  {d.date()}: {len(filled[d])} members")