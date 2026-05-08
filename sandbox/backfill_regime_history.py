# backfill_regime_history.py
# One-time script to fetch full FRED macro history for regime reconstruction.
# Fetches from 2007-01-01 to cover the full backtest range (prices go to 1995
# but regime needs 252d VIX lookback before first computation date ~2008).
# Uses save_regime_data which now appends+deduplicates — safe to re-run.

from data.fetchers.fred_fetcher import get_all_regime_data
from data.storage import save_regime_data

START = "2007-01-01"
END   = "2026-03-13"

print(f"Fetching full FRED regime history {START} → {END}")
data = get_all_regime_data(START, END)

for key, df in data.items():
    save_regime_data(key, df)
    print(f"  Saved {key}: {len(df)} rows")

print("Done. Ready to run regime reconstruction.")