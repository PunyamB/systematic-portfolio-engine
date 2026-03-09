# notebooks/fetch_backtest_data.py
# One-time historical data fetcher for backtesting.
# Fetches:
#   1. Point-in-time S&P 500 constituent history (additions/removals since 1957)
#   2. Prices from 2006-01-01 for all ever-constituent tickers (3 chunks to avoid 5000-row cap)
#   3. Quarterly financials for all ever-constituent tickers (120 quarters = 30 years)
#   4. Key metrics for all ever-constituent tickers (120 quarters = 30 years)
# Saves to data/backtest/ — separate from live pipeline data.
# Resumable — rerun after failure and it picks up where it left off.
# Run once. Takes ~60-90 mins for full ~800 ticker universe.

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import date

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fetchers.fmp_fetcher import get_daily_prices, get_quarterly_financials, get_key_metrics, _get
from utils.broker_health import get_env
from utils.config_loader import get_config
from data.storage import load_constituents

cfg     = get_config()
API_KEY = get_env("FMP_API_KEY")

BACKTEST_DIR = Path("data/backtest")
PRICE_START  = "1995-01-01"
PRICE_CHUNK2 = "2005-01-01"   # chunk 1: 1995-2005 (~2500 rows)
PRICE_CHUNK3 = "2015-01-01"   # chunk 2: 2005-2015 (~2500 rows)
                               # chunk 3: 2015-present (~2500 rows)
PRICE_END    = date.today().strftime("%Y-%m-%d")  # chunk 3: 2020-present (~1500 rows)
BATCH_SLEEP  = 0.3            # seconds between API calls

BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# STEP 1 — POINT-IN-TIME CONSTITUENT HISTORY
# ------------------------------------------------------------

def fetch_constituent_history() -> pd.DataFrame:
    """
    Fetches full S&P 500 addition/removal history from FMP.
    Reconstructs point-in-time membership for any given date.
    """
    print("[backtest] Fetching S&P 500 constituent history...")

    url  = f"https://financialmodelingprep.com/stable/historical-sp500-constituent?apikey={API_KEY}"
    resp = requests.get(url)
    data = resp.json()

    if not isinstance(data, list) or not data:
        print("[backtest] Failed to fetch constituent history")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    out = BACKTEST_DIR / "constituent_history.parquet"
    df.to_parquet(out, index=False)
    print(f"[backtest] Constituent history saved: {len(df)} changes | {out}")
    return df


def get_all_ever_tickers(constituent_history: pd.DataFrame,
                          current_constituents: pd.DataFrame) -> list:
    """
    Returns all tickers ever in the S&P 500 — union of:
    - current 503 constituents
    - all historical additions
    - all historical removals
    Used to fetch prices for full survivorship-bias-free universe.
    """
    current            = set(current_constituents["ticker"].tolist())
    historical_adds    = set(constituent_history["symbol"].dropna().tolist())
    historical_removes = set(constituent_history["removedTicker"].dropna().tolist())
    all_tickers        = current | historical_adds | historical_removes

    # Filter out None/NaN/empty/non-string
    all_tickers = {t for t in all_tickers if isinstance(t, str) and len(t) > 0}
    return sorted(list(all_tickers))


def build_universe_on_date(constituent_history: pd.DataFrame,
                            as_of_date: date,
                            current_constituents: pd.DataFrame) -> list:
    """
    Reconstructs S&P 500 membership on a given historical date.
    Starts from current constituents and rolls back additions/removals.
    Returns list of tickers active on as_of_date.
    """
    cutoff  = pd.Timestamp(as_of_date)
    members = set(current_constituents["ticker"].tolist())

    # Roll back: undo additions after as_of_date, re-add removals after as_of_date
    future_changes = constituent_history[constituent_history["date"] > cutoff]

    for _, row in future_changes.iterrows():
        added   = row.get("symbol")
        removed = row.get("removedTicker")

        if pd.notna(added) and added in members:
            members.discard(added)

        if pd.notna(removed) and removed not in members:
            members.add(removed)

    return sorted(list(members))


# ------------------------------------------------------------
# STEP 2 — HISTORICAL PRICES
# ------------------------------------------------------------

def fetch_historical_prices(tickers: list) -> pd.DataFrame:
    """
    Fetches prices from PRICE_START to PRICE_END for all tickers.
    Splits into three date range chunks to avoid 5000-row API cap.
    Saves incrementally every 50 tickers to allow resume on failure.
    """
    out_file  = BACKTEST_DIR / "prices.parquet"
    done_file = BACKTEST_DIR / "prices_done_tickers.txt"

    # Load already-fetched tickers to allow resume
    done_tickers = set()
    if done_file.exists():
        with open(done_file) as f:
            done_tickers = set(f.read().splitlines())

    existing_dfs = []
    if out_file.exists():
        existing_dfs.append(pd.read_parquet(out_file))

    remaining = [t for t in tickers if t not in done_tickers]
    print(f"[backtest] Fetching prices: {len(remaining)} tickers remaining "
          f"({len(done_tickers)} already done)")

    batch = []
    for i, ticker in enumerate(remaining):
        try:
            # Three chunks to stay under 5000-row cap per call
            df1 = get_daily_prices(ticker, PRICE_START,  PRICE_CHUNK2)
            time.sleep(BATCH_SLEEP)
            df2 = get_daily_prices(ticker, PRICE_CHUNK2, PRICE_CHUNK3)
            time.sleep(BATCH_SLEEP)
            df3 = get_daily_prices(ticker, PRICE_CHUNK3, PRICE_END)
            time.sleep(BATCH_SLEEP)

            df = pd.concat([df1, df2, df3], ignore_index=True)
            df = df.drop_duplicates(subset=["date", "ticker"])

            if not df.empty:
                batch.append(df)

            done_tickers.add(ticker)

        except Exception as e:
            print(f"[backtest] Price fetch failed for {ticker}: {e}")

        # Checkpoint every 50 tickers
        if (i + 1) % 50 == 0:
            if batch:
                combined = pd.concat(existing_dfs + batch, ignore_index=True)
                combined = combined.drop_duplicates(subset=["date", "ticker"])
                combined.to_parquet(out_file, index=False)
                existing_dfs = [combined]
                batch = []

            with open(done_file, "w") as f:
                f.write("\n".join(done_tickers))

            print(f"[backtest] Prices: {i+1}/{len(remaining)} done | checkpoint saved")

    # Final save
    if batch:
        combined = pd.concat(existing_dfs + batch, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"])
        combined.to_parquet(out_file, index=False)

    with open(done_file, "w") as f:
        f.write("\n".join(done_tickers))

    final = pd.read_parquet(out_file)
    print(f"[backtest] Prices complete: {len(final)} rows | "
          f"{final['ticker'].nunique()} tickers | "
          f"earliest: {final['date'].min().date()} | "
          f"latest: {final['date'].max().date()}")
    return final


# ------------------------------------------------------------
# STEP 3 — HISTORICAL FINANCIALS
# ------------------------------------------------------------

def fetch_historical_financials(tickers: list) -> pd.DataFrame:
    """
    Fetches quarterly financials for all tickers.
    FMP returns up to 80 quarters (~20 years) per ticker.
    Saves incrementally with resume support.
    """
    out_file  = BACKTEST_DIR / "financials.parquet"
    done_file = BACKTEST_DIR / "financials_done_tickers.txt"

    done_tickers = set()
    if done_file.exists():
        with open(done_file) as f:
            done_tickers = set(f.read().splitlines())

    existing_dfs = []
    if out_file.exists():
        existing_dfs.append(pd.read_parquet(out_file))

    remaining = [t for t in tickers if t not in done_tickers]
    print(f"[backtest] Fetching financials: {len(remaining)} tickers remaining")

    batch = []
    for i, ticker in enumerate(remaining):
        try:
            df = get_quarterly_financials(ticker)
            time.sleep(BATCH_SLEEP)

            if not df.empty:
                batch.append(df)

            done_tickers.add(ticker)

        except Exception as e:
            print(f"[backtest] Financials fetch failed for {ticker}: {e}")

        if (i + 1) % 50 == 0:
            if batch:
                combined = pd.concat(existing_dfs + batch, ignore_index=True)
                combined = combined.drop_duplicates(subset=["date", "ticker"])
                combined.to_parquet(out_file, index=False)
                existing_dfs = [combined]
                batch = []

            with open(done_file, "w") as f:
                f.write("\n".join(done_tickers))

            print(f"[backtest] Financials: {i+1}/{len(remaining)} done | checkpoint saved")

    if batch:
        combined = pd.concat(existing_dfs + batch, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"])
        combined.to_parquet(out_file, index=False)

    with open(done_file, "w") as f:
        f.write("\n".join(done_tickers))

    final = pd.read_parquet(out_file)
    print(f"[backtest] Financials complete: {len(final)} rows | "
          f"{final['ticker'].nunique()} tickers | "
          f"earliest: {final['date'].min().date()} | "
          f"latest: {final['date'].max().date()}")
    return final


# ------------------------------------------------------------
# STEP 4 — HISTORICAL KEY METRICS
# ------------------------------------------------------------

def fetch_historical_key_metrics(tickers: list) -> pd.DataFrame:
    """
    Fetches quarterly key metrics (P/E, EV/EBITDA, ROE, ROIC) for all tickers.
    FMP returns up to 80 quarters (~20 years) per ticker.
    Saves incrementally with resume support.
    """
    out_file  = BACKTEST_DIR / "key_metrics.parquet"
    done_file = BACKTEST_DIR / "key_metrics_done_tickers.txt"

    done_tickers = set()
    if done_file.exists():
        with open(done_file) as f:
            done_tickers = set(f.read().splitlines())

    existing_dfs = []
    if out_file.exists():
        existing_dfs.append(pd.read_parquet(out_file))

    remaining = [t for t in tickers if t not in done_tickers]
    print(f"[backtest] Fetching key metrics: {len(remaining)} tickers remaining")

    batch = []
    for i, ticker in enumerate(remaining):
        try:
            df = get_key_metrics(ticker)
            time.sleep(BATCH_SLEEP)

            if not df.empty:
                batch.append(df)

            done_tickers.add(ticker)

        except Exception as e:
            print(f"[backtest] Key metrics fetch failed for {ticker}: {e}")

        if (i + 1) % 50 == 0:
            if batch:
                combined = pd.concat(existing_dfs + batch, ignore_index=True)
                combined = combined.drop_duplicates(subset=["date", "ticker"])
                combined.to_parquet(out_file, index=False)
                existing_dfs = [combined]
                batch = []

            with open(done_file, "w") as f:
                f.write("\n".join(done_tickers))

            print(f"[backtest] Key metrics: {i+1}/{len(remaining)} done | checkpoint saved")

    if batch:
        combined = pd.concat(existing_dfs + batch, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"])
        combined.to_parquet(out_file, index=False)

    with open(done_file, "w") as f:
        f.write("\n".join(done_tickers))

    final = pd.read_parquet(out_file)
    print(f"[backtest] Key metrics complete: {len(final)} rows | "
          f"{final['ticker'].nunique()} tickers | "
          f"earliest: {final['date'].min().date()} | "
          f"latest: {final['date'].max().date()}")
    return final


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n[backtest] ================================================")
    print(f"[backtest] Starting historical data fetch")
    print(f"[backtest] Price range: {PRICE_START} to {PRICE_END} (3 chunks)")
    print(f"[backtest] Fundamentals: 120 quarters (~30 years)")
    print(f"[backtest] Output: {BACKTEST_DIR}")
    print(f"[backtest] ================================================\n")

    # Step 1 — constituent history (point-in-time universe reconstruction)
    constituent_history = fetch_constituent_history()

    # Step 2 — build full ever-constituent ticker list
    current_constituents = load_constituents()
    all_tickers = get_all_ever_tickers(constituent_history, current_constituents)
    print(f"[backtest] Total ever-constituent tickers: {len(all_tickers)}")

    # Save full ticker list for reference
    pd.Series(all_tickers).to_csv(
        BACKTEST_DIR / "all_tickers.csv", index=False, header=False
    )

    # Step 3 — prices (2006 to present, 3 chunks per ticker)
    fetch_historical_prices(all_tickers)

    # Step 4 — quarterly financials (80 quarters)
    fetch_historical_financials(all_tickers)

    # Step 5 — key metrics (80 quarters)
    fetch_historical_key_metrics(all_tickers)

    print(f"\n[backtest] ================================================")
    print(f"[backtest] All data fetched. Ready for backtest notebook.")
    print(f"[backtest] ================================================\n")