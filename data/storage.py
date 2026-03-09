# data/storage.py
# Handles all read/write operations for the pipeline.
# Data is stored as Parquet files and queryable via DuckDB.
# Every module reads/writes through here — no direct file access elsewhere.

import pandas as pd
import duckdb
import json
import os
from pathlib import Path
from datetime import date, datetime
from utils.config_loader import get_config

cfg = get_config()

RAW_DIR       = Path(cfg["paths"]["raw_data"])
PROCESSED_DIR = Path(cfg["paths"]["processed_data"])
SNAPSHOTS_DIR = Path(cfg["paths"]["snapshots"])

# Fetch timestamp file — tracks when each dataset was last fully refreshed
FETCH_TIMESTAMPS_FILE = PROCESSED_DIR / "fetch_timestamps.json"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, SNAPSHOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# PARQUET READ / WRITE
# ------------------------------------------------------------

def save_parquet(df: pd.DataFrame, folder: Path, filename: str) -> None:
    """
    Saves a DataFrame as a Parquet file.
    Overwrites if file already exists.
    """
    path = folder / filename
    df.to_parquet(path, index=False)
    print(f"[storage] Saved {len(df)} rows to {path}")


def load_parquet(folder: Path, filename: str) -> pd.DataFrame:
    """
    Loads a Parquet file as a DataFrame.
    Returns empty DataFrame if file doesn't exist.
    """
    path = folder / filename
    if not path.exists():
        print(f"[storage] File not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"[storage] Loaded {len(df)} rows from {path}")
    return df


# ------------------------------------------------------------
# FETCH TIMESTAMP TRACKING
# Tracks when financials and key_metrics were last fully refreshed.
# Used to avoid unnecessary daily re-fetches of quarterly data.
# ------------------------------------------------------------

def _load_fetch_timestamps() -> dict:
    if not FETCH_TIMESTAMPS_FILE.exists():
        return {}
    with open(FETCH_TIMESTAMPS_FILE, "r") as f:
        return json.load(f)


def _save_fetch_timestamps(timestamps: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(FETCH_TIMESTAMPS_FILE, "w") as f:
        json.dump(timestamps, f)


def get_last_fetch_date(dataset: str) -> date | None:
    """
    Returns the last full-refresh date for a dataset (financials, key_metrics).
    Returns None if never fetched.
    """
    timestamps = _load_fetch_timestamps()
    val = timestamps.get(dataset)
    if val is None:
        return None
    return datetime.strptime(val, "%Y-%m-%d").date()


def mark_fetched(dataset: str, fetch_date: date = None) -> None:
    """
    Records that a full refresh was done for a dataset today.
    Call after successfully saving financials or key_metrics.
    """
    if fetch_date is None:
        fetch_date = date.today()
    timestamps = _load_fetch_timestamps()
    timestamps[dataset] = fetch_date.strftime("%Y-%m-%d")
    _save_fetch_timestamps(timestamps)


# ------------------------------------------------------------
# STANDARD SAVE / LOAD SHORTCUTS
# data_dir parameter allows backtest to point at data/backtest/
# instead of the default data/raw/ used by the live pipeline.
# ------------------------------------------------------------

def save_prices(df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, "prices.parquet")

def load_prices(data_dir: Path = None) -> pd.DataFrame:
    folder = Path(data_dir) if data_dir else RAW_DIR
    return load_parquet(folder, "prices.parquet")

def save_financials(df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, "financials.parquet")

def load_financials(data_dir: Path = None) -> pd.DataFrame:
    folder = Path(data_dir) if data_dir else RAW_DIR
    return load_parquet(folder, "financials.parquet")

def save_key_metrics(df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, "key_metrics.parquet")

def load_key_metrics(data_dir: Path = None) -> pd.DataFrame:
    folder = Path(data_dir) if data_dir else RAW_DIR
    return load_parquet(folder, "key_metrics.parquet")

def save_constituents(df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, "constituents.parquet")

def load_constituents(data_dir: Path = None) -> pd.DataFrame:
    folder = Path(data_dir) if data_dir else RAW_DIR
    return load_parquet(folder, "constituents.parquet")

def save_regime_data(key: str, df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, f"regime_{key}.parquet")

def load_regime_data(key: str) -> pd.DataFrame:
    return load_parquet(RAW_DIR, f"regime_{key}.parquet")

def save_signals(df: pd.DataFrame) -> None:
    save_parquet(df, PROCESSED_DIR, "signals.parquet")

def load_signals() -> pd.DataFrame:
    return load_parquet(PROCESSED_DIR, "signals.parquet")

def save_portfolio(df: pd.DataFrame) -> None:
    save_parquet(df, PROCESSED_DIR, "portfolio.parquet")

def load_portfolio() -> pd.DataFrame:
    return load_parquet(PROCESSED_DIR, "portfolio.parquet")


# ------------------------------------------------------------
# INCREMENTAL PRICE HELPERS
# ------------------------------------------------------------

def get_last_price_date() -> date | None:
    """
    Returns the most recent date present in prices.parquet.
    Returns None if the file doesn't exist or is empty.
    Used by pipeline_data.py to determine how far back to fetch.
    """
    existing = load_prices()
    if existing.empty:
        return None
    return pd.to_datetime(existing["date"]).max().date()


def append_prices(new_df: pd.DataFrame) -> None:
    """
    Appends new price rows to prices.parquet.
    Deduplicates on (ticker, date) so re-running never creates duplicates.
    Only saves if new_df has rows.
    """
    if new_df.empty:
        print("[storage] No new price rows to append")
        return

    existing = load_prices()

    if existing.empty:
        save_prices(new_df)
        return

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    save_prices(combined)
    print(f"[storage] Prices updated: {len(existing)} existing + {len(new_df)} new = {len(combined)} total rows")


# ------------------------------------------------------------
# DAILY SNAPSHOT (AUDIT TRAIL)
# ------------------------------------------------------------

def save_snapshot(df: pd.DataFrame, label: str, run_date: date = None) -> None:
    """
    Saves a dated snapshot for audit trail.
    Label examples: signals, regime, proposed_trades, fills
    """
    if run_date is None:
        run_date = date.today()
    filename = f"{run_date.strftime('%Y%m%d')}_{label}.parquet"
    save_parquet(df, SNAPSHOTS_DIR, filename)


def load_snapshot(label: str, run_date: date) -> pd.DataFrame:
    filename = f"{run_date.strftime('%Y%m%d')}_{label}.parquet"
    return load_parquet(SNAPSHOTS_DIR, filename)


# ------------------------------------------------------------
# DUCKDB QUERIES
# ------------------------------------------------------------

def query(sql: str) -> pd.DataFrame:
    """
    Runs a SQL query across any Parquet files using DuckDB.
    Reference files directly in SQL using their full path.

    Example:
        query("SELECT * FROM 'data/raw/prices.parquet' WHERE ticker = 'AAPL'")
    """
    con = duckdb.connect()
    result = con.execute(sql).fetchdf()
    con.close()
    return result


def get_prices_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Returns all price rows for a single ticker using DuckDB.
    """
    path = str(RAW_DIR / "prices.parquet")
    return query(f"SELECT * FROM '{path}' WHERE ticker = '{ticker}' ORDER BY date")


def get_latest_prices(as_of_date: str) -> pd.DataFrame:
    """
    Returns the most recent price for each ticker on or before as_of_date.
    Used by portfolio valuation and risk monitoring.
    """
    path = str(RAW_DIR / "prices.parquet")
    return query(f"""
        SELECT ticker, date, close, volume
        FROM '{path}'
        WHERE date <= '{as_of_date}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
    """)