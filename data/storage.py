# data/storage.py
# Handles all read/write operations for the pipeline.
# Data is stored as Parquet files and queryable via DuckDB.
# Every module reads/writes through here — no direct file access elsewhere.
#
# SESSION CACHE: Large Parquet files (prices, financials, key_metrics,
# constituents) are cached in memory after first load. Cache is cleared
# at pipeline start via clear_cache(). This eliminates 15+ redundant
# disk reads per pipeline run.

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
LOGS_DIR      = Path("logs")

# Fetch timestamp file — tracks when each dataset was last fully refreshed
FETCH_TIMESTAMPS_FILE = PROCESSED_DIR / "fetch_timestamps.json"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, SNAPSHOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# SESSION CACHE
# ============================================================

_cache = {}


def clear_cache():
    """Clears all cached DataFrames. Call at pipeline start."""
    global _cache
    _cache.clear()


def _get_cached(key: str) -> pd.DataFrame | None:
    return _cache.get(key)


def _set_cached(key: str, df: pd.DataFrame) -> None:
    _cache[key] = df


def _invalidate(key: str) -> None:
    _cache.pop(key, None)


# ============================================================
# PARQUET READ / WRITE
# ============================================================

def save_parquet(df: pd.DataFrame, folder: Path, filename: str) -> None:
    """Saves a DataFrame as a Parquet file. Overwrites if exists."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    df.to_parquet(path, index=False)
    print(f"[storage] Saved {len(df)} rows to {path}")


def load_parquet(folder: Path, filename: str) -> pd.DataFrame:
    """Loads a Parquet file as a DataFrame. Returns empty if missing."""
    path = folder / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df


def append_parquet(df: pd.DataFrame, folder: Path, filename: str) -> None:
    """
    Appends rows to an existing Parquet file.
    Creates the file if it doesn't exist.
    Used for append-only logs (decision_log, portfolio_history, etc.).
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_parquet(path, index=False)


# ============================================================
# FETCH TIMESTAMP TRACKING
# ============================================================

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
    timestamps = _load_fetch_timestamps()
    val = timestamps.get(dataset)
    if val is None:
        return None
    return datetime.strptime(val, "%Y-%m-%d").date()


def mark_fetched(dataset: str, fetch_date: date = None) -> None:
    if fetch_date is None:
        fetch_date = date.today()
    timestamps = _load_fetch_timestamps()
    timestamps[dataset] = fetch_date.strftime("%Y-%m-%d")
    _save_fetch_timestamps(timestamps)


# ============================================================
# STANDARD SAVE / LOAD — WITH SESSION CACHE
# data_dir parameter allows backtest to point at data/backtest/
# ============================================================

def save_prices(df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, "prices.parquet")
    _invalidate("prices")

def load_prices(data_dir: Path = None) -> pd.DataFrame:
    if data_dir is not None:
        return load_parquet(Path(data_dir), "prices.parquet")
    cached = _get_cached("prices")
    if cached is not None:
        return cached
    df = load_parquet(RAW_DIR, "prices.parquet")
    _set_cached("prices", df)
    return df

def save_financials(df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, "financials.parquet")
    _invalidate("financials")

def load_financials(data_dir: Path = None) -> pd.DataFrame:
    if data_dir is not None:
        return load_parquet(Path(data_dir), "financials.parquet")
    cached = _get_cached("financials")
    if cached is not None:
        return cached
    df = load_parquet(RAW_DIR, "financials.parquet")
    _set_cached("financials", df)
    return df

def save_key_metrics(df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, "key_metrics.parquet")
    _invalidate("key_metrics")

def load_key_metrics(data_dir: Path = None) -> pd.DataFrame:
    if data_dir is not None:
        return load_parquet(Path(data_dir), "key_metrics.parquet")
    cached = _get_cached("key_metrics")
    if cached is not None:
        return cached
    df = load_parquet(RAW_DIR, "key_metrics.parquet")
    _set_cached("key_metrics", df)
    return df

def save_constituents(df: pd.DataFrame) -> None:
    save_parquet(df, RAW_DIR, "constituents.parquet")
    _invalidate("constituents")

def load_constituents(data_dir: Path = None) -> pd.DataFrame:
    if data_dir is not None:
        return load_parquet(Path(data_dir), "constituents.parquet")
    cached = _get_cached("constituents")
    if cached is not None:
        return cached
    df = load_parquet(RAW_DIR, "constituents.parquet")
    _set_cached("constituents", df)
    return df

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
    _invalidate("portfolio")

def load_portfolio() -> pd.DataFrame:
    cached = _get_cached("portfolio")
    if cached is not None:
        return cached
    df = load_parquet(PROCESSED_DIR, "portfolio.parquet")
    _set_cached("portfolio", df)
    return df


# ============================================================
# SPY PRICES — loaded once per session
# ============================================================

def load_spy_prices() -> pd.DataFrame:
    """
    Loads SPY prices from the prices parquet.
    Cached separately since it's needed by risk monitor independently.
    """
    cached = _get_cached("spy_prices")
    if cached is not None:
        return cached
    prices = load_prices()
    if prices.empty:
        return pd.DataFrame()
    spy = prices[prices["ticker"] == "SPY"].copy()
    _set_cached("spy_prices", spy)
    return spy


# ============================================================
# INCREMENTAL PRICE HELPERS
# ============================================================

def get_last_price_date() -> date | None:
    existing = load_prices()
    if existing.empty:
        return None
    return pd.to_datetime(existing["date"]).max().date()


def append_prices(new_df: pd.DataFrame) -> None:
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


# ============================================================
# DAILY SNAPSHOT (AUDIT TRAIL)
# ============================================================

def save_snapshot(data, label: str, run_date: date = None) -> None:
    """
    Saves a dated snapshot for audit trail.
    Accepts DataFrame or dict (dicts converted to single-row DataFrame).
    Label examples: signals, regime, optimizer, risk, portfolio
    """
    if run_date is None:
        run_date = date.today()
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    if isinstance(data, pd.DataFrame) and data.empty:
        return
    filename = f"{run_date.strftime('%Y%m%d')}_{label}.parquet"
    save_parquet(data, SNAPSHOTS_DIR, filename)


def load_snapshot(label: str, run_date: date) -> pd.DataFrame:
    filename = f"{run_date.strftime('%Y%m%d')}_{label}.parquet"
    return load_parquet(SNAPSHOTS_DIR, filename)


# ============================================================
# LOGGING — APPEND-ONLY PARQUET FILES
# ============================================================

def append_decision_log(record: dict) -> None:
    """Appends one row to logs/decision_log.parquet."""
    df = pd.DataFrame([record])
    append_parquet(df, LOGS_DIR, "decision_log.parquet")


def load_decision_log() -> pd.DataFrame:
    return load_parquet(LOGS_DIR, "decision_log.parquet")


def append_portfolio_history(positions_df: pd.DataFrame, run_date: date) -> None:
    """
    Appends current portfolio positions to logs/portfolio_history.parquet.
    Each row = one position on one date. Empty portfolio records a single
    row with ticker='CASH_ONLY' so the date still appears in history.
    """
    if positions_df.empty:
        empty_row = pd.DataFrame([{
            "date": run_date, "ticker": "CASH_ONLY", "shares": 0,
            "market_value": 0.0, "weight": 0.0, "cost_basis": 0.0,
            "unrealized_pnl": 0.0, "entry_date": None,
            "stop_price": None, "sector": ""
        }])
        append_parquet(empty_row, LOGS_DIR, "portfolio_history.parquet")
        return
    positions_df = positions_df.copy()
    positions_df["date"] = run_date
    append_parquet(positions_df, LOGS_DIR, "portfolio_history.parquet")


def load_portfolio_history() -> pd.DataFrame:
    return load_parquet(LOGS_DIR, "portfolio_history.parquet")


def append_pipeline_history(health_record: dict) -> None:
    """Appends pipeline health record to logs/pipeline_history.parquet."""
    df = pd.DataFrame([health_record])
    append_parquet(df, LOGS_DIR, "pipeline_history.parquet")


def load_pipeline_history() -> pd.DataFrame:
    return load_parquet(LOGS_DIR, "pipeline_history.parquet")


def append_execution_log(records: pd.DataFrame) -> None:
    """Appends execution fill records to logs/execution_log.parquet."""
    if records.empty:
        return
    append_parquet(records, LOGS_DIR, "execution_log.parquet")


def load_execution_log() -> pd.DataFrame:
    return load_parquet(LOGS_DIR, "execution_log.parquet")


# ============================================================
# DUCKDB QUERIES
# ============================================================

def query(sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    result = con.execute(sql).fetchdf()
    con.close()
    return result


def get_prices_for_ticker(ticker: str) -> pd.DataFrame:
    path = str(RAW_DIR / "prices.parquet")
    return query(f"SELECT * FROM '{path}' WHERE ticker = '{ticker}' ORDER BY date")


def get_latest_prices(as_of_date: str) -> pd.DataFrame:
    path = str(RAW_DIR / "prices.parquet")
    return query(f"""
        SELECT ticker, date, close, volume
        FROM '{path}'
        WHERE date <= '{as_of_date}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
    """)
