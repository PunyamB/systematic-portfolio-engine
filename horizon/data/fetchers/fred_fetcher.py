# data/fetchers/fred_fetcher.py
# Fetches macro data from FRED for regime detection.
# Covers: VIX, yield curve (10Y-2Y spread), PMI, HYG/LQD prices.
# All series IDs verified against FRED database.

import requests
import pandas as pd
from utils.config_loader import get_env

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED Series IDs
SERIES = {
    "vix":        "VIXCLS",        # CBOE VIX daily
    "t10y":       "DGS10",         # 10-Year Treasury yield
    "t2y":        "DGS2",          # 2-Year Treasury yield
    "pmi":        "MANEMP",        # ISM Manufacturing (proxy, monthly)
    "hyg":        "BAMLH0A0HYM2",  # HY spread (proxy for HYG)
    "lqd":        "BAMLC0A0CM",    # IG spread (proxy for LQD)
}


def _get_series(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches a single FRED series as a DataFrame.
    Returns columns: date, value.
    """
    api_key = get_env("FRED_API_KEY")

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date
    }

    response = requests.get(BASE_URL, params=params, timeout=30)

    if response.status_code != 200:
        raise ConnectionError(f"FRED request failed [{response.status_code}]: {series_id}")

    data = response.json()

    if "observations" not in data:
        raise ValueError(f"FRED returned no observations for {series_id}")

    df = pd.DataFrame(data["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


def get_vix(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns daily VIX levels.
    Used for L1 market stress regime detection.
    """
    df = _get_series(SERIES["vix"], start_date, end_date)
    df = df.rename(columns={"value": "vix"})
    print(f"[fred_fetcher] VIX fetched: {len(df)} rows")
    return df


def get_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns 10Y-2Y yield spread.
    Positive = normal curve, negative = inverted (recession signal).
    Used for L2 economic cycle regime detection.
    """
    t10 = _get_series(SERIES["t10y"], start_date, end_date).rename(columns={"value": "t10y"})
    t2  = _get_series(SERIES["t2y"],  start_date, end_date).rename(columns={"value": "t2y"})

    df = pd.merge(t10, t2, on="date", how="inner")
    df["spread_10y2y"] = df["t10y"] - df["t2y"]
    print(f"[fred_fetcher] Yield curve fetched: {len(df)} rows")
    return df[["date", "t10y", "t2y", "spread_10y2y"]]


def get_credit_spreads(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns HY and IG credit spreads.
    Rising spreads = risk-off signal.
    Used for L2 economic cycle regime detection.
    """
    hyg = _get_series(SERIES["hyg"], start_date, end_date).rename(columns={"value": "hy_spread"})
    lqd = _get_series(SERIES["lqd"], start_date, end_date).rename(columns={"value": "ig_spread"})

    df = pd.merge(hyg, lqd, on="date", how="inner")
    print(f"[fred_fetcher] Credit spreads fetched: {len(df)} rows")
    return df


def get_all_regime_data(start_date: str, end_date: str) -> dict:
    """
    Fetches all macro series needed for regime detection in one call.
    Returns a dict of DataFrames keyed by series name.
    """
    print(f"[fred_fetcher] Fetching all regime data from {start_date} to {end_date}")

    return {
        "vix":            get_vix(start_date, end_date),
        "yield_curve":    get_yield_curve(start_date, end_date),
        "credit_spreads": get_credit_spreads(start_date, end_date)
    }