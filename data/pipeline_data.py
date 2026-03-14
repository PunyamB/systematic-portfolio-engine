# data/pipeline_data.py
# Daily data refresh module. Runs at the start of every pipeline execution.
# Fetches all required data, validates quality gates, and saves to storage.
# Every downstream module reads from storage — nothing hits the API directly.
#
# INCREMENTAL FETCH STRATEGY:
# - Prices: only fetches new rows since last date in parquet, then appends.
#           Full 2-year backfill only on first run.
# - Financials / key metrics: quarterly data — full re-fetch once per week.
#           Skipped on daily runs if fetched within the last 7 days.
# - Regime data (FRED): fetched daily, lightweight.
# - Constituents: cached, refreshed weekly or on force flag.

import pandas as pd
from datetime import date, timedelta
from data.fetchers.fmp_fetcher import (
    get_sp500_constituents,
    get_bulk_prices,
    get_bulk_financials,
    get_bulk_key_metrics,
    validate_data_quality
)
from data.fetchers.fred_fetcher import get_all_regime_data
from data.storage import (
    save_constituents, load_constituents,
    append_prices, get_last_price_date,
    save_financials, load_financials,
    save_key_metrics, load_key_metrics,
    save_regime_data,
    get_last_fetch_date, mark_fetched
)
from utils.notifications import notify

FINANCIALS_REFRESH_DAYS = 7   # re-fetch financials once per week
KEY_METRICS_REFRESH_DAYS = 7  # re-fetch key metrics once per week
PRICE_INITIAL_LOOKBACK   = 504  # ~2 years of prices on first run


def _needs_refresh(dataset: str, threshold_days: int, run_date: date) -> bool:
    """
    Returns True if the dataset has never been fetched,
    or if the last fetch was more than threshold_days ago.
    """
    last = get_last_fetch_date(dataset)
    if last is None:
        return True
    return (run_date - last).days >= threshold_days


def run_data_refresh(
    run_date: date = None,
    force_constituent_refresh: bool = False,
    force_financials_refresh: bool = False
) -> bool:
    """
    Incremental data refresh pipeline. Returns True if all quality gates pass.
    Called at the start of main pipeline before any signal computation.

    Steps:
    1. Refresh S&P 500 constituents (weekly or forced)
    2. Fetch new prices only (from last date in parquet to today)
    3. Fetch financials and key metrics (weekly only, skipped if recent)
    4. Fetch macro/regime data from FRED (daily, lightweight)
    5. Validate data quality gates
    """
    if run_date is None:
        run_date = date.today()

    end_date = run_date.strftime("%Y-%m-%d")

    print(f"[pipeline_data] Starting data refresh for {run_date}")
    notify(f"Data refresh started for {run_date}", level="info")

    # ----------------------------------------------------------
    # STEP 1 — S&P 500 CONSTITUENTS
    # Cached — only refresh weekly or when forced
    # ----------------------------------------------------------
    constituents = load_constituents()

    if constituents.empty or force_constituent_refresh:
        print("[pipeline_data] Refreshing S&P 500 constituents")
        constituents = get_sp500_constituents()
        save_constituents(constituents)
    else:
        print(f"[pipeline_data] Using cached constituents: {len(constituents)} tickers")

    tickers = constituents["ticker"].tolist()

    # ----------------------------------------------------------
    # STEP 2 — PRICES (INCREMENTAL)
    # Only fetch rows newer than the last date already in parquet.
    # Full 2-year backfill on first run.
    # ----------------------------------------------------------
    last_price_date = get_last_price_date()

    if last_price_date is None:
        # First run — full backfill
        start_date = (run_date - timedelta(days=PRICE_INITIAL_LOOKBACK)).strftime("%Y-%m-%d")
        print(f"[pipeline_data] First price fetch — full backfill from {start_date}")
    else:
        # Incremental — fetch only from day after last stored date
        start_date = (last_price_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[pipeline_data] Incremental price fetch from {start_date} to {end_date}")

    if start_date > end_date:
        print("[pipeline_data] Prices already up to date — skipping fetch")
    else:
        new_prices = get_bulk_prices(tickers, start_date, end_date)

        if not validate_data_quality(new_prices, "prices"):
            notify("Data refresh halted: prices failed quality gate", level="critical")
            return False

        append_prices(new_prices)

    # ----------------------------------------------------------
    # STEP 3 — FINANCIALS (WEEKLY)
    # Quarterly data — no point re-fetching daily.
    # Re-fetches in full once per week to pick up new earnings reports.
    # ----------------------------------------------------------
    if _needs_refresh("financials", FINANCIALS_REFRESH_DAYS, run_date) or force_financials_refresh:
        print(f"[pipeline_data] Fetching financials for {len(tickers)} tickers")
        financials = get_bulk_financials(tickers)

        if not validate_data_quality(financials, "financials"):
            notify("Data refresh halted: financials failed quality gate", level="critical")
            return False

        save_financials(financials)
        mark_fetched("financials", run_date)
    else:
        last = get_last_fetch_date("financials")
        print(f"[pipeline_data] Financials up to date (last fetched {last}) — skipping")

    # ----------------------------------------------------------
    # STEP 4 — KEY METRICS (WEEKLY)
    # Same rationale as financials — quarterly data.
    # ----------------------------------------------------------
    if _needs_refresh("key_metrics", KEY_METRICS_REFRESH_DAYS, run_date) or force_financials_refresh:
        print(f"[pipeline_data] Fetching key metrics for {len(tickers)} tickers")
        key_metrics = get_bulk_key_metrics(tickers)
        save_key_metrics(key_metrics)
        mark_fetched("key_metrics", run_date)
    else:
        last = get_last_fetch_date("key_metrics")
        print(f"[pipeline_data] Key metrics up to date (last fetched {last}) — skipping")

    # ----------------------------------------------------------
    # STEP 5 — MACRO / REGIME DATA (DAILY)
    # Backfills 1 year of history if insufficient rows stored.
    # ----------------------------------------------------------
    print("[pipeline_data] Fetching macro data from FRED")

    from data.storage import load_regime_data
    vix_stored = load_regime_data("vix")
    if vix_stored.empty or len(vix_stored) < 252:
        print("[pipeline_data] VIX history insufficient — backfilling 1 year")
        regime_start = (run_date - timedelta(days=365)).strftime("%Y-%m-%d")
    else:
        regime_start = (run_date - timedelta(days=30)).strftime("%Y-%m-%d")

    regime_data = get_all_regime_data(regime_start, end_date)

    for key, df in regime_data.items():
        if df.empty:
            notify(f"Data refresh warning: FRED {key} returned empty", level="warning")
        else:
            save_regime_data(key, df)

    # ----------------------------------------------------------
    # STEP 6 — SUMMARY
    # ----------------------------------------------------------
    from data.storage import load_prices
    prices = load_prices()

    print(f"[pipeline_data] Data refresh complete for {run_date}")
    print(f"[pipeline_data] Prices: {len(prices)} rows, {prices['ticker'].nunique() if not prices.empty else 0} tickers")
    print(f"[pipeline_data] Regime data: {list(regime_data.keys())}")

    notify(
        f"Data refresh complete for {run_date}\n"
        f"Prices: {prices['ticker'].nunique() if not prices.empty else 0} tickers\n"
        f"Financials refresh: {'yes' if _needs_refresh('financials', FINANCIALS_REFRESH_DAYS, run_date) else 'skipped'}",
        level="info"
    )

    return True