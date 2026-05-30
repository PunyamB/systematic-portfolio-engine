# data/fetchers/fmp_fetcher.py
# Fetches all required data from Financial Modeling Prep.
# Base URL: https://financialmodelingprep.com/stable
# All endpoints verified against FMP stable API docs.
#
# TWO FETCH MODES:
# - Sync (_get, get_*):        used by live pipeline — simple, no dependencies
# - Async (_aget, async_bulk_*): used by backtest fetcher — concurrent, much faster
#
# Async bulk functions require: pip install aiohttp

import asyncio
import requests
import pandas as pd
from utils.config_loader import get_env
from utils.notifications import notify

BASE_URL         = "https://financialmodelingprep.com/stable"
ASYNC_SEMAPHORE  = 10   # max concurrent requests — safe for FMP Premium


# ============================================================
# SYNC CORE (live pipeline)
# ============================================================

def _get(endpoint: str, params: dict = {}) -> dict | list:
    """
    Core synchronous GET request to FMP. Raises on failure.
    """
    api_key = get_env("FMP_API_KEY")
    params["apikey"] = api_key
    url = f"{BASE_URL}{endpoint}"

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise ConnectionError(f"FMP request failed [{response.status_code}]: {url}")

    data = response.json()

    if isinstance(data, dict) and "Error Message" in data:
        raise ValueError(f"FMP error: {data['Error Message']}")

    return data


# ============================================================
# ASYNC CORE (backtest fetcher)
# ============================================================

async def _aget(session, semaphore, endpoint: str, params: dict = {}) -> dict | list:
    """
    Core async GET request to FMP.
    Uses semaphore to cap concurrent requests at ASYNC_SEMAPHORE.
    Returns empty list on any failure — backtest skips failed tickers gracefully.
    """
    import aiohttp

    api_key = get_env("FMP_API_KEY")
    params  = {**params, "apikey": api_key}
    url     = f"{BASE_URL}{endpoint}"

    async with semaphore:
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    # Rate limited — wait and retry once
                    await asyncio.sleep(2)
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as retry:
                        if retry.status != 200:
                            return []
                        data = await retry.json()
                elif resp.status != 200:
                    return []
                else:
                    data = await resp.json()

            if isinstance(data, dict) and "Error Message" in data:
                return []
            return data

        except Exception:
            return []


# ============================================================
# SYNC SINGLE-TICKER FUNCTIONS (used by live pipeline + backtest sequential fallback)
# ============================================================

def get_sp500_constituents() -> pd.DataFrame:
    """
    Returns current S&P 500 constituents with ticker and sector.
    Endpoint: /sp500-constituent
    """
    data = _get("/sp500-constituent")
    df   = pd.DataFrame(data)
    df   = df[["symbol", "name", "sector", "subSector"]].rename(columns={
        "symbol":    "ticker",
        "name":      "company_name",
        "sector":    "sector",
        "subSector": "sub_sector"
    })
    print(f"[fmp_fetcher] Fetched {len(df)} S&P 500 constituents")
    return df


def get_daily_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns daily OHLCV for a single ticker.
    Endpoint: /historical-price-eod/full
    Dates in YYYY-MM-DD format.
    """
    data = _get("/historical-price-eod/full", params={
        "symbol": ticker,
        "from":   start_date,
        "to":     end_date
    })
    if not data or not isinstance(data, list):
        return pd.DataFrame()

    df           = pd.DataFrame(data)
    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["date"])

    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("date").reset_index(drop=True)


def get_quarterly_financials(ticker: str) -> pd.DataFrame:
    """
    Returns merged quarterly financials for a single ticker.
    Combines income statement, balance sheet, and cash flow.
    """
    inc_data = _get("/income-statement",       params={"symbol": ticker, "period": "quarter", "limit": 120})
    bs_data  = _get("/balance-sheet-statement", params={"symbol": ticker, "period": "quarter", "limit": 120})
    cf_data  = _get("/cash-flow-statement",     params={"symbol": ticker, "period": "quarter", "limit": 120})

    if not inc_data or not bs_data or not cf_data:
        return pd.DataFrame()

    return _merge_financials(ticker, inc_data, bs_data, cf_data)


def get_key_metrics(ticker: str) -> pd.DataFrame:
    """
    Returns quarterly key metrics for a single ticker.
    Used for: P/E, EV/EBITDA, ROE, ROIC.
    """
    data = _get("/key-metrics", params={"symbol": ticker, "period": "quarter", "limit": 120})
    if not data:
        return pd.DataFrame()
    return _parse_key_metrics(ticker, data)


def get_dividends(ticker: str) -> pd.DataFrame:
    data = _get("/dividends", params={"symbol": ticker})
    if not data:
        return pd.DataFrame()
    df           = pd.DataFrame(data)
    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_splits(ticker: str) -> pd.DataFrame:
    data = _get("/splits", params={"symbol": ticker})
    if not data:
        return pd.DataFrame()
    df           = pd.DataFrame(data)
    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_financial_scores(ticker: str) -> pd.DataFrame:
    data = _get("/financial-scores", params={"symbol": ticker})
    if not data:
        return pd.DataFrame()
    df           = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    df["ticker"] = ticker
    cols         = ["ticker", "piotroskiScore", "altmanZScore"]
    return df[[c for c in cols if c in df.columns]]


# ============================================================
# SHARED PARSE HELPERS (used by both sync and async paths)
# ============================================================

def _merge_financials(ticker: str, inc_data: list, bs_data: list, cf_data: list) -> pd.DataFrame:
    """Merges income, balance sheet, and cash flow into one DataFrame."""
    inc = pd.DataFrame(inc_data)
    bs  = pd.DataFrame(bs_data)
    cf  = pd.DataFrame(cf_data)

    for df in [inc, bs, cf]:
        df["date"] = pd.to_datetime(df["date"])

    inc_cols = ["date", "revenue", "grossProfit", "operatingIncome",
                "netIncome", "eps", "epsDiluted", "weightedAverageShsOut",
                "ebitda", "costOfRevenue"]
    bs_cols  = ["date", "totalAssets", "totalCurrentAssets", "totalCurrentLiabilities",
                "totalStockholdersEquity", "totalEquity", "longTermDebt",
                "totalDebt", "inventory", "accountReceivables",
                "cashAndCashEquivalents", "netReceivables"]
    cf_cols  = ["date", "operatingCashFlow", "freeCashFlow", "capitalExpenditure", "netIncome"]

    inc = inc[[c for c in inc_cols if c in inc.columns]]
    bs  = bs[[c for c in bs_cols if c in bs.columns]]
    cf  = cf[[c for c in cf_cols if c in cf.columns]].rename(columns={"netIncome": "netIncomeCF"})

    merged           = inc.merge(bs, on="date", how="inner").merge(cf, on="date", how="inner")
    merged["ticker"] = ticker

    if "grossProfit" in merged.columns and "revenue" in merged.columns:
        merged["grossMargin"] = merged["grossProfit"] / merged["revenue"].replace(0, None)
    if "netIncome" in merged.columns and "totalStockholdersEquity" in merged.columns:
        merged["roe"] = merged["netIncome"] / merged["totalStockholdersEquity"].replace(0, None)

    return merged.sort_values("date").reset_index(drop=True)


def _parse_key_metrics(ticker: str, data: list) -> pd.DataFrame:
    """Parses raw key metrics API response into standard DataFrame."""
    df           = pd.DataFrame(data)
    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["date"])

    if "earningsYield" in df.columns:
        df["peRatio"] = df["earningsYield"].apply(
            lambda x: round(1 / x, 4) if x and x != 0 else None
        )
    else:
        df["peRatio"] = None

    df = df.rename(columns={
        "evToEBITDA":              "evToEbitda",
        "returnOnEquity":          "roe",
        "returnOnInvestedCapital": "roic",
    })

    cols = ["date", "ticker", "peRatio", "evToEbitda", "roe", "roic"]
    return df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)


def _parse_prices(ticker: str, data: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Parses raw price API response into standard DataFrame."""
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    df           = pd.DataFrame(data)
    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["date"])
    cols         = ["date", "ticker", "open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)


# ============================================================
# ASYNC SINGLE-TICKER FUNCTIONS
# ============================================================

async def _async_get_prices(session, semaphore, ticker: str,
                              start_date: str, end_date: str) -> pd.DataFrame:
    data = await _aget(session, semaphore, "/historical-price-eod/full", {
        "symbol": ticker, "from": start_date, "to": end_date
    })
    return _parse_prices(ticker, data, start_date, end_date)


async def _async_get_financials(session, semaphore, ticker: str) -> pd.DataFrame:
    params = {"symbol": ticker, "period": "quarter", "limit": 120}
    inc, bs, cf = await asyncio.gather(
        _aget(session, semaphore, "/income-statement",        params),
        _aget(session, semaphore, "/balance-sheet-statement", params),
        _aget(session, semaphore, "/cash-flow-statement",     params),
    )
    if not inc or not bs or not cf:
        return pd.DataFrame()
    return _merge_financials(ticker, inc, bs, cf)


async def _async_get_key_metrics(session, semaphore, ticker: str) -> pd.DataFrame:
    data = await _aget(session, semaphore, "/key-metrics", {
        "symbol": ticker, "period": "quarter", "limit": 120
    })
    if not data:
        return pd.DataFrame()
    return _parse_key_metrics(ticker, data)


# ============================================================
# ASYNC BULK FUNCTIONS (backtest fetcher)
# ============================================================

async def _async_bulk(fetch_fn, tickers: list, label: str,
                       checkpoint_file: str = None,
                       done_tickers: set = None) -> list:
    """
    Generic async bulk fetch runner.
    Runs fetch_fn concurrently for all tickers with semaphore limiting.
    Prints progress every 100 tickers.
    Returns list of DataFrames (empties excluded).
    """
    import aiohttp

    semaphore = asyncio.Semaphore(ASYNC_SEMAPHORE)
    results   = []
    completed = 0

    async with aiohttp.ClientSession() as session:
        tasks = {
            ticker: asyncio.create_task(fetch_fn(session, semaphore, ticker))
            for ticker in tickers
            if done_tickers is None or ticker not in done_tickers
        }

        for ticker, task in tasks.items():
            try:
                df = await task
                if not df.empty:
                    results.append(df)
                if done_tickers is not None:
                    done_tickers.add(ticker)
            except Exception as e:
                print(f"[fmp_fetcher] {label} failed for {ticker}: {e}")

            completed += 1
            if completed % 100 == 0:
                print(f"[fmp_fetcher] {label}: {completed}/{len(tasks)} done")

            # Checkpoint every 50 if file provided
            if checkpoint_file and completed % 50 == 0 and done_tickers is not None:
                with open(checkpoint_file, "w") as f:
                    f.write("\n".join(done_tickers))

    return results


def async_bulk_prices(tickers: list, start_date: str, end_date: str,
                       done_tickers: set = None,
                       checkpoint_file: str = None) -> pd.DataFrame:
    """
    Async bulk price fetch. Drop-in replacement for get_bulk_prices.
    Runs in existing or new event loop.
    """
    async def _run():
        import aiohttp
        semaphore = asyncio.Semaphore(ASYNC_SEMAPHORE)
        results   = []
        completed = 0
        done      = done_tickers or set()

        remaining = [t for t in tickers if t not in done]
        print(f"[fmp_fetcher] Async price fetch: {len(remaining)} tickers "
              f"({len(tickers) - len(remaining)} already done)")

        async with aiohttp.ClientSession() as session:
            tasks = [
                (t, asyncio.create_task(
                    _async_get_prices(session, semaphore, t, start_date, end_date)
                ))
                for t in remaining
            ]
            for ticker, task in tasks:
                try:
                    df = await task
                    if not df.empty:
                        results.append(df)
                    done.add(ticker)
                except Exception as e:
                    print(f"[fmp_fetcher] Price failed: {ticker}: {e}")

                completed += 1
                if completed % 100 == 0:
                    print(f"[fmp_fetcher] Prices: {completed}/{len(remaining)} done")
                if checkpoint_file and completed % 50 == 0:
                    with open(checkpoint_file, "w") as f:
                        f.write("\n".join(done))

        return results

    dfs = asyncio.run(_run())
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"[fmp_fetcher] Async prices complete: {len(combined)} rows | "
          f"{combined['ticker'].nunique() if not combined.empty else 0} tickers")
    return combined


def async_bulk_financials(tickers: list,
                           done_tickers: set = None,
                           checkpoint_file: str = None) -> pd.DataFrame:
    """
    Async bulk financials fetch. Drop-in replacement for get_bulk_financials.
    """
    async def _run():
        import aiohttp
        semaphore = asyncio.Semaphore(ASYNC_SEMAPHORE)
        results   = []
        completed = 0
        done      = done_tickers or set()
        remaining = [t for t in tickers if t not in done]

        print(f"[fmp_fetcher] Async financials fetch: {len(remaining)} tickers "
              f"({len(tickers) - len(remaining)} already done)")

        async with aiohttp.ClientSession() as session:
            tasks = [
                (t, asyncio.create_task(_async_get_financials(session, semaphore, t)))
                for t in remaining
            ]
            for ticker, task in tasks:
                try:
                    df = await task
                    if not df.empty:
                        results.append(df)
                    done.add(ticker)
                except Exception as e:
                    print(f"[fmp_fetcher] Financials failed: {ticker}: {e}")

                completed += 1
                if completed % 100 == 0:
                    print(f"[fmp_fetcher] Financials: {completed}/{len(remaining)} done")
                if checkpoint_file and completed % 50 == 0:
                    with open(checkpoint_file, "w") as f:
                        f.write("\n".join(done))

        return results

    dfs = asyncio.run(_run())
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"[fmp_fetcher] Async financials complete: {len(combined)} rows | "
          f"{combined['ticker'].nunique() if not combined.empty else 0} tickers")
    return combined


def async_bulk_key_metrics(tickers: list,
                            done_tickers: set = None,
                            checkpoint_file: str = None) -> pd.DataFrame:
    """
    Async bulk key metrics fetch. Drop-in replacement for get_bulk_key_metrics.
    """
    async def _run():
        import aiohttp
        semaphore = asyncio.Semaphore(ASYNC_SEMAPHORE)
        results   = []
        completed = 0
        done      = done_tickers or set()
        remaining = [t for t in tickers if t not in done]

        print(f"[fmp_fetcher] Async key metrics fetch: {len(remaining)} tickers "
              f"({len(tickers) - len(remaining)} already done)")

        async with aiohttp.ClientSession() as session:
            tasks = [
                (t, asyncio.create_task(_async_get_key_metrics(session, semaphore, t)))
                for t in remaining
            ]
            for ticker, task in tasks:
                try:
                    df = await task
                    if not df.empty:
                        results.append(df)
                    done.add(ticker)
                except Exception as e:
                    print(f"[fmp_fetcher] Key metrics failed: {ticker}: {e}")

                completed += 1
                if completed % 100 == 0:
                    print(f"[fmp_fetcher] Key metrics: {completed}/{len(remaining)} done")
                if checkpoint_file and completed % 50 == 0:
                    with open(checkpoint_file, "w") as f:
                        f.write("\n".join(done))

        return results

    dfs = asyncio.run(_run())
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"[fmp_fetcher] Async key metrics complete: {len(combined)} rows | "
          f"{combined['ticker'].nunique() if not combined.empty else 0} tickers")
    return combined


# ============================================================
# SYNC BULK FUNCTIONS (live pipeline — unchanged behavior)
# ============================================================

def get_bulk_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    all_data, failed = [], []
    for i, ticker in enumerate(tickers):
        try:
            df = get_daily_prices(ticker, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        except Exception:
            failed.append(ticker)
        if (i + 1) % 50 == 0:
            print(f"[fmp_fetcher] Fetched prices: {i+1}/{len(tickers)}")
    if failed:
        print(f"[fmp_fetcher] Failed tickers ({len(failed)}): {failed[:10]}")
    combined = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    print(f"[fmp_fetcher] Bulk price fetch complete: {len(combined)} rows, "
          f"{combined['ticker'].nunique() if not combined.empty else 0} tickers")
    return combined


def get_bulk_financials(tickers: list) -> pd.DataFrame:
    all_data, failed = [], []
    for i, ticker in enumerate(tickers):
        try:
            df = get_quarterly_financials(ticker)
            if not df.empty:
                all_data.append(df)
        except Exception:
            failed.append(ticker)
        if (i + 1) % 50 == 0:
            print(f"[fmp_fetcher] Fetched financials: {i+1}/{len(tickers)}")
    if failed:
        print(f"[fmp_fetcher] Failed financials ({len(failed)}): {failed[:10]}")
    combined = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    print(f"[fmp_fetcher] Bulk financials complete: {len(combined)} rows")
    return combined


def get_bulk_key_metrics(tickers: list) -> pd.DataFrame:
    all_data, failed = [], []
    for i, ticker in enumerate(tickers):
        try:
            df = get_key_metrics(ticker)
            if not df.empty:
                all_data.append(df)
        except Exception:
            failed.append(ticker)
        if (i + 1) % 50 == 0:
            print(f"[fmp_fetcher] Fetched key metrics: {i+1}/{len(tickers)}")
    if failed:
        print(f"[fmp_fetcher] Failed key metrics ({len(failed)}): {failed[:10]}")
    combined = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    print(f"[fmp_fetcher] Bulk key metrics complete: {len(combined)} rows")
    return combined


# ============================================================
# VALIDATION
# ============================================================

def validate_data_quality(df: pd.DataFrame, label: str) -> bool:
    if df.empty:
        notify(f"Data quality failure: {label} returned empty DataFrame", level="critical")
        return False
    missing_pct = df.isnull().mean().mean()
    if missing_pct > 0.05:
        notify(f"Data quality gate failed for {label}: {missing_pct:.2%} missing", level="critical")
        return False
    return True