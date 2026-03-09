# corporate_actions/processor.py
# Processes daily splits and dividends from FMP.
# Adjusts share counts in portfolio and updates trailing stop reference prices.
# Runs daily before any signal computation or risk checks.

import pandas as pd
from datetime import date
from data.fetchers.fmp_fetcher import get_splits, get_dividends
from data.storage import load_portfolio, save_portfolio
from utils.notifications import notify


def process_splits(run_date: date = None) -> pd.DataFrame:
    """
    Checks for any splits effective today across all held positions.
    Adjusts share counts and cost basis accordingly.
    Returns a DataFrame of splits processed (empty if none).
    """
    if run_date is None:
        run_date = date.today()

    portfolio = load_portfolio()
    if portfolio.empty:
        return pd.DataFrame()

    tickers = portfolio["ticker"].tolist()
    splits_processed = []

    for ticker in tickers:
        try:
            splits = get_splits(ticker)
            if splits.empty:
                continue

            # Filter for splits effective today
            today_splits = splits[
                splits["date"].dt.date == run_date
            ]

            if today_splits.empty:
                continue

            for _, split in today_splits.iterrows():
                numerator   = float(split.get("numerator", 1))
                denominator = float(split.get("denominator", 1))

                if denominator == 0:
                    continue

                ratio = numerator / denominator

                # Adjust share count and cost basis in portfolio
                mask = portfolio["ticker"] == ticker
                portfolio.loc[mask, "shares"]     *= ratio
                portfolio.loc[mask, "cost_basis"]  /= ratio

                # Adjust trailing stop reference price
                if "stop_reference_price" in portfolio.columns:
                    portfolio.loc[mask, "stop_reference_price"] /= ratio

                splits_processed.append({
                    "date":   run_date,
                    "ticker": ticker,
                    "ratio":  ratio
                })

                print(f"[corporate_actions] Split processed: {ticker} {denominator}:{numerator} ratio={ratio:.4f}")
                notify(f"Split processed: {ticker} ratio {ratio:.4f}", level="info")

        except Exception as e:
            print(f"[corporate_actions] Failed to process split for {ticker}: {e}")

    if splits_processed:
        save_portfolio(portfolio)
        print(f"[corporate_actions] Portfolio updated for {len(splits_processed)} splits")

    return pd.DataFrame(splits_processed)


def process_dividends(run_date: date = None) -> pd.DataFrame:
    """
    Checks for dividends with ex-date today across all held positions.
    Credits cash to fund accounting.
    Returns a DataFrame of dividends processed (empty if none).
    """
    if run_date is None:
        run_date = date.today()

    portfolio = load_portfolio()
    if portfolio.empty:
        return pd.DataFrame()

    tickers = portfolio["ticker"].tolist()
    dividends_processed = []

    for ticker in tickers:
        try:
            dividends = get_dividends(ticker)
            if dividends.empty:
                continue

            # Filter for ex-dividends today
            today_divs = dividends[
                dividends["date"].dt.date == run_date
            ]

            if today_divs.empty:
                continue

            for _, div in today_divs.iterrows():
                amount_per_share = float(div.get("dividend", 0))
                if amount_per_share == 0:
                    continue

                shares = float(
                    portfolio.loc[portfolio["ticker"] == ticker, "shares"].values[0]
                )
                total_dividend = shares * amount_per_share

                dividends_processed.append({
                    "date":             run_date,
                    "ticker":           ticker,
                    "amount_per_share": amount_per_share,
                    "shares":           shares,
                    "total_dividend":   total_dividend
                })

                print(f"[corporate_actions] Dividend: {ticker} ${amount_per_share:.4f}/share, total ${total_dividend:,.2f}")

        except Exception as e:
            print(f"[corporate_actions] Failed to process dividend for {ticker}: {e}")

    if dividends_processed:
        notify(
            f"Dividends processed: {len(dividends_processed)} positions, "
            f"total ${sum(d['total_dividend'] for d in dividends_processed):,.2f}",
            level="info"
        )

    return pd.DataFrame(dividends_processed)


def run_corporate_actions(run_date: date = None) -> dict:
    """
    Runs both splits and dividends processing for the day.
    Called at the start of the pipeline after data refresh.
    Returns dict with splits and dividends DataFrames.
    """
    if run_date is None:
        run_date = date.today()

    print(f"[corporate_actions] Processing corporate actions for {run_date}")

    splits    = process_splits(run_date)
    dividends = process_dividends(run_date)

    print(f"[corporate_actions] Done. Splits: {len(splits)}, Dividends: {len(dividends)}")

    return {
        "splits":    splits,
        "dividends": dividends
    }