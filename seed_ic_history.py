# seed_ic_history.py
# Seeds IC history from historical data so the live combiner
# uses IC-IR weights from day one instead of equal weights.
#
# Computes signal scores for each trading day in the seed period,
# then correlates them with realized 21-day forward returns to
# build ic_history.parquet.
#
# Usage:
#   python seed_ic_history.py                    # seeds from 2025-01-01 to latest
#   python seed_ic_history.py 2025-06-01         # seeds from a specific start date
#
# After running:
#   - data/processed/ic_history.parquet will contain IC records
#   - Next pipeline run's combiner will use IC-IR weights
#   - Force rebalance to apply: python -c "from pipeline.runner import run_pipeline; run_pipeline(force_rebalance=True)"

import sys
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

# Project imports
from data.storage import load_prices, load_financials, load_key_metrics, load_constituents
from signals.momentum_12_1 import compute as momentum_12_1
from signals.earnings_momentum import compute as earnings_momentum
from signals.pe_zscore import compute as pe_zscore
from signals.pb_zscore import compute as pb_zscore
from signals.ev_ebitda_zscore import compute as ev_ebitda_zscore
from signals.roe_stability import compute as roe_stability
from signals.gross_margin_trend import compute as gross_margin_trend
from signals.piotroski import compute as piotroski
from signals.earnings_accruals import compute as earnings_accruals
from signals.short_term_reversal import compute as short_term_reversal
from signals.rsi_extremes import compute as rsi_extremes

IC_HISTORY_FILE = Path("data/processed/ic_history.parquet")
FORWARD_DAYS = 21  # 21-day forward return for IC evaluation

ALL_SIGNALS = {
    "momentum_12_1":       momentum_12_1,
    "earnings_momentum":   earnings_momentum,
    "pe_zscore":           pe_zscore,
    "pb_zscore":           pb_zscore,
    "ev_ebitda_zscore":    ev_ebitda_zscore,
    "roe_stability":       roe_stability,
    "gross_margin_trend":  gross_margin_trend,
    "piotroski":           piotroski,
    "earnings_accruals":   earnings_accruals,
    "short_term_reversal": short_term_reversal,
    "rsi_extremes":        rsi_extremes,
}


def get_trading_days(prices_df: pd.DataFrame, start_date: str, end_date: str) -> list:
    """Returns sorted list of unique trading dates in the price data within range."""
    dates = pd.to_datetime(prices_df["date"]).dt.date.unique()
    dates = sorted([d for d in dates if pd.Timestamp(start_date).date() <= d <= pd.Timestamp(end_date).date()])
    return dates


def compute_forward_returns(prices_df: pd.DataFrame, as_of_date, forward_days: int = 21) -> pd.Series:
    """
    Computes forward returns for all tickers from as_of_date.
    forward_return = price[T + forward_days] / price[T] - 1
    Returns Series indexed by ticker.
    """
    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    as_of_ts = pd.Timestamp(as_of_date)

    # Get close on as_of_date
    today_prices = prices_df[prices_df["date"] == as_of_ts].set_index("ticker")["close"]

    # Get close ~forward_days trading days later
    future_dates = sorted(prices_df[prices_df["date"] > as_of_ts]["date"].unique())

    if len(future_dates) < forward_days:
        return pd.Series(dtype=float)

    target_date = future_dates[forward_days - 1]
    future_prices = prices_df[prices_df["date"] == target_date].set_index("ticker")["close"]

    # Align tickers
    common = today_prices.index.intersection(future_prices.index)
    if len(common) == 0:
        return pd.Series(dtype=float)

    fwd_returns = (future_prices[common] / today_prices[common]) - 1.0
    return fwd_returns


def seed_ic_history(start_date: str = "2025-01-02", end_date: str = None):
    """
    Main seeding function.
    Loops through trading days, computes signals and forward returns,
    calculates IC per signal, and writes to ic_history.parquet.
    """
    print(f"\n[seed_ic] ============================================================")
    print(f"[seed_ic] IC History Seeding")
    print(f"[seed_ic] ============================================================\n")

    # Load all data once (uses cache)
    print("[seed_ic] Loading data...")
    prices = load_prices()
    if prices.empty:
        print("[seed_ic] ERROR: No price data available")
        return

    prices["date"] = pd.to_datetime(prices["date"])

    if end_date is None:
        # Stop FORWARD_DAYS before the last available date
        # (need forward returns to exist)
        all_dates = sorted(prices["date"].dt.date.unique())
        if len(all_dates) < FORWARD_DAYS + 10:
            print("[seed_ic] ERROR: Not enough price history")
            return
        end_date = str(all_dates[-(FORWARD_DAYS + 1)])

    trading_days = get_trading_days(prices, start_date, end_date)
    print(f"[seed_ic] Seed period: {start_date} to {end_date}")
    print(f"[seed_ic] Trading days to process: {len(trading_days)}")

    if len(trading_days) == 0:
        print("[seed_ic] No trading days in range")
        return

    # Load existing IC history to avoid duplicates
    existing_ic = pd.DataFrame()
    if IC_HISTORY_FILE.exists():
        existing_ic = pd.read_parquet(IC_HISTORY_FILE)
        existing_ic["date"] = pd.to_datetime(existing_ic["date"])
        existing_dates = set(existing_ic["date"].dt.date.unique())
        trading_days = [d for d in trading_days if d not in existing_dates]
        print(f"[seed_ic] Skipping {len(existing_dates)} already-computed dates")
        print(f"[seed_ic] Remaining days to compute: {len(trading_days)}")

    if len(trading_days) == 0:
        print("[seed_ic] All dates already seeded")
        return

    all_ic_rows = []
    failed_days = 0

    for i, trade_date in enumerate(trading_days):
        try:
            # Compute forward returns
            fwd_returns = compute_forward_returns(prices, trade_date, FORWARD_DAYS)
            if fwd_returns.empty or len(fwd_returns) < 50:
                failed_days += 1
                continue

            # Compute each signal and calculate IC
            for signal_name, compute_fn in ALL_SIGNALS.items():
                try:
                    scores_df = compute_fn(trade_date)
                    if scores_df.empty or "raw_score" not in scores_df.columns:
                        continue

                    scores = scores_df.set_index("ticker")["raw_score"]

                    # Align tickers
                    common = scores.index.intersection(fwd_returns.index)
                    if len(common) < 30:
                        continue

                    # Spearman rank correlation
                    ic = float(scores[common].corr(fwd_returns[common], method="spearman"))

                    if not np.isnan(ic):
                        all_ic_rows.append({
                            "date":        trade_date,
                            "signal_name": signal_name,
                            "ic":          ic,
                        })

                except Exception:
                    continue  # skip failed signals silently

            if (i + 1) % 10 == 0:
                print(f"[seed_ic] Progress: {i + 1}/{len(trading_days)} days | {len(all_ic_rows)} IC records")

        except Exception as e:
            failed_days += 1
            if (i + 1) % 10 == 0:
                print(f"[seed_ic] Day {trade_date} failed: {e}")

    if not all_ic_rows:
        print("[seed_ic] No IC records computed — check data availability")
        return

    new_ic = pd.DataFrame(all_ic_rows)
    new_ic["date"] = pd.to_datetime(new_ic["date"])

    # Merge with existing
    if not existing_ic.empty:
        combined = pd.concat([existing_ic, new_ic], ignore_index=True)
    else:
        combined = new_ic

    combined = combined.sort_values(["signal_name", "date"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["date", "signal_name"], keep="last")
    combined.to_parquet(IC_HISTORY_FILE, index=False)

    # Summary
    n_dates = combined["date"].nunique()
    n_signals = combined["signal_name"].nunique()

    print(f"\n[seed_ic] ============================================================")
    print(f"[seed_ic] IC HISTORY SEEDED")
    print(f"[seed_ic] Total IC records: {len(combined)}")
    print(f"[seed_ic] Unique dates: {n_dates}")
    print(f"[seed_ic] Signals covered: {n_signals}")
    print(f"[seed_ic] Failed days: {failed_days}")
    print(f"[seed_ic] ============================================================")

    # Print IC summary per signal
    print(f"\n[seed_ic] IC Summary (mean | std | IC-IR):")
    print(f"{'Signal':<25} {'Mean IC':>10} {'Std IC':>10} {'IC-IR':>10}")
    print("-" * 58)
    for signal_name in sorted(ALL_SIGNALS.keys()):
        sig_ic = combined[combined["signal_name"] == signal_name]["ic"]
        if len(sig_ic) >= 5:
            mean_ic = sig_ic.mean()
            std_ic = sig_ic.std()
            ic_ir = mean_ic / std_ic if std_ic > 0 else 0
            print(f"{signal_name:<25} {mean_ic:>10.4f} {std_ic:>10.4f} {ic_ir:>10.4f}")
        else:
            print(f"{signal_name:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print(f"\n[seed_ic] Next step: python -c \"from pipeline.runner import run_pipeline; run_pipeline(force_rebalance=True)\"")


if __name__ == "__main__":
    start = "2025-01-02"
    if len(sys.argv) > 1:
        start = sys.argv[1]

    seed_ic_history(start_date=start)
