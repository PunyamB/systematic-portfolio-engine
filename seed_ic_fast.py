# seed_ic_fast.py
# Seeds IC history from precomputed backtest data.
# Uses signals_history.parquet and forward_returns.parquet
# which already have signal scores and forward returns
# for 204 monthly dates from 2009-2025.
#
# Usage:
#   python seed_ic_fast.py                # seeds from 2024-01-01 onward
#   python seed_ic_fast.py 2020-01-01     # seeds from a specific start date

import sys
import pandas as pd
import numpy as np
from pathlib import Path

IC_HISTORY_FILE = Path("data/processed/ic_history.parquet")
SIGNALS_FILE    = Path("data/backtest/precomputed/signals_history.parquet")
FWD_FILE        = Path("data/backtest/precomputed/forward_returns.parquet")

SIGNAL_NAMES = [
    "momentum_12_1",
    "earnings_momentum",
    "pe_zscore",
    "pb_zscore",
    "ev_ebitda_zscore",
    "roe_stability",
    "gross_margin_trend",
    "piotroski",
    "earnings_accruals",
    "short_term_reversal",
    "rsi_extremes",
]


def seed_ic(start_date: str = "2024-01-01"):
    print(f"\n[seed_ic] ============================================================")
    print(f"[seed_ic] Fast IC History Seeding (from backtest precomputed data)")
    print(f"[seed_ic] ============================================================\n")

    # Load precomputed data
    sig = pd.read_parquet(SIGNALS_FILE)
    fwd = pd.read_parquet(FWD_FILE)

    sig["date"] = pd.to_datetime(sig["date"])
    fwd["date"] = pd.to_datetime(fwd["date"])

    # Filter to seed period
    sig = sig[sig["date"] >= start_date]
    fwd = fwd[fwd["date"] >= start_date]

    dates = sorted(sig["date"].unique())
    print(f"[seed_ic] Dates in range: {len(dates)} ({dates[0].date()} to {dates[-1].date()})")

    # Compute IC per signal per date
    all_ic_rows = []

    for dt in dates:
        sig_day = sig[sig["date"] == dt]
        fwd_day = fwd[fwd["date"] == dt]

        # Use 1-month forward return for IC
        if "fwd_1m" not in fwd_day.columns:
            continue

        merged = sig_day.merge(fwd_day[["ticker", "fwd_1m"]], on="ticker", how="inner")
        merged = merged.dropna(subset=["fwd_1m"])

        if len(merged) < 30:
            continue

        for signal_name in SIGNAL_NAMES:
            if signal_name not in merged.columns:
                continue

            signal_vals = merged[signal_name].dropna()
            fwd_vals = merged.loc[signal_vals.index, "fwd_1m"]

            if len(signal_vals) < 30:
                continue

            ic = float(signal_vals.corr(fwd_vals, method="spearman"))

            if not np.isnan(ic):
                all_ic_rows.append({
                    "date":        dt,
                    "signal_name": signal_name,
                    "ic":          ic,
                })

    if not all_ic_rows:
        print("[seed_ic] No IC records computed")
        return

    new_ic = pd.DataFrame(all_ic_rows)

    # Merge with any existing IC history
    if IC_HISTORY_FILE.exists():
        existing = pd.read_parquet(IC_HISTORY_FILE)
        existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, new_ic], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "signal_name"], keep="last")
    else:
        combined = new_ic

    combined = combined.sort_values(["signal_name", "date"]).reset_index(drop=True)

    IC_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(IC_HISTORY_FILE, index=False)

    n_dates = combined["date"].nunique()
    n_signals = combined["signal_name"].nunique()

    print(f"\n[seed_ic] ============================================================")
    print(f"[seed_ic] IC HISTORY SEEDED")
    print(f"[seed_ic] Total IC records: {len(combined)}")
    print(f"[seed_ic] Unique dates: {n_dates}")
    print(f"[seed_ic] Signals covered: {n_signals}")
    print(f"[seed_ic] ============================================================")

    # IC summary
    print(f"\n{'Signal':<25} {'Mean IC':>10} {'Std IC':>10} {'IC-IR':>10} {'Count':>8}")
    print("-" * 68)
    for signal_name in SIGNAL_NAMES:
        sig_ic = combined[combined["signal_name"] == signal_name]["ic"]
        if len(sig_ic) >= 3:
            mean_ic = sig_ic.mean()
            std_ic = sig_ic.std()
            ic_ir = mean_ic / std_ic if std_ic > 0 else 0
            print(f"{signal_name:<25} {mean_ic:>10.4f} {std_ic:>10.4f} {ic_ir:>10.4f} {len(sig_ic):>8}")
        else:
            print(f"{signal_name:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {len(sig_ic):>8}")

    print(f"\n[seed_ic] Next steps:")
    print(f"  python -c \"from pipeline.runner import run_pipeline; run_pipeline(force_rebalance=True)\"")
    print(f"  python approve.py")
    print(f"  python execute.py")


if __name__ == "__main__":
    start = "2024-01-01"
    if len(sys.argv) > 1:
        start = sys.argv[1]

    seed_ic(start_date=start)
