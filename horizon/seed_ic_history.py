"""seed_ic_history.py - Seed Horizon's ic_history.parquet."""
import sys
import pandas as pd
from pathlib import Path
from datetime import date

# IC panel lives in test1_crsp
IC_PANEL_PATH = Path(r"D:\Projects\StrategyResearchLab\experiments\test1_crsp\results\precomputed\ic_panel.parquet")

# Signals + fwd returns live in exp006
SIGNALS_PATH = Path(r"D:\Projects\StrategyResearchLab\experiments\exp006_extended_wf\data\precomputed\signals_history.parquet")
FWDRET_PATH  = Path(r"D:\Projects\StrategyResearchLab\experiments\exp006_extended_wf\data\precomputed\forward_returns.parquet")

HORIZON_BASE = Path(__file__).resolve().parent
TARGET_PATH  = HORIZON_BASE / "data" / "processed" / "ic_history.parquet"

SIGNAL_NAMES = [
    "momentum_12_1", "earnings_momentum", "pe_zscore", "pb_zscore",
    "ev_ebitda_zscore", "roe_stability", "gross_margin_trend", "piotroski",
    "earnings_accruals", "short_term_reversal", "rsi_extremes", "revenue_growth",
    "low_volatility", "fcf_yield", "volume_momentum",
]


def load_ic_panel_long() -> pd.DataFrame:
    print(f"[seed] Loading IC panel: {IC_PANEL_PATH}")
    ic_wide = pd.read_parquet(IC_PANEL_PATH)
    print(f"[seed]   shape: {ic_wide.shape}")
    print(f"[seed]   dates: {ic_wide.index.min()} to {ic_wide.index.max()}")

    long = ic_wide.reset_index().melt(id_vars="date", var_name="signal_name", value_name="ic")
    long = long.dropna(subset=["ic"])
    long["date"] = pd.to_datetime(long["date"]).dt.date
    long = long[long["signal_name"].isin(SIGNAL_NAMES)]
    print(f"[seed]   long-format rows: {len(long):,}")
    return long.sort_values(["signal_name", "date"]).reset_index(drop=True)


def compute_ic_for_date(signals_history, forward_returns, run_date):
    sh_day = signals_history[signals_history["date"] == run_date]
    fr_day = forward_returns[forward_returns["date"] == run_date]
    if sh_day.empty or fr_day.empty:
        return {}
    fr_series = fr_day.set_index("ticker")["fwd_1m"]
    ic_values = {}
    for sig in SIGNAL_NAMES:
        if sig not in sh_day.columns:
            continue
        scores = sh_day.set_index("ticker")[sig].dropna()
        aligned = pd.concat([scores, fr_series], axis=1).dropna()
        aligned.columns = ["score", "fwd"]
        if len(aligned) < 10:
            continue
        ic = float(aligned["score"].corr(aligned["fwd"], method="spearman"))
        if pd.notna(ic):
            ic_values[sig] = ic
    return ic_values


def backfill_gap(existing: pd.DataFrame) -> pd.DataFrame:
    last_existing = existing["date"].max()
    print(f"[seed] Last existing IC date: {last_existing}")

    if not SIGNALS_PATH.exists() or not FWDRET_PATH.exists():
        print(f"[seed] signals_history or forward_returns missing — cannot backfill")
        return existing

    sh = pd.read_parquet(SIGNALS_PATH)
    fr = pd.read_parquet(FWDRET_PATH)
    sh["date"] = pd.to_datetime(sh["date"])
    fr["date"] = pd.to_datetime(fr["date"])

    cutoff = pd.Timestamp(last_existing) + pd.Timedelta(days=1)
    today  = pd.Timestamp(date.today())
    target_dates = sorted(d for d in sh["date"].unique()
                          if pd.Timestamp(d) >= cutoff and pd.Timestamp(d) <= today)

    if not target_dates:
        print(f"[seed] No backfill dates available")
        return existing

    print(f"[seed] Backfilling {len(target_dates)} dates from {target_dates[0].date()} to {target_dates[-1].date()}")
    new_rows = []
    for d in target_dates:
        ic_day = compute_ic_for_date(sh, fr, d)
        for sig, ic_val in ic_day.items():
            new_rows.append({"date": d.date(), "signal_name": sig, "ic": ic_val})

    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    print(f"[seed] Added {len(new_df):,} new IC rows")
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "signal_name"], keep="last")
    return combined.sort_values(["signal_name", "date"]).reset_index(drop=True)


def main():
    if TARGET_PATH.exists():
        confirm = input(f"{TARGET_PATH} already exists. Overwrite? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    ic_long = load_ic_panel_long()
    ic_long = backfill_gap(ic_long)

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    ic_long.to_parquet(TARGET_PATH, index=False)

    print(f"\n[seed] DONE")
    print(f"[seed]   target: {TARGET_PATH}")
    print(f"[seed]   total rows: {len(ic_long):,}")
    print(f"[seed]   date range: {ic_long['date'].min()} to {ic_long['date'].max()}")
    print(f"[seed]   signals: {ic_long['signal_name'].nunique()}")
    print(f"\nPer-signal row counts:")
    print(ic_long.groupby("signal_name").size())


if __name__ == "__main__":
    main()
