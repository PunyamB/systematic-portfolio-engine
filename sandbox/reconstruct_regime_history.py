# sandbox/reconstruct_regime_history.py
# Reconstructs point-in-time regime states across the full backtest period.
# Uses correct per-date S&P 500 membership for breadth computation.
# Output: data/backtest/regime_history.parquet

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import date as date_type

from data.storage import load_regime_data
from regime.detector import compute_economic_cycle, compute_composite_regime
from utils.config_loader import get_config

cfg = get_config()
R = cfg["regime"]

PRICES_PATH  = Path("data/backtest/prices.parquet")
UNIVERSE_PKL = Path("data/backtest/universe_history.pkl")
OUTPUT_PATH  = Path("data/backtest/regime_history.parquet")

# ── Load data ──
print("Loading prices...")
prices = pd.read_parquet(PRICES_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["ticker", "date"])

print("Loading VIX...")
vix_df = load_regime_data("vix")
vix_df["date"] = pd.to_datetime(vix_df["date"])
vix_df = vix_df.sort_values("date")

print("Loading universe history...")
with open(UNIVERSE_PKL, "rb") as f:
    universe_history = pickle.load(f)

# ── Get monthly trading dates from 2009 ──
all_dates = sorted(prices["date"].unique())
monthly_dates = []
seen = set()
for d in all_dates:
    key = (d.year, d.month)
    if key not in seen:
        seen.add(key)
        monthly_dates.append(d)

monthly_dates = [d for d in monthly_dates if d.year >= 2009]
print(f"Computing regime for {len(monthly_dates)} monthly dates...")

# ── Point-in-time breadth ──
def compute_breadth_pit(as_of: pd.Timestamp, members: set) -> float:
    """
    Computes breadth using only S&P 500 members on as_of date.
    % of members trading above their 200d MA.
    """
    cutoff = as_of
    lookback_start = cutoff - pd.Timedelta(days=300)  # 300 calendar days covers 200 trading days

    ticker_prices = prices[
        (prices["ticker"].isin(members)) &
        (prices["date"] >= lookback_start) &
        (prices["date"] <= cutoff)
    ]

    if ticker_prices.empty:
        return 0.5

    # Compute 200d MA per ticker
    above = 0
    valid = 0
    for ticker, grp in ticker_prices.groupby("ticker"):
        grp = grp.sort_values("date")
        if len(grp) < 200:
            continue
        ma_200 = grp["close"].tail(200).mean()
        last_close = grp["close"].iloc[-1]
        valid += 1
        if last_close > ma_200:
            above += 1

    return above / valid if valid > 0 else 0.5

# ── Reconstruct regime ──
records = []
for i, dt in enumerate(monthly_dates):
    as_of = dt.date()
    ts    = pd.Timestamp(as_of)

    try:
        # ── L1: VIX ──
        vix_slice = vix_df[vix_df["date"] <= ts]
        stress_state = "elevated"
        vix_val      = None
        vix_ratio    = None
        breadth      = None

        if len(vix_slice) >= R["vix_lookback"]:
            vix_val   = float(vix_slice["vix"].iloc[-1])
            avg_vix   = float(vix_slice["vix"].tail(R["vix_lookback"]).mean())
            vix_ratio = round(vix_val / avg_vix, 4) if avg_vix > 0 else 1.0

            # Point-in-time members
            members = universe_history.get(ts, set())
            if not members:
                # Use nearest available date
                available = [d for d in universe_history.keys() if d <= ts]
                if available:
                    members = universe_history[max(available)]

            breadth = compute_breadth_pit(ts, members)
            breadth = round(breadth, 4)

            if vix_ratio < R["vix_elevated_threshold"] and breadth > R["breadth_high"]:
                stress_state = "low_stress"
            elif vix_ratio > R["vix_crisis_threshold"] and breadth < R["breadth_low"]:
                stress_state = "crisis"
            else:
                stress_state = "elevated"

        # ── L2: Economic cycle ──
        l2 = compute_economic_cycle(as_of)

        # ── Composite ──
        composite = compute_composite_regime(stress_state, l2["cycle_state"])

        records.append({
            "date":           dt,
            "composite":      composite,
            "stress_state":   stress_state,
            "cycle_state":    l2["cycle_state"],
            "vix":            vix_val,
            "vix_ratio":      vix_ratio,
            "breadth":        breadth,
            "yield_spread":   l2["yield_spread"],
            "curve_inverted": l2["curve_inverted"],
            "credit_trend":   l2["credit_spread_trend"],
        })

        if i % 12 == 0:
            print(f"  {as_of} → {composite:8s} | VIX={vix_val} ratio={vix_ratio} breadth={breadth} members={len(members) if members else 0}")

    except Exception as e:
        print(f"  SKIP {as_of}: {e}")

# ── Save ──
df = pd.DataFrame(records)
df = df.sort_values("date").reset_index(drop=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)

print(f"\nDone. {len(df)} regime records saved to {OUTPUT_PATH}")
print("\nRegime distribution:")
print(df["composite"].value_counts())
print("\nSample:")
print(df[["date","composite","vix","breadth","yield_spread"]].tail(10).to_string())