"""
Module 1: Data Loader
=====================
Loads price data from SPE backtest parquets and constructs properly
aligned loss series for both SPY and the Meridian backtest portfolio.

Losses are negated log returns: positive values = losses.

Meridian NAV is chained from walk-forward windows: daily returns are
computed within each window (avoiding NAV resets at window boundaries),
then compounded into a single continuous $1M NAV series.

Usage:
    from evt_tail_risk.m1_data_loader import load_spy_losses, load_meridian_losses, load_all

    spy = load_spy_losses()        # pd.DataFrame with columns [date, loss]
    mer = load_meridian_losses()   # pd.DataFrame with columns [date, loss]
    spy, mer, meta = load_all()    # Both + metadata dict
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from evt_tail_risk import config


def load_spy_losses(save=True):
    """
    Load SPY prices from evt_tail_risk/data/spy_prices.parquet,
    compute daily log-losses (negated log returns).

    Returns:
        pd.DataFrame with columns [date, loss]
    """
    spy_path = os.path.join(config.DATA_DIR, "spy_prices.parquet")
    if not os.path.exists(spy_path):
        raise FileNotFoundError(
            f"SPY prices not found at {spy_path}. Run evt_setup.py first."
        )

    df = pd.read_parquet(spy_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Use adjClose if available, otherwise close
    price_col = "adjClose" if "adjClose" in df.columns else "close"
    df = df[["date", price_col]].dropna().copy()

    # Log returns, then negate (positive = loss)
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df["loss"] = -df["log_return"]
    df = df.dropna(subset=["loss"]).reset_index(drop=True)

    result = df[["date", "loss"]].copy()

    if save:
        out_path = os.path.join(config.DATA_DIR, "spy_losses.parquet")
        result.to_parquet(out_path, index=False)

    return result


def load_meridian_losses(save=True):
    """
    Chain EXP006 walk-forward NAV windows into a single continuous
    $1M NAV series, then compute log-losses.

    For each window, daily returns are computed internally (avoiding
    NAV resets at window boundaries). Returns are then compounded
    sequentially to produce one continuous NAV from $1M.

    Also saves the continuous NAV series to meridian_nav.parquet.

    Returns:
        pd.DataFrame with columns [date, loss]
    """
    wf_dir = config.WF_RESULTS_DIR
    if not os.path.exists(wf_dir):
        raise FileNotFoundError(f"Walk-forward results not found at {wf_dir}")

    # Find all EXP006 NAV files
    nav_files = [
        f for f in os.listdir(wf_dir)
        if f.startswith("nav_window_") and "exp006_monthly" in f
    ]

    if not nav_files:
        raise FileNotFoundError(f"No EXP006 NAV files found in {wf_dir}")

    # Sort: numbered windows first, holdout last
    nav_files = sorted(
        nav_files,
        key=lambda x: (0 if "holdout" not in x else 1, x)
    )

    # Compute daily returns WITHIN each window (skip first day per window)
    all_returns = []
    for f in nav_files:
        df = pd.read_parquet(os.path.join(wf_dir, f))
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "nav"]].sort_values("date").reset_index(drop=True)

        # Daily return ratio within this window
        df["daily_return"] = df["nav"] / df["nav"].shift(1)
        df = df.dropna(subset=["daily_return"])  # drops first row (reset boundary)
        all_returns.append(df[["date", "daily_return"]])

    # Concatenate and sort
    combined = pd.concat(all_returns, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["date"], keep="first")

    # Chain into one continuous NAV from $1M
    combined["nav"] = 1_000_000 * combined["daily_return"].cumprod()

    # Log-losses (positive = loss)
    combined["loss"] = -np.log(combined["daily_return"])

    result = combined[["date", "loss"]].copy()

    if save:
        out_path = os.path.join(config.DATA_DIR, "meridian_losses.parquet")
        result.to_parquet(out_path, index=False)

        nav_path = os.path.join(config.DATA_DIR, "meridian_nav.parquet")
        combined[["date", "nav"]].to_parquet(nav_path, index=False)

    return result


def _compute_stats(series, name):
    """Compute summary statistics for a loss series."""
    from scipy import stats as sp_stats
    return {
        "name": name,
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "skewness": float(sp_stats.skew(series)),
        "kurtosis": float(sp_stats.kurtosis(series)),  # excess kurtosis
        "median": float(series.median()),
        "q95": float(series.quantile(0.95)),
        "q99": float(series.quantile(0.99)),
    }


def _validate_series(df, name):
    """Run validation checks on a loss series. Returns list of issues."""
    issues = []

    # Check for NaN / Inf
    n_nan = df["loss"].isna().sum()
    n_inf = np.isinf(df["loss"]).sum()
    if n_nan > 0:
        issues.append(f"{name}: {n_nan} NaN values in loss series")
    if n_inf > 0:
        issues.append(f"{name}: {n_inf} infinite values in loss series")

    # Check for gaps > 3 business days
    dates = pd.to_datetime(df["date"])
    gaps = dates.diff().dt.days
    big_gaps = gaps[gaps > 5]  # 5 calendar days ~ 3 business days
    if len(big_gaps) > 0:
        max_gap = big_gaps.max()
        gap_idx = big_gaps.idxmax()
        gap_date = dates.iloc[gap_idx]
        issues.append(
            f"{name}: {len(big_gaps)} gaps > 5 calendar days "
            f"(max {max_gap} days near {gap_date.date()})"
        )

    return issues


def load_all():
    """
    Load both loss series, validate, compute stats, save metadata.

    Returns:
        (spy_losses, meridian_losses, metadata_dict)
    """
    print("Loading SPY losses...")
    spy = load_spy_losses(save=True)
    print(f"  SPY: {len(spy)} observations, {spy['date'].min().date()} to {spy['date'].max().date()}")

    print("Loading Meridian losses...")
    mer = load_meridian_losses(save=True)
    print(f"  Meridian: {len(mer)} observations, {mer['date'].min().date()} to {mer['date'].max().date()}")

    # Validate
    print("\nValidating...")
    all_issues = []
    all_issues.extend(_validate_series(spy, "SPY"))
    all_issues.extend(_validate_series(mer, "Meridian"))

    if all_issues:
        print("  WARNINGS:")
        for issue in all_issues:
            print(f"    - {issue}")
    else:
        print("  All checks passed.")

    # Stats
    spy_stats = _compute_stats(spy["loss"], "SPY")
    mer_stats = _compute_stats(mer["loss"], "Meridian")

    print(f"\nSummary Statistics:")
    print(f"  {'':20s} {'SPY':>12s} {'Meridian':>12s}")
    print(f"  {'─'*44}")
    for key in ["count", "mean", "std", "min", "max", "skewness", "kurtosis", "q95", "q99"]:
        sv = spy_stats[key]
        mv = mer_stats[key]
        if key == "count":
            print(f"  {key:20s} {sv:>12,d} {mv:>12,d}")
        else:
            print(f"  {key:20s} {sv:>12.6f} {mv:>12.6f}")

    # Save metadata
    metadata = {
        "spy": {
            "date_range": [str(spy["date"].min().date()), str(spy["date"].max().date())],
            "stats": spy_stats,
        },
        "meridian": {
            "date_range": [str(mer["date"].min().date()), str(mer["date"].max().date())],
            "stats": mer_stats,
        },
        "issues": all_issues,
    }

    meta_path = os.path.join(config.DATA_DIR, "loss_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")

    return spy, mer, metadata


if __name__ == "__main__":
    spy, mer, meta = load_all()
