"""
EVT Tail Risk Project Setup
============================
Creates directory structure, fetches SPY prices from FMP (1993-present),
and writes all initial module files.

Run from SPE root:
    cd D:\Projects\SystematicPortfolioEngine
    SirAlgotsAlot\Scripts\Activate.ps1
    python evt_setup.py
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path

SPE_ROOT = Path(r"D:\Projects\SystematicPortfolioEngine")
EVT_ROOT = SPE_ROOT / "evt_tail_risk"
FMP_KEY = "chiezlHmDSi0a5A8OUPwMxhOBMuEIkSq"

# ── 1. Create directory structure ────────────────────────────────
print("=" * 60)
print("1. Creating directory structure...")
print("=" * 60)

dirs = [
    EVT_ROOT,
    EVT_ROOT / "data",
    EVT_ROOT / "outputs",
    EVT_ROOT / "notebooks",
    EVT_ROOT / "tests",
]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {d.relative_to(SPE_ROOT)}")

# ── 2. Write __init__.py ─────────────────────────────────────────
print("\n2. Writing __init__.py...")
init_path = EVT_ROOT / "__init__.py"
init_path.write_text('"""EVT Tail Risk Decomposition — Project B1"""\n', encoding="utf-8")
print(f"  Written: {init_path.relative_to(SPE_ROOT)}")

# ── 3. Write config.py ──────────────────────────────────────────
print("\n3. Writing config.py...")
config_code = '''"""
EVT Tail Risk — Configuration
==============================
Single source of truth for all EVT-specific parameters.
"""

# ── Rolling / Window ────────────────────────────────────────────
ROLLING_WINDOW = 500            # Trading days for rolling estimation (~2 years)
COV_LOOKBACK = 252              # Not used directly in EVT, kept for reference

# ── Confidence Levels ───────────────────────────────────────────
CONFIDENCE_LEVELS = [0.95, 0.99, 0.995]

# ── Threshold Selection (Module 3) ──────────────────────────────
THRESHOLD_QUANTILE = 0.95       # Default threshold as quantile (fallback)
MIN_EXCEEDANCES = 50            # Minimum exceedances for GPD fit
MAX_EXCEEDANCES = 250           # Upper bound for threshold selection

# ── GPD Estimation (Module 4) ───────────────────────────────────
GPD_SYNTHETIC_N = 10_000        # Sample size for synthetic recovery test
GPD_SYNTHETIC_REPS = 100        # Number of repetitions for coverage test
GPD_SYNTHETIC_XI = 0.25         # True xi for synthetic test
GPD_SYNTHETIC_SIGMA = 1.0       # True sigma for synthetic test

# ── GARCH (Module 6) ───────────────────────────────────────────
GARCH_P = 1                     # GARCH lag order
GARCH_Q = 1                     # ARCH lag order

# ── Rolling Backtest (Module 7) ─────────────────────────────────
ROLLING_REFIT_FREQ = 5          # Refit GPD/GARCH every N days
BACKTEST_START_BUFFER = 500     # Initial fitting window (days)

# ── Crisis Periods ──────────────────────────────────────────────
CRISIS_PERIODS = {
    "Dotcom":       ("2000-03-10", "2002-10-09"),
    "GFC":          ("2007-10-09", "2009-03-09"),
    "EU Debt":      ("2011-07-01", "2011-10-04"),
    "COVID":        ("2020-02-19", "2020-03-23"),
    "Rate Hikes":   ("2022-01-03", "2022-10-12"),
}

# ── Plot Style ──────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.figsize": (12, 6),
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "lines.linewidth": 1.2,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

COLORS = {
    "navy":    "#1B2A4A",
    "accent":  "#2E75B6",
    "red":     "#D32F2F",
    "green":   "#388E3C",
    "orange":  "#F57C00",
    "gray":    "#757575",
    "light":   "#E8F0FE",
}

# ── Data Paths (relative to EVT project root) ──────────────────
import os as _os
_EVT_DIR = _os.path.dirname(_os.path.abspath(__file__))
DATA_DIR = _os.path.join(_EVT_DIR, "data")
OUTPUT_DIR = _os.path.join(_EVT_DIR, "outputs")

# ── Source Data Paths (relative to SPE root) ────────────────────
_SPE_DIR = _os.path.dirname(_EVT_DIR)
BACKTEST_PRICES_PATH = _os.path.join(_SPE_DIR, "data", "backtest", "prices.parquet")
WF_RESULTS_DIR = _os.path.join(_SPE_DIR, "data", "backtest", "wf_results")
REGIME_HISTORY_PATH = _os.path.join(_SPE_DIR, "data", "backtest", "regime_history.parquet")
NAV_HISTORY_PATH = _os.path.join(_SPE_DIR, "data", "processed", "nav_history.parquet")
'''

config_path = EVT_ROOT / "config.py"
config_path.write_text(config_code, encoding="utf-8")
print(f"  Written: {config_path.relative_to(SPE_ROOT)}")

# ── 4. Fetch SPY from FMP ───────────────────────────────────────
print("\n4. Fetching SPY prices from FMP (1993-present)...")
spy_out = EVT_ROOT / "data" / "spy_prices.parquet"

if spy_out.exists():
    existing = pd.read_parquet(spy_out)
    print(f"  SPY already exists: {len(existing)} rows, {existing['date'].min()} to {existing['date'].max()}")
    print("  Skipping fetch. Delete evt_tail_risk/data/spy_prices.parquet to re-fetch.")
else:
    url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol=SPY&from=1993-01-01&apikey={FMP_KEY}"
    )
    print(f"  Requesting: {url[:80]}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list) and len(data) > 0:
        df = pd.DataFrame(data)
        # Keep only the columns we need
        keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "adjClose"] if c in df.columns]
        df = df[keep_cols].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(spy_out, index=False)
        print(f"  Saved: {spy_out.relative_to(SPE_ROOT)}")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        print(f"  ERROR: FMP returned unexpected data. Type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")
        print(f"  Response preview: {str(data)[:300]}")

# ── 5. Write m1_data_loader.py ───────────────────────────────────
print("\n5. Writing m1_data_loader.py...")
m1_code = '''"""
Module 1: Data Loader
=====================
Loads price data from SPE backtest parquets and constructs properly
aligned loss series for both SPY and the Meridian backtest portfolio.

Losses are negated log returns: positive values = losses.

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
    Stitch EXP006 walk-forward NAV windows into a single continuous
    daily return series, then compute log-losses.

    Reads nav_window_*_exp006_monthly.parquet from wf_results/.
    Windows are sorted by date and concatenated (test periods only,
    non-overlapping).

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

    # Load and concatenate, sorted by date
    frames = []
    for f in nav_files:
        df = pd.read_parquet(os.path.join(wf_dir, f))
        # Index is date for these files
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "nav"]].copy()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # Remove duplicate dates (window boundaries)
    combined = combined.drop_duplicates(subset=["date"], keep="first")

    # Log returns from NAV series, then negate
    combined["log_return"] = np.log(combined["nav"] / combined["nav"].shift(1))
    combined["loss"] = -combined["log_return"]
    combined = combined.dropna(subset=["loss"]).reset_index(drop=True)

    result = combined[["date", "loss"]].copy()

    if save:
        out_path = os.path.join(config.DATA_DIR, "meridian_losses.parquet")
        result.to_parquet(out_path, index=False)

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
    print("\\nValidating...")
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

    print("\\nSummary Statistics:")
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
    print(f"\\nMetadata saved to: {meta_path}")

    return spy, mer, metadata


if __name__ == "__main__":
    spy, mer, meta = load_all()
'''

m1_path = EVT_ROOT / "m1_data_loader.py"
m1_path.write_text(m1_code, encoding="utf-8")
print(f"  Written: {m1_path.relative_to(SPE_ROOT)}")

# ── 6. Write m8_visualizer.py (scaffold) ────────────────────────
print("\n6. Writing m8_visualizer.py (scaffold)...")
m8_code = '''"""
Module 8: Visualizer (Scaffold)
===============================
Centralized plotting module. All EVT plots are generated here.
Modules 2-7 call these functions rather than containing their own plotting code.

This is the scaffold version with the first few plot functions.
Remaining functions will be added as modules are built.

Usage:
    from evt_tail_risk.m8_visualizer import plot_loss_histogram, plot_qq_normal
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from evt_tail_risk import config


def _apply_style():
    """Apply consistent EVT plot styling."""
    plt.rcParams.update(config.PLOT_STYLE)


def _save_fig(fig, name, dpi=150):
    """Save figure to outputs/."""
    import os
    path = os.path.join(config.OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_loss_histogram(losses, name="SPY", bins=100, save=True):
    """
    Histogram of loss series with fitted normal overlay.
    Shows visual departure from Gaussian, especially in tails.
    """
    _apply_style()
    c = config.COLORS

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(losses, bins=bins, density=True, alpha=0.6,
            color=c["accent"], edgecolor="white", linewidth=0.3, label="Empirical")

    # Fitted normal
    mu, sigma = losses.mean(), losses.std()
    x = np.linspace(losses.min(), losses.max(), 300)
    ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), color=c["red"],
            linewidth=2, label=f"Normal(mu={mu:.4f}, sigma={sigma:.4f})")

    ax.set_title(f"{name} Daily Loss Distribution", fontsize=14, color=c["navy"])
    ax.set_xlabel("Loss (negative log return)")
    ax.set_ylabel("Density")
    ax.legend()

    if save:
        return _save_fig(fig, f"loss_histogram_{name.lower()}")
    return fig


def plot_qq_normal(losses, name="SPY", save=True):
    """
    QQ plot of losses against normal distribution.
    Curvature in tails confirms heavy-tail behavior.
    """
    _apply_style()
    c = config.COLORS

    fig, ax = plt.subplots(figsize=(8, 8))
    res = sp_stats.probplot(losses, dist="norm", plot=None)
    theoretical, ordered = res[0]

    ax.scatter(theoretical, ordered, s=8, alpha=0.5, color=c["accent"], label="Data")

    # 45-degree reference line
    lims = [min(theoretical.min(), ordered.min()), max(theoretical.max(), ordered.max())]
    ax.plot(lims, lims, color=c["red"], linewidth=1.5, linestyle="--", label="Normal reference")

    ax.set_title(f"{name} QQ Plot vs Normal", fontsize=14, color=c["navy"])
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Sample Quantiles")
    ax.legend()

    if save:
        return _save_fig(fig, f"qq_normal_{name.lower()}")
    return fig


def plot_rolling_vol(losses, dates, name="SPY", window=252, save=True):
    """
    Rolling 252-day volatility with crisis period bands.
    Shows time-varying risk levels.
    """
    _apply_style()
    c = config.COLORS

    series = pd.Series(losses, index=pd.to_datetime(dates))
    rolling_vol = series.rolling(window).std() * np.sqrt(252)  # Annualized

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_vol.index, rolling_vol.values, color=c["navy"], linewidth=1)

    # Shade crisis periods
    for crisis_name, (start, end) in config.CRISIS_PERIODS.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        if start_dt >= rolling_vol.index.min() and start_dt <= rolling_vol.index.max():
            ax.axvspan(start_dt, end_dt, alpha=0.15, color=c["red"], label=crisis_name)

    ax.set_title(f"{name} Rolling {window}-Day Annualized Volatility", fontsize=14, color=c["navy"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    if save:
        return _save_fig(fig, f"rolling_vol_{name.lower()}")
    return fig


def plot_drawdown(losses, dates, name="SPY", save=True):
    """
    Drawdown time series with crisis annotations.
    Shows cumulative peak-to-trough losses.
    """
    _apply_style()
    c = config.COLORS

    # Reconstruct cumulative returns from losses (loss = -log_return)
    log_returns = -pd.Series(losses, index=pd.to_datetime(dates))
    cum_returns = log_returns.cumsum()
    running_max = cum_returns.cummax()
    drawdown = cum_returns - running_max

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color=c["red"], alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color=c["red"], linewidth=0.5)

    # Crisis annotations
    for crisis_name, (start, end) in config.CRISIS_PERIODS.items():
        start_dt = pd.Timestamp(start)
        if start_dt >= drawdown.index.min() and start_dt <= drawdown.index.max():
            ax.axvline(start_dt, color=c["gray"], linewidth=0.8, linestyle="--", alpha=0.7)
            ax.text(start_dt, ax.get_ylim()[0] * 0.9, f" {crisis_name}",
                    fontsize=8, color=c["gray"], rotation=90, va="bottom")

    ax.set_title(f"{name} Drawdown (Log Returns)", fontsize=14, color=c["navy"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")

    if save:
        return _save_fig(fig, f"drawdown_{name.lower()}")
    return fig


if __name__ == "__main__":
    # Quick test: load data and generate scaffold plots
    from evt_tail_risk.m1_data_loader import load_spy_losses, load_meridian_losses

    print("Generating scaffold plots...")

    spy = load_spy_losses(save=False)
    plot_loss_histogram(spy["loss"].values, name="SPY")
    plot_qq_normal(spy["loss"].values, name="SPY")
    plot_rolling_vol(spy["loss"].values, spy["date"].values, name="SPY")
    plot_drawdown(spy["loss"].values, spy["date"].values, name="SPY")

    mer = load_meridian_losses(save=False)
    plot_loss_histogram(mer["loss"].values, name="Meridian")
    plot_qq_normal(mer["loss"].values, name="Meridian")
    plot_rolling_vol(mer["loss"].values, mer["date"].values, name="Meridian")
    plot_drawdown(mer["loss"].values, mer["date"].values, name="Meridian")

    print("\\nScaffold plots complete.")
'''

m8_path = EVT_ROOT / "m8_visualizer.py"
m8_path.write_text(m8_code, encoding="utf-8")
print(f"  Written: {m8_path.relative_to(SPE_ROOT)}")

# ── Done ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print(f"""
Next steps:
  1. Run M1 data loader:
     python -m evt_tail_risk.m1_data_loader

  2. Run M8 scaffold plots:
     python -m evt_tail_risk.m8_visualizer

  3. Check outputs in:
     evt_tail_risk/data/       (parquet files)
     evt_tail_risk/outputs/    (plots)
""")
