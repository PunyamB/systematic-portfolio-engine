# signals/decay_tracker.py
# Signal decay monitoring module.
# Tracks how quickly each signal's rank correlation decays after computation.
# Fits exponential decay to rank correlation series per factor.
# Triggers drift-rebalance if rank correlation drops below 0.65 threshold.
# Runs daily between rebalances to monitor signal freshness.

import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from scipy.optimize import curve_fit
from data.storage import load_signals, load_parquet, save_parquet
from utils.config_loader import get_config
from utils.notifications import notify

cfg = get_config()

DECAY_FILE      = Path("data/processed/signal_decay.parquet")
DECAY_THRESHOLD = cfg["risk"]["signal_decay_threshold"]  # 0.65

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


# ------------------------------------------------------------
# EXPONENTIAL DECAY FITTING
# ------------------------------------------------------------

def _exp_decay(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential decay function: a * exp(-b * t)"""
    return a * np.exp(-b * t)


def compute_half_life(correlation_series: pd.Series) -> float:
    """
    Fits exponential decay to a rank correlation time series.
    Returns half-life in days — how long until correlation drops to 0.5.
    Returns None if fit fails.
    """
    if len(correlation_series) < 5:
        return None

    y = correlation_series.values
    x = np.arange(len(y), dtype=float)

    try:
        popt, _ = curve_fit(
            _exp_decay, x, y,
            p0=[1.0, 0.1],
            bounds=([0, 0], [2, 10]),
            maxfev=1000
        )
        a, b = popt
        if b <= 0:
            return None
        half_life = np.log(2) / b
        return float(half_life)
    except Exception:
        return None


# ------------------------------------------------------------
# RANK CORRELATION TRACKING
# ------------------------------------------------------------

def compute_rank_correlation(
    baseline_signals: pd.DataFrame,
    current_signals: pd.DataFrame,
    signal_name: str
) -> float:
    """
    Computes Spearman rank correlation between baseline signal scores
    and current signal scores for a given signal.
    baseline_signals: signals DataFrame from last rebalance day
    current_signals:  signals DataFrame from today
    Returns float 0-1, or None if insufficient overlap.
    """
    if signal_name not in baseline_signals.columns:
        return None
    if signal_name not in current_signals.columns:
        return None

    baseline = baseline_signals[["ticker", signal_name]].dropna()
    current  = current_signals[["ticker", signal_name]].dropna()

    merged = baseline.merge(current, on="ticker", how="inner", suffixes=("_base", "_curr"))

    if len(merged) < 10:
        return None

    corr = merged[f"{signal_name}_base"].corr(
        merged[f"{signal_name}_curr"],
        method="spearman"
    )
    return float(corr)


# ------------------------------------------------------------
# DECAY HISTORY
# ------------------------------------------------------------

def load_decay_history() -> pd.DataFrame:
    if not DECAY_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(DECAY_FILE)


def save_decay_entry(run_date: date, correlations: dict, half_lives: dict) -> None:
    """
    Appends daily decay readings to decay history parquet.
    correlations: dict of signal_name -> rank_correlation
    half_lives:   dict of signal_name -> half_life_days
    """
    history = load_decay_history()
    new_rows = []

    for signal_name in SIGNAL_NAMES:
        new_rows.append({
            "date":             run_date,
            "signal_name":      signal_name,
            "rank_correlation": correlations.get(signal_name),
            "half_life_days":   half_lives.get(signal_name),
        })

    new_df = pd.DataFrame(new_rows)
    if not history.empty:
        # Remove existing entries for same date
        history = history[history["date"] != pd.Timestamp(run_date)]
        history = pd.concat([history, new_df], ignore_index=True)
    else:
        history = new_df

    history = history.sort_values(["signal_name", "date"]).reset_index(drop=True)
    history.to_parquet(DECAY_FILE, index=False)


# ------------------------------------------------------------
# DRIFT-REBALANCE TRIGGER CHECK
# ------------------------------------------------------------

def check_decay_trigger(correlations: dict) -> list:
    """
    Checks if any signal's rank correlation has dropped below threshold.
    Returns list of signal names that triggered — empty if none.
    These signals have decayed enough to warrant an interim rebalance.
    """
    triggered = []
    for signal_name, corr in correlations.items():
        if corr is not None and corr < DECAY_THRESHOLD:
            triggered.append(signal_name)
            print(f"[decay_tracker] Decay trigger: {signal_name} correlation={corr:.4f} < {DECAY_THRESHOLD}")

    return triggered


# ------------------------------------------------------------
# FULL DECAY TRACKING RUN
# ------------------------------------------------------------

def run_decay_tracker(
    run_date: date = None,
    baseline_date: date = None
) -> dict:
    """
    Full decay tracking run.
    Loads signals from today and from baseline (last rebalance day).
    Computes rank correlations and half-lives per signal.
    Triggers drift-rebalance notification if any signal decays below threshold.

    run_date:      today's date
    baseline_date: last rebalance date (signals saved on that day)
    Returns dict with correlations, half_lives, triggered signals.
    """
    if run_date is None:
        run_date = date.today()

    print(f"[decay_tracker] Running decay check for {run_date}")

    # Load today's signals
    current_signals = load_signals()
    if current_signals.empty:
        print("[decay_tracker] No current signals — skipping")
        return {"correlations": {}, "half_lives": {}, "triggered": []}

    # Load baseline signals (from last rebalance)
    if baseline_date is not None:
        from data.storage import load_snapshot
        baseline_signals = load_snapshot("signals", baseline_date)
    else:
        # No baseline — use current as baseline (first run)
        baseline_signals = current_signals.copy()

    if baseline_signals.empty:
        print("[decay_tracker] No baseline signals — using current as baseline")
        baseline_signals = current_signals.copy()

    # Compute rank correlations
    correlations = {}
    for signal_name in SIGNAL_NAMES:
        corr = compute_rank_correlation(baseline_signals, current_signals, signal_name)
        correlations[signal_name] = corr

    # Compute half-lives from decay history
    history    = load_decay_history()
    half_lives = {}

    for signal_name in SIGNAL_NAMES:
        if not history.empty:
            signal_history = history[
                history["signal_name"] == signal_name
            ]["rank_correlation"].dropna()
            half_lives[signal_name] = compute_half_life(signal_history)
        else:
            half_lives[signal_name] = None

    # Save decay entry
    save_decay_entry(run_date, correlations, half_lives)

    # Check for drift-rebalance triggers
    triggered = check_decay_trigger(correlations)

    if triggered:
        notify(
            f"Signal decay trigger for {run_date}\n"
            f"Signals below threshold: {triggered}\n"
            f"Consider interim rebalance",
            level="warning"
        )
    else:
        print(f"[decay_tracker] No decay triggers — all signals above {DECAY_THRESHOLD}")

    valid_corrs = {k: v for k, v in correlations.items() if v is not None}
    if valid_corrs:
        print(f"[decay_tracker] Mean correlation: {np.mean(list(valid_corrs.values())):.4f}")

    return {
        "correlations": correlations,
        "half_lives":   half_lives,
        "triggered":    triggered
    }