# signals/combiner.py
# Signal combination module.
# Runs all 10 signals, cross-sectionally Z-scores and winsorizes each one,
# then combines using IC-IR weighting with regime-conditional multipliers.
# IC history stored in data/processed/ic_history.parquet.
# Output: single composite score per ticker, saved to data/processed/signals.parquet.

import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from data.storage import save_signals, load_parquet, save_parquet
from utils.config_loader import get_config
from utils.notifications import notify

# Import all 10 signals
from signals.momentum_12_1      import compute as momentum_12_1
from signals.earnings_momentum  import compute as earnings_momentum
from signals.pe_zscore          import compute as pe_zscore
from signals.pb_zscore          import compute as pb_zscore
from signals.ev_ebitda_zscore   import compute as ev_ebitda_zscore
from signals.roe_stability      import compute as roe_stability
from signals.gross_margin_trend import compute as gross_margin_trend
from signals.piotroski          import compute as piotroski
from signals.earnings_accruals  import compute as earnings_accruals
from signals.short_term_reversal import compute as short_term_reversal
from signals.rsi_extremes       import compute as rsi_extremes

cfg = get_config()
SC  = cfg["signals"]

IC_HISTORY_FILE = Path("data/processed/ic_history.parquet")
WINSOR_CLIP     = 3.0   # clip Z-scores beyond 3 std
IC_FLOOR        = SC.get("ic_floor", 0.0)
IC_LOOKBACK     = SC.get("ic_lookback", 63)

ALL_SIGNALS = {
    "momentum_12_1":      momentum_12_1,
    "earnings_momentum":  earnings_momentum,
    "pe_zscore":          pe_zscore,
    "pb_zscore":          pb_zscore,
    "ev_ebitda_zscore":   ev_ebitda_zscore,
    "roe_stability":      roe_stability,
    "gross_margin_trend": gross_margin_trend,
    "piotroski":          piotroski,
    "earnings_accruals":  earnings_accruals,
    "short_term_reversal": short_term_reversal,
    "rsi_extremes":       rsi_extremes,
}


# ------------------------------------------------------------
# CROSS-SECTIONAL NORMALIZATION
# ------------------------------------------------------------

def zscore_winsorize(series: pd.Series) -> pd.Series:
    """
    Cross-sectionally Z-scores a signal then winsorizes at ±3 std.
    Returns normalized series with mean~0 and std~1.
    """
    mean = series.mean()
    std  = series.std()

    if std == 0:
        return pd.Series(0.0, index=series.index)

    zscored = (series - mean) / std
    return zscored.clip(-WINSOR_CLIP, WINSOR_CLIP)


# ------------------------------------------------------------
# IC HISTORY
# ------------------------------------------------------------

def load_ic_history() -> pd.DataFrame:
    if not IC_HISTORY_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(IC_HISTORY_FILE)


def update_ic_history(signal_scores: pd.DataFrame, forward_returns: pd.Series, run_date: date) -> None:
    """
    Computes IC (rank correlation) between each signal score and
    realized forward returns, appends to ic_history.parquet.
    Called after returns are available (next rebalance day).
    forward_returns: Series indexed by ticker with 1-period return.
    """
    if signal_scores.empty or forward_returns.empty:
        return

    history = load_ic_history()
    new_rows = []

    for signal_name in ALL_SIGNALS.keys():
        if signal_name not in signal_scores.columns:
            continue

        scores  = signal_scores[["ticker", signal_name]].dropna()
        aligned = scores.merge(
            forward_returns.rename("fwd_return").reset_index(),
            on="ticker", how="inner"
        )

        if len(aligned) < 10:
            continue

        ic = float(aligned[signal_name].corr(aligned["fwd_return"], method="spearman"))
        new_rows.append({
            "date":        run_date,
            "signal_name": signal_name,
            "ic":          ic
        })

    if not new_rows:
        return

    new_df = pd.DataFrame(new_rows)
    if not history.empty:
        history = pd.concat([history, new_df], ignore_index=True)
    else:
        history = new_df

    history = history.sort_values(["signal_name", "date"]).reset_index(drop=True)
    history.to_parquet(IC_HISTORY_FILE, index=False)
    print(f"[combiner] IC history updated: {len(new_rows)} signals for {run_date}")


# ------------------------------------------------------------
# IC-IR WEIGHTS
# ------------------------------------------------------------

def compute_ic_weights(regime: str) -> dict:
    """
    Computes IC-IR signal weights.
    weight = max(0, IC_mean / IC_std) — IC-IR with zero floor.
    Applies regime multipliers from settings.yaml.
    Falls back to equal weighting if insufficient IC history.
    Returns dict of signal_name -> weight (normalized to sum to 1).
    """
    history = load_ic_history()
    regime_multipliers = SC.get("regime_multipliers", {}).get(regime, {})

    # Equal weight fallback
    equal_weight = 1.0 / len(ALL_SIGNALS)
    weights = {s: equal_weight for s in ALL_SIGNALS}

    if not history.empty and len(history) >= IC_LOOKBACK:
        ic_weights = {}

        for signal_name in ALL_SIGNALS.keys():
            signal_ic = history[history["signal_name"] == signal_name]["ic"]

            if len(signal_ic) < 5:
                ic_weights[signal_name] = equal_weight
                continue

            ic_mean = float(signal_ic.tail(IC_LOOKBACK).mean())
            ic_std  = float(signal_ic.tail(IC_LOOKBACK).std())

            if ic_std == 0:
                ic_weights[signal_name] = equal_weight
                continue

            # IC-IR with zero floor
            ic_ir = max(0.0, ic_mean / ic_std)
            ic_weights[signal_name] = ic_ir

        total = sum(ic_weights.values())
        if total > 0:
            weights = {k: v / total for k, v in ic_weights.items()}
        else:
            weights = {s: equal_weight for s in ALL_SIGNALS}

        print(f"[combiner] Using IC-IR weights for regime={regime}")
    else:
        print(f"[combiner] Insufficient IC history — using equal weights")

    # Apply regime multipliers
    for signal_name, multiplier in regime_multipliers.items():
        if signal_name in weights:
            weights[signal_name] *= multiplier

    # Renormalize after regime adjustment
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


# ------------------------------------------------------------
# MAIN COMBINATION RUN
# ------------------------------------------------------------

def run_combiner(run_date: date = None, regime: str = "recovery") -> pd.DataFrame:
    """
    Full signal combination run.
    1. Runs all 10 signals
    2. Cross-sectionally Z-scores and winsorizes each
    3. Combines using IC-IR weights with regime multipliers
    4. Returns composite score DataFrame saved to signals.parquet

    regime: composite regime string from regime/detector.py
    """
    if run_date is None:
        run_date = date.today()

    print(f"[combiner] Running signal combination for {run_date} | regime={regime}")

    # ----------------------------------------------------------
    # STEP 1 — Run all signals
    # ----------------------------------------------------------
    raw_scores = {}
    for signal_name, compute_fn in ALL_SIGNALS.items():
        try:
            df = compute_fn(run_date)
            if not df.empty and "raw_score" in df.columns:
                raw_scores[signal_name] = df.set_index("ticker")["raw_score"]
            else:
                print(f"[combiner] {signal_name} returned empty — skipping")
        except Exception as e:
            print(f"[combiner] {signal_name} failed: {e}")

    if not raw_scores:
        print("[combiner] No signals computed — aborting")
        notify("Signal combiner failed — no signals computed", level="critical")
        return pd.DataFrame()

    # ----------------------------------------------------------
    # STEP 2 — Normalize each signal
    # ----------------------------------------------------------
    normalized = {}
    for signal_name, scores in raw_scores.items():
        normalized[signal_name] = zscore_winsorize(scores)

    scores_df = pd.DataFrame(normalized)
    scores_df.index.name = "ticker"

    # ----------------------------------------------------------
    # STEP 3 — Compute weights and combine
    # ----------------------------------------------------------
    weights = compute_ic_weights(regime)

    composite      = pd.Series(0.0, index=scores_df.index)
    weight_totals  = pd.Series(0.0, index=scores_df.index)

    for signal_name, weight in weights.items():
        if signal_name in scores_df.columns:
            signal_vals = scores_df[signal_name]
            available   = signal_vals.notna()
            composite  += weight * signal_vals.fillna(0.0)
            weight_totals[available] += weight

    # Normalize by sum of weights for available signals only
    weight_totals = weight_totals.replace(0, np.nan)
    composite     = composite / weight_totals

    # Minimum signal coverage floor — exclude tickers with fewer than 6 signals
    signal_count = scores_df.notna().sum(axis=1)
    composite    = composite.where(signal_count >= 6, other=np.nan)
    composite    = composite.fillna(0.0)

    # Final cross-sectional rank (percentile 0-1)
    composite_rank = composite.rank(pct=True)

    result = pd.DataFrame({
        "ticker":          composite.index,
        "date":            run_date,
        "composite_score": composite.values,
        "composite_rank":  composite_rank.values,
    }).reset_index(drop=True)

    # Add individual normalized scores for diagnostics
    scores_reset = scores_df.reset_index()
    result = result.merge(scores_reset, on="ticker", how="left")

    save_signals(result)

    print(f"[combiner] Done. {len(result)} tickers scored | "
          f"top5: {result.nlargest(5, 'composite_score')['ticker'].tolist()}")

    return result