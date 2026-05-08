"""
Module 5: Rule-Based Comparison
Compare selected MS model against Meridian's rule-based regime detector
on the 2009-2026 overlap period.

Analyses:
  1. Agreement rate (Cohen's kappa)
  2. Confusion matrix
  3. Transition timing (early warning analysis)
  4. Soft vs hard probability comparison
  5. Regime-conditional return distributions

Inputs:
  - regime_switching/data/best_model_probs.parquet
  - regime_switching/data/model_comparison.json
  - data/backtest/regime_history.parquet (rule-based, 2009-2026)
  - regime_switching/data/returns.parquet

Outputs:
  - regime_switching/data/comparison_results.json
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from regime_switching.config import (
    DATA_DIR, OUTPUT_DIR, REGIME_HISTORY_PATH, COMPARISON_START,
    CRISIS_EVENTS, INTEGRATION_CRITERIA,
)


def load_data():
    """Load MS probs, rule-based regimes, and returns."""
    # MS model probs
    ms_probs = pd.read_parquet(DATA_DIR / "best_model_probs.parquet")

    # Model comparison to get selected model info
    with open(DATA_DIR / "model_comparison.json") as f:
        comparison = json.load(f)
    selected = comparison["selected_model"]

    # Rule-based regime history
    rb = pd.read_parquet(REGIME_HISTORY_PATH)
    rb["date"] = pd.to_datetime(rb["date"])
    rb = rb.set_index("date")

    # Returns
    returns_df = pd.read_parquet(DATA_DIR / "returns.parquet")

    return ms_probs, rb, returns_df, selected


def align_overlap(ms_probs, rb):
    """Align MS and rule-based on overlap period (2009+, monthly rule-based dates)."""
    start = pd.Timestamp(COMPARISON_START)

    # Rule-based is monthly, MS is daily
    # Expand rule-based to daily by forward-filling
    rb_trimmed = rb[rb.index >= start][["composite"]].copy()
    rb_trimmed = rb_trimmed.rename(columns={"composite": "rb_regime"})

    ms_trimmed = ms_probs[ms_probs.index >= start].copy()

    # Merge on date — forward-fill rule-based to daily
    combined = ms_trimmed.join(rb_trimmed, how="left")
    combined["rb_regime"] = combined["rb_regime"].ffill()
    combined = combined.dropna(subset=["rb_regime"])

    return combined


def map_ms_to_rb_states(combined, K):
    """
    Map MS state labels to rule-based labels for comparison.
    MS: bull, bear, crisis (3-state) or bull, bear (2-state)
    RB: bull, recovery, bear, crisis
    """
    # For kappa computation, map both to common set
    # MS doesn't have 'recovery' — map RB recovery to bull for comparison
    rb_mapped = combined["rb_regime"].map({
        "bull": "bull",
        "recovery": "bull",  # recovery ~ transitional bull
        "bear": "bear",
        "crisis": "crisis",
    })

    ms_regime = combined["regime"].copy()

    if K == 2:
        # Map RB crisis -> bear for 2-state comparison
        rb_mapped = rb_mapped.map({
            "bull": "bull",
            "bear": "bear",
            "crisis": "bear",
        })

    return ms_regime, rb_mapped


def compute_agreement(ms_regime, rb_mapped):
    """Compute Cohen's kappa and agreement rate."""
    # Drop any remaining NaN
    valid = ~(ms_regime.isna() | rb_mapped.isna())
    ms_clean = ms_regime[valid]
    rb_clean = rb_mapped[valid]

    agreement_rate = float((ms_clean == rb_clean).mean())
    kappa = float(cohen_kappa_score(rb_clean, ms_clean))

    # Confusion matrix
    labels = sorted(set(ms_clean) | set(rb_clean))
    cm = confusion_matrix(rb_clean, ms_clean, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": cm.tolist(),
        "rows_are": "rule_based",
        "cols_are": "ms_model",
    }

    return {
        "agreement_rate": agreement_rate,
        "cohen_kappa": kappa,
        "kappa_interpretation": (
            "substantial" if kappa > 0.6 else
            "moderate" if kappa > 0.4 else
            "fair" if kappa > 0.2 else
            "slight"
        ),
        "n_compared": int(valid.sum()),
        "confusion_matrix": cm_dict,
    }


def compute_early_warning(combined, K):
    """
    For each crisis event, find when MS first signals danger vs rule-based.
    MS signal: P(bear) + P(crisis) > 0.5 (or P(bear) > 0.5 for 2-state)
    RB signal: regime != bull/recovery
    """
    events = {
        "GFC": ("2008-06-01", "2008-12-31"),
        "COVID": ("2020-01-01", "2020-04-30"),
        "Rate Hikes 2022": ("2021-11-01", "2022-06-30"),
    }

    results = {}

    for event_name, (window_start, window_end) in events.items():
        window = combined[
            (combined.index >= pd.Timestamp(window_start)) &
            (combined.index <= pd.Timestamp(window_end))
        ].copy()

        if len(window) == 0:
            results[event_name] = {"status": "no data in window"}
            continue

        # MS: when does P(non-bull) first exceed 0.5?
        if K == 3:
            if "smoothed_bear" in window.columns and "smoothed_crisis" in window.columns:
                ms_danger = window["smoothed_bear"] + window["smoothed_crisis"]
            else:
                ms_danger = 1 - window["smoothed_bull"]
        else:
            ms_danger = window["smoothed_bear"] if "smoothed_bear" in window.columns else (1 - window["smoothed_bull"])

        ms_trigger_mask = ms_danger > 0.5
        ms_first = window.index[ms_trigger_mask][0] if ms_trigger_mask.any() else None

        # RB: when does regime first leave bull/recovery?
        rb_danger = window["rb_regime"].isin(["bear", "crisis"])
        rb_first = window.index[rb_danger][0] if rb_danger.any() else None

        if ms_first is not None and rb_first is not None:
            lead_days = (rb_first - ms_first).days
            results[event_name] = {
                "ms_first_signal": str(ms_first.date()),
                "rb_first_signal": str(rb_first.date()),
                "ms_lead_days": lead_days,
                "ms_earlier": lead_days > 0,
            }
        else:
            results[event_name] = {
                "ms_first_signal": str(ms_first.date()) if ms_first else "none",
                "rb_first_signal": str(rb_first.date()) if rb_first else "none",
                "ms_lead_days": None,
                "ms_earlier": None,
            }

    # Average lead across events with data
    leads = [r["ms_lead_days"] for r in results.values() if r.get("ms_lead_days") is not None]
    avg_lead = float(np.mean(leads)) if leads else None

    return {
        "events": results,
        "avg_lead_days": avg_lead,
        "n_events_measured": len(leads),
    }


def compute_regime_conditional_returns(combined, returns_df):
    """Compare return distributions under MS vs RB regimes."""
    returns = returns_df.loc[combined.index, "log_return"]

    results = {}

    # MS regime stats
    ms_stats = {}
    for regime in combined["regime"].unique():
        mask = combined["regime"] == regime
        r = returns[mask]
        if len(r) > 0:
            ms_stats[regime] = {
                "n_days": int(len(r)),
                "mean_ann": float(r.mean() * 252),
                "vol_ann": float(r.std() * np.sqrt(252)),
                "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            }
    results["ms_regimes"] = ms_stats

    # RB regime stats
    rb_stats = {}
    for regime in combined["rb_regime"].unique():
        if pd.isna(regime):
            continue
        mask = combined["rb_regime"] == regime
        r = returns[mask]
        if len(r) > 0:
            rb_stats[regime] = {
                "n_days": int(len(r)),
                "mean_ann": float(r.mean() * 252),
                "vol_ann": float(r.std() * np.sqrt(252)),
                "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            }
    results["rb_regimes"] = rb_stats

    return results


def evaluate_integration_criteria(agreement, early_warning, K):
    """Evaluate the 3 integration criteria."""
    criteria = {}

    # C1: Kappa > 0.40
    c1_pass = agreement["cohen_kappa"] >= INTEGRATION_CRITERIA["C1_kappa_threshold"]
    criteria["C1_kappa"] = {
        "value": agreement["cohen_kappa"],
        "threshold": INTEGRATION_CRITERIA["C1_kappa_threshold"],
        "pass": bool(c1_pass),
    }

    # C2: Early warning >= 3 days
    avg_lead = early_warning["avg_lead_days"]
    c2_pass = avg_lead is not None and avg_lead >= INTEGRATION_CRITERIA["C2_early_warning_days"]
    criteria["C2_early_warning"] = {
        "value": avg_lead,
        "threshold": INTEGRATION_CRITERIA["C2_early_warning_days"],
        "pass": bool(c2_pass) if avg_lead is not None else False,
    }

    # C3: Walk-forward improvement — evaluated in M6, placeholder here
    criteria["C3_walkforward"] = {
        "value": None,
        "threshold": "CAGR > 0 AND Sharpe > 0 vs hard regime",
        "pass": None,
        "note": "Evaluated in Module 6 (integration backtest)",
    }

    criteria["all_pass"] = bool(c1_pass and c2_pass)
    criteria["note"] = "C3 pending M6. Current assessment based on C1+C2 only."

    return criteria


def run():
    """Main entry point."""
    print("=" * 60)
    print("B2 Module 5: Rule-Based Comparison")
    print("=" * 60)

    ms_probs, rb, returns_df, selected = load_data()

    # Get K from selected model name
    K = int(selected.split("-")[1])

    print(f"  Selected model: {selected}")
    print(f"  MS probs: {len(ms_probs)} rows")
    print(f"  Rule-based: {len(rb)} rows, {rb.index.min().date()} to {rb.index.max().date()}")
    print()

    # Align overlap
    combined = align_overlap(ms_probs, rb)
    print(f"  Overlap period: {combined.index.min().date()} to {combined.index.max().date()}")
    print(f"  Overlap days: {len(combined)}")
    print()

    # Map states
    ms_regime, rb_mapped = map_ms_to_rb_states(combined, K)

    # Agreement
    print("[1/4] Agreement analysis...")
    agreement = compute_agreement(ms_regime, rb_mapped)
    print(f"  Agreement rate: {agreement['agreement_rate']:.1%}")
    print(f"  Cohen's kappa: {agreement['cohen_kappa']:.3f} ({agreement['kappa_interpretation']})")
    print(f"  Confusion matrix labels: {agreement['confusion_matrix']['labels']}")
    cm = np.array(agreement["confusion_matrix"]["matrix"])
    labels = agreement["confusion_matrix"]["labels"]
    print(f"  {'':>12} " + " ".join(f"{l:>8}" for l in labels) + "  (MS model)")
    for i, l in enumerate(labels):
        print(f"  {l:>12} " + " ".join(f"{cm[i,j]:>8}" for j in range(len(labels))))
    print(f"  {'(rule-based)':>12}")
    print()

    # Early warning
    print("[2/4] Early warning analysis...")
    early_warning = compute_early_warning(combined, K)
    for event, info in early_warning["events"].items():
        if info.get("ms_lead_days") is not None:
            direction = "earlier" if info["ms_earlier"] else "later"
            print(f"  {event}: MS {abs(info['ms_lead_days'])} days {direction} "
                  f"(MS: {info['ms_first_signal']}, RB: {info['rb_first_signal']})")
        else:
            print(f"  {event}: {info.get('status', 'incomplete data')}")
    if early_warning["avg_lead_days"] is not None:
        print(f"  Average lead: {early_warning['avg_lead_days']:.1f} days")
    print()

    # Regime-conditional returns
    print("[3/4] Regime-conditional returns...")
    regime_returns = compute_regime_conditional_returns(combined, returns_df)

    print("  MS regime returns (overlap period):")
    for regime, stats in regime_returns["ms_regimes"].items():
        print(f"    {regime}: {stats['n_days']} days, "
              f"mean={stats['mean_ann']:.4f}, vol={stats['vol_ann']:.4f}, "
              f"sharpe={stats['sharpe']:.3f}")

    print("  RB regime returns (overlap period):")
    for regime, stats in regime_returns["rb_regimes"].items():
        print(f"    {regime}: {stats['n_days']} days, "
              f"mean={stats['mean_ann']:.4f}, vol={stats['vol_ann']:.4f}, "
              f"sharpe={stats['sharpe']:.3f}")
    print()

    # Integration criteria
    print("[4/4] Integration criteria evaluation...")
    criteria = evaluate_integration_criteria(agreement, early_warning, K)
    print(f"  C1 (kappa >= {criteria['C1_kappa']['threshold']}): "
          f"{'PASS' if criteria['C1_kappa']['pass'] else 'FAIL'} "
          f"(kappa={criteria['C1_kappa']['value']:.3f})")
    print(f"  C2 (lead >= {criteria['C2_early_warning']['threshold']} days): "
          f"{'PASS' if criteria['C2_early_warning']['pass'] else 'FAIL'} "
          f"(avg lead={criteria['C2_early_warning']['value']})")
    print(f"  C3 (walk-forward): Pending M6")
    print(f"  C1+C2 assessment: {'PASS' if criteria['all_pass'] else 'FAIL'}")
    print()

    # Save
    print("Saving...")
    results = {
        "selected_model": selected,
        "overlap_period": {
            "start": str(combined.index.min().date()),
            "end": str(combined.index.max().date()),
            "n_days": len(combined),
        },
        "agreement": agreement,
        "early_warning": early_warning,
        "regime_conditional_returns": regime_returns,
        "integration_criteria": criteria,
    }

    with open(DATA_DIR / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  comparison_results.json: saved")

    print()
    print("=" * 60)
    print("Module 5 COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run()
