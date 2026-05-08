"""
Module 4: Model Selection
Compare all 4 model variants (2-state fixed, 3-state fixed, 2-state TVTP, 3-state TVTP)
on BIC/AIC, regime quality diagnostics, and covariate significance.

Inputs:
  - regime_switching/data/ms{2,3}_fixed_results.json
  - regime_switching/data/ms{2,3}_tvtp_results.json
  - regime_switching/data/ms{2,3}_{fixed,tvtp}_probs.parquet

Outputs:
  - regime_switching/data/model_comparison.json
  - regime_switching/data/best_model_probs.parquet
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

from regime_switching.config import DATA_DIR, OUTPUT_DIR


def load_results():
    """Load all 4 model results."""
    models = {}
    for prefix, label in [
        ("ms2_fixed", "MS-2-Fixed"),
        ("ms3_fixed", "MS-3-Fixed"),
        ("ms2_tvtp", "MS-2-TVTP"),
        ("ms3_tvtp", "MS-3-TVTP"),
    ]:
        path = DATA_DIR / f"{prefix}_results.json"
        with open(path) as f:
            models[label] = json.load(f)
        models[label]["_prefix"] = prefix
    return models


def compute_regime_quality(models):
    """Compute regime quality diagnostics for each model."""
    returns_df = pd.read_parquet(DATA_DIR / "returns.parquet")
    returns = returns_df["log_return"].values

    for name, info in models.items():
        prefix = info["_prefix"]
        probs_df = pd.read_parquet(DATA_DIR / f"{prefix}_probs.parquet")

        K = info["K"]
        labels = info["state_labels"]

        # Regime-conditional return stats
        regime_stats = {}
        for label in labels:
            mask = probs_df["regime"] == label
            n_days = int(mask.sum())
            if n_days > 0:
                r_regime = returns[mask.values]
                regime_stats[label] = {
                    "n_days": n_days,
                    "pct_of_sample": float(n_days / len(returns) * 100),
                    "mean_return_ann": float(r_regime.mean() * 252),
                    "vol_ann": float(r_regime.std() * np.sqrt(252)),
                    "min": float(r_regime.min()),
                    "max": float(r_regime.max()),
                }
            else:
                regime_stats[label] = {"n_days": 0, "pct_of_sample": 0.0}

        info["regime_quality"] = regime_stats

        # Regime separation: difference between bull and worst state means
        mu_ann = info["mu_annualized"]
        info["regime_separation"] = float(mu_ann[0] - mu_ann[-1])

        # Count regime transitions
        regimes = probs_df["regime"].values
        transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        info["n_transitions"] = transitions
        info["avg_transitions_per_year"] = float(transitions / (len(regimes) / 252))

    return models


def select_best(models):
    """Select best model by BIC (lower is better)."""
    best_name = min(models, key=lambda k: models[k]["bic"])
    return best_name


def run():
    """Main entry point."""
    print("=" * 60)
    print("B2 Module 4: Model Selection")
    print("=" * 60)

    models = load_results()
    models = compute_regime_quality(models)
    best_name = select_best(models)

    # Print comparison table
    print()
    print(f"{'Model':<16} {'K':>3} {'LL':>12} {'n_params':>10} {'BIC':>12} {'AIC':>12} {'Transitions':>12}")
    print("-" * 80)

    for name in ["MS-2-Fixed", "MS-3-Fixed", "MS-2-TVTP", "MS-3-TVTP"]:
        info = models[name]
        marker = " <-- BEST" if name == best_name else ""
        print(f"{name:<16} {info['K']:>3} {info['log_likelihood']:>12.2f} "
              f"{info['n_params']:>10} {info['bic']:>12.2f} {info['aic']:>12.2f} "
              f"{info.get('n_transitions', 'N/A'):>12}{marker}")

    print()
    print(f"Selected model: {best_name}")
    print()

    # Print regime quality for best model
    best = models[best_name]
    print(f"Regime quality ({best_name}):")
    for label in best["state_labels"]:
        rq = best["regime_quality"][label]
        if rq["n_days"] > 0:
            print(f"  {label}: {rq['n_days']} days ({rq['pct_of_sample']:.1f}%), "
                  f"mean={rq['mean_return_ann']:.4f}, vol={rq['vol_ann']:.4f}")

    print(f"  Regime separation (bull - worst): {best['regime_separation']:.4f}")
    print(f"  Avg transitions/year: {best['avg_transitions_per_year']:.1f}")
    print()

    # Print regime durations
    print(f"Regime durations ({best_name}):")
    for label in best["state_labels"]:
        dur = best["regime_durations"][label]
        print(f"  {label}: {dur:.1f} days avg")
    print()

    # TVTP vs Fixed comparison
    if "TVTP" in best_name:
        k_str = best_name.split("-")[1]
        fixed_name = f"MS-{k_str}-Fixed"
        fixed = models[fixed_name]
        ll_improvement = best["log_likelihood"] - fixed["log_likelihood"]
        bic_improvement = fixed["bic"] - best["bic"]  # positive = TVTP better
        print(f"TVTP vs Fixed-TP ({k_str}-state):")
        print(f"  LL improvement: {ll_improvement:.2f}")
        print(f"  BIC improvement: {bic_improvement:.2f}")
        print(f"  Extra parameters: {best['n_params'] - fixed['n_params']}")
        print(f"  Covariates justify their cost: {'YES' if bic_improvement > 0 else 'NO'}")
    print()

    # Copy best model probs as the canonical file
    best_prefix = best["_prefix"]
    best_probs = pd.read_parquet(DATA_DIR / f"{best_prefix}_probs.parquet")
    best_probs.to_parquet(DATA_DIR / "best_model_probs.parquet")
    print(f"Saved best_model_probs.parquet ({best_name}): {len(best_probs)} rows")

    # Save comparison JSON
    comparison = {
        "selected_model": best_name,
        "comparison_table": {},
    }
    for name in ["MS-2-Fixed", "MS-3-Fixed", "MS-2-TVTP", "MS-3-TVTP"]:
        info = models[name]
        comparison["comparison_table"][name] = {
            "K": info["K"],
            "log_likelihood": info["log_likelihood"],
            "n_params": info["n_params"],
            "bic": info["bic"],
            "aic": info["aic"],
            "converged": info["converged"],
            "regime_durations": info["regime_durations"],
            "regime_separation": info["regime_separation"],
            "n_transitions": info.get("n_transitions"),
            "avg_transitions_per_year": info.get("avg_transitions_per_year"),
            "regime_quality": info.get("regime_quality"),
        }

    if "TVTP" in best_name:
        k_str = best_name.split("-")[1]
        fixed_name = f"MS-{k_str}-Fixed"
        comparison["tvtp_vs_fixed"] = {
            "ll_improvement": float(best["log_likelihood"] - models[fixed_name]["log_likelihood"]),
            "bic_improvement": float(models[fixed_name]["bic"] - best["bic"]),
            "extra_params": best["n_params"] - models[fixed_name]["n_params"],
            "covariates_justified": bool(models[fixed_name]["bic"] > best["bic"]),
        }

    with open(DATA_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("Saved model_comparison.json")

    print()
    print("=" * 60)
    print("Module 4 COMPLETE")
    print("=" * 60)

    return best_name, models


if __name__ == "__main__":
    run()
