"""
Module 3: Threshold Selection
==============================
Selects the threshold u above which exceedances will be modeled by the GPD.
This is the most critical methodological decision in the project.

Too low: GPD assumptions violated (including non-extreme observations).
Too high: too few exceedances for reliable estimation.

Three diagnostic methods:
  1. Mean Residual Life (MRL) plot — visual, choose where linearity begins
  2. Parameter Stability plot — GPD params vs threshold, choose flat region
  3. Automated quantile search — fits GPD at many thresholds, picks best
     by KS test p-value within the [MIN_EXCEEDANCES, MAX_EXCEEDANCES] range

Target: 50-250 exceedances (config.MIN_EXCEEDANCES to config.MAX_EXCEEDANCES).

Usage:
    from evt_tail_risk.m3_threshold import run_threshold_selection

    results = run_threshold_selection()  # runs all methods, saves results + plots
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from evt_tail_risk import config


def compute_mrl(losses, n_points=200):
    """
    Compute Mean Residual Life: E[X - u | X > u] for a range of thresholds u.

    If GPD is valid above u, MRL should be approximately linear in u.

    Returns:
        thresholds: array of threshold values
        mrl_values: mean excess at each threshold
        mrl_ci: 95% confidence interval half-width at each threshold
    """
    lo = np.percentile(losses, 50)
    hi = np.percentile(losses, 99.5)
    thresholds = np.linspace(lo, hi, n_points)

    mrl_values = []
    mrl_ci = []

    for u in thresholds:
        exceedances = losses[losses > u] - u
        n_exc = len(exceedances)
        if n_exc < 5:
            mrl_values.append(np.nan)
            mrl_ci.append(np.nan)
        else:
            mean_excess = exceedances.mean()
            std_excess = exceedances.std()
            ci = 1.96 * std_excess / np.sqrt(n_exc)
            mrl_values.append(mean_excess)
            mrl_ci.append(ci)

    return thresholds, np.array(mrl_values), np.array(mrl_ci)


def compute_parameter_stability(losses, n_points=80):
    """
    Fit GPD at many thresholds and track how parameters change.
    Plot reparameterized scale (sigma_u - xi*u) and shape (xi) vs u.
    Parameters should stabilize (flat region) at the correct threshold.

    Returns:
        thresholds, xi_values, xi_ci, sigma_star_values, sigma_star_ci, n_exceedances
    """
    lo = np.percentile(losses, 85)
    hi = np.percentile(losses, 99.5)
    thresholds = np.linspace(lo, hi, n_points)

    xi_values = []
    xi_ci = []
    sigma_star_values = []
    sigma_star_ci = []
    n_exc_list = []

    for u in thresholds:
        exceedances = losses[losses > u] - u
        n_exc = len(exceedances)

        if n_exc < 20:
            xi_values.append(np.nan)
            xi_ci.append(np.nan)
            sigma_star_values.append(np.nan)
            sigma_star_ci.append(np.nan)
            n_exc_list.append(n_exc)
            continue

        try:
            xi, loc, sigma = sp_stats.genpareto.fit(exceedances, floc=0)
            sigma_star = sigma - xi * u

            # Approximate standard errors
            se_xi = (1 + xi) / np.sqrt(n_exc) if (1 + xi) > 0 else np.nan
            se_sigma = sigma * np.sqrt(2 * (1 + xi)) / np.sqrt(n_exc) if (1 + xi) > 0 else np.nan

            xi_values.append(xi)
            xi_ci.append(1.96 * se_xi)
            sigma_star_values.append(sigma_star)
            sigma_star_ci.append(1.96 * se_sigma)
            n_exc_list.append(n_exc)

        except Exception:
            xi_values.append(np.nan)
            xi_ci.append(np.nan)
            sigma_star_values.append(np.nan)
            sigma_star_ci.append(np.nan)
            n_exc_list.append(n_exc)

    return (
        thresholds,
        np.array(xi_values),
        np.array(xi_ci),
        np.array(sigma_star_values),
        np.array(sigma_star_ci),
        np.array(n_exc_list),
    )


def automated_threshold_search(losses):
    """
    Automated threshold selection: fits GPD at many quantile thresholds,
    picks the one with highest KS p-value (best fit) within the valid
    exceedance count range [MIN_EXCEEDANCES, MAX_EXCEEDANCES].

    Returns:
        dict with best threshold and the full search results table.
    """
    quantiles = np.linspace(0.90, 0.995, 50)
    results = []

    for q in quantiles:
        u = np.percentile(losses, q * 100)
        exceedances = losses[losses > u] - u
        n_exc = len(exceedances)

        if n_exc < 10:
            continue

        try:
            xi, loc, sigma = sp_stats.genpareto.fit(exceedances, floc=0)
            ks_stat, ks_pval = sp_stats.kstest(exceedances, "genpareto", args=(xi, 0, sigma))

            results.append({
                "quantile": float(q),
                "threshold": float(u),
                "n_exceedances": int(n_exc),
                "xi": float(xi),
                "sigma": float(sigma),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_pval),
            })
        except Exception:
            continue

    if not results:
        raise RuntimeError("No valid GPD fits found across threshold range")

    df = pd.DataFrame(results)

    # Filter to valid exceedance range
    valid = df[
        (df["n_exceedances"] >= config.MIN_EXCEEDANCES)
        & (df["n_exceedances"] <= config.MAX_EXCEEDANCES)
    ]

    if len(valid) == 0:
        print("  WARNING: No thresholds in [50, 250] exceedance range. Using best overall.")
        valid = df

    # Pick highest KS p-value (best fit)
    best_idx = valid["ks_pvalue"].idxmax()
    best = valid.loc[best_idx].to_dict()

    return {
        "best": best,
        "search_table": results,
    }


def select_threshold(series_name="SPY"):
    """
    Run all threshold selection methods on a single loss series.

    Args:
        series_name: "SPY" or "Meridian"

    Returns:
        dict with chosen threshold, diagnostics, and all intermediate results
    """
    loss_path = os.path.join(config.DATA_DIR, f"{series_name.lower()}_losses.parquet")
    df = pd.read_parquet(loss_path)
    losses = df["loss"].values

    print(f"\n{'='*60}")
    print(f"THRESHOLD SELECTION: {series_name}")
    print(f"{'='*60}")
    print(f"  Total observations: {len(losses)}")

    # 1. Mean Residual Life
    print("\n  1. Computing Mean Residual Life plot...")
    mrl_thresholds, mrl_values, mrl_ci = compute_mrl(losses)

    # 2. Parameter Stability
    print("  2. Computing Parameter Stability plot...")
    ps_thresholds, xi_vals, xi_ci, ss_vals, ss_ci, n_exc = compute_parameter_stability(losses)

    # 3. Automated search
    print("  3. Running automated threshold search...")
    auto_results = automated_threshold_search(losses)
    best = auto_results["best"]

    print(f"\n  SELECTED THRESHOLD:")
    print(f"    u = {best['threshold']:.6f} (quantile {best['quantile']:.3f})")
    print(f"    Exceedances: {best['n_exceedances']}")
    print(f"    GPD shape (xi): {best['xi']:.4f}")
    print(f"    GPD scale (sigma): {best['sigma']:.6f}")
    print(f"    KS test: stat={best['ks_stat']:.4f}, p={best['ks_pvalue']:.4f}")

    if best["ks_pvalue"] < 0.05:
        print(f"    WARNING: KS p-value < 0.05 — threshold may be suspect")
    else:
        print(f"    KS test PASSED (p > 0.05) — GPD fit is acceptable")

    # Generate plots
    from evt_tail_risk.m8_visualizer import plot_mrl, plot_parameter_stability

    print(f"\n  Generating diagnostic plots...")
    plot_mrl(mrl_thresholds, mrl_values, mrl_ci,
             chosen_u=best["threshold"], name=series_name)
    plot_parameter_stability(ps_thresholds, xi_vals, xi_ci, ss_vals, ss_ci,
                             chosen_u=best["threshold"], name=series_name)

    return {
        "series": series_name,
        "threshold": best["threshold"],
        "quantile": best["quantile"],
        "n_exceedances": best["n_exceedances"],
        "xi": best["xi"],
        "sigma": best["sigma"],
        "ks_stat": best["ks_stat"],
        "ks_pvalue": best["ks_pvalue"],
        "ks_passed": best["ks_pvalue"] >= 0.05,
        "search_table": auto_results["search_table"],
    }


def run_threshold_selection():
    """
    Run threshold selection on both SPY and Meridian.
    Saves results to threshold_results.json.

    Returns:
        dict with results for both series
    """
    spy_results = select_threshold("SPY")
    mer_results = select_threshold("Meridian")

    output = {
        "spy": {k: v for k, v in spy_results.items() if k != "search_table"},
        "meridian": {k: v for k, v in mer_results.items() if k != "search_table"},
    }

    out_path = os.path.join(config.DATA_DIR, "threshold_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in [("SPY", spy_results), ("Meridian", mer_results)]:
        print(f"  {name:10s} | u={r['threshold']:.6f} | q={r['quantile']:.3f} | "
              f"n={r['n_exceedances']} | xi={r['xi']:.4f} | "
              f"KS p={r['ks_pvalue']:.4f} {'PASS' if r['ks_passed'] else 'FAIL'}")

    return output


if __name__ == "__main__":
    run_threshold_selection()
