"""
Module 5: EVT Risk Measures
============================
Computes VaR and Expected Shortfall at multiple confidence levels
using the fitted GPD from M4, and compares against empirical tail.

Formulas:
  EVT-VaR at level p:
    VaR_p = u + (sigma/xi) * [(n/N_u * (1-p))^(-xi) - 1]

  EVT-ES at level p:
    ES_p = VaR_p / (1 - xi) + (sigma - xi*u) / (1 - xi)
    Valid for xi < 1.

  CIs via delta method propagation from GPD parameter CIs.

Usage:
    from evt_tail_risk.m5_risk_measures import run_risk_measures

    results = run_risk_measures()
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from evt_tail_risk import config


def compute_evt_var(p, threshold, xi, sigma, n_total, n_exc):
    """
    Compute EVT-VaR at confidence level p.

    Args:
        p: confidence level (e.g., 0.99)
        threshold: selected threshold u
        xi: GPD shape parameter
        sigma: GPD scale parameter
        n_total: total observations
        n_exc: number of exceedances

    Returns:
        float: VaR estimate
    """
    zeta = n_exc / n_total  # exceedance rate

    if xi != 0:
        var = threshold + (sigma / xi) * ((n_total / n_exc * (1 - p)) ** (-xi) - 1)
    else:
        var = threshold + sigma * np.log(n_total / n_exc * (1 - p))

    return float(var)


def compute_evt_es(p, threshold, xi, sigma, n_total, n_exc):
    """
    Compute EVT-ES (Expected Shortfall) at confidence level p.
    Valid for xi < 1.

    Args:
        Same as compute_evt_var

    Returns:
        float: ES estimate
    """
    if xi >= 1:
        return float("inf")

    var = compute_evt_var(p, threshold, xi, sigma, n_total, n_exc)
    es = var / (1 - xi) + (sigma - xi * threshold) / (1 - xi)

    return float(es)


def compute_var_ci(p, threshold, xi, sigma, xi_se, sigma_se, n_total, n_exc):
    """
    Approximate 95% CI for VaR via delta method.
    Perturbs xi and sigma by +/- 1.96*SE and computes VaR at each.

    Returns:
        (var_lo, var_hi)
    """
    var_center = compute_evt_var(p, threshold, xi, sigma, n_total, n_exc)

    # Perturb parameters
    vars_perturbed = []
    for xi_delta in [-1, 0, 1]:
        for sigma_delta in [-1, 0, 1]:
            xi_p = xi + xi_delta * 1.96 * xi_se
            sigma_p = sigma + sigma_delta * 1.96 * sigma_se
            if sigma_p > 0:  # scale must be positive
                try:
                    v = compute_evt_var(p, threshold, xi_p, sigma_p, n_total, n_exc)
                    vars_perturbed.append(v)
                except Exception:
                    pass

    if vars_perturbed:
        return (float(min(vars_perturbed)), float(max(vars_perturbed)))
    return (float(var_center), float(var_center))


def compute_es_ci(p, threshold, xi, sigma, xi_se, sigma_se, n_total, n_exc):
    """
    Approximate 95% CI for ES via delta method.

    Returns:
        (es_lo, es_hi)
    """
    es_center = compute_evt_es(p, threshold, xi, sigma, n_total, n_exc)

    es_perturbed = []
    for xi_delta in [-1, 0, 1]:
        for sigma_delta in [-1, 0, 1]:
            xi_p = xi + xi_delta * 1.96 * xi_se
            sigma_p = sigma + sigma_delta * 1.96 * sigma_se
            if sigma_p > 0 and xi_p < 1:
                try:
                    e = compute_evt_es(p, threshold, xi_p, sigma_p, n_total, n_exc)
                    es_perturbed.append(e)
                except Exception:
                    pass

    if es_perturbed:
        return (float(min(es_perturbed)), float(max(es_perturbed)))
    return (float(es_center), float(es_center))


def compute_empirical_var_es(losses, p):
    """
    Compute empirical VaR and ES at confidence level p.

    VaR = quantile of the loss distribution
    ES = mean of losses exceeding VaR

    Returns:
        (empirical_var, empirical_es)
    """
    var = float(np.percentile(losses, p * 100))
    exceedances = losses[losses >= var]
    es = float(exceedances.mean()) if len(exceedances) > 0 else var

    return var, es


def compute_risk_measures(losses, gpd_fit, series_name="SPY"):
    """
    Compute VaR and ES at all confidence levels for a single series.

    Args:
        losses: array of loss values
        gpd_fit: dict from M4 with xi, sigma, threshold, etc.
        series_name: for labeling

    Returns:
        dict with risk measures at each confidence level
    """
    threshold = gpd_fit["threshold"]
    xi = gpd_fit["xi"]
    sigma = gpd_fit["sigma"]
    xi_se = gpd_fit["xi_se"]
    sigma_se = gpd_fit["sigma_se"]
    n_total = gpd_fit["n_total"]
    n_exc = gpd_fit["n_exceedances"]

    measures = {}

    for p in config.CONFIDENCE_LEVELS:
        level_key = f"{p:.1%}".replace(".", "").replace("%", "pct")

        # EVT estimates
        evt_var = compute_evt_var(p, threshold, xi, sigma, n_total, n_exc)
        evt_es = compute_evt_es(p, threshold, xi, sigma, n_total, n_exc)

        # CIs
        var_ci = compute_var_ci(p, threshold, xi, sigma, xi_se, sigma_se, n_total, n_exc)
        es_ci = compute_es_ci(p, threshold, xi, sigma, xi_se, sigma_se, n_total, n_exc)

        # Empirical estimates
        emp_var, emp_es = compute_empirical_var_es(losses, p)

        # Ratio: EVT / Empirical (>1 means EVT is more conservative)
        var_ratio = evt_var / emp_var if emp_var != 0 else float("inf")
        es_ratio = evt_es / emp_es if emp_es != 0 else float("inf")

        measures[level_key] = {
            "confidence_level": float(p),
            "evt_var": evt_var,
            "evt_var_ci": var_ci,
            "evt_es": evt_es,
            "evt_es_ci": es_ci,
            "empirical_var": emp_var,
            "empirical_es": emp_es,
            "var_ratio_evt_vs_empirical": float(var_ratio),
            "es_ratio_evt_vs_empirical": float(es_ratio),
        }

    return measures


def run_risk_measures():
    """
    Compute risk measures for both SPY and Meridian.
    Saves results to risk_measures.json.

    Returns:
        dict with all risk measures
    """
    # Load GPD fit results from M4
    gpd_path = os.path.join(config.DATA_DIR, "gpd_fit.json")
    with open(gpd_path) as f:
        gpd_fits = json.load(f)

    results = {}

    for series_name, key in [("SPY", "spy"), ("Meridian", "meridian")]:
        print(f"\n{'='*60}")
        print(f"RISK MEASURES: {series_name}")
        print(f"{'='*60}")

        # Load losses
        loss_path = os.path.join(config.DATA_DIR, f"{key}_losses.parquet")
        df = pd.read_parquet(loss_path)
        losses = df["loss"].values

        gpd_fit = gpd_fits[key]
        measures = compute_risk_measures(losses, gpd_fit, series_name)

        # Print results
        print(f"\n  {'Level':>8s} | {'EVT VaR':>10s} {'[CI]':>22s} | {'Emp VaR':>10s} | {'Ratio':>6s} | {'EVT ES':>10s} {'[CI]':>22s} | {'Emp ES':>10s} | {'Ratio':>6s}")
        print(f"  {'-'*120}")

        for level_key, m in measures.items():
            p = m["confidence_level"]
            print(
                f"  {p:>7.1%} | {m['evt_var']:>10.6f} [{m['evt_var_ci'][0]:.6f}, {m['evt_var_ci'][1]:.6f}] | "
                f"{m['empirical_var']:>10.6f} | {m['var_ratio_evt_vs_empirical']:>6.2f} | "
                f"{m['evt_es']:>10.6f} [{m['evt_es_ci'][0]:.6f}, {m['evt_es_ci'][1]:.6f}] | "
                f"{m['empirical_es']:>10.6f} | {m['es_ratio_evt_vs_empirical']:>6.2f}"
            )

        results[key] = measures

    # Validation checks
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")

    all_valid = True
    for key, series_name in [("spy", "SPY"), ("meridian", "Meridian")]:
        measures = results[key]
        # At 99.5%, EVT VaR should exceed empirical (accounts for tail heaviness)
        m995 = measures["995pct"]
        if m995["evt_var"] > m995["empirical_var"]:
            print(f"  {series_name}: EVT VaR > Empirical VaR at 99.5% — PASS (ratio {m995['var_ratio_evt_vs_empirical']:.2f})")
        else:
            print(f"  {series_name}: EVT VaR <= Empirical VaR at 99.5% — UNEXPECTED")
            all_valid = False

        # CIs should be non-degenerate
        for level_key, m in measures.items():
            if m["evt_var_ci"][0] == m["evt_var_ci"][1]:
                print(f"  {series_name}: Degenerate CI at {m['confidence_level']:.1%} — WARNING")
                all_valid = False

    if all_valid:
        print("  All validation checks PASSED.")

    # Save
    out_path = os.path.join(config.DATA_DIR, "risk_measures.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    run_risk_measures()
