"""
Module 4: GPD Estimator
========================
Fits the Generalized Pareto Distribution to threshold exceedances via
Maximum Likelihood Estimation. Produces parameter estimates with
confidence intervals and full diagnostic validation.

Key steps:
  1. Synthetic recovery test — validate estimator on known GPD data
     before touching real data (100 reps, verify 95% CI coverage)
  2. Fit GPD to real exceedances from M3 threshold
  3. Compute CIs via Fisher information (delta method)
  4. Generate 4-panel diagnostic plot (QQ, PP, return level, density)
  5. Run KS goodness-of-fit test

Usage:
    from evt_tail_risk.m4_gpd import run_gpd_estimation

    results = run_gpd_estimation()  # fits both series, saves results + plots
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import approx_fprime
from evt_tail_risk import config


def synthetic_recovery_test():
    """
    Validate the GPD estimator on synthetic data before real fitting.

    Generate GPD_SYNTHETIC_REPS samples of size GPD_SYNTHETIC_N from
    GPD(xi=0.25, sigma=1.0), fit MLE, check that 95% CI contains
    the true parameters ~95% of the time.

    Returns:
        dict with coverage rates and pass/fail status
    """
    true_xi = config.GPD_SYNTHETIC_XI
    true_sigma = config.GPD_SYNTHETIC_SIGMA
    n = config.GPD_SYNTHETIC_N
    reps = config.GPD_SYNTHETIC_REPS

    xi_covered = 0
    sigma_covered = 0
    fit_failures = 0

    rng = np.random.default_rng(42)

    for i in range(reps):
        # Generate synthetic GPD data
        sample = sp_stats.genpareto.rvs(c=true_xi, loc=0, scale=true_sigma, size=n, random_state=rng)

        try:
            xi_hat, _, sigma_hat = sp_stats.genpareto.fit(sample, floc=0)

            # Approximate standard errors
            n_obs = len(sample)
            se_xi = (1 + xi_hat) / np.sqrt(n_obs) if (1 + xi_hat) > 0 else np.inf
            se_sigma = sigma_hat * np.sqrt(2 * (1 + xi_hat)) / np.sqrt(n_obs) if (1 + xi_hat) > 0 else np.inf

            # 95% CI
            xi_lo = xi_hat - 1.96 * se_xi
            xi_hi = xi_hat + 1.96 * se_xi
            sigma_lo = sigma_hat - 1.96 * se_sigma
            sigma_hi = sigma_hat + 1.96 * se_sigma

            if xi_lo <= true_xi <= xi_hi:
                xi_covered += 1
            if sigma_lo <= true_sigma <= sigma_hi:
                sigma_covered += 1

        except Exception:
            fit_failures += 1

    valid_reps = reps - fit_failures
    xi_coverage = xi_covered / valid_reps if valid_reps > 0 else 0
    sigma_coverage = sigma_covered / valid_reps if valid_reps > 0 else 0

    # Pass if coverage is within reasonable range of 95% (say 85-100%)
    xi_pass = 0.85 <= xi_coverage <= 1.0
    sigma_pass = 0.85 <= sigma_coverage <= 1.0

    return {
        "true_xi": true_xi,
        "true_sigma": true_sigma,
        "n_samples": n,
        "n_reps": reps,
        "fit_failures": fit_failures,
        "xi_coverage": float(xi_coverage),
        "sigma_coverage": float(sigma_coverage),
        "xi_pass": xi_pass,
        "sigma_pass": sigma_pass,
        "overall_pass": xi_pass and sigma_pass,
    }


def fit_gpd(losses, threshold, series_name="SPY"):
    """
    Fit GPD to exceedances above threshold via MLE.

    Args:
        losses: array of loss values
        threshold: selected threshold u from M3
        series_name: for labeling

    Returns:
        dict with xi, sigma, CIs, log-likelihood, diagnostics
    """
    exceedances = losses[losses > threshold] - threshold
    n_exc = len(exceedances)
    n_total = len(losses)

    if n_exc < config.MIN_EXCEEDANCES:
        raise ValueError(
            f"Only {n_exc} exceedances (need >= {config.MIN_EXCEEDANCES}). "
            f"Lower the threshold."
        )

    # MLE fit
    xi_hat, _, sigma_hat = sp_stats.genpareto.fit(exceedances, floc=0)

    # Log-likelihood at MLE
    log_lik = np.sum(sp_stats.genpareto.logpdf(exceedances, c=xi_hat, loc=0, scale=sigma_hat))

    # Standard errors via approximate Fisher information
    se_xi = (1 + xi_hat) / np.sqrt(n_exc) if (1 + xi_hat) > 0 else np.nan
    se_sigma = sigma_hat * np.sqrt(2 * (1 + xi_hat)) / np.sqrt(n_exc) if (1 + xi_hat) > 0 else np.nan

    # 95% CIs
    xi_ci = (float(xi_hat - 1.96 * se_xi), float(xi_hat + 1.96 * se_xi))
    sigma_ci = (float(sigma_hat - 1.96 * se_sigma), float(sigma_hat + 1.96 * se_sigma))

    # KS goodness-of-fit
    ks_stat, ks_pval = sp_stats.kstest(exceedances, "genpareto", args=(xi_hat, 0, sigma_hat))

    # Tail type interpretation
    if xi_hat > 0.5:
        tail_type = "Very heavy tail (infinite variance)"
    elif xi_hat > 0:
        tail_type = "Heavy tail (Frechet domain, finite variance)"
    elif xi_hat == 0:
        tail_type = "Exponential tail (Gumbel domain)"
    else:
        tail_type = "Bounded tail (Weibull domain)"

    return {
        "series": series_name,
        "threshold": float(threshold),
        "n_exceedances": int(n_exc),
        "n_total": int(n_total),
        "exceedance_rate": float(n_exc / n_total),
        "xi": float(xi_hat),
        "xi_se": float(se_xi),
        "xi_ci_95": xi_ci,
        "sigma": float(sigma_hat),
        "sigma_se": float(se_sigma),
        "sigma_ci_95": sigma_ci,
        "log_likelihood": float(log_lik),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "ks_passed": bool(ks_pval >= 0.05),
        "tail_type": tail_type,
        "exceedances": exceedances,  # kept for plotting, not saved to JSON
    }


def run_gpd_estimation():
    """
    Run full GPD estimation pipeline:
      1. Synthetic recovery test
      2. Fit GPD on SPY and Meridian exceedances
      3. Generate 4-panel diagnostic plots
      4. Save results

    Returns:
        dict with all results
    """
    from evt_tail_risk.m8_visualizer import plot_gpd_4panel

    # 1. Synthetic recovery test
    print("=" * 60)
    print("SYNTHETIC RECOVERY TEST")
    print("=" * 60)

    synth = synthetic_recovery_test()
    print(f"  True params: xi={synth['true_xi']}, sigma={synth['true_sigma']}")
    print(f"  Samples: {synth['n_samples']}, Reps: {synth['n_reps']}")
    print(f"  Fit failures: {synth['fit_failures']}")
    print(f"  Xi coverage:    {synth['xi_coverage']:.1%} {'PASS' if synth['xi_pass'] else 'FAIL'}")
    print(f"  Sigma coverage: {synth['sigma_coverage']:.1%} {'PASS' if synth['sigma_pass'] else 'FAIL'}")
    print(f"  Overall: {'PASS' if synth['overall_pass'] else 'FAIL'}")

    if not synth["overall_pass"]:
        print("  WARNING: Synthetic test failed. Estimator may have issues.")
        print("  Proceeding with caution...")

    # 2. Load threshold results from M3
    thresh_path = os.path.join(config.DATA_DIR, "threshold_results.json")
    with open(thresh_path) as f:
        thresh = json.load(f)

    results = {"synthetic_test": synth}

    for series_name, key in [("SPY", "spy"), ("Meridian", "meridian")]:
        print(f"\n{'='*60}")
        print(f"GPD ESTIMATION: {series_name}")
        print(f"{'='*60}")

        # Load losses
        loss_path = os.path.join(config.DATA_DIR, f"{key}_losses.parquet")
        df = pd.read_parquet(loss_path)
        losses = df["loss"].values

        # Get threshold from M3
        u = thresh[key]["threshold"]
        print(f"  Threshold (from M3): {u:.6f}")

        # Fit GPD
        fit = fit_gpd(losses, u, series_name)

        print(f"\n  RESULTS:")
        print(f"    Exceedances: {fit['n_exceedances']} / {fit['n_total']} ({fit['exceedance_rate']:.1%})")
        print(f"    Shape (xi):  {fit['xi']:.4f} +/- {fit['xi_se']:.4f}  CI: [{fit['xi_ci_95'][0]:.4f}, {fit['xi_ci_95'][1]:.4f}]")
        print(f"    Scale (sigma): {fit['sigma']:.6f} +/- {fit['sigma_se']:.6f}  CI: [{fit['sigma_ci_95'][0]:.6f}, {fit['sigma_ci_95'][1]:.6f}]")
        print(f"    Log-likelihood: {fit['log_likelihood']:.2f}")
        print(f"    Tail type: {fit['tail_type']}")
        print(f"    KS test: stat={fit['ks_stat']:.4f}, p={fit['ks_pvalue']:.4f} {'PASS' if fit['ks_passed'] else 'FAIL'}")

        # Generate 4-panel diagnostic
        print(f"\n  Generating 4-panel diagnostic plot...")
        plot_gpd_4panel(
            exceedances=fit["exceedances"],
            xi=fit["xi"],
            sigma=fit["sigma"],
            threshold=u,
            n_total=fit["n_total"],
            name=series_name,
        )

        # Store results (without exceedances array)
        results[key] = {k: v for k, v in fit.items() if k != "exceedances"}

    # Save
    out_path = os.path.join(config.DATA_DIR, "gpd_fit.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Synthetic test: {'PASS' if synth['overall_pass'] else 'FAIL'}")
    for key in ["spy", "meridian"]:
        r = results[key]
        print(f"  {r['series']:10s} | xi={r['xi']:.4f} [{r['xi_ci_95'][0]:.4f}, {r['xi_ci_95'][1]:.4f}] | "
              f"sigma={r['sigma']:.6f} | KS p={r['ks_pvalue']:.4f} {'PASS' if r['ks_passed'] else 'FAIL'}")

    return results


if __name__ == "__main__":
    run_gpd_estimation()
