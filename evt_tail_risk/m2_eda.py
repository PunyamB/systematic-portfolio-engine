"""
Module 2: Exploratory Data Analysis
====================================
Characterizes the distributional properties of the loss series.
Establishes that returns are non-normal with heavy tails, justifying EVT.

Runs 5 statistical tests:
  - Jarque-Bera (joint normality on skewness + kurtosis)
  - Anderson-Darling (tail-weighted normality)
  - Shapiro-Wilk (general normality, subsampled if N > 5000)
  - Ljung-Box on |returns| (volatility clustering)
  - Ljung-Box on returns^2 (ARCH effects)

Generates 6 diagnostic plots via m8_visualizer.

Usage:
    from evt_tail_risk.m2_eda import run_eda

    results = run_eda()  # runs all tests, saves results + plots
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from evt_tail_risk import config


def _jarque_bera(losses, name):
    """Jarque-Bera test for joint normality (skewness + kurtosis)."""
    stat, pval = sp_stats.jarque_bera(losses)
    skew = sp_stats.skew(losses)
    kurt = sp_stats.kurtosis(losses)  # excess
    return {
        "test": "Jarque-Bera",
        "series": name,
        "statistic": float(stat),
        "p_value": float(pval),
        "reject_at_001": bool(pval < 0.01),
        "skewness": float(skew),
        "excess_kurtosis": float(kurt),
    }


def _anderson_darling(losses, name):
    """Anderson-Darling test (tail-weighted normality)."""
    result = sp_stats.anderson(losses, dist="norm")
    # Use 1% significance level (index 4 in critical values)
    crit_1pct = result.critical_values[4]
    reject = result.statistic > crit_1pct
    return {
        "test": "Anderson-Darling",
        "series": name,
        "statistic": float(result.statistic),
        "critical_value_1pct": float(crit_1pct),
        "reject_at_001": bool(reject),
    }


def _shapiro_wilk(losses, name, max_n=5000):
    """
    Shapiro-Wilk test for normality.
    Subsamples to max_n if series is longer (SW has N limit).
    """
    if len(losses) > max_n:
        rng = np.random.default_rng(42)
        sample = rng.choice(losses, size=max_n, replace=False)
        subsampled = True
    else:
        sample = losses
        subsampled = False

    stat, pval = sp_stats.shapiro(sample)
    return {
        "test": "Shapiro-Wilk",
        "series": name,
        "statistic": float(stat),
        "p_value": float(pval),
        "reject_at_001": bool(pval < 0.01),
        "subsampled": subsampled,
        "sample_size": int(len(sample)),
    }


def _ljung_box_abs(losses, name, lags=20):
    """Ljung-Box test on |returns| for serial correlation (volatility clustering)."""
    abs_losses = np.abs(losses)
    lb = acorr_ljungbox(abs_losses, lags=lags, return_df=True)
    # Report at lag 10 and lag 20
    results = {}
    for lag in [10, 20]:
        if lag <= lags:
            row = lb.loc[lag]
            results[f"lag_{lag}"] = {
                "statistic": float(row["lb_stat"]),
                "p_value": float(row["lb_pvalue"]),
                "significant": bool(row["lb_pvalue"] < 0.01),
            }
    return {
        "test": "Ljung-Box |returns|",
        "series": name,
        "lags_tested": lags,
        "results": results,
    }


def _ljung_box_sq(losses, name, lags=20):
    """Ljung-Box test on returns^2 for ARCH effects."""
    sq_losses = losses ** 2
    lb = acorr_ljungbox(sq_losses, lags=lags, return_df=True)
    results = {}
    for lag in [10, 20]:
        if lag <= lags:
            row = lb.loc[lag]
            results[f"lag_{lag}"] = {
                "statistic": float(row["lb_stat"]),
                "p_value": float(row["lb_pvalue"]),
                "significant": bool(row["lb_pvalue"] < 0.01),
            }
    return {
        "test": "Ljung-Box returns^2",
        "series": name,
        "lags_tested": lags,
        "results": results,
    }


def run_tests(losses, name):
    """Run all 5 statistical tests on a loss series."""
    results = []
    results.append(_jarque_bera(losses, name))
    results.append(_anderson_darling(losses, name))
    results.append(_shapiro_wilk(losses, name))
    results.append(_ljung_box_abs(losses, name))
    results.append(_ljung_box_sq(losses, name))
    return results


def print_results(all_results):
    """Print test results in a readable format."""
    for r in all_results:
        test = r["test"]
        series = r["series"]

        if test in ("Jarque-Bera", "Shapiro-Wilk"):
            reject = r["reject_at_001"]
            print(f"  {series:10s} | {test:25s} | stat={r['statistic']:12.2f} | p={r['p_value']:.2e} | {'REJECT' if reject else 'FAIL TO REJECT'}")
            if test == "Jarque-Bera":
                print(f"  {'':10s} |   skewness={r['skewness']:.4f}, excess_kurtosis={r['excess_kurtosis']:.4f}")

        elif test == "Anderson-Darling":
            reject = r["reject_at_001"]
            print(f"  {series:10s} | {test:25s} | stat={r['statistic']:12.2f} | crit(1%)={r['critical_value_1pct']:.4f} | {'REJECT' if reject else 'FAIL TO REJECT'}")

        elif "Ljung-Box" in test:
            for lag_key, lag_res in r["results"].items():
                sig = lag_res["significant"]
                print(f"  {series:10s} | {test:25s} | {lag_key}: stat={lag_res['statistic']:10.2f} | p={lag_res['p_value']:.2e} | {'SIGNIFICANT' if sig else 'not sig'}")


def run_eda():
    """
    Run full EDA: load data, run all tests, generate all plots, save results.

    Returns:
        dict with all test results
    """
    from evt_tail_risk.m1_data_loader import load_spy_losses, load_meridian_losses
    from evt_tail_risk.m8_visualizer import (
        plot_loss_histogram, plot_qq_normal, plot_rolling_vol,
        plot_drawdown, plot_acf_volatility, plot_tail_probability
    )

    # Load data
    print("Loading data...")
    spy = load_spy_losses(save=False)
    mer = pd.read_parquet(os.path.join(config.DATA_DIR, "meridian_losses.parquet"))

    spy_losses = spy["loss"].values
    mer_losses = mer["loss"].values

    # Run tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    print("\nNormality Tests (expect: REJECT for all)")
    print("-" * 70)
    spy_results = run_tests(spy_losses, "SPY")
    mer_results = run_tests(mer_losses, "Meridian")

    print_results(spy_results)
    print()
    print_results(mer_results)

    # Check that all normality tests reject
    normality_tests = ["Jarque-Bera", "Anderson-Darling", "Shapiro-Wilk"]
    all_reject = True
    for r in spy_results + mer_results:
        if r["test"] in normality_tests:
            if not r.get("reject_at_001", False):
                all_reject = False
                print(f"\n  WARNING: {r['series']} {r['test']} did NOT reject at 1%")

    if all_reject:
        print("\n  All normality tests REJECT at 1%. Non-normality confirmed.")

    # Check Ljung-Box significance
    lb_tests = ["Ljung-Box |returns|", "Ljung-Box returns^2"]
    lb_all_sig = True
    for r in spy_results + mer_results:
        if r["test"] in lb_tests:
            for lag_res in r["results"].values():
                if not lag_res["significant"]:
                    lb_all_sig = False

    if lb_all_sig:
        print("  All Ljung-Box tests significant. Volatility clustering confirmed.")
        print("  GARCH filtering justified for FHS benchmark (Module 6).")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    for name, losses, dates in [
        ("SPY", spy_losses, spy["date"].values),
        ("Meridian", mer_losses, mer["date"].values),
    ]:
        print(f"\n{name}:")
        plot_loss_histogram(losses, name=name)
        plot_qq_normal(losses, name=name)
        plot_rolling_vol(losses, dates, name=name)
        plot_drawdown(losses, dates, name=name)
        plot_acf_volatility(losses, name=name)
        plot_tail_probability(losses, name=name)

    # Save results
    output = {
        "spy": spy_results,
        "meridian": mer_results,
        "normality_rejected": all_reject,
        "volatility_clustering_confirmed": lb_all_sig,
    }

    out_path = os.path.join(config.DATA_DIR, "eda_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_eda()
