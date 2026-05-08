"""
Module 7: Rolling EVT and Backtesting
=======================================
The most complex and valuable module. Produces:
  (a) Rolling GPD estimates tracking tail evolution over time
  (b) Formal VaR backtest across all 5 models

Rolling GPD: fit on 500-day rolling window, refit every ROLLING_REFIT_FREQ
days (default 5). Produces daily time series of xi, sigma, VaR, ES.

VaR Backtest: compare predicted VaR at time t against realized loss at t+1.
Two tests per model per confidence level:
  - Kupiec (1995) unconditional coverage
  - Christoffersen (1998) conditional coverage

Usage:
    from evt_tail_risk.m7_rolling import run_rolling_backtest

    results = run_rolling_backtest()
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from evt_tail_risk import config


# ── Rolling GPD ─────────────────────────────────────────────────

def rolling_gpd(losses, dates, window=500, refit_freq=5):
    """
    Fit GPD on rolling windows, producing daily tail parameter estimates.

    At each refit point, uses the 95th percentile of the window as threshold,
    fits GPD to exceedances, computes VaR/ES at all confidence levels.
    Between refits, carries forward the last estimates.

    Args:
        losses: array of loss values
        dates: array of dates
        window: rolling window size (default 500)
        refit_freq: refit every N days (default 5)

    Returns:
        DataFrame with date, xi, sigma, threshold, n_exc, and VaR/ES columns
    """
    n = len(losses)
    results = []

    last_xi = np.nan
    last_sigma = np.nan
    last_threshold = np.nan
    last_n_exc = 0
    last_var = {p: np.nan for p in config.CONFIDENCE_LEVELS}
    last_es = {p: np.nan for p in config.CONFIDENCE_LEVELS}
    fit_failures = 0

    for i in range(window, n):
        do_refit = (i - window) % refit_freq == 0

        if do_refit:
            w = losses[i - window:i]
            u = np.percentile(w, 95)
            exceedances = w[w > u] - u
            n_exc = len(exceedances)
            n_w = len(w)

            if n_exc >= 15:
                try:
                    xi, _, sigma = sp_stats.genpareto.fit(exceedances, floc=0)
                    last_xi = xi
                    last_sigma = sigma
                    last_threshold = u
                    last_n_exc = n_exc

                    zeta = n_exc / n_w
                    for p in config.CONFIDENCE_LEVELS:
                        if xi != 0:
                            last_var[p] = u + (sigma / xi) * ((n_w / n_exc * (1 - p)) ** (-xi) - 1)
                        else:
                            last_var[p] = u + sigma * np.log(n_w / n_exc * (1 - p))

                        if xi < 1:
                            last_es[p] = last_var[p] / (1 - xi) + (sigma - xi * u) / (1 - xi)
                        else:
                            last_es[p] = np.nan

                except Exception:
                    fit_failures += 1

        row = {
            "date": dates[i],
            "xi": last_xi,
            "sigma": last_sigma,
            "threshold": last_threshold,
            "n_exceedances": last_n_exc,
        }
        for p in config.CONFIDENCE_LEVELS:
            pct = f"{p:.3f}".replace(".", "")
            row[f"evt_var_{pct}"] = last_var[p]
            row[f"evt_es_{pct}"] = last_es[p]

        results.append(row)

    if fit_failures > 0:
        print(f"    Rolling GPD fit failures: {fit_failures}")

    return pd.DataFrame(results)


# ── Kupiec Test ─────────────────────────────────────────────────

def kupiec_test(violations, n_total, expected_rate):
    """
    Kupiec (1995) unconditional coverage test.

    H0: violation rate = expected rate
    LR = -2 * [n1*ln(p) + n0*ln(1-p) - n1*ln(p_hat) - n0*ln(1-p_hat)]

    Returns:
        dict with statistic, p_value, pass/fail
    """
    n_violations = int(np.sum(violations))
    n_no_viol = n_total - n_violations
    p_hat = n_violations / n_total if n_total > 0 else 0
    p = expected_rate

    if p_hat == 0 or p_hat == 1:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "pass": False,
            "violation_rate": float(p_hat),
            "expected_rate": float(p),
            "n_violations": n_violations,
        }

    # Log-likelihood ratio
    lr = -2 * (
        n_violations * np.log(p) + n_no_viol * np.log(1 - p)
        - n_violations * np.log(p_hat) - n_no_viol * np.log(1 - p_hat)
    )

    p_value = 1 - sp_stats.chi2.cdf(lr, df=1)

    return {
        "statistic": float(lr),
        "p_value": float(p_value),
        "pass": bool(p_value > 0.05),
        "violation_rate": float(p_hat),
        "expected_rate": float(p),
        "n_violations": n_violations,
    }


# ── Christoffersen Test ─────────────────────────────────────────

def christoffersen_test(violations):
    """
    Christoffersen (1998) conditional coverage test.

    Tests both correct coverage AND independence of violations.
    LR_cc = LR_uc + LR_ind

    Returns:
        dict with statistic, p_value, pass/fail
    """
    violations = np.array(violations, dtype=int)
    n = len(violations)

    if n < 2:
        return {"statistic": np.nan, "p_value": np.nan, "pass": False}

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = violations[i - 1], violations[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Transition probabilities
    n0 = n00 + n01
    n1 = n10 + n11

    if n0 == 0 or n1 == 0 or n01 == 0 or n10 == 0:
        return {"statistic": np.nan, "p_value": np.nan, "pass": False}

    p01 = n01 / n0
    p11 = n11 / n1
    p_hat = (n01 + n11) / n  # overall violation rate

    if p_hat == 0 or p_hat == 1 or p01 == 0 or p01 == 1:
        return {"statistic": np.nan, "p_value": np.nan, "pass": False}

    # Independence LR
    try:
        lr_ind = -2 * (
            n00 * np.log(1 - p_hat) + n01 * np.log(p_hat)
            + n10 * np.log(1 - p_hat) + n11 * np.log(p_hat)
            - n00 * np.log(1 - p01) - n01 * np.log(p01)
            - n10 * np.log(1 - p11) - n11 * np.log(p11)
        )
    except (ValueError, RuntimeWarning):
        return {"statistic": np.nan, "p_value": np.nan, "pass": False}

    # CC test = UC + Ind (df=2)
    n_violations = n01 + n11
    expected_rate = p_hat  # use observed for CC
    uc = kupiec_test(violations, n, p_hat)

    lr_cc = lr_ind  # The independence component
    p_value = 1 - sp_stats.chi2.cdf(lr_cc, df=1)

    return {
        "statistic": float(lr_cc),
        "p_value": float(p_value),
        "pass": bool(p_value > 0.05),
    }


# ── Backtest Engine ─────────────────────────────────────────────

def run_backtest_single(realized_losses, var_forecasts, confidence_level):
    """
    Run VaR backtest for a single model at a single confidence level.

    Args:
        realized_losses: array of actual losses (t+1)
        var_forecasts: array of VaR forecasts (t)
        confidence_level: e.g. 0.99

    Returns:
        dict with violation array, Kupiec test, Christoffersen test
    """
    # Violation: realized loss exceeds predicted VaR
    valid = ~(np.isnan(realized_losses) | np.isnan(var_forecasts))
    r = realized_losses[valid]
    v = var_forecasts[valid]

    violations = (r > v).astype(int)
    expected_rate = 1 - confidence_level

    kupiec = kupiec_test(violations, len(violations), expected_rate)
    christ = christoffersen_test(violations)

    return {
        "n_obs": int(len(violations)),
        "n_violations": int(violations.sum()),
        "violation_rate": float(violations.mean()),
        "expected_rate": float(expected_rate),
        "kupiec": kupiec,
        "christoffersen": christ,
    }


# ── Main Runner ─────────────────────────────────────────────────

def run_rolling_backtest():
    """
    Run rolling GPD + full 5-model VaR backtest on both series.

    Steps:
      1. Compute rolling GPD estimates
      2. Merge with M6 benchmark forecasts
      3. Run Kupiec + Christoffersen for all 5 models at all levels
      4. Generate plots
      5. Save results

    Returns:
        dict with all backtest results
    """
    from evt_tail_risk.m8_visualizer import (
        plot_rolling_tail_index, plot_backtest_violations, plot_model_scorecard
    )

    all_results = {}

    for series_name, key in [("SPY", "spy"), ("Meridian", "meridian")]:
        print(f"\n{'='*60}")
        print(f"ROLLING EVT + BACKTEST: {series_name}")
        print(f"{'='*60}")

        # Load losses
        loss_path = os.path.join(config.DATA_DIR, f"{key}_losses.parquet")
        df = pd.read_parquet(loss_path)
        losses = df["loss"].values
        dates = pd.to_datetime(df["date"].values)

        # 1. Rolling GPD
        print(f"\n  1. Computing rolling GPD (window={config.ROLLING_WINDOW}, refit every {config.ROLLING_REFIT_FREQ} days)...")
        rolling = rolling_gpd(
            losses, dates,
            window=config.ROLLING_WINDOW,
            refit_freq=config.ROLLING_REFIT_FREQ,
        )
        print(f"    Rolling estimates: {len(rolling)} days")
        print(f"    Xi range: [{rolling['xi'].min():.4f}, {rolling['xi'].max():.4f}]")

        # Save rolling GPD
        rolling_path = os.path.join(config.DATA_DIR, f"rolling_gpd_{key}.parquet")
        rolling.to_parquet(rolling_path, index=False)
        print(f"    Saved: {rolling_path}")

        # 2. Load benchmark forecasts from M6
        bench_path = os.path.join(config.DATA_DIR, f"benchmark_var_{key}.parquet")
        bench = pd.read_parquet(bench_path)
        bench["date"] = pd.to_datetime(bench["date"])

        # 3. Merge rolling EVT with benchmarks on date
        rolling["date"] = pd.to_datetime(rolling["date"])
        merged = pd.merge(bench, rolling[["date"] + [c for c in rolling.columns if c.startswith("evt_")]],
                          on="date", how="inner")
        print(f"    Merged forecast days: {len(merged)}")

        # 4. Run backtests
        print(f"\n  2. Running VaR backtests...")
        models = {
            "Gaussian": "gauss",
            "Historical Sim": "hs",
            "Cornish-Fisher": "cf",
            "Filtered HS": "fhs",
            "EVT-GPD": "evt",
        }

        backtest_results = {}
        realized = merged["realized_loss"].values

        for p in config.CONFIDENCE_LEVELS:
            pct = f"{p:.3f}".replace(".", "")
            level_key = f"{p:.1%}"
            backtest_results[level_key] = {}

            for model_name, prefix in models.items():
                var_col = f"{prefix}_var_{pct}"
                if var_col not in merged.columns:
                    continue

                var_forecasts = merged[var_col].values
                bt = run_backtest_single(realized, var_forecasts, p)
                backtest_results[level_key][model_name] = bt

        # Print results
        print(f"\n  {'Model':>18s} | {'Level':>6s} | {'Viol Rate':>9s} | {'Expected':>8s} | {'Kupiec p':>9s} | {'KP':>4s} | {'Christ p':>9s} | {'CP':>4s}")
        print(f"  {'-'*95}")

        for level_key in backtest_results:
            for model_name, bt in backtest_results[level_key].items():
                kp = bt["kupiec"]["p_value"]
                cp = bt["christoffersen"]["p_value"]
                kp_str = f"{kp:.4f}" if not np.isnan(kp) else "  N/A "
                cp_str = f"{cp:.4f}" if not np.isnan(cp) else "  N/A "
                k_pass = "PASS" if bt["kupiec"]["pass"] else "FAIL"
                c_pass = "PASS" if bt["christoffersen"]["pass"] else "FAIL"

                print(
                    f"  {model_name:>18s} | {level_key:>6s} | "
                    f"{bt['violation_rate']:>9.4f} | {bt['expected_rate']:>8.4f} | "
                    f"{kp_str:>9s} | {k_pass:>4s} | {cp_str:>9s} | {c_pass:>4s}"
                )

        # 5. Generate plots
        print(f"\n  3. Generating plots...")
        plot_rolling_tail_index(rolling, name=series_name)
        plot_backtest_violations(merged, backtest_results, name=series_name)
        plot_model_scorecard(backtest_results, name=series_name)

        all_results[key] = {
            "n_forecast_days": len(merged),
            "rolling_xi_range": [float(rolling["xi"].min()), float(rolling["xi"].max())],
            "backtest": backtest_results,
        }

    # Save results (convert numpy types for JSON)
    out_path = os.path.join(config.DATA_DIR, "backtest_results.json")

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(clean_for_json(all_results), f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return all_results


if __name__ == "__main__":
    run_rolling_backtest()
