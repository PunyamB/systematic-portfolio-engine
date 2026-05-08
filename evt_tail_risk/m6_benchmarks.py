"""
Module 6: Benchmark Risk Models
=================================
Implements 4 competing VaR/ES models to benchmark against EVT-GPD.
Demonstrates that EVT adds value, not just complexity.

Models:
  1. Gaussian — VaR = mu + z_p * sigma (rolling 252d)
  2. Historical Simulation — empirical quantile of rolling 500d window
  3. Cornish-Fisher — normal VaR adjusted for skewness and kurtosis
  4. Filtered Historical Simulation (FHS) — GARCH(1,1) standardized
     residuals resampled, rescaled by conditional volatility

All models produce one-day-ahead VaR/ES forecasts at 95%, 99%, 99.5%.
EVT-GPD rolling forecasts are added in M7.

Usage:
    from evt_tail_risk.m6_benchmarks import run_benchmarks

    results = run_benchmarks()
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from evt_tail_risk import config


def gaussian_var_es(losses, window=252):
    """
    Gaussian VaR/ES: rolling mean + z_p * sigma.

    Returns:
        DataFrame with columns: date, var_95, var_99, var_995, es_95, es_99, es_995
    """
    n = len(losses)
    results = []

    for i in range(window, n):
        w = losses[i - window:i]
        mu = w.mean()
        sigma = w.std()

        row = {}
        for p in config.CONFIDENCE_LEVELS:
            z = sp_stats.norm.ppf(p)
            var = mu + z * sigma
            # ES for normal: mu + sigma * phi(z) / (1-p)
            es = mu + sigma * sp_stats.norm.pdf(z) / (1 - p)
            pct = f"{p:.3f}".replace(".", "")
            row[f"var_{pct}"] = var
            row[f"es_{pct}"] = es

        results.append(row)

    return results


def historical_sim_var_es(losses, window=500):
    """
    Historical Simulation: empirical quantile of rolling window.

    Returns:
        list of dicts with var/es at each level
    """
    n = len(losses)
    results = []

    for i in range(window, n):
        w = losses[i - window:i]

        row = {}
        for p in config.CONFIDENCE_LEVELS:
            var = np.percentile(w, p * 100)
            # ES = mean of losses exceeding VaR
            exceedances = w[w >= var]
            es = exceedances.mean() if len(exceedances) > 0 else var
            pct = f"{p:.3f}".replace(".", "")
            row[f"var_{pct}"] = var
            row[f"es_{pct}"] = es

        results.append(row)

    return results


def cornish_fisher_var_es(losses, window=252):
    """
    Cornish-Fisher VaR: normal VaR adjusted for skewness and kurtosis.

    CF expansion: z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3*z)*K/24 - (2*z^3 - 5*z)*S^2/36
    where S = skewness, K = excess kurtosis, z = normal quantile.

    Returns:
        list of dicts with var/es at each level
    """
    n = len(losses)
    results = []

    for i in range(window, n):
        w = losses[i - window:i]
        mu = w.mean()
        sigma = w.std()
        S = sp_stats.skew(w)
        K = sp_stats.kurtosis(w)  # excess

        row = {}
        for p in config.CONFIDENCE_LEVELS:
            z = sp_stats.norm.ppf(p)
            # Cornish-Fisher adjustment
            z_cf = (z
                    + (z**2 - 1) * S / 6
                    + (z**3 - 3 * z) * K / 24
                    - (2 * z**3 - 5 * z) * S**2 / 36)
            var = mu + z_cf * sigma

            # ES approximation: use CF-adjusted quantile in normal ES formula
            es = mu + sigma * sp_stats.norm.pdf(z_cf) / (1 - p)

            pct = f"{p:.3f}".replace(".", "")
            row[f"var_{pct}"] = var
            row[f"es_{pct}"] = es

        results.append(row)

    return results


def fhs_var_es(losses, window=500, garch_refit_freq=20):
    """
    Filtered Historical Simulation:
      1. Fit GARCH(1,1) to returns in rolling window
      2. Extract standardized residuals z_t = r_t / sigma_t
      3. VaR = current sigma * quantile(z_t)
      4. ES = current sigma * mean(z_t exceeding VaR quantile)

    Refits GARCH every garch_refit_freq days for speed.

    Returns:
        list of dicts with var/es at each level
    """
    from arch import arch_model

    n = len(losses)
    results = []

    # Negate losses back to returns for GARCH (GARCH models returns, not losses)
    returns = -losses * 100  # percentage returns for numerical stability

    cached_resid = None
    cached_vol = None
    last_fit = -garch_refit_freq  # force fit on first iteration

    for i in range(window, n):
        # Refit GARCH periodically
        if i - last_fit >= garch_refit_freq:
            w = returns[i - window:i]
            try:
                model = arch_model(w, vol="Garch", p=config.GARCH_P,
                                   q=config.GARCH_Q, mean="Constant",
                                   rescale=False)
                fit = model.fit(disp="off", show_warning=False)
                cached_resid = fit.resid.values / fit.conditional_volatility.values
                cached_vol = fit.conditional_volatility.values[-1]
                last_fit = i
            except Exception:
                # If GARCH fails, fall back to simple vol
                cached_resid = w / w.std()
                cached_vol = w.std()
                last_fit = i

        if cached_resid is None:
            continue

        # Current conditional vol (use last fitted value as proxy)
        curr_vol = cached_vol

        row = {}
        for p in config.CONFIDENCE_LEVELS:
            # VaR from standardized residuals (remember: these are returns, not losses)
            # We want the p-th quantile of losses, which is the (1-p) quantile of returns
            z_quantile = np.percentile(-cached_resid, p * 100)  # negate for loss space
            var = curr_vol * z_quantile / 100  # convert back from percentage

            # ES
            z_exceed = -cached_resid[-cached_resid <= -z_quantile]  # loss space
            if len(z_exceed) > 0:
                es = curr_vol * z_exceed.mean() / 100
            else:
                es = var

            pct = f"{p:.3f}".replace(".", "")
            row[f"var_{pct}"] = var
            row[f"es_{pct}"] = es

        results.append(row)

    return results


def run_benchmarks():
    """
    Run all 4 benchmark models on both SPY and Meridian.
    Saves daily VaR/ES forecasts to benchmark_var.parquet.

    Returns:
        dict with summary info
    """
    output = {}

    for series_name, key in [("SPY", "spy"), ("Meridian", "meridian")]:
        print(f"\n{'='*60}")
        print(f"BENCHMARK MODELS: {series_name}")
        print(f"{'='*60}")

        # Load losses
        loss_path = os.path.join(config.DATA_DIR, f"{key}_losses.parquet")
        df = pd.read_parquet(loss_path)
        losses = df["loss"].values
        dates = pd.to_datetime(df["date"].values)

        n = len(losses)
        gauss_window = 252
        hs_window = config.ROLLING_WINDOW  # 500

        # Use the larger window as the common start point
        start_idx = hs_window  # 500

        print(f"  Total observations: {n}")
        print(f"  Forecast start index: {start_idx} ({dates[start_idx].date()})")
        print(f"  Forecast days: {n - start_idx}")

        # Run models
        print(f"\n  Running Gaussian (window={gauss_window})...")
        gauss = gaussian_var_es(losses, window=gauss_window)

        print(f"  Running Historical Simulation (window={hs_window})...")
        hs = historical_sim_var_es(losses, window=hs_window)

        print(f"  Running Cornish-Fisher (window={gauss_window})...")
        cf = cornish_fisher_var_es(losses, window=gauss_window)

        print(f"  Running Filtered Historical Simulation (window={hs_window})...")
        fhs = fhs_var_es(losses, window=hs_window)

        # Align all models to the same date range (start at hs_window = 500)
        # Gaussian and CF start at index 252, HS and FHS start at index 500
        # We align everything to start at index 500
        gauss_offset = start_idx - gauss_window  # 500 - 252 = 248
        cf_offset = start_idx - gauss_window

        forecast_dates = dates[start_idx:]
        n_forecasts = len(forecast_dates)

        # Build aligned DataFrames
        rows = []
        for i in range(n_forecasts):
            row = {"date": forecast_dates[i], "realized_loss": losses[start_idx + i]}

            # Gaussian (started at index 252, so offset by 248)
            gi = gauss_offset + i
            if 0 <= gi < len(gauss):
                for col, val in gauss[gi].items():
                    row[f"gauss_{col}"] = val

            # HS (started at index 500, aligned)
            hi = i
            if 0 <= hi < len(hs):
                for col, val in hs[hi].items():
                    row[f"hs_{col}"] = val

            # CF (started at index 252, offset by 248)
            ci = cf_offset + i
            if 0 <= ci < len(cf):
                for col, val in cf[ci].items():
                    row[f"cf_{col}"] = val

            # FHS (started at index 500, aligned)
            fi = i
            if 0 <= fi < len(fhs):
                for col, val in fhs[fi].items():
                    row[f"fhs_{col}"] = val

            rows.append(row)

        result_df = pd.DataFrame(rows)

        # Check for NaN
        n_nan = result_df.isna().sum().sum()
        n_rows = len(result_df)
        print(f"\n  Results: {n_rows} forecast days")
        print(f"  NaN values: {n_nan}")

        # Quick summary at 99% VaR
        for model in ["gauss", "hs", "cf", "fhs"]:
            col = f"{model}_var_0990"
            if col in result_df.columns:
                mean_var = result_df[col].mean()
                print(f"  {model:6s} mean 99% VaR: {mean_var:.6f}")

        # Save
        out_path = os.path.join(config.DATA_DIR, f"benchmark_var_{key}.parquet")
        result_df.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")

        output[key] = {
            "n_forecasts": n_rows,
            "n_nan": int(n_nan),
            "date_range": [str(forecast_dates[0].date()), str(forecast_dates[-1].date())],
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key in ["spy", "meridian"]:
        r = output[key]
        print(f"  {key:10s} | {r['n_forecasts']} forecasts | {r['date_range'][0]} to {r['date_range'][1]} | NaN: {r['n_nan']}")

    return output


if __name__ == "__main__":
    run_benchmarks()
