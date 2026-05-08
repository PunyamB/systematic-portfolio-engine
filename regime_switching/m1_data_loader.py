"""
Module 1: Data Loader
Loads SPY returns and macro covariates, aligns dates, standardizes covariates.

Inputs:
  - evt_tail_risk/data/spy_prices.parquet (SPY OHLCV, 1993-2026)
  - StrategyResearchLab/data/raw/macro_features.parquet (VIX, yield curve, credit spread)
  - StrategyResearchLab/data/raw/breadth.parquet (optional 4th covariate)

Outputs:
  - regime_switching/data/returns.parquet (daily SPY log returns)
  - regime_switching/data/covariates.parquet (standardized macro covariates)
  - regime_switching/data/covariates_raw.parquet (raw covariates for plotting)
  - regime_switching/data/data_metadata.json (summary stats, date range, diagnostics)
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

from regime_switching.config import (
    SPY_PRICES_PATH, MACRO_FEATURES_PATH, BREADTH_PATH,
    DATA_DIR, SAMPLE_START, SAMPLE_END, COVARIATES,
)


def load_spy_returns():
    """Load SPY prices and compute daily log returns."""
    spy = pd.read_parquet(SPY_PRICES_PATH)
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").reset_index(drop=True)

    # Daily log returns
    spy["log_return"] = np.log(spy["close"] / spy["close"].shift(1))
    spy = spy.dropna(subset=["log_return"])

    returns = spy[["date", "log_return"]].copy()
    returns = returns.set_index("date")
    return returns


def load_macro_covariates():
    """Load macro features (VIX, yield curve, credit spread) from StrategyResearchLab."""
    mf = pd.read_parquet(MACRO_FEATURES_PATH)
    mf.index = pd.to_datetime(mf.index)
    mf.index.name = "date"
    return mf[COVARIATES]


def load_breadth():
    """Load market breadth (optional 4th covariate)."""
    br = pd.read_parquet(BREADTH_PATH)
    br.index = pd.to_datetime(br.index)
    br.index.name = "date"
    return br


def align_and_trim(returns, covariates):
    """Align returns and covariates on common trading dates within sample period."""
    start = pd.Timestamp(SAMPLE_START)
    end = pd.Timestamp(SAMPLE_END)

    # Trim to sample period
    returns = returns[(returns.index >= start) & (returns.index <= end)]
    covariates = covariates[(covariates.index >= start) & (covariates.index <= end)]

    # Forward fill covariates (FRED has gaps on holidays)
    covariates = covariates.ffill()

    # Inner join on dates
    common_dates = returns.index.intersection(covariates.index)
    returns = returns.loc[common_dates].sort_index()
    covariates = covariates.loc[common_dates].sort_index()

    return returns, covariates


def standardize_covariates(covariates):
    """Zero mean, unit variance. Store stats for out-of-sample application."""
    stats = {
        "mean": {k: float(v) for k, v in covariates.mean().items()},
        "std": {k: float(v) for k, v in covariates.std().items()},
    }
    standardized = (covariates - covariates.mean()) / covariates.std()
    return standardized, stats


def run_diagnostics(returns, covariates):
    """Run validation checks and compute summary statistics."""
    diagnostics = {}

    # 1. Date alignment check
    assert returns.index.equals(covariates.index), "Date mismatch between returns and covariates"
    diagnostics["n_observations"] = len(returns)
    diagnostics["date_range"] = {
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
    }

    # 2. NaN check
    ret_nan = int(returns.isna().sum().sum())
    cov_nan = int(covariates.isna().sum().sum())
    assert ret_nan == 0, f"NaN in returns: {ret_nan}"
    assert cov_nan == 0, f"NaN in covariates: {cov_nan}"
    diagnostics["nan_count"] = {"returns": ret_nan, "covariates": cov_nan}

    # 3. Returns summary stats
    r = returns["log_return"]
    diagnostics["returns_stats"] = {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurt()),
        "min": float(r.min()),
        "max": float(r.max()),
        "annualized_mean": float(r.mean() * 252),
        "annualized_vol": float(r.std() * np.sqrt(252)),
    }

    # 4. Covariate correlation matrix
    corr = covariates.corr()
    diagnostics["covariate_correlations"] = {
        k: {k2: float(v2) for k2, v2 in v.items()}
        for k, v in corr.to_dict().items()
    }
    max_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).max().max()
    diagnostics["max_pairwise_correlation"] = float(max_corr)
    if max_corr > 0.85:
        diagnostics["collinearity_warning"] = f"Max pairwise correlation {max_corr:.3f} > 0.85"

    # 5. ADF test on returns (should reject unit root)
    adf_result = adfuller(r, maxlag=20)
    diagnostics["adf_test"] = {
        "statistic": float(adf_result[0]),
        "p_value": float(adf_result[1]),
        "stationary": bool(adf_result[1] < 0.01),
    }

    # 6. Covariate summary stats (pre-standardization)
    cov_stats = {}
    for col in covariates.columns:
        cov_stats[col] = {
            "mean": float(covariates[col].mean()),
            "std": float(covariates[col].std()),
            "min": float(covariates[col].min()),
            "max": float(covariates[col].max()),
        }
    diagnostics["covariate_stats"] = cov_stats

    return diagnostics


def run():
    """Main entry point. Load, align, standardize, validate, save."""
    print("=" * 60)
    print("B2 Module 1: Data Loader")
    print("=" * 60)

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    print()
    print("[1/5] Loading SPY returns...")
    returns = load_spy_returns()
    print(f"  SPY returns loaded: {len(returns)} obs, "
          f"{returns.index.min().date()} to {returns.index.max().date()}")

    print()
    print("[2/5] Loading macro covariates...")
    covariates_raw = load_macro_covariates()
    print(f"  Macro features loaded: {len(covariates_raw)} obs, "
          f"{covariates_raw.index.min().date()} to {covariates_raw.index.max().date()}")

    # 2. Align and trim
    print()
    print("[3/5] Aligning and trimming to sample period...")
    returns, covariates_raw = align_and_trim(returns, covariates_raw)
    print(f"  Aligned: {len(returns)} common trading days")
    print(f"  Period: {returns.index.min().date()} to {returns.index.max().date()}")

    # 3. Run diagnostics on raw covariates (before standardization)
    print()
    print("[4/5] Running diagnostics...")
    diagnostics = run_diagnostics(returns, covariates_raw)

    rs = diagnostics["returns_stats"]
    print(f"  Observations: {diagnostics['n_observations']}")
    print(f"  Returns: mean={rs['annualized_mean']:.4f}, "
          f"vol={rs['annualized_vol']:.4f}, "
          f"skew={rs['skewness']:.3f}, "
          f"kurt={rs['kurtosis']:.3f}")
    print(f"  ADF p-value: {diagnostics['adf_test']['p_value']:.6f} "
          f"({'stationary' if diagnostics['adf_test']['stationary'] else 'NON-STATIONARY'})")
    print(f"  Max pairwise correlation: {diagnostics['max_pairwise_correlation']:.3f}")

    if "collinearity_warning" in diagnostics:
        print(f"  WARNING: {diagnostics['collinearity_warning']}")

    # Print correlation matrix
    print()
    print("  Correlation matrix:")
    corr = diagnostics["covariate_correlations"]
    cols = list(corr.keys())
    header = "                      " + "  ".join(f"{c:>20}" for c in cols)
    print(header)
    for c1 in cols:
        row = f"  {c1:>20}" + "  ".join(f"{corr[c1][c2]:>20.3f}" for c2 in cols)
        print(row)

    # 4. Standardize covariates
    covariates_std, std_stats = standardize_covariates(covariates_raw)
    diagnostics["standardization"] = std_stats

    # 5. Save outputs
    print()
    print("[5/5] Saving outputs...")
    returns.to_parquet(DATA_DIR / "returns.parquet")
    covariates_std.to_parquet(DATA_DIR / "covariates.parquet")
    covariates_raw.to_parquet(DATA_DIR / "covariates_raw.parquet")

    with open(DATA_DIR / "data_metadata.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"  returns.parquet: {len(returns)} rows")
    print(f"  covariates.parquet: {len(covariates_std)} rows ({list(covariates_std.columns)})")
    print(f"  covariates_raw.parquet: {len(covariates_raw)} rows")
    print(f"  data_metadata.json: saved")

    print()
    print("=" * 60)
    print("Module 1 COMPLETE")
    print("=" * 60)

    return returns, covariates_std, diagnostics


if __name__ == "__main__":
    run()
