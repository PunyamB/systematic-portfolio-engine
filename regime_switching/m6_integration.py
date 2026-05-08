"""
Module 6: Meridian Integration Backtest (CORRECTED)

Walk-forward backtest comparing soft TVTP regime weights vs hard regime switching.

Pipeline:
  Phase 1: Fit TVTP per window with full saving (8 restarts, max 500 iter)
           Save filtered probs + all artifacts to regime_switching/data/window_fits/window_{id}/
  Phase 2: Run modified EXP006 engine (run_soft_backtest.py) using saved soft probs
           Recalibrates lambda/ra per window with soft regime active
  Phase 3: Compare aggregate soft vs hard NAVs

Inputs:
  - regime_switching/data/returns.parquet, covariates_raw.parquet
  - StrategyResearchLab/exp006 walk_forward_log.json (window definitions)
  - EXP006 NAV files (hard regime baseline)

Outputs (per window):
  regime_switching/data/window_fits/window_{id}/
    - tvtp_result.pkl (full result: mu, sigma, coeffs, P_all, filtered, smoothed)
    - filtered_probs.parquet (P(bull), P(bear), P(crisis) daily)
    - smoothed_probs.parquet
    - covariate_stats.json
    - em_convergence_log.json
    - all_restarts.pkl
    - soft_composite_scores.parquet (filled by run_soft_backtest)
    - soft_multipliers.parquet (filled by run_soft_backtest)
    - optimizer_weights.parquet (filled by run_soft_backtest)
    - daily_nav.parquet (filled by run_soft_backtest)
    - trade_log.parquet (filled by run_soft_backtest)

Aggregate:
  regime_switching/data/integration_backtest.parquet
  regime_switching/data/integration_metrics.json
  regime_switching/data/walk_forward_log_soft.json
"""
import pandas as pd
import numpy as np
import json
import time
import pickle
import subprocess
import sys
from pathlib import Path

from regime_switching.config import DATA_DIR, EM_MAX_ITER
from regime_switching.m3_tvtp_ms import (
    em_tvtp_ms, label_states,
    compute_all_transition_matrices, tvtp_hamilton_filter, tvtp_smoother,
)

# Override EM_MAX_ITER for this run (500 cap)
import regime_switching.m3_tvtp_ms as m3_mod
m3_mod.EM_MAX_ITER = 500

EXP006_DIR = Path("D:/Projects/StrategyResearchLab/experiments/exp006_extended_wf")
EXP006_WF = EXP006_DIR / "data" / "wf_results"
WINDOW_FITS_DIR = DATA_DIR / "window_fits"


def load_wf_log():
    """Load EXP006 walk-forward log."""
    with open(EXP006_DIR / "walk_forward_log.json") as f:
        wf = json.load(f)
    return wf["windows"], wf["final_holdout"]


def fit_tvtp_for_window_full(returns_full, covariates_full, train_end, n_restarts=8):
    """
    Fit TVTP with 8 restarts, capture ALL restart results + convergence logs.
    Returns: (best_result, all_restarts, convergence_log, z_mean, z_std)
    """
    train_mask = returns_full.index <= pd.Timestamp(train_end)
    r_train = returns_full.loc[train_mask, "log_return"].values
    z_train_raw = covariates_full.loc[train_mask].values

    z_mean = z_train_raw.mean(axis=0)
    z_std = z_train_raw.std(axis=0)
    z_train = (z_train_raw - z_mean) / z_std

    K = 3
    d = z_train.shape[1]

    all_restarts = []
    convergence_log = []
    best_result = None
    best_ll = -np.inf

    for restart in range(n_restarts):
        t0 = time.time()

        # Random init
        mu_init = np.array([
            np.random.uniform(0.0002, 0.001),
            np.random.uniform(-0.0005, 0.0002),
            np.random.uniform(-0.003, -0.0005),
        ])
        sigma_init = np.array([
            np.random.uniform(0.005, 0.010),
            np.random.uniform(0.010, 0.020),
            np.random.uniform(0.020, 0.040),
        ])
        coeffs_init = {}
        for i in range(K):
            c = np.random.randn(K - 1, 1 + d) * 0.05
            for j_idx in range(K - 1):
                c[j_idx, 0] = np.random.uniform(-4.0, -2.0)
            coeffs_init[i] = c

        init_params = {"mu": mu_init, "sigma": sigma_init, "coeffs": coeffs_init}

        try:
            result = em_tvtp_ms(r_train, z_train, K, init_params=init_params)
            elapsed = time.time() - t0

            restart_info = {
                "restart_id": restart + 1,
                "log_likelihood": float(result["log_likelihood"]),
                "n_iter": result["n_iter"],
                "converged": bool(result["converged"]),
                "time_seconds": elapsed,
            }
            convergence_log.append(restart_info)
            all_restarts.append(result)

            print(f"      Restart {restart + 1}/{n_restarts}: "
                  f"LL={result['log_likelihood']:.2f}, "
                  f"iter={result['n_iter']}, conv={result['converged']}, "
                  f"time={elapsed:.0f}s")

            if result["log_likelihood"] > best_ll:
                best_ll = result["log_likelihood"]
                best_result = result

        except Exception as e:
            elapsed = time.time() - t0
            print(f"      Restart {restart + 1}/{n_restarts}: FAILED ({e})")
            convergence_log.append({
                "restart_id": restart + 1,
                "failed": True,
                "error": str(e),
                "time_seconds": elapsed,
            })

    if best_result is None:
        raise RuntimeError(f"All {n_restarts} restarts failed")

    best_result = label_states(best_result)
    return best_result, all_restarts, convergence_log, z_mean, z_std


def compute_filtered_probs_full_sample(result, returns_full, covariates_full, z_mean, z_std):
    """
    Run filter forward on full sample with frozen training params.
    Filtered = causal (only past data at each point).
    Returns full sample of filtered probs (use only test period for backtest).
    """
    r_all = returns_full["log_return"].values
    z_all_raw = covariates_full.values
    z_all = (z_all_raw - z_mean) / z_std

    P_all = compute_all_transition_matrices(z_all, result["coeffs"], K=3)
    filtered, _, _ = tvtp_hamilton_filter(
        r_all, result["mu"], result["sigma"], P_all
    )
    smoothed, _ = tvtp_smoother(filtered, _, P_all) if False else (None, None)

    # Recompute smoothed properly
    _, _, predicted = tvtp_hamilton_filter(r_all, result["mu"], result["sigma"], P_all)
    smoothed, _ = tvtp_smoother(filtered, predicted, P_all)

    state_labels = result["state_labels"]
    filtered_df = pd.DataFrame(
        filtered,
        index=returns_full.index,
        columns=[f"p_{label}" for label in state_labels],
    )
    smoothed_df = pd.DataFrame(
        smoothed,
        index=returns_full.index,
        columns=[f"p_{label}" for label in state_labels],
    )
    return filtered_df, smoothed_df


def save_window_artifacts(win_dir, result, all_restarts, convergence_log,
                           z_mean, z_std, filtered_df, smoothed_df, covariate_names):
    """Save all per-window TVTP artifacts."""
    win_dir.mkdir(parents=True, exist_ok=True)

    # Full result (model parameters)
    result_to_save = {
        "mu": result["mu"].tolist(),
        "sigma": result["sigma"].tolist(),
        "state_labels": result["state_labels"],
        "log_likelihood": float(result["log_likelihood"]),
        "n_iter": result["n_iter"],
        "converged": bool(result["converged"]),
        "coeffs": {k: v.tolist() for k, v in result["coeffs"].items()},
        "P_all_mean": result["P_all"].mean(axis=0).tolist(),
    }
    with open(win_dir / "tvtp_result.pkl", "wb") as f:
        pickle.dump(result, f)
    with open(win_dir / "tvtp_result.json", "w") as f:
        json.dump(result_to_save, f, indent=2)

    # Filtered probs (causal, used in backtest)
    filtered_df.to_parquet(win_dir / "filtered_probs.parquet")

    # Smoothed probs (analysis only)
    smoothed_df.to_parquet(win_dir / "smoothed_probs.parquet")

    # Covariate stats
    with open(win_dir / "covariate_stats.json", "w") as f:
        json.dump({
            "covariates": covariate_names,
            "mean": z_mean.tolist(),
            "std": z_std.tolist(),
        }, f, indent=2)

    # EM convergence log
    with open(win_dir / "em_convergence_log.json", "w") as f:
        json.dump(convergence_log, f, indent=2)

    # All restart results
    with open(win_dir / "all_restarts.pkl", "wb") as f:
        pickle.dump(all_restarts, f)


def phase1_fit_all_windows(returns_full, covariates_full, windows, holdout, n_restarts=8):
    """Phase 1: Fit TVTP for every window, save artifacts."""
    print("=" * 60)
    print("Phase 1: TVTP fits per window")
    print("=" * 60)

    all_windows = windows + [holdout]
    covariate_names = list(covariates_full.columns)

    for w in all_windows:
        wid = w["window_id"]
        train_end = w["train_end"]
        win_dir = WINDOW_FITS_DIR / f"window_{wid}"

        # Skip if already done
        if (win_dir / "filtered_probs.parquet").exists() and            (win_dir / "tvtp_result.pkl").exists():
            print(f"  Window {wid}: artifacts exist, skipping fit")
            continue

        print()
        print(f"Window {wid}: train to {train_end}")
        t0 = time.time()

        try:
            result, all_restarts, convergence_log, z_mean, z_std =                 fit_tvtp_for_window_full(returns_full, covariates_full, train_end, n_restarts)

            filtered_df, smoothed_df = compute_filtered_probs_full_sample(
                result, returns_full, covariates_full, z_mean, z_std
            )

            save_window_artifacts(
                win_dir, result, all_restarts, convergence_log,
                z_mean, z_std, filtered_df, smoothed_df, covariate_names
            )

            elapsed = time.time() - t0
            print(f"    Best LL={result['log_likelihood']:.2f} | "
                  f"saved to {win_dir.name}/ | total time={elapsed:.0f}s")

        except Exception as e:
            print(f"    FAILED: {e}")
            continue


def phase2_run_soft_backtest():
    """Phase 2: Run modified EXP006 engine with soft regime."""
    print()
    print("=" * 60)
    print("Phase 2: Soft regime backtest via modified EXP006 engine")
    print("=" * 60)

    soft_script = Path("regime_switching/run_soft_backtest.py").resolve()
    print(f"  Running: python {soft_script}")
    print()

    result = subprocess.run(
        [sys.executable, str(soft_script)],
        cwd=str(soft_script.parent.parent),  # SPE root
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  Soft backtest FAILED with exit {result.returncode}")
        return False
    return True


def compute_metrics(nav_series):
    """CAGR, Sharpe, MaxDD."""
    rets = nav_series.pct_change().dropna()
    n_y = len(rets) / 252
    if n_y <= 0 or len(rets) == 0:
        return {"cagr": 0, "sharpe": 0, "max_dd": 0, "n_days": 0}

    cagr = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / n_y) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    max_dd = float(((nav_series / nav_series.cummax()) - 1).min())
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "n_days": int(len(rets)),
    }


def phase3_compare(windows, holdout):
    """Phase 3: Aggregate soft vs hard comparison."""
    print()
    print("=" * 60)
    print("Phase 3: Soft vs Hard comparison")
    print("=" * 60)

    all_windows = windows + [holdout]
    soft_navs = []
    hard_navs = []
    per_window = []

    for w in all_windows:
        wid = w["window_id"]
        soft_path = WINDOW_FITS_DIR / f"window_{wid}" / "daily_nav.parquet"
        hard_path = EXP006_WF / (
            "nav_window_holdout_mv_monthly.parquet" if wid == "holdout"
            else f"nav_window_{wid}_mv_monthly.parquet"
        )

        if not soft_path.exists():
            print(f"  Window {wid}: soft NAV missing, skipping")
            continue
        if not hard_path.exists():
            print(f"  Window {wid}: hard NAV missing, skipping")
            continue

        soft_nav = pd.read_parquet(soft_path)["nav"]
        hard_nav = pd.read_parquet(hard_path)["nav"]

        soft_navs.append(soft_nav)
        hard_navs.append(hard_nav)

        soft_m = compute_metrics(soft_nav)
        hard_m = compute_metrics(hard_nav)

        per_window.append({
            "window_id": wid,
            "soft": soft_m,
            "hard": hard_m,
            "soft_better_cagr": soft_m["cagr"] > hard_m["cagr"],
            "soft_better_sharpe": soft_m["sharpe"] > hard_m["sharpe"],
        })

        print(f"  Window {wid}: Soft CAGR={soft_m['cagr']:.4f} Sharpe={soft_m['sharpe']:.3f} | "
              f"Hard CAGR={hard_m['cagr']:.4f} Sharpe={hard_m['sharpe']:.3f}")

    if not soft_navs:
        print("  No data to compare")
        return None

    # Stitch with NAV continuity
    def stitch(nav_list):
        running = 1_000_000
        pieces = []
        for nav in nav_list:
            scale = running / nav.iloc[0]
            scaled = nav * scale
            running = scaled.iloc[-1]
            pieces.append(scaled)
        return pd.concat(pieces).sort_index()

    soft_chained = stitch(soft_navs)
    hard_chained = stitch(hard_navs)

    soft_agg = compute_metrics(soft_chained)
    hard_agg = compute_metrics(hard_chained)

    print()
    print(f"  AGGREGATE (chained NAVs):")
    print(f"    Soft: CAGR={soft_agg['cagr']:.4f}, Sharpe={soft_agg['sharpe']:.3f}, MaxDD={soft_agg['max_dd']:.4f}")
    print(f"    Hard: CAGR={hard_agg['cagr']:.4f}, Sharpe={hard_agg['sharpe']:.3f}, MaxDD={hard_agg['max_dd']:.4f}")
    print()
    print(f"    CAGR improvement: {soft_agg['cagr'] - hard_agg['cagr']:+.4f}")
    print(f"    Sharpe improvement: {soft_agg['sharpe'] - hard_agg['sharpe']:+.3f}")

    c3_pass = (soft_agg["cagr"] > hard_agg["cagr"]) and (soft_agg["sharpe"] > hard_agg["sharpe"])
    print(f"    C3 (soft beats hard on both): {'PASS' if c3_pass else 'FAIL'}")

    soft_wins_cagr = sum(1 for m in per_window if m["soft_better_cagr"])
    soft_wins_sharpe = sum(1 for m in per_window if m["soft_better_sharpe"])
    n = len(per_window)
    print(f"    Soft won CAGR: {soft_wins_cagr}/{n} windows")
    print(f"    Soft won Sharpe: {soft_wins_sharpe}/{n} windows")

    # Save
    soft_chained.to_frame("nav").to_parquet(DATA_DIR / "integration_backtest.parquet")

    integration_results = {
        "aggregate": {
            "soft": soft_agg,
            "hard": hard_agg,
            "cagr_improvement": float(soft_agg["cagr"] - hard_agg["cagr"]),
            "sharpe_improvement": float(soft_agg["sharpe"] - hard_agg["sharpe"]),
            "c3_pass": bool(c3_pass),
        },
        "per_window": per_window,
        "windows_won_cagr": soft_wins_cagr,
        "windows_won_sharpe": soft_wins_sharpe,
        "n_windows": n,
    }

    with open(DATA_DIR / "integration_metrics.json", "w") as f:
        json.dump(integration_results, f, indent=2, default=str)

    # Update comparison_results.json with C3
    try:
        with open(DATA_DIR / "comparison_results.json") as f:
            comp = json.load(f)
        comp["integration_criteria"]["C3_walkforward"] = {
            "value": {
                "cagr_improvement": float(soft_agg["cagr"] - hard_agg["cagr"]),
                "sharpe_improvement": float(soft_agg["sharpe"] - hard_agg["sharpe"]),
            },
            "threshold": "CAGR > 0 AND Sharpe > 0",
            "pass": bool(c3_pass),
        }
        comp["integration_criteria"]["all_pass"] = bool(
            comp["integration_criteria"]["C1_kappa"]["pass"] and
            comp["integration_criteria"]["C2_early_warning"]["pass"] and
            c3_pass
        )
        with open(DATA_DIR / "comparison_results.json", "w") as f:
            json.dump(comp, f, indent=2)
    except Exception as e:
        print(f"  Could not update comparison_results.json: {e}")

    return integration_results


def run():
    """Main: 3-phase orchestrator."""
    print("=" * 60)
    print("B2 Module 6: Meridian Integration Backtest (CORRECTED)")
    print("=" * 60)
    print()

    # Load data
    returns_full = pd.read_parquet(DATA_DIR / "returns.parquet")
    covariates_full = pd.read_parquet(DATA_DIR / "covariates_raw.parquet")
    windows, holdout = load_wf_log()

    print(f"  Returns: {len(returns_full)} obs")
    print(f"  Covariates: {covariates_full.shape}")
    print(f"  Windows: {len(windows)} + holdout")
    print()

    WINDOW_FITS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: TVTP fits with full saving
    phase1_fit_all_windows(returns_full, covariates_full, windows, holdout, n_restarts=8)

    # Phase 2: Soft backtest through modified EXP006 engine
    success = phase2_run_soft_backtest()
    if not success:
        print("  Phase 2 failed. Stopping.")
        return None

    # Phase 3: Compare
    results = phase3_compare(windows, holdout)

    print()
    print("=" * 60)
    print("Module 6 COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run()
