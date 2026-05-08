"""
Module 2: Fixed Transition Markov-Switching
Implements standard MS models with constant transition probabilities (2-state and 3-state).
Serves as baseline for TVTP comparison and validation of custom Hamilton filter.

Steps:
  1. Synthetic recovery test (known parameters, verify recovery)
  2. Fit 2-state and 3-state on real SPY returns
  3. Cross-validate against statsmodels MarkovRegression
  4. Save filtered/smoothed probabilities and parameters

Inputs:
  - regime_switching/data/returns.parquet

Outputs:
  - regime_switching/data/ms2_fixed_results.json
  - regime_switching/data/ms3_fixed_results.json
  - regime_switching/data/ms2_fixed_probs.parquet
  - regime_switching/data/ms3_fixed_probs.parquet
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import norm
from scipy.special import logsumexp

from regime_switching.config import (
    DATA_DIR, EM_MAX_ITER, EM_TOL, N_RESTARTS, FILTER_CLAMP,
    SYNTHETIC_N, SYNTHETIC_REPS,
)


# ──────────────────────────────────────────────────────────
# Core: Hamilton Filter + EM for Fixed Transition MS
# ──────────────────────────────────────────────────────────

def _emission_logpdf(r, mu, sigma):
    """Log probability density of observation r under N(mu, sigma^2)."""
    return norm.logpdf(r, loc=mu, scale=sigma)


def hamilton_filter(returns, mu, sigma, trans_mat, init_prob=None):
    """
    Hamilton filter forward pass with constant transition matrix.

    Args:
        returns: (T,) array of log returns
        mu: (K,) regime means
        sigma: (K,) regime std devs
        trans_mat: (K, K) transition matrix, trans_mat[i,j] = P(S_t=j | S_{t-1}=i)
        init_prob: (K,) initial state probabilities (default: stationary dist)

    Returns:
        filtered: (T, K) filtered probabilities P(S_t=k | r_1..r_t)
        log_likelihood: total log-likelihood
        predicted: (T, K) predicted probabilities P(S_t=k | r_1..r_{t-1})
    """
    T = len(returns)
    K = len(mu)

    if init_prob is None:
        # Stationary distribution: solve pi = pi @ P
        eigenvalues, eigenvectors = np.linalg.eig(trans_mat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        init_prob = np.real(eigenvectors[:, idx])
        init_prob = init_prob / init_prob.sum()
        init_prob = np.maximum(init_prob, FILTER_CLAMP)

    filtered = np.zeros((T, K))
    predicted = np.zeros((T, K))
    log_likelihood = 0.0

    for t in range(T):
        # Prediction step
        if t == 0:
            pred = init_prob
        else:
            pred = filtered[t - 1] @ trans_mat  # (K,) @ (K,K) -> (K,)

        pred = np.maximum(pred, FILTER_CLAMP)
        pred = pred / pred.sum()
        predicted[t] = pred

        # Update step
        log_lik_k = np.array([_emission_logpdf(returns[t], mu[k], sigma[k]) for k in range(K)])

        # Log-space for numerical stability
        log_joint = log_lik_k + np.log(pred)
        log_marginal = logsumexp(log_joint)
        log_likelihood += log_marginal

        filtered[t] = np.exp(log_joint - log_marginal)
        filtered[t] = np.maximum(filtered[t], FILTER_CLAMP)
        filtered[t] = filtered[t] / filtered[t].sum()

    return filtered, log_likelihood, predicted


def kim_smoother(filtered, predicted, trans_mat):
    """
    Kim (1994) backward smoother.

    Returns:
        smoothed: (T, K) smoothed probabilities P(S_t=k | r_1..r_T)
        xi: (T-1, K, K) joint smoothed P(S_{t-1}=i, S_t=j | all data)
    """
    T, K = filtered.shape
    smoothed = np.zeros((T, K))
    xi = np.zeros((T - 1, K, K))

    smoothed[-1] = filtered[-1]

    for t in range(T - 2, -1, -1):
        for i in range(K):
            for j in range(K):
                if predicted[t + 1, j] > FILTER_CLAMP:
                    xi[t, i, j] = (filtered[t, i] * trans_mat[i, j] *
                                   smoothed[t + 1, j] / predicted[t + 1, j])
                else:
                    xi[t, i, j] = 0.0

        smoothed[t] = xi[t].sum(axis=1)
        s_sum = smoothed[t].sum()
        if s_sum > 0:
            smoothed[t] = smoothed[t] / s_sum
        smoothed[t] = np.maximum(smoothed[t], FILTER_CLAMP)
        smoothed[t] = smoothed[t] / smoothed[t].sum()

    return smoothed, xi


def em_fixed_ms(returns, K, max_iter=EM_MAX_ITER, tol=EM_TOL, init_params=None):
    """
    EM algorithm for fixed-transition Markov-switching model.

    Args:
        returns: (T,) array
        K: number of states
        max_iter: max EM iterations
        tol: relative LL convergence threshold
        init_params: optional dict with mu, sigma, trans_mat

    Returns:
        dict with mu, sigma, trans_mat, filtered, smoothed, log_likelihood, n_iter, converged
    """
    T = len(returns)

    # Initialize parameters
    if init_params is not None:
        mu = init_params["mu"].copy()
        sigma = init_params["sigma"].copy()
        trans_mat = init_params["trans_mat"].copy()
    else:
        # Data-driven initialization
        percentiles = np.linspace(10, 90, K + 2)[1:-1]
        mu = np.percentile(returns, percentiles) * 252  # annualize roughly
        mu = mu / 252  # back to daily
        sigma = np.full(K, returns.std()) * np.linspace(0.5, 2.0, K)
        trans_mat = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(trans_mat, 0.95)

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # E-step
        filtered, log_likelihood, predicted = hamilton_filter(returns, mu, sigma, trans_mat)
        smoothed, xi = kim_smoother(filtered, predicted, trans_mat)

        # Check convergence
        if prev_ll > -np.inf:
            rel_change = abs(log_likelihood - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < tol:
                return {
                    "mu": mu, "sigma": sigma, "trans_mat": trans_mat,
                    "filtered": filtered, "smoothed": smoothed,
                    "log_likelihood": log_likelihood,
                    "n_iter": iteration + 1, "converged": True,
                }
        prev_ll = log_likelihood

        # M-step: emissions
        gamma = smoothed  # (T, K)
        for k in range(K):
            w = gamma[:, k]
            w_sum = w.sum()
            if w_sum > 1e-10:
                mu[k] = np.sum(w * returns) / w_sum
                sigma[k] = np.sqrt(np.sum(w * (returns - mu[k]) ** 2) / w_sum)
                sigma[k] = max(sigma[k], 1e-6)  # floor

        # M-step: transitions (closed-form for fixed MS)
        for i in range(K):
            for j in range(K):
                num = xi[:, i, j].sum()
                den = gamma[:-1, i].sum()
                if den > 1e-10:
                    trans_mat[i, j] = num / den
                else:
                    trans_mat[i, j] = 1.0 / K

            # Normalize row
            row_sum = trans_mat[i].sum()
            if row_sum > 0:
                trans_mat[i] = trans_mat[i] / row_sum

    return {
        "mu": mu, "sigma": sigma, "trans_mat": trans_mat,
        "filtered": filtered, "smoothed": smoothed,
        "log_likelihood": log_likelihood,
        "n_iter": max_iter, "converged": False,
    }


def fit_with_restarts(returns, K, n_restarts=8):
    """
    Fit MS model with multiple random restarts, keep best.
    """
    best_result = None
    best_ll = -np.inf

    for restart in range(n_restarts):
        # Random initialization
        if K == 2:
            mu_init = np.array([
                np.random.uniform(0.0001, 0.001),
                np.random.uniform(-0.001, -0.0001),
            ])
            sigma_init = np.array([
                np.random.uniform(0.005, 0.012),
                np.random.uniform(0.015, 0.035),
            ])
        else:  # K == 3
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

        # High persistence diagonal
        diag = np.random.uniform(0.93, 0.99, K)
        trans_init = np.full((K, K), 0.0)
        for i in range(K):
            off_diag = (1 - diag[i]) / (K - 1)
            trans_init[i] = off_diag
            trans_init[i, i] = diag[i]

        init_params = {"mu": mu_init, "sigma": sigma_init, "trans_mat": trans_init}

        try:
            result = em_fixed_ms(returns, K, init_params=init_params)
            if result["log_likelihood"] > best_ll:
                best_ll = result["log_likelihood"]
                best_result = result
        except Exception as e:
            continue  # skip failed restarts

    if best_result is None:
        raise RuntimeError(f"All {n_restarts} restarts failed for K={K}")

    return best_result


def label_states(result):
    """Label states by mean return: highest mu = bull (0), lowest = crisis/bear."""
    mu = result["mu"]
    order = np.argsort(mu)[::-1]  # descending by mean

    result["mu"] = mu[order]
    result["sigma"] = result["sigma"][order]
    result["trans_mat"] = result["trans_mat"][order][:, order]
    result["filtered"] = result["filtered"][:, order]
    result["smoothed"] = result["smoothed"][:, order]

    K = len(mu)
    if K == 2:
        result["state_labels"] = ["bull", "bear"]
    else:
        result["state_labels"] = ["bull", "bear", "crisis"]

    return result


# ──────────────────────────────────────────────────────────
# Synthetic Recovery Test
# ──────────────────────────────────────────────────────────

def generate_synthetic_ms(T, mu, sigma, trans_mat):
    """Generate synthetic data from a known MS model."""
    K = len(mu)
    states = np.zeros(T, dtype=int)
    returns = np.zeros(T)

    # Stationary distribution for initial state
    eigenvalues, eigenvectors = np.linalg.eig(trans_mat.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()

    states[0] = np.random.choice(K, p=pi)
    returns[0] = np.random.normal(mu[states[0]], sigma[states[0]])

    for t in range(1, T):
        states[t] = np.random.choice(K, p=trans_mat[states[t - 1]])
        returns[t] = np.random.normal(mu[states[t]], sigma[states[t]])

    return returns, states


def synthetic_recovery_test(K=2, n_reps=SYNTHETIC_REPS, n_obs=SYNTHETIC_N):
    """
    Generate data from known MS model, fit, check parameter recovery.
    Returns coverage rates for mu and sigma.
    """
    if K == 2:
        true_mu = np.array([0.0005, -0.0005])
        true_sigma = np.array([0.008, 0.020])
        true_trans = np.array([[0.98, 0.02], [0.05, 0.95]])
    else:
        true_mu = np.array([0.0005, -0.0001, -0.002])
        true_sigma = np.array([0.008, 0.015, 0.030])
        true_trans = np.array([
            [0.97, 0.02, 0.01],
            [0.03, 0.94, 0.03],
            [0.02, 0.03, 0.95],
        ])

    mu_recovered = 0
    sigma_recovered = 0

    for rep in range(n_reps):
        returns, _ = generate_synthetic_ms(n_obs, true_mu, true_sigma, true_trans)

        try:
            result = fit_with_restarts(returns, K, n_restarts=3)
            result = label_states(result)

            # Check if true params are within 50% of estimated
            # (loose criterion since we care about rough recovery, not exact)
            mu_close = all(
                abs(result["mu"][k] - true_mu[k]) < 3 * true_sigma[k] / np.sqrt(n_obs)
                for k in range(K)
            )
            sigma_close = all(
                abs(result["sigma"][k] - true_sigma[k]) / true_sigma[k] < 0.5
                for k in range(K)
            )

            if mu_close:
                mu_recovered += 1
            if sigma_close:
                sigma_recovered += 1

        except Exception:
            continue

    return {
        "K": K,
        "n_reps": n_reps,
        "mu_recovery_rate": mu_recovered / n_reps,
        "sigma_recovery_rate": sigma_recovered / n_reps,
        "true_mu": true_mu.tolist(),
        "true_sigma": true_sigma.tolist(),
    }


# ──────────────────────────────────────────────────────────
# statsmodels Cross-Validation
# ──────────────────────────────────────────────────────────

def cross_validate_statsmodels(returns, custom_result, K):
    """
    Fit statsmodels MarkovRegression and compare log-likelihood and params
    against our custom implementation.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    # statsmodels expects a Series with datetime index
    mod = MarkovRegression(returns, k_regimes=K, trend="c", switching_variance=True)
    sm_result = mod.fit(search_reps=20, em_iter=300)

    # Extract statsmodels params (sorted by mean for comparison)
    sm_mu = np.array([sm_result.params[f"const[{k}]"] for k in range(K)])
    sm_sigma = np.array([np.sqrt(sm_result.params[f"sigma2[{k}]"]) for k in range(K)])
    sm_ll = sm_result.llf

    # Sort by mean (same ordering as our label_states)
    sm_order = np.argsort(sm_mu)[::-1]
    sm_mu = sm_mu[sm_order]
    sm_sigma = sm_sigma[sm_order]

    # Compare
    ll_diff = abs(custom_result["log_likelihood"] - sm_ll)
    mu_diff = np.max(np.abs(custom_result["mu"] - sm_mu))
    sigma_diff = np.max(np.abs(custom_result["sigma"] - sm_sigma))

    return {
        "statsmodels_ll": float(sm_ll),
        "custom_ll": float(custom_result["log_likelihood"]),
        "ll_difference": float(ll_diff),
        "mu_max_difference": float(mu_diff),
        "sigma_max_difference": float(sigma_diff),
        "ll_match": bool(ll_diff < 10),  # within 10 log-lik points
        "params_match": bool(mu_diff < 1e-3 and sigma_diff < 1e-3),
        "statsmodels_mu": sm_mu.tolist(),
        "statsmodels_sigma": sm_sigma.tolist(),
        "statsmodels_bic": float(sm_result.bic),
        "statsmodels_aic": float(sm_result.aic),
    }


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def _serialize_result(result, K):
    """Convert result dict to JSON-serializable format."""
    T = len(result["filtered"])
    n_params = 2 * K + K * (K - 1)  # emission + transition
    return {
        "K": K,
        "state_labels": result["state_labels"],
        "mu": result["mu"].tolist(),
        "sigma": result["sigma"].tolist(),
        "mu_annualized": (result["mu"] * 252).tolist(),
        "sigma_annualized": (result["sigma"] * np.sqrt(252)).tolist(),
        "trans_mat": result["trans_mat"].tolist(),
        "log_likelihood": float(result["log_likelihood"]),
        "n_params": n_params,
        "bic": float(-2 * result["log_likelihood"] + n_params * np.log(T)),
        "aic": float(-2 * result["log_likelihood"] + 2 * n_params),
        "n_iter": result["n_iter"],
        "converged": bool(result["converged"]),
        "regime_durations": {
            result["state_labels"][k]: float(1.0 / (1.0 - result["trans_mat"][k, k]))
            for k in range(K)
        },
    }


def run():
    """Main entry point for Module 2."""
    print("=" * 60)
    print("B2 Module 2: Fixed Transition Markov-Switching")
    print("=" * 60)

    # Load data
    returns_df = pd.read_parquet(DATA_DIR / "returns.parquet")
    returns = returns_df["log_return"].values
    dates = returns_df.index

    # ── Synthetic recovery tests ──
    print()
    print("[1/4] Synthetic recovery tests...")

    synth_2 = synthetic_recovery_test(K=2, n_reps=5, n_obs=2000)
    print(f"  2-state: mu recovery={synth_2['mu_recovery_rate']:.0%}, "
          f"sigma recovery={synth_2['sigma_recovery_rate']:.0%}")

    synth_3 = synthetic_recovery_test(K=3, n_reps=5, n_obs=2000)
    print(f"  3-state: mu recovery={synth_3['mu_recovery_rate']:.0%}, "
          f"sigma recovery={synth_3['sigma_recovery_rate']:.0%}")

    if synth_2["mu_recovery_rate"] < 0.7:
        print("  WARNING: 2-state mu recovery below 70%")
    if synth_3["mu_recovery_rate"] < 0.6:
        print("  WARNING: 3-state mu recovery below 60%")

    # ── Fit 2-state on real data ──
    print()
    print("[2/4] Fitting 2-state fixed MS on SPY returns...")
    result_2 = fit_with_restarts(returns, K=2)
    result_2 = label_states(result_2)

    info_2 = _serialize_result(result_2, K=2)
    print(f"  Converged: {result_2['converged']} (iter={result_2['n_iter']})")
    print(f"  Log-likelihood: {result_2['log_likelihood']:.2f}")
    print(f"  BIC: {info_2['bic']:.2f}")
    for k, label in enumerate(result_2["state_labels"]):
        dur = info_2["regime_durations"][label]
        print(f"  {label}: mu={info_2['mu_annualized'][k]:.4f} (ann), "
              f"sigma={info_2['sigma_annualized'][k]:.4f} (ann), "
              f"avg duration={dur:.1f} days")

    # ── Fit 3-state on real data ──
    print()
    print("[3/4] Fitting 3-state fixed MS on SPY returns...")
    result_3 = fit_with_restarts(returns, K=3)
    result_3 = label_states(result_3)

    info_3 = _serialize_result(result_3, K=3)
    print(f"  Converged: {result_3['converged']} (iter={result_3['n_iter']})")
    print(f"  Log-likelihood: {result_3['log_likelihood']:.2f}")
    print(f"  BIC: {info_3['bic']:.2f}")
    for k, label in enumerate(result_3["state_labels"]):
        dur = info_3["regime_durations"][label]
        print(f"  {label}: mu={info_3['mu_annualized'][k]:.4f} (ann), "
              f"sigma={info_3['sigma_annualized'][k]:.4f} (ann), "
              f"avg duration={dur:.1f} days")

    # ── Cross-validate against statsmodels ──
    print()
    print("[4/4] Cross-validating against statsmodels...")

    returns_series = returns_df["log_return"]

    cv_2 = cross_validate_statsmodels(returns_series, result_2, K=2)
    print(f"  2-state: LL diff={cv_2['ll_difference']:.2f}, "
          f"mu max diff={cv_2['mu_max_difference']:.6f}, "
          f"sigma max diff={cv_2['sigma_max_difference']:.6f}")
    print(f"    LL match: {cv_2['ll_match']}, Params match: {cv_2['params_match']}")

    cv_3 = cross_validate_statsmodels(returns_series, result_3, K=3)
    print(f"  3-state: LL diff={cv_3['ll_difference']:.2f}, "
          f"mu max diff={cv_3['mu_max_difference']:.6f}, "
          f"sigma max diff={cv_3['sigma_max_difference']:.6f}")
    print(f"    LL match: {cv_3['ll_match']}, Params match: {cv_3['params_match']}")

    # ── Save results ──
    print()
    print("Saving results...")

    # JSON results
    info_2["synthetic_test"] = synth_2
    info_2["cross_validation"] = cv_2
    with open(DATA_DIR / "ms2_fixed_results.json", "w") as f:
        json.dump(info_2, f, indent=2)

    info_3["synthetic_test"] = synth_3
    info_3["cross_validation"] = cv_3
    with open(DATA_DIR / "ms3_fixed_results.json", "w") as f:
        json.dump(info_3, f, indent=2)

    # Probability parquets
    for result, K, label in [(result_2, 2, "ms2"), (result_3, 3, "ms3")]:
        probs_df = pd.DataFrame(index=dates)
        for k in range(K):
            state_label = result["state_labels"][k]
            probs_df[f"filtered_{state_label}"] = result["filtered"][:, k]
            probs_df[f"smoothed_{state_label}"] = result["smoothed"][:, k]
        probs_df["regime"] = np.array(result["state_labels"])[
            result["smoothed"].argmax(axis=1)
        ]
        probs_df.to_parquet(DATA_DIR / f"{label}_fixed_probs.parquet")
        print(f"  {label}_fixed_probs.parquet: {len(probs_df)} rows")

    print(f"  ms2_fixed_results.json: saved")
    print(f"  ms3_fixed_results.json: saved")

    print()
    print("=" * 60)
    print("Module 2 COMPLETE")
    print("=" * 60)

    return result_2, result_3


if __name__ == "__main__":
    run()
