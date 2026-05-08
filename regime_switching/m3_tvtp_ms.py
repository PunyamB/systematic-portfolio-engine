"""
Module 3: TVTP Markov-Switching (Vectorized)
Time-Varying Transition Probability MS with logistic transition probabilities.
Custom Hamilton filter with time-varying transition matrices.
EM with numerical M-step (L-BFGS-B) for logistic coefficients.

Vectorized implementation: inner loops replaced with numpy operations
for 10-30x speedup over pure Python loops.

Inputs:
  - regime_switching/data/returns.parquet
  - regime_switching/data/covariates.parquet

Outputs:
  - regime_switching/data/ms2_tvtp_results.json
  - regime_switching/data/ms3_tvtp_results.json
  - regime_switching/data/ms2_tvtp_probs.parquet
  - regime_switching/data/ms3_tvtp_probs.parquet
  - regime_switching/data/tvtp_coefficients.json
"""
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from scipy.stats import norm
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize

from regime_switching.config import (
    DATA_DIR, EM_MAX_ITER, EM_TOL, FILTER_CLAMP,
    LOGISTIC_CLAMP, LBFGS_MAXITER, SYNTHETIC_N,
)


# ──────────────────────────────────────────────────────────
# Vectorized Logistic Transition Matrices
# ──────────────────────────────────────────────────────────

def compute_all_transition_matrices(Z, coeffs, K):
    """
    Vectorized: compute (T, K, K) transition matrices from covariates.

    Z: (T, d) standardized covariates
    coeffs: dict[int -> (K-1, 1+d) array] keyed by origin state
    Returns: (T, K, K) array
    """
    T, d = Z.shape
    Z_aug = np.column_stack([np.ones(T), Z])  # (T, 1+d)

    P_all = np.zeros((T, K, K))

    for i in range(K):
        # coeffs[i] is (K-1, 1+d) — one row per non-reference destination
        # Compute logits for all T at once: (T, K-1) = (T, 1+d) @ (1+d, K-1)
        logits_nonref = Z_aug @ coeffs[i].T  # (T, K-1)
        logits_nonref = np.clip(logits_nonref, -LOGISTIC_CLAMP, LOGISTIC_CLAMP)

        # Full logits: reference state 0 has logit=0
        logits_full = np.zeros((T, K))
        logits_full[:, 1:] = logits_nonref  # states 1..K-1

        # Softmax across destination states for each t
        # logsumexp along axis=1, then exp
        log_norm = logsumexp(logits_full, axis=1, keepdims=True)  # (T, 1)
        P_all[:, i, :] = np.exp(logits_full - log_norm)

    return P_all


# ──────────────────────────────────────────────────────────
# Vectorized Hamilton Filter
# ──────────────────────────────────────────────────────────

def tvtp_hamilton_filter(returns, mu, sigma, P_all, init_prob=None):
    """
    Hamilton filter with time-varying transition matrices.
    Emission densities vectorized. Filter loop is sequential (unavoidable).

    returns: (T,)
    mu: (K,)
    sigma: (K,)
    P_all: (T, K, K)
    """
    T = len(returns)
    K = len(mu)

    if init_prob is None:
        init_prob = np.ones(K) / K

    # Precompute ALL emission log-densities: (T, K)
    log_emissions = np.zeros((T, K))
    for k in range(K):
        log_emissions[:, k] = norm.logpdf(returns, loc=mu[k], scale=sigma[k])

    filtered = np.zeros((T, K))
    predicted = np.zeros((T, K))
    log_likelihood = 0.0

    # Filter loop (sequential — each step depends on previous)
    prev_filtered = init_prob.copy()

    for t in range(T):
        # Prediction: prev_filtered @ P_all[t]
        if t == 0:
            pred = init_prob.copy()
        else:
            pred = prev_filtered @ P_all[t]

        pred = np.maximum(pred, FILTER_CLAMP)
        pred /= pred.sum()
        predicted[t] = pred

        # Update
        log_joint = log_emissions[t] + np.log(pred)
        log_marginal = logsumexp(log_joint)
        log_likelihood += log_marginal

        filt = np.exp(log_joint - log_marginal)
        filt = np.maximum(filt, FILTER_CLAMP)
        filt /= filt.sum()
        filtered[t] = filt
        prev_filtered = filt

    return filtered, log_likelihood, predicted


# ──────────────────────────────────────────────────────────
# Vectorized Kim Smoother
# ──────────────────────────────────────────────────────────

def tvtp_smoother(filtered, predicted, P_all):
    """
    Kim (1994) backward smoother — vectorized per-timestep.

    Returns:
        smoothed: (T, K)
        xi: (T-1, K, K)
    """
    T, K = filtered.shape
    smoothed = np.zeros((T, K))
    xi = np.zeros((T - 1, K, K))

    smoothed[-1] = filtered[-1]

    for t in range(T - 2, -1, -1):
        # xi[t, i, j] = filtered[t,i] * P[t+1,i,j] * smoothed[t+1,j] / predicted[t+1,j]
        # Vectorized: outer product structure
        # numerator[i,j] = filtered[t,i] * P_all[t+1,i,j] * smoothed[t+1,j]
        pred_safe = np.maximum(predicted[t + 1], FILTER_CLAMP)
        ratio = smoothed[t + 1] / pred_safe  # (K,)

        # xi[t] = diag(filtered[t]) @ P_all[t+1] @ diag(ratio)
        xi_t = filtered[t, :, None] * P_all[t + 1] * ratio[None, :]  # (K, K)
        xi[t] = xi_t

        sm = xi_t.sum(axis=1)  # (K,)
        s_sum = sm.sum()
        if s_sum > 0:
            sm /= s_sum
        sm = np.maximum(sm, FILTER_CLAMP)
        sm /= sm.sum()
        smoothed[t] = sm

    return smoothed, xi


# ──────────────────────────────────────────────────────────
# Vectorized Logistic M-step
# ──────────────────────────────────────────────────────────

def _logistic_objective_and_grad(params_flat, Z_aug, xi_i, K):
    """
    Combined objective + gradient for origin state i.
    Vectorized over time.

    params_flat: ((K-1)*(1+d),)
    Z_aug: (T-1, 1+d)
    xi_i: (T-1, K) joint smoothed from state i to all states
    """
    T = Z_aug.shape[0]
    d_aug = Z_aug.shape[1]
    n_dest = K - 1
    params = params_flat.reshape(n_dest, d_aug)

    # Compute logits for all t: (T, K-1) = (T, d_aug) @ (d_aug, K-1)
    logits_nonref = Z_aug @ params.T  # (T, K-1)
    logits_nonref = np.clip(logits_nonref, -LOGISTIC_CLAMP, LOGISTIC_CLAMP)

    logits_full = np.zeros((T, K))
    logits_full[:, 1:] = logits_nonref

    # Log-probs and probs: (T, K)
    log_norm = logsumexp(logits_full, axis=1, keepdims=True)
    log_probs = logits_full - log_norm
    probs = np.exp(log_probs)

    # Objective: -sum_t sum_j xi_i[t,j] * log_probs[t,j]
    # Only where xi_i > tiny threshold
    mask = xi_i > 1e-15
    neg_Q = -np.sum(xi_i[mask] * log_probs[mask])

    # Gradient
    xi_sum = xi_i.sum(axis=1, keepdims=True)  # (T, 1)
    # For each non-ref destination j: grad_j = sum_t (probs[t,j] * xi_sum[t] - xi_i[t,j]) * Z_aug[t]
    residuals = probs[:, 1:] * xi_sum - xi_i[:, 1:]  # (T, K-1)
    grad = residuals.T @ Z_aug  # (K-1, d_aug)

    return neg_Q, grad.ravel()


def optimize_logistic_params(Z, xi, coeffs, K):
    """
    M-step: optimize logistic transition parameters for all origin states.
    Vectorized objective and gradient.
    """
    T_minus_1 = xi.shape[0]
    d = Z.shape[1]
    Z_aug = np.column_stack([np.ones(T_minus_1), Z[:T_minus_1]])

    new_coeffs = {}

    for i in range(K):
        x0 = coeffs[i].ravel()
        xi_i = xi[:, i, :]

        result = minimize(
            _logistic_objective_and_grad,
            x0,
            args=(Z_aug, xi_i, K),
            jac=True,  # function returns (obj, grad) together
            method="L-BFGS-B",
            options={"maxiter": LBFGS_MAXITER, "ftol": 1e-10},
        )

        new_coeffs[i] = result.x.reshape(K - 1, 1 + d)

    return new_coeffs


# ──────────────────────────────────────────────────────────
# TVTP EM Algorithm
# ──────────────────────────────────────────────────────────

def em_tvtp_ms(returns, Z, K, max_iter=EM_MAX_ITER, tol=EM_TOL, init_params=None):
    """
    EM algorithm for TVTP-MS model. Vectorized internals.
    """
    T = len(returns)
    d = Z.shape[1]

    if init_params is not None:
        mu = init_params["mu"].copy()
        sigma = init_params["sigma"].copy()
        coeffs = {i: init_params["coeffs"][i].copy() for i in range(K)}
    else:
        percentiles = np.linspace(10, 90, K + 2)[1:-1]
        mu = np.percentile(returns, percentiles)
        sigma = np.full(K, returns.std()) * np.linspace(0.5, 2.0, K)
        coeffs = {}
        for i in range(K):
            coeffs[i] = np.random.randn(K - 1, 1 + d) * 0.01
            for j_idx in range(K - 1):
                coeffs[i][j_idx, 0] = -3.0

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # E-step
        P_all = compute_all_transition_matrices(Z, coeffs, K)
        filtered, log_likelihood, predicted = tvtp_hamilton_filter(returns, mu, sigma, P_all)
        smoothed, xi = tvtp_smoother(filtered, predicted, P_all)

        # Convergence check
        if prev_ll > -np.inf:
            rel_change = abs(log_likelihood - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < tol:
                return {
                    "mu": mu, "sigma": sigma, "coeffs": coeffs,
                    "filtered": filtered, "smoothed": smoothed,
                    "P_all": P_all,
                    "log_likelihood": log_likelihood,
                    "n_iter": iteration + 1, "converged": True,
                }
        prev_ll = log_likelihood

        # M-step: emissions (vectorized)
        gamma = smoothed
        for k in range(K):
            w = gamma[:, k]
            w_sum = w.sum()
            if w_sum > 1e-10:
                mu[k] = np.dot(w, returns) / w_sum
                sigma[k] = np.sqrt(np.dot(w, (returns - mu[k]) ** 2) / w_sum)
                sigma[k] = max(sigma[k], 1e-6)

        # M-step: logistic transitions (vectorized objective + gradient)
        coeffs = optimize_logistic_params(Z, xi, coeffs, K)

    # Did not converge — run final pass
    P_all = compute_all_transition_matrices(Z, coeffs, K)
    filtered, log_likelihood, predicted = tvtp_hamilton_filter(returns, mu, sigma, P_all)
    smoothed, xi = tvtp_smoother(filtered, predicted, P_all)

    return {
        "mu": mu, "sigma": sigma, "coeffs": coeffs,
        "filtered": filtered, "smoothed": smoothed,
        "P_all": P_all,
        "log_likelihood": log_likelihood,
        "n_iter": max_iter, "converged": False,
    }


def fit_tvtp_with_restarts(returns, Z, K, n_restarts=8):
    """Fit TVTP-MS with multiple random restarts."""
    best_result = None
    best_ll = -np.inf
    d = Z.shape[1]

    for restart in range(n_restarts):
        t0 = time.time()

        if K == 2:
            mu_init = np.array([
                np.random.uniform(0.0001, 0.001),
                np.random.uniform(-0.001, -0.0001),
            ])
            sigma_init = np.array([
                np.random.uniform(0.005, 0.012),
                np.random.uniform(0.015, 0.035),
            ])
        else:
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
            result = em_tvtp_ms(returns, Z, K, init_params=init_params)
            elapsed = time.time() - t0
            print(f"    Restart {restart + 1}/{n_restarts}: "
                  f"LL={result['log_likelihood']:.2f}, "
                  f"iter={result['n_iter']}, "
                  f"conv={result['converged']}, "
                  f"time={elapsed:.1f}s")

            if result["log_likelihood"] > best_ll:
                best_ll = result["log_likelihood"]
                best_result = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    Restart {restart + 1}/{n_restarts}: FAILED ({e}), time={elapsed:.1f}s")
            continue

    if best_result is None:
        raise RuntimeError(f"All {n_restarts} TVTP restarts failed for K={K}")

    return best_result


def label_states(result):
    """Label states by mean return: highest mu = bull."""
    mu = result["mu"]
    K = len(mu)
    order = np.argsort(mu)[::-1]

    result["mu"] = mu[order]
    result["sigma"] = result["sigma"][order]
    result["filtered"] = result["filtered"][:, order]
    result["smoothed"] = result["smoothed"][:, order]
    if "P_all" in result:
        result["P_all"] = result["P_all"][:, order][:, :, order]

    new_coeffs = {}
    for new_i, old_i in enumerate(order):
        new_coeffs[new_i] = result["coeffs"][old_i]
    result["coeffs"] = new_coeffs

    if K == 2:
        result["state_labels"] = ["bull", "bear"]
    else:
        result["state_labels"] = ["bull", "bear", "crisis"]

    return result


def compute_covariate_significance(coeffs, K, covariate_names):
    """Report logistic coefficients per transition."""
    results = {}
    for i in range(K):
        state_results = {}
        for j_idx in range(K - 1):
            j = j_idx + 1
            row = coeffs[i][j_idx]
            transition_name = f"state{i}_to_state{j}"
            coefs = {"intercept": float(row[0])}
            for c_idx, cname in enumerate(covariate_names):
                coefs[cname] = float(row[1 + c_idx])
            state_results[transition_name] = coefs
        results[f"from_state_{i}"] = state_results
    return results


def _serialize_result(result, K, T):
    """Convert result to JSON-serializable format."""
    d_covariate = result["coeffs"][0].shape[1] - 1
    n_emission = 2 * K
    n_transition = K * (K - 1) * (1 + d_covariate)
    n_params = n_emission + n_transition
    avg_P = result["P_all"].mean(axis=0)

    return {
        "K": K,
        "state_labels": result["state_labels"],
        "mu": result["mu"].tolist(),
        "sigma": result["sigma"].tolist(),
        "mu_annualized": (result["mu"] * 252).tolist(),
        "sigma_annualized": (result["sigma"] * np.sqrt(252)).tolist(),
        "avg_transition_matrix": avg_P.tolist(),
        "log_likelihood": float(result["log_likelihood"]),
        "n_params": n_params,
        "bic": float(-2 * result["log_likelihood"] + n_params * np.log(T)),
        "aic": float(-2 * result["log_likelihood"] + 2 * n_params),
        "n_iter": result["n_iter"],
        "converged": bool(result["converged"]),
        "regime_durations": {
            result["state_labels"][k]: float(1.0 / (1.0 - avg_P[k, k]))
            for k in range(K)
        },
    }


# ──────────────────────────────────────────────────────────
# Synthetic Test
# ──────────────────────────────────────────────────────────

def synthetic_tvtp_test(n_obs=2000):
    """Quick synthetic test: 2-state TVTP with 1 covariate."""
    K = 2
    true_mu = np.array([0.0005, -0.0005])
    true_sigma = np.array([0.008, 0.020])
    true_coeffs = {
        0: np.array([[-3.0, 0.5]]),
        1: np.array([[-2.5, -0.3]]),
    }

    z = np.random.randn(n_obs, 1)
    states = np.zeros(n_obs, dtype=int)
    returns = np.zeros(n_obs)
    states[0] = 0
    returns[0] = np.random.normal(true_mu[0], true_sigma[0])

    for t in range(1, n_obs):
        z_aug = np.array([1.0, z[t, 0]])
        logits = np.zeros(K)
        logits[1] = np.clip(true_coeffs[states[t-1]][0] @ z_aug, -LOGISTIC_CLAMP, LOGISTIC_CLAMP)
        probs = softmax(logits)
        states[t] = np.random.choice(K, p=probs)
        returns[t] = np.random.normal(true_mu[states[t]], true_sigma[states[t]])

    result = fit_tvtp_with_restarts(returns, z, K, n_restarts=3)
    result = label_states(result)

    return {
        "converged": bool(result["converged"]),
        "log_likelihood": float(result["log_likelihood"]),
        "mu_fitted": result["mu"].tolist(),
        "mu_true": true_mu.tolist(),
        "mu_recovery": bool(all(abs(result["mu"][k] - true_mu[k]) < 0.001 for k in range(K))),
    }


def run():
    """Main entry point."""
    print("=" * 60)
    print("B2 Module 3: TVTP Markov-Switching (Vectorized)")
    print("=" * 60)

    returns_df = pd.read_parquet(DATA_DIR / "returns.parquet")
    covariates_df = pd.read_parquet(DATA_DIR / "covariates.parquet")
    returns = returns_df["log_return"].values
    Z = covariates_df.values
    dates = returns_df.index
    T = len(returns)
    covariate_names = list(covariates_df.columns)

    print(f"  Data: {T} observations, {Z.shape[1]} covariates")
    print()

    # Synthetic test
    print("[1/3] Synthetic TVTP test (2-state, 1 covariate)...")
    t0 = time.time()
    synth = synthetic_tvtp_test()
    print(f"  Result: converged={synth['converged']}, mu_recovery={synth['mu_recovery']}, time={time.time()-t0:.1f}s")
    print()

    # 2-state TVTP
    print("[2/3] Fitting 2-state TVTP (8 restarts)...")
    t0 = time.time()
    result_2 = fit_tvtp_with_restarts(returns, Z, K=2, n_restarts=8)
    result_2 = label_states(result_2)
    info_2 = _serialize_result(result_2, K=2, T=T)
    print(f"  DONE in {time.time()-t0:.0f}s | LL={info_2['log_likelihood']:.2f} | BIC={info_2['bic']:.2f}")
    for k, label in enumerate(result_2["state_labels"]):
        print(f"  {label}: mu={info_2['mu_annualized'][k]:.4f}, "
              f"sigma={info_2['sigma_annualized'][k]:.4f}, "
              f"dur={info_2['regime_durations'][label]:.1f}d")
    print()

    # 3-state TVTP
    print("[3/3] Fitting 3-state TVTP (8 restarts)...")
    t0 = time.time()
    result_3 = fit_tvtp_with_restarts(returns, Z, K=3, n_restarts=8)
    result_3 = label_states(result_3)
    info_3 = _serialize_result(result_3, K=3, T=T)
    print(f"  DONE in {time.time()-t0:.0f}s | LL={info_3['log_likelihood']:.2f} | BIC={info_3['bic']:.2f}")
    for k, label in enumerate(result_3["state_labels"]):
        print(f"  {label}: mu={info_3['mu_annualized'][k]:.4f}, "
              f"sigma={info_3['sigma_annualized'][k]:.4f}, "
              f"dur={info_3['regime_durations'][label]:.1f}d")
    print()

    # Covariate significance
    sig_2 = compute_covariate_significance(result_2["coeffs"], K=2, covariate_names=covariate_names)
    sig_3 = compute_covariate_significance(result_3["coeffs"], K=3, covariate_names=covariate_names)

    print("2-state TVTP coefficients:")
    for fs, trans in sig_2.items():
        for tn, coefs in trans.items():
            print(f"  {tn}: " + ", ".join(f"{k}={v:.4f}" for k, v in coefs.items()))

    print("3-state TVTP coefficients:")
    for fs, trans in sig_3.items():
        for tn, coefs in trans.items():
            print(f"  {tn}: " + ", ".join(f"{k}={v:.4f}" for k, v in coefs.items()))
    print()

    # Save
    print("Saving...")
    info_2["synthetic_test"] = synth
    info_2["covariate_significance"] = sig_2
    with open(DATA_DIR / "ms2_tvtp_results.json", "w") as f:
        json.dump(info_2, f, indent=2)

    info_3["covariate_significance"] = sig_3
    with open(DATA_DIR / "ms3_tvtp_results.json", "w") as f:
        json.dump(info_3, f, indent=2)

    with open(DATA_DIR / "tvtp_coefficients.json", "w") as f:
        json.dump({"2_state": sig_2, "3_state": sig_3}, f, indent=2)

    for result, K, label in [(result_2, 2, "ms2"), (result_3, 3, "ms3")]:
        probs_df = pd.DataFrame(index=dates)
        for k in range(K):
            sl = result["state_labels"][k]
            probs_df[f"filtered_{sl}"] = result["filtered"][:, k]
            probs_df[f"smoothed_{sl}"] = result["smoothed"][:, k]
        probs_df["regime"] = np.array(result["state_labels"])[
            result["smoothed"].argmax(axis=1)
        ]
        probs_df.to_parquet(DATA_DIR / f"{label}_tvtp_probs.parquet")
        print(f"  {label}_tvtp_probs.parquet: {len(probs_df)} rows")

    print("  All JSON files saved")
    print()
    print("=" * 60)
    print("Module 3 COMPLETE")
    print("=" * 60)

    return result_2, result_3


if __name__ == "__main__":
    run()
