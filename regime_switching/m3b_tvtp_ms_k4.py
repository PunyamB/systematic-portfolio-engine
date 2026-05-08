"""
M3b: K=4 TVTP Markov-Switching (per-window walk-forward fits)

Same architecture as M3 (K=3 TVTP), extended to K=4 states.
Custom Hamilton filter with K x K time-varying transition matrix.
EM with logistic M-step for transition coefficients.

States (sorted by mu after fitting): crisis / bear / recovery / bull
Covariates: VIX, yield curve slope, credit spread (same as K=3, apples-to-apples)

Per-window walk-forward: 18 windows (1-17 + holdout), expanding training data.
For each window, fit on training period, apply forward through filter on test period.

Outputs saved to regime_switching/data/k4_extension/window_fits/window_X/:
  - tvtp_result.pkl + .json     - Best of 8 restarts
  - filtered_probs.parquet       - Causal probabilities (used for backtest)
  - smoothed_probs.parquet       - Uses future info (analysis only)
  - covariate_stats.json         - Training mean/std for standardization
  - em_convergence_log.json      - LL traces across all restarts
  - all_restarts.pkl             - All 8 restart results
"""

import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============== CONFIGURATION ==============
N_STATES = 4
N_RESTARTS = 8
EM_MAX_ITER = 700
EM_TOL = 1e-6
FILTER_CLAMP = 1e-10
LBFGS_MAXITER = 100
SEED = 42

DATA = Path('regime_switching/data')
OUT = DATA / 'k4_extension' / 'window_fits'
OUT.mkdir(parents=True, exist_ok=True)

# Walk-forward windows (must match EXP006 / B2)
# Each window has training_start, training_end, test_start, test_end
# Training is expanding (always starts from 1997)
WINDOWS = [
    {'id': 1,  'train': ('1997-01-02', '2004-12-31'), 'test': ('2005-01-03', '2005-12-30')},
    {'id': 2,  'train': ('1997-01-02', '2005-12-30'), 'test': ('2006-01-03', '2006-12-29')},
    {'id': 3,  'train': ('1997-01-02', '2006-12-29'), 'test': ('2007-01-03', '2007-12-28')},
    {'id': 4,  'train': ('1997-01-02', '2007-12-28'), 'test': ('2008-01-02', '2008-12-30')},
    {'id': 5,  'train': ('1997-01-02', '2008-12-30'), 'test': ('2009-01-02', '2009-12-30')},
    {'id': 6,  'train': ('1997-01-02', '2009-12-30'), 'test': ('2010-01-04', '2010-12-30')},
    {'id': 7,  'train': ('1997-01-02', '2010-12-30'), 'test': ('2011-01-03', '2011-12-30')},
    {'id': 8,  'train': ('1997-01-02', '2011-12-30'), 'test': ('2012-01-03', '2012-12-28')},
    {'id': 9,  'train': ('1997-01-02', '2012-12-28'), 'test': ('2013-01-02', '2013-12-30')},
    {'id': 10, 'train': ('1997-01-02', '2013-12-30'), 'test': ('2014-01-02', '2014-12-30')},
    {'id': 11, 'train': ('1997-01-02', '2014-12-30'), 'test': ('2015-01-02', '2015-12-30')},
    {'id': 12, 'train': ('1997-01-02', '2015-12-30'), 'test': ('2016-01-04', '2016-12-30')},
    {'id': 13, 'train': ('1997-01-02', '2016-12-30'), 'test': ('2017-01-03', '2017-12-29')},
    {'id': 14, 'train': ('1997-01-02', '2017-12-29'), 'test': ('2018-01-02', '2018-12-28')},
    {'id': 15, 'train': ('1997-01-02', '2018-12-28'), 'test': ('2019-01-02', '2019-12-30')},
    {'id': 16, 'train': ('1997-01-02', '2019-12-30'), 'test': ('2020-01-02', '2020-12-30')},
    {'id': 17, 'train': ('1997-01-02', '2020-12-30'), 'test': ('2021-01-04', '2021-12-30')},
    {'id': 'holdout', 'train': ('1997-01-02', '2021-12-30'), 'test': ('2022-01-03', '2026-03-06')},
]

COVARIATE_COLS = ['vix_level', 'yield_curve_slope', 'credit_spread']

# ============== TVTP COMPONENTS ==============

def tvtp_transition_matrix(z_t, intercepts, coefficients):
    """
    Compute time-varying transition matrix at time t given covariates z_t.
    
    intercepts: shape (K, K-1) -- a_ij for i in 0..K-1, j in 1..K-1 (j=0 is reference)
    coefficients: shape (K, K-1, d) -- b_ij for each origin state, destination, covariate
    z_t: shape (d,) -- covariates at time t
    
    Returns: transition matrix P_t of shape (K, K), row i sums to 1
    """
    K = intercepts.shape[0]
    P_t = np.zeros((K, K))
    for i in range(K):
        # Compute logits for destinations (j=0 is reference, logit=0)
        logits = np.zeros(K)
        for j_idx in range(K - 1):
            j = j_idx + 1  # destinations 1..K-1
            logits[j] = intercepts[i, j_idx] + coefficients[i, j_idx] @ z_t
        # Clip to prevent overflow
        logits = np.clip(logits, -20, 20)
        P_t[i] = softmax(logits)
    return P_t

def tvtp_transition_matrices_vectorized(Z, intercepts, coefficients):
    """Vectorized: compute all P_t for t=1..T at once."""
    K = intercepts.shape[0]
    T_z, d = Z.shape
    # logits shape: (T, K, K), [t, i, j] = a_ij + b_ij @ z_t (with j=0 reference = 0)
    logits = np.zeros((T_z, K, K))
    for i in range(K):
        for j_idx in range(K - 1):
            j = j_idx + 1
            logits[:, i, j] = intercepts[i, j_idx] + Z @ coefficients[i, j_idx]
    logits = np.clip(logits, -20, 20)
    # Softmax along axis=2 (destinations)
    logits_max = logits.max(axis=2, keepdims=True)
    exp = np.exp(logits - logits_max)
    P_t = exp / exp.sum(axis=2, keepdims=True)
    return P_t  # shape (T, K, K)

def hamilton_filter_tvtp(r, Z, mu, sigma, intercepts, coefficients, pi0):
    """
    Forward filter with time-varying transitions.
    Z[t] is covariate at time t. P_t = tvtp(z_t).
    Returns log-likelihood and filtered probabilities.
    """
    T = len(r)
    K = len(mu)
    
    # Pre-compute all transition matrices (vectorized, fast)
    P_all = tvtp_transition_matrices_vectorized(Z, intercepts, coefficients)
    
    log_alpha = np.zeros((T, K))
    log_emit = norm.logpdf(r[0], loc=mu, scale=sigma)
    log_alpha[0] = np.log(np.maximum(pi0, FILTER_CLAMP)) + log_emit
    log_norm = logsumexp(log_alpha[0])
    log_alpha[0] -= log_norm
    log_lik = log_norm
    
    for t in range(1, T):
        P_t = P_all[t]
        log_P_t = np.log(np.maximum(P_t, FILTER_CLAMP))
        log_pred = logsumexp(log_alpha[t-1][:, None] + log_P_t, axis=0)
        log_emit = norm.logpdf(r[t], loc=mu, scale=sigma)
        log_alpha[t] = log_pred + log_emit
        log_norm = logsumexp(log_alpha[t])
        log_alpha[t] -= log_norm
        log_lik += log_norm
    
    return log_lik, np.exp(log_alpha), P_all

def kim_smoother_tvtp(filtered, P_all):
    """Backward smoother with time-varying P. Returns smoothed and joint xi."""
    T, K = filtered.shape
    smoothed = np.zeros_like(filtered)
    smoothed[-1] = filtered[-1]
    xi = np.zeros((T-1, K, K))
    
    for t in range(T-2, -1, -1):
        P_t1 = P_all[t+1]  # transition from t to t+1
        pred = filtered[t] @ P_t1
        pred = np.maximum(pred, FILTER_CLAMP)
        smoothed[t] = filtered[t] * (P_t1 @ (smoothed[t+1] / pred))
        smoothed[t] /= smoothed[t].sum()
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = filtered[t, i] * P_t1[i, j] * smoothed[t+1, j] / pred[j]
        xi[t] /= xi[t].sum()
    
    return smoothed, xi

def m_step_logistic(xi, Z, K, init_intercepts, init_coefficients):
    """
    M-step for logistic transition parameters.
    Maximize Q = sum_t sum_i sum_j xi[t,i,j] * log P(j | i, z_t)
    Decomposes by origin state i.
    """
    T_minus_1 = xi.shape[0]
    d = Z.shape[1]
    
    new_intercepts = init_intercepts.copy()
    new_coefficients = init_coefficients.copy()
    
    for i in range(K):
        # Optimize over (a_ij, b_ij) for j=1..K-1
        # Pack params as: [a_i1, b_i1[0..d-1], a_i2, b_i2[0..d-1], a_i3, b_i3[0..d-1]]
        # j=0 is reference (a_i0 = 0, b_i0 = 0)
        
        n_params_per_dest = 1 + d
        n_total_params = (K - 1) * n_params_per_dest
        
        x0 = np.zeros(n_total_params)
        for j_idx in range(K - 1):
            x0[j_idx * n_params_per_dest] = init_intercepts[i, j_idx]
            x0[j_idx * n_params_per_dest + 1: (j_idx + 1) * n_params_per_dest] = init_coefficients[i, j_idx]
        
        # xi[t, i, :] gives expected transitions from i at time t (note: xi indexed t from 0..T-2 but corresponds to t+1)
        xi_i = xi[:, i, :]  # shape (T-1, K)
        Z_used = Z[1:1+T_minus_1]  # covariates at t+1 (the destination time)
        
        def neg_q(params):
            intercepts_i = np.zeros(K - 1)
            coeffs_i = np.zeros((K - 1, d))
            for j_idx in range(K - 1):
                intercepts_i[j_idx] = params[j_idx * n_params_per_dest]
                coeffs_i[j_idx] = params[j_idx * n_params_per_dest + 1: (j_idx + 1) * n_params_per_dest]
            
            # Compute log P(j | i, z_t) for all t
            logits = np.zeros((T_minus_1, K))
            for j_idx in range(K - 1):
                j = j_idx + 1
                logits[:, j] = intercepts_i[j_idx] + Z_used @ coeffs_i[j_idx]
            logits = np.clip(logits, -20, 20)
            log_P = logits - logsumexp(logits, axis=1, keepdims=True)
            
            # Negative Q for this origin state
            return -np.sum(xi_i * log_P)
        
        try:
            res = minimize(neg_q, x0, method='L-BFGS-B', options={'maxiter': LBFGS_MAXITER})
            opt_params = res.x
        except Exception:
            opt_params = x0  # fall back to previous
        
        for j_idx in range(K - 1):
            new_intercepts[i, j_idx] = opt_params[j_idx * n_params_per_dest]
            new_coefficients[i, j_idx] = opt_params[j_idx * n_params_per_dest + 1: (j_idx + 1) * n_params_per_dest]
    
    return new_intercepts, new_coefficients

def em_fit_tvtp_k4(r, Z, n_iter, tol, seed):
    """EM for TVTP-MS with K=4 states."""
    rng = np.random.default_rng(seed)
    K = N_STATES
    d = Z.shape[1]
    T = len(r)
    
    # Initialize emissions: quantile-based + noise
    quantiles = np.quantile(r, np.linspace(0.1, 0.9, K))
    mu = np.sort(quantiles + rng.normal(0, 0.0003, K))
    sigma = np.abs(rng.normal(r.std(), r.std() * 0.3, K))
    sigma = np.sort(sigma)
    
    # Initialize logistic params: intercepts shape (K, K-1), coefficients shape (K, K-1, d)
    # Set intercepts so diagonal of P is high (~0.95)
    intercepts = rng.normal(0, 0.1, (K, K - 1))
    coefficients = rng.normal(0, 0.05, (K, K - 1, d))
    
    # Bias intercepts so P(stay) >= P(switch). For state i, j=0 is reference.
    # If i==0: stay means j=0, so we want a_0j (j>=1) to be negative. Set a_0j = -3.
    # If i>=1: stay means j=i, so a_ii should be high. Set a_i,(i)_idx where i_idx is index of i in destinations.
    for i in range(K):
        for j_idx in range(K - 1):
            j = j_idx + 1
            if i == 0:
                # j=0 is reference (stay), j>=1 is switch
                intercepts[i, j_idx] = -3.0 + rng.normal(0, 0.2)
            elif j == i:
                # Stay
                intercepts[i, j_idx] = 3.0 + rng.normal(0, 0.2)
            else:
                # Switch to a different non-reference state
                intercepts[i, j_idx] = -1.0 + rng.normal(0, 0.2)
    
    pi0 = np.ones(K) / K
    
    ll_history = []
    prev_ll = -np.inf
    converged = False
    
    for it in range(n_iter):
        # E-step
        ll, filtered, P_all = hamilton_filter_tvtp(r, Z, mu, sigma, intercepts, coefficients, pi0)
        smoothed, xi = kim_smoother_tvtp(filtered, P_all)
        ll_history.append(ll)
        
        # M-step: emissions
        gamma = smoothed
        Nk = gamma.sum(axis=0)
        mu = (gamma * r[:, None]).sum(axis=0) / Nk
        diff_sq = (r[:, None] - mu) ** 2
        sigma = np.sqrt((gamma * diff_sq).sum(axis=0) / Nk)
        sigma = np.maximum(sigma, 1e-5)
        
        # M-step: TVTP transition coefficients (numerical optimization)
        intercepts, coefficients = m_step_logistic(xi, Z, K, intercepts, coefficients)
        
        pi0 = smoothed[0]
        
        if it > 0 and abs(ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol:
            converged = True
            break
        prev_ll = ll
    
    # Final pass with final params
    ll_final, filtered_final, P_all_final = hamilton_filter_tvtp(r, Z, mu, sigma, intercepts, coefficients, pi0)
    smoothed_final, _ = kim_smoother_tvtp(filtered_final, P_all_final)
    
    return {
        'mu': mu, 'sigma': sigma,
        'intercepts': intercepts, 'coefficients': coefficients,
        'pi0': pi0,
        'log_likelihood': ll_final,
        'filtered': filtered_final,
        'smoothed': smoothed_final,
        'P_all': P_all_final,
        'n_iter': it + 1,
        'converged': converged,
        'll_history': ll_history,
    }

def causal_filter_only(r, Z, mu, sigma, intercepts, coefficients, pi0):
    """Apply Hamilton filter forward (causal) with frozen params. For test-period probabilities."""
    ll, filtered, P_all = hamilton_filter_tvtp(r, Z, mu, sigma, intercepts, coefficients, pi0)
    return filtered

# ============== MAIN: PER-WINDOW FITS ==============

print('=' * 70)
print('M3b: K=4 TVTP MARKOV-SWITCHING (per-window walk-forward)')
print('=' * 70)
print(f'States: {N_STATES}, Restarts: {N_RESTARTS}, Max iter: {EM_MAX_ITER}')
print(f'Covariates: {COVARIATE_COLS}')
print(f'Windows: {len(WINDOWS)}')
print()

# Load full series
returns = pd.read_parquet(DATA / 'returns.parquet')
returns.index = pd.to_datetime(returns.index)
covariates_raw = pd.read_parquet(DATA / 'covariates_raw.parquet')
covariates_raw.index = pd.to_datetime(covariates_raw.index)

# Align
common_idx = returns.index.intersection(covariates_raw.index)
returns = returns.loc[common_idx]
covariates_raw = covariates_raw.loc[common_idx]
r_full = returns['log_return'].values if 'log_return' in returns.columns else returns.iloc[:, 0].values

print(f'Full sample: {len(r_full)} obs, {returns.index[0].date()} to {returns.index[-1].date()}')
print()

# State labels by ascending mu
STATE_LABELS = ['crisis', 'bear', 'recovery', 'bull']

overall_start = time.time()

for w in WINDOWS:
    wid = w['id']
    win_dir = OUT / f'window_{wid}'
    win_dir.mkdir(exist_ok=True)
    
    # Skip if already done
    if (win_dir / 'tvtp_result.json').exists():
        print(f'Window {wid}: already complete, skipping')
        continue
    
    print(f'Window {wid}: train {w["train"][0]} to {w["train"][1]}, test {w["test"][0]} to {w["test"][1]}')
    win_start_time = time.time()
    
    train_start, train_end = pd.Timestamp(w['train'][0]), pd.Timestamp(w['train'][1])
    test_start, test_end = pd.Timestamp(w['test'][0]), pd.Timestamp(w['test'][1])
    
    train_mask = (returns.index >= train_start) & (returns.index <= train_end)
    full_mask = (returns.index >= train_start) & (returns.index <= test_end)
    
    r_train = r_full[train_mask]
    r_full_window = r_full[full_mask]
    full_idx = returns.index[full_mask]
    
    # Standardize covariates using TRAINING stats only
    cov_train = covariates_raw.loc[train_mask, COVARIATE_COLS]
    cov_mean = cov_train.mean()
    cov_std = cov_train.std()
    cov_standardized = (covariates_raw[COVARIATE_COLS] - cov_mean) / cov_std
    Z_train = cov_standardized.loc[train_mask].values
    Z_full = cov_standardized.loc[full_mask].values
    assert len(Z_train) == len(r_train), f'Z_train len {len(Z_train)} != r_train len {len(r_train)}'
    assert len(Z_full) == len(r_full_window), f'Z_full len {len(Z_full)} != r_full_window len {len(r_full_window)}'
    
    # Save covariate stats
    with open(win_dir / 'covariate_stats.json', 'w') as f:
        json.dump({
            'mean': cov_mean.to_dict(),
            'std': cov_std.to_dict(),
            'n_train_obs': int(len(r_train)),
        }, f, indent=2)
    
    # Run restarts
    all_restarts = []
    for restart in range(N_RESTARTS):
        seed = SEED + restart * 17
        print(f'  Restart {restart+1}/{N_RESTARTS} (seed {seed})... ', end='', flush=True)
        try:
            t0 = time.time()
            result = em_fit_tvtp_k4(r_train, Z_train, EM_MAX_ITER, EM_TOL, seed)
            result['seed'] = seed
            result['restart_idx'] = restart
            result['runtime_sec'] = time.time() - t0
            all_restarts.append(result)
            print(f'LL={result["log_likelihood"]:.2f}, iter={result["n_iter"]}, conv={result["converged"]}, time={result["runtime_sec"]:.0f}s')
        except Exception as e:
            print(f'FAILED: {e}')
    
    if not all_restarts:
        print(f'  Window {wid}: ALL RESTARTS FAILED, skipping')
        continue
    
    # Best
    best = max(all_restarts, key=lambda x: x['log_likelihood'])
    
    # Sort states by mu
    order = np.argsort(best['mu'])
    mu_sorted = best['mu'][order]
    sigma_sorted = best['sigma'][order]
    intercepts_sorted = best['intercepts'][order]
    coefficients_sorted = best['coefficients'][order]
    pi0_sorted = best['pi0'][order]
    
    # Reorder intercepts/coefficients along destination axis too
    # Original: intercepts[i, j_idx] where j_idx in 0..K-2 corresponds to dest j=1..K-1
    # After sorting i, we also need to sort destinations.
    # The destination space is parameterized with j=0 as reference. After sorting,
    # the new state ordering may have a different reference. To keep things simple,
    # we re-derive the full transition matrix and sort that, accepting that we lose
    # the (intercept, coefficient) representation but keep the computed probabilities.
    # For inference (filter forward), we need P_t. Let's just store reordered P_all
    # and skip storing the logistic params after sorting.
    
    P_all_sorted = best['P_all'][:, order, :][:, :, order]
    filtered_sorted = best['filtered'][:, order]
    smoothed_sorted = best['smoothed'][:, order]
    
    # Apply forward filter to test period using the FROZEN training-period params
    # We need to extend P_all to cover test period using same logistic params (frozen)
    # Re-compute P_all over full window using training-period intercepts/coefficients
    # (Note: we can't use the sorted ones because ordering of destinations matters for the logistic.
    #  We use the original best params and sort the OUTPUT probabilities afterward.)
    
    # Run causal filter on full window with frozen (unsorted) params
    filtered_full = causal_filter_only(
        r_full_window, Z_full,
        best['mu'], best['sigma'],
        best['intercepts'], best['coefficients'],
        best['pi0']
    )
    # Then reorder columns by mu sort
    filtered_full_sorted = filtered_full[:, order]
    
    # Smoothed full: also do that
    _, _, P_all_full = hamilton_filter_tvtp(
        r_full_window, Z_full,
        best['mu'], best['sigma'],
        best['intercepts'], best['coefficients'],
        best['pi0']
    )
    smoothed_full, _ = kim_smoother_tvtp(filtered_full, P_all_full)
    smoothed_full_sorted = smoothed_full[:, order]
    
    # Save filtered + smoothed probs over FULL window (training + test)
    probs_full_df = pd.DataFrame(
        filtered_full_sorted,
        index=full_idx,
        columns=[f'p_{l}' for l in STATE_LABELS]
    )
    probs_full_df.to_parquet(win_dir / 'filtered_probs.parquet')
    
    smoothed_full_df = pd.DataFrame(
        smoothed_full_sorted,
        index=full_idx,
        columns=[f'p_{l}_smoothed' for l in STATE_LABELS]
    )
    smoothed_full_df.to_parquet(win_dir / 'smoothed_probs.parquet')
    
    # Result JSON (params)
    occupancies = [smoothed_sorted[:, i].mean() for i in range(N_STATES)]
    result_summary = {
        'window_id': str(wid),
        'n_states': N_STATES,
        'state_labels': STATE_LABELS,
        'best_seed': int(best['seed']),
        'best_restart_idx': int(best['restart_idx']),
        'log_likelihood_train': float(best['log_likelihood']),
        'n_iter_best': int(best['n_iter']),
        'converged_best': bool(best['converged']),
        'mu_daily': mu_sorted.tolist(),
        'mu_annual': (mu_sorted * 252).tolist(),
        'sigma_daily': sigma_sorted.tolist(),
        'sigma_annual': (sigma_sorted * np.sqrt(252)).tolist(),
        'pi0': pi0_sorted.tolist(),
        'training_occupancy': occupancies,
        'n_train_obs': int(len(r_train)),
        'n_test_obs': int(((returns.index >= test_start) & (returns.index <= test_end)).sum()),
    }
    with open(win_dir / 'tvtp_result.json', 'w') as f:
        json.dump(result_summary, f, indent=2)
    
    # Pickle full result (with raw arrays for later analysis)
    pickle_payload = {
        'window_id': wid,
        'state_labels': STATE_LABELS,
        'mu_unsorted': best['mu'],
        'sigma_unsorted': best['sigma'],
        'intercepts_unsorted': best['intercepts'],
        'coefficients_unsorted': best['coefficients'],
        'pi0_unsorted': best['pi0'],
        'order_to_sort': order,
        'mu_sorted': mu_sorted,
        'sigma_sorted': sigma_sorted,
        'log_likelihood': best['log_likelihood'],
        'n_iter': best['n_iter'],
        'converged': best['converged'],
        'covariate_cols': COVARIATE_COLS,
        'cov_mean': cov_mean.values,
        'cov_std': cov_std.values,
    }
    with open(win_dir / 'tvtp_result.pkl', 'wb') as f:
        pickle.dump(pickle_payload, f)
    
    # Convergence log
    conv_log = {
        'restarts': [
            {
                'idx': int(rs['restart_idx']),
                'seed': int(rs['seed']),
                'final_ll': float(rs['log_likelihood']),
                'n_iter': int(rs['n_iter']),
                'converged': bool(rs['converged']),
                'runtime_sec': float(rs.get('runtime_sec', 0)),
                'll_history': [float(x) for x in rs['ll_history']],
            }
            for rs in all_restarts
        ]
    }
    with open(win_dir / 'em_convergence_log.json', 'w') as f:
        json.dump(conv_log, f, indent=2)
    
    # All restarts pickle
    with open(win_dir / 'all_restarts.pkl', 'wb') as f:
        pickle.dump(all_restarts, f)
    
    win_elapsed = time.time() - win_start_time
    print(f'  Window {wid} complete in {win_elapsed/60:.1f} min')
    print(f'  States (annualized): mu={[f"{m*252*100:.1f}%" for m in mu_sorted]}, sigma={[f"{s*np.sqrt(252)*100:.1f}%" for s in sigma_sorted]}')
    print(f'  Training occupancy: {[f"{o*100:.1f}%" for o in occupancies]}')
    print()

total_elapsed = time.time() - overall_start
print('=' * 70)
print(f'ALL WINDOWS COMPLETE in {total_elapsed/3600:.2f} hours')
print(f'Outputs in: {OUT}')
print('=' * 70)

