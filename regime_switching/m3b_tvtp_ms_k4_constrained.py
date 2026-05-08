"""
M3b-Constrained: K=4 TVTP MS with economic constraints

Same architecture as m3b_tvtp_ms_k4.py with three additions:
  1. mu bounded to [-100%, +100%] annualized (clip during M-step)
  2. Diagonal transition probabilities pushed toward >= 0.90 via penalty
  3. Initialization seeded from M2b K=4 fixed-MS parameters (not random)

Outputs to regime_switching/data/k4_extension_constrained/window_fits/window_X/
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

# Constraints
MU_ANNUAL_BOUND = 1.0          # +/- 100% annualized
MU_DAILY_BOUND = MU_ANNUAL_BOUND / 252
MIN_DIAGONAL_PROB = 0.90        # P(stay) >= 0.90
DIAGONAL_PENALTY_WEIGHT = 50.0  # Penalty strength for low diagonals

DATA = Path('regime_switching/data')
OUT = DATA / 'k4_extension_constrained' / 'window_fits'
OUT.mkdir(parents=True, exist_ok=True)

# Load M2b K=4 reference parameters for initialization
M2B_PATH = DATA / 'm2b_k4_diagnostic' / 'ms4_fixed_results.json'
if not M2B_PATH.exists():
    raise FileNotFoundError(f'M2b results not found at {M2B_PATH}. Run m2b_fixed_ms_k4.py first.')

with open(M2B_PATH) as f:
    m2b = json.load(f)
M2B_MU_DAILY = np.array(m2b['mu_daily'])         # crisis, bear, recovery, bull (sorted by mu)
M2B_SIGMA_DAILY = np.array(m2b['sigma_daily'])
M2B_TRANSITION = np.array(m2b['transition_matrix'])

print(f'M2b K=4 reference (sorted by mu):')
print(f'  mu (annual): {[f"{m*252*100:.1f}%" for m in M2B_MU_DAILY]}')
print(f'  sigma (annual): {[f"{s*np.sqrt(252)*100:.1f}%" for s in M2B_SIGMA_DAILY]}')
print()

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
STATE_LABELS = ['crisis', 'bear', 'recovery', 'bull']

# ============== TVTP COMPONENTS ==============

def tvtp_transition_matrices_vectorized(Z, intercepts, coefficients):
    K = intercepts.shape[0]
    T_z, d = Z.shape
    logits = np.zeros((T_z, K, K))
    for i in range(K):
        for j_idx in range(K - 1):
            j = j_idx + 1
            logits[:, i, j] = intercepts[i, j_idx] + Z @ coefficients[i, j_idx]
    logits = np.clip(logits, -20, 20)
    logits_max = logits.max(axis=2, keepdims=True)
    exp = np.exp(logits - logits_max)
    return exp / exp.sum(axis=2, keepdims=True)

def hamilton_filter_tvtp(r, Z, mu, sigma, intercepts, coefficients, pi0):
    T = len(r)
    K = len(mu)
    P_all = tvtp_transition_matrices_vectorized(Z, intercepts, coefficients)
    log_alpha = np.zeros((T, K))
    log_emit = norm.logpdf(r[0], loc=mu, scale=sigma)
    log_alpha[0] = np.log(np.maximum(pi0, FILTER_CLAMP)) + log_emit
    log_norm = logsumexp(log_alpha[0])
    log_alpha[0] -= log_norm
    log_lik = log_norm
    for t in range(1, T):
        log_P_t = np.log(np.maximum(P_all[t], FILTER_CLAMP))
        log_pred = logsumexp(log_alpha[t-1][:, None] + log_P_t, axis=0)
        log_emit = norm.logpdf(r[t], loc=mu, scale=sigma)
        log_alpha[t] = log_pred + log_emit
        log_norm = logsumexp(log_alpha[t])
        log_alpha[t] -= log_norm
        log_lik += log_norm
    return log_lik, np.exp(log_alpha), P_all

def kim_smoother_tvtp(filtered, P_all):
    T, K = filtered.shape
    smoothed = np.zeros_like(filtered)
    smoothed[-1] = filtered[-1]
    xi = np.zeros((T-1, K, K))
    for t in range(T-2, -1, -1):
        P_t1 = P_all[t+1]
        pred = filtered[t] @ P_t1
        pred = np.maximum(pred, FILTER_CLAMP)
        smoothed[t] = filtered[t] * (P_t1 @ (smoothed[t+1] / pred))
        smoothed[t] /= smoothed[t].sum()
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = filtered[t, i] * P_t1[i, j] * smoothed[t+1, j] / pred[j]
        xi[t] /= xi[t].sum()
    return smoothed, xi

def average_diagonal_prob(intercepts, coefficients, Z):
    """Mean diagonal P(stay in state) across time, for penalty term."""
    K = intercepts.shape[0]
    P_all = tvtp_transition_matrices_vectorized(Z, intercepts, coefficients)
    diag = np.array([P_all[:, i, i].mean() for i in range(K)])
    return diag

def m_step_logistic_constrained(xi, Z, K, init_intercepts, init_coefficients):
    """
    M-step with diagonal-persistence penalty.
    Maximize: Q - lambda * sum_i max(0, MIN_DIAGONAL_PROB - mean(P_t[i,i]))^2
    """
    T_minus_1 = xi.shape[0]
    d = Z.shape[1]
    new_intercepts = init_intercepts.copy()
    new_coefficients = init_coefficients.copy()
    
    for i in range(K):
        n_params_per_dest = 1 + d
        n_total_params = (K - 1) * n_params_per_dest
        x0 = np.zeros(n_total_params)
        for j_idx in range(K - 1):
            x0[j_idx * n_params_per_dest] = init_intercepts[i, j_idx]
            x0[j_idx * n_params_per_dest + 1: (j_idx + 1) * n_params_per_dest] = init_coefficients[i, j_idx]
        
        xi_i = xi[:, i, :]
        Z_used = Z[1:1+T_minus_1]
        
        def neg_q_with_penalty(params):
            intercepts_i = np.zeros(K - 1)
            coeffs_i = np.zeros((K - 1, d))
            for j_idx in range(K - 1):
                intercepts_i[j_idx] = params[j_idx * n_params_per_dest]
                coeffs_i[j_idx] = params[j_idx * n_params_per_dest + 1: (j_idx + 1) * n_params_per_dest]
            
            logits = np.zeros((T_minus_1, K))
            for j_idx in range(K - 1):
                j = j_idx + 1
                logits[:, j] = intercepts_i[j_idx] + Z_used @ coeffs_i[j_idx]
            logits = np.clip(logits, -20, 20)
            log_P = logits - logsumexp(logits, axis=1, keepdims=True)
            
            neg_q = -np.sum(xi_i * log_P)
            
            # Diagonal persistence penalty: P[i,i] should average >= MIN_DIAGONAL_PROB
            # P[i,i]: when i==0 it's exp(0)/sum_exp, when i>=1 it's exp(logit_i)/sum_exp
            P_stay = np.exp(log_P[:, i])
            mean_P_stay = P_stay.mean()
            shortfall = max(0, MIN_DIAGONAL_PROB - mean_P_stay)
            penalty = DIAGONAL_PENALTY_WEIGHT * shortfall ** 2 * T_minus_1
            
            return neg_q + penalty
        
        try:
            res = minimize(neg_q_with_penalty, x0, method='L-BFGS-B', options={'maxiter': LBFGS_MAXITER})
            opt_params = res.x
        except Exception:
            opt_params = x0
        
        for j_idx in range(K - 1):
            new_intercepts[i, j_idx] = opt_params[j_idx * n_params_per_dest]
            new_coefficients[i, j_idx] = opt_params[j_idx * n_params_per_dest + 1: (j_idx + 1) * n_params_per_dest]
    
    return new_intercepts, new_coefficients

def init_from_m2b(rng, d, perturb_scale=1.0):
    """
    Initialize emission params from M2b K=4 fixed-MS reference, with small noise.
    Initialize logistic params from M2b transition matrix as starting point.
    """
    # Emissions: M2b values + small noise
    mu = M2B_MU_DAILY + rng.normal(0, 0.0001 * perturb_scale, N_STATES)
    sigma = M2B_SIGMA_DAILY * (1 + rng.normal(0, 0.05 * perturb_scale, N_STATES))
    sigma = np.maximum(sigma, 1e-5)
    
    # Logistic init: solve for intercepts that produce M2b's transition matrix at z=0 (training mean)
    # M2B_TRANSITION[i, j] = exp(a_ij) / sum_k exp(a_ik), with a_i0 = 0
    # So a_ij = log(M2B_TRANSITION[i, j] / M2B_TRANSITION[i, 0])
    intercepts = np.zeros((N_STATES, N_STATES - 1))
    for i in range(N_STATES):
        ref_prob = max(M2B_TRANSITION[i, 0], 1e-6)
        for j_idx in range(N_STATES - 1):
            j = j_idx + 1
            target_prob = max(M2B_TRANSITION[i, j], 1e-6)
            intercepts[i, j_idx] = np.log(target_prob / ref_prob) + rng.normal(0, 0.1 * perturb_scale)
    
    # Coefficients: small random (covariates start with small effect)
    coefficients = rng.normal(0, 0.1 * perturb_scale, (N_STATES, N_STATES - 1, d))
    
    return mu, sigma, intercepts, coefficients

def em_fit_tvtp_k4_constrained(r, Z, n_iter, tol, seed):
    rng = np.random.default_rng(seed)
    K = N_STATES
    d = Z.shape[1]
    
    mu, sigma, intercepts, coefficients = init_from_m2b(rng, d)
    pi0 = np.ones(K) / K
    
    ll_history = []
    prev_ll = -np.inf
    converged = False
    
    for it in range(n_iter):
        ll, filtered, P_all = hamilton_filter_tvtp(r, Z, mu, sigma, intercepts, coefficients, pi0)
        smoothed, xi = kim_smoother_tvtp(filtered, P_all)
        ll_history.append(ll)
        
        # M-step: emissions WITH BOUND
        gamma = smoothed
        Nk = gamma.sum(axis=0)
        mu_unconstrained = (gamma * r[:, None]).sum(axis=0) / Nk
        # Constraint 1: clip mu to [-MU_DAILY_BOUND, +MU_DAILY_BOUND]
        mu = np.clip(mu_unconstrained, -MU_DAILY_BOUND, MU_DAILY_BOUND)
        diff_sq = (r[:, None] - mu) ** 2
        sigma = np.sqrt((gamma * diff_sq).sum(axis=0) / Nk)
        sigma = np.maximum(sigma, 1e-5)
        
        # M-step: TVTP transitions with diagonal persistence penalty
        intercepts, coefficients = m_step_logistic_constrained(xi, Z, K, intercepts, coefficients)
        
        pi0 = smoothed[0]
        
        if it > 0 and abs(ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol:
            converged = True
            break
        prev_ll = ll
    
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
        'mean_diagonal': average_diagonal_prob(intercepts, coefficients, Z),
    }

def causal_filter_only(r, Z, mu, sigma, intercepts, coefficients, pi0):
    _, filtered, _ = hamilton_filter_tvtp(r, Z, mu, sigma, intercepts, coefficients, pi0)
    return filtered

# ============== MAIN ==============

print('=' * 70)
print('M3b-CONSTRAINED: K=4 TVTP MS WITH ECONOMIC CONSTRAINTS')
print('=' * 70)
print(f'Constraints:')
print(f'  mu in [{-MU_ANNUAL_BOUND*100:.0f}%, +{MU_ANNUAL_BOUND*100:.0f}%] annualized')
print(f'  Diagonal transition >= {MIN_DIAGONAL_PROB} (penalty weight {DIAGONAL_PENALTY_WEIGHT})')
print(f'  Init from M2b K=4 fixed-MS reference (small noise)')
print(f'States: {N_STATES}, Restarts: {N_RESTARTS}, Max iter: {EM_MAX_ITER}')
print(f'Covariates: {COVARIATE_COLS}')
print()

returns = pd.read_parquet(DATA / 'returns.parquet')
returns.index = pd.to_datetime(returns.index)
covariates_raw = pd.read_parquet(DATA / 'covariates_raw.parquet')
covariates_raw.index = pd.to_datetime(covariates_raw.index)
common_idx = returns.index.intersection(covariates_raw.index)
returns = returns.loc[common_idx]
covariates_raw = covariates_raw.loc[common_idx]
r_full = returns['log_return'].values if 'log_return' in returns.columns else returns.iloc[:, 0].values

print(f'Full sample: {len(r_full)} obs, {returns.index[0].date()} to {returns.index[-1].date()}')
print()

overall_start = time.time()

for w in WINDOWS:
    wid = w['id']
    win_dir = OUT / f'window_{wid}'
    win_dir.mkdir(exist_ok=True)
    
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
    
    cov_train = covariates_raw.loc[train_mask, COVARIATE_COLS]
    cov_mean = cov_train.mean()
    cov_std = cov_train.std()
    cov_standardized = (covariates_raw[COVARIATE_COLS] - cov_mean) / cov_std
    Z_train = cov_standardized.loc[train_mask].values
    Z_full = cov_standardized.loc[full_mask].values
    assert len(Z_train) == len(r_train), f'Z_train {len(Z_train)} != r_train {len(r_train)}'
    assert len(Z_full) == len(r_full_window), f'Z_full {len(Z_full)} != r_full_window {len(r_full_window)}'
    
    with open(win_dir / 'covariate_stats.json', 'w') as f:
        json.dump({
            'mean': cov_mean.to_dict(),
            'std': cov_std.to_dict(),
            'n_train_obs': int(len(r_train)),
        }, f, indent=2)
    
    all_restarts = []
    for restart in range(N_RESTARTS):
        seed = SEED + restart * 17
        print(f'  Restart {restart+1}/{N_RESTARTS} (seed {seed})... ', end='', flush=True)
        try:
            t0 = time.time()
            result = em_fit_tvtp_k4_constrained(r_train, Z_train, EM_MAX_ITER, EM_TOL, seed)
            result['seed'] = seed
            result['restart_idx'] = restart
            result['runtime_sec'] = time.time() - t0
            all_restarts.append(result)
            mean_diag = result['mean_diagonal']
            print(f'LL={result["log_likelihood"]:.2f}, iter={result["n_iter"]}, conv={result["converged"]}, '
                  f'meanP_stay={mean_diag.mean():.3f}, time={result["runtime_sec"]:.0f}s')
        except Exception as e:
            print(f'FAILED: {e}')
    
    if not all_restarts:
        print(f'  Window {wid}: ALL RESTARTS FAILED, skipping')
        continue
    
    best = max(all_restarts, key=lambda x: x['log_likelihood'])
    
    order = np.argsort(best['mu'])
    mu_sorted = best['mu'][order]
    sigma_sorted = best['sigma'][order]
    pi0_sorted = best['pi0'][order]
    smoothed_sorted = best['smoothed'][:, order]
    diag_sorted = best['mean_diagonal'][order]
    
    filtered_full = causal_filter_only(
        r_full_window, Z_full,
        best['mu'], best['sigma'],
        best['intercepts'], best['coefficients'],
        best['pi0']
    )
    filtered_full_sorted = filtered_full[:, order]
    
    _, _, P_all_full = hamilton_filter_tvtp(
        r_full_window, Z_full,
        best['mu'], best['sigma'],
        best['intercepts'], best['coefficients'],
        best['pi0']
    )
    smoothed_full, _ = kim_smoother_tvtp(filtered_full, P_all_full)
    smoothed_full_sorted = smoothed_full[:, order]
    
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
        'mean_diagonal_prob': diag_sorted.tolist(),
        'n_train_obs': int(len(r_train)),
        'n_test_obs': int(((returns.index >= test_start) & (returns.index <= test_end)).sum()),
        'constraints': {
            'mu_annual_bound': MU_ANNUAL_BOUND,
            'min_diagonal_prob': MIN_DIAGONAL_PROB,
            'diagonal_penalty_weight': DIAGONAL_PENALTY_WEIGHT,
        }
    }
    with open(win_dir / 'tvtp_result.json', 'w') as f:
        json.dump(result_summary, f, indent=2)
    
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
        'mean_diagonal': best['mean_diagonal'],
        'n_iter': best['n_iter'],
        'converged': best['converged'],
        'covariate_cols': COVARIATE_COLS,
        'cov_mean': cov_mean.values,
        'cov_std': cov_std.values,
    }
    with open(win_dir / 'tvtp_result.pkl', 'wb') as f:
        pickle.dump(pickle_payload, f)
    
    conv_log = {
        'restarts': [
            {
                'idx': int(rs['restart_idx']),
                'seed': int(rs['seed']),
                'final_ll': float(rs['log_likelihood']),
                'n_iter': int(rs['n_iter']),
                'converged': bool(rs['converged']),
                'runtime_sec': float(rs.get('runtime_sec', 0)),
                'mean_diagonal': [float(d) for d in rs['mean_diagonal']],
                'll_history': [float(x) for x in rs['ll_history']],
            }
            for rs in all_restarts
        ]
    }
    with open(win_dir / 'em_convergence_log.json', 'w') as f:
        json.dump(conv_log, f, indent=2)
    
    with open(win_dir / 'all_restarts.pkl', 'wb') as f:
        pickle.dump(all_restarts, f)
    
    win_elapsed = time.time() - win_start_time
    print(f'  Window {wid} complete in {win_elapsed/60:.1f} min')
    print(f'  States (annualized): mu={[f"{m*252*100:.1f}%" for m in mu_sorted]}, sigma={[f"{s*np.sqrt(252)*100:.1f}%" for s in sigma_sorted]}')
    print(f'  Diagonal P_stay: {[f"{d:.3f}" for d in diag_sorted]}')
    print(f'  Training occupancy: {[f"{o*100:.1f}%" for o in occupancies]}')
    print()

total_elapsed = time.time() - overall_start
print('=' * 70)
print(f'ALL WINDOWS COMPLETE in {total_elapsed/3600:.2f} hours')
print(f'Outputs in: {OUT}')
print('=' * 70)
