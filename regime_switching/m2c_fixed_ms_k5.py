"""
M2c: Fixed-Transition Markov-Switching with K=5 states (Diagnostic)

Same purpose as M2b: cheap diagnostic before committing to a 5-state TVTP refit.
We test whether equity returns support a 5-state regime structure.

Decision criteria (same as M2b but tweaked for K=5):
  C1: All states have occupancy >= 3% (relaxed from 5% per M2b learning -
      crisis-type rare states should not be penalized for being rare)
  C2: Adjacent state mu separation >= 0.0003 daily (same)
  C3: Adjacent state sigma ratio >= 1.15 (same)
  C4: BIC prefers K=5 over K=4

Outputs saved to regime_switching/data/m2c_k5_diagnostic/.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.special import logsumexp

# ---- Configuration ----
N_STATES = 5
N_RESTARTS = 15           # Bumped from 12 (more states = more local optima)
EM_MAX_ITER = 600         # Bumped from 500 (more states = slower convergence)
EM_TOL = 1e-6
FILTER_CLAMP = 1e-10
SEED = 42

# Decision thresholds
MIN_OCCUPANCY = 0.03      # 3% — relaxed per M2b learning
MIN_MU_SEPARATION = 0.0003
MIN_SIGMA_RATIO = 1.15

# ---- Paths ----
DATA = Path('regime_switching/data')
OUT = DATA / 'm2c_k5_diagnostic'
OUT.mkdir(exist_ok=True)

print('=' * 70)
print('M2c: K=5 FIXED-TRANSITION MS DIAGNOSTIC')
print('=' * 70)
print(f'Output dir: {OUT}')
print(f'States: {N_STATES}, Restarts: {N_RESTARTS}, Max iter: {EM_MAX_ITER}')
print()

# ---- Load returns ----
returns = pd.read_parquet(DATA / 'returns.parquet')
returns.index = pd.to_datetime(returns.index)
r = returns['log_return'].values if 'log_return' in returns.columns else returns.iloc[:, 0].values
T = len(r)
print(f'Loaded {T} return observations: {returns.index[0].date()} to {returns.index[-1].date()}')
print(f'Mean: {r.mean()*100:.4f}%, Std: {r.std()*100:.4f}%')
print()

# ---- Hamilton filter (fixed transition matrix) ----
def hamilton_filter(r, mu, sigma, P, pi0):
    K = len(mu)
    T = len(r)
    log_alpha = np.zeros((T, K))
    log_emit = norm.logpdf(r[0], loc=mu, scale=sigma)
    log_alpha[0] = np.log(np.maximum(pi0, FILTER_CLAMP)) + log_emit
    log_norm = logsumexp(log_alpha[0])
    log_alpha[0] -= log_norm
    log_lik = log_norm
    log_P = np.log(np.maximum(P, FILTER_CLAMP))
    for t in range(1, T):
        log_pred = logsumexp(log_alpha[t-1][:, None] + log_P, axis=0)
        log_emit = norm.logpdf(r[t], loc=mu, scale=sigma)
        log_alpha[t] = log_pred + log_emit
        log_norm = logsumexp(log_alpha[t])
        log_alpha[t] -= log_norm
        log_lik += log_norm
    return log_lik, np.exp(log_alpha)

def kim_smoother(filtered, P):
    T, K = filtered.shape
    smoothed = np.zeros_like(filtered)
    smoothed[-1] = filtered[-1]
    xi = np.zeros((T-1, K, K))
    for t in range(T-2, -1, -1):
        pred = filtered[t] @ P
        pred = np.maximum(pred, FILTER_CLAMP)
        smoothed[t] = filtered[t] * (P @ (smoothed[t+1] / pred))
        smoothed[t] /= smoothed[t].sum()
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = filtered[t, i] * P[i, j] * smoothed[t+1, j] / pred[j]
        xi[t] /= xi[t].sum()
    return smoothed, xi

def em_fit(r, K, n_iter, tol, seed):
    rng = np.random.default_rng(seed)
    quantiles = np.quantile(r, np.linspace(0.1, 0.9, K))
    mu = np.sort(quantiles + rng.normal(0, 0.0003, K))
    sigma = np.abs(rng.normal(r.std(), r.std()*0.3, K))
    sigma = np.sort(sigma)
    P = np.full((K, K), (1 - 0.95) / (K-1))
    np.fill_diagonal(P, 0.95)
    pi0 = np.ones(K) / K
    ll_history = []
    prev_ll = -np.inf
    converged = False
    for it in range(n_iter):
        ll, filtered = hamilton_filter(r, mu, sigma, P, pi0)
        smoothed, xi = kim_smoother(filtered, P)
        ll_history.append(ll)
        gamma = smoothed
        Nk = gamma.sum(axis=0)
        mu = (gamma * r[:, None]).sum(axis=0) / Nk
        diff_sq = (r[:, None] - mu) ** 2
        sigma = np.sqrt((gamma * diff_sq).sum(axis=0) / Nk)
        sigma = np.maximum(sigma, 1e-5)
        Nij = xi.sum(axis=0)
        P = Nij / Nij.sum(axis=1, keepdims=True)
        P = np.maximum(P, FILTER_CLAMP)
        P = P / P.sum(axis=1, keepdims=True)
        pi0 = smoothed[0]
        if it > 0 and abs(ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol:
            converged = True
            break
        prev_ll = ll
    ll_final, filtered_final = hamilton_filter(r, mu, sigma, P, pi0)
    smoothed_final, _ = kim_smoother(filtered_final, P)
    return {
        'mu': mu, 'sigma': sigma, 'P': P, 'pi0': pi0,
        'log_likelihood': ll_final,
        'filtered': filtered_final, 'smoothed': smoothed_final,
        'n_iter': it + 1, 'converged': converged,
        'll_history': ll_history,
    }

# ---- Multi-restart fit ----
print(f'Running {N_RESTARTS} EM restarts...')
all_restarts = []
for restart in range(N_RESTARTS):
    seed = SEED + restart * 17
    print(f'  Restart {restart+1}/{N_RESTARTS} (seed {seed})... ', end='', flush=True)
    try:
        result = em_fit(r, N_STATES, EM_MAX_ITER, EM_TOL, seed)
        result['seed'] = seed
        result['restart_idx'] = restart
        all_restarts.append(result)
        print(f'LL={result["log_likelihood"]:.2f}, iter={result["n_iter"]}, conv={result["converged"]}')
    except Exception as e:
        print(f'FAILED: {e}')

best = max(all_restarts, key=lambda x: x['log_likelihood'])
print()
print(f'Best LL: {best["log_likelihood"]:.2f} (restart {best["restart_idx"]+1}, seed {best["seed"]})')
print()

# ---- Sort states by mu (ascending) ----
order = np.argsort(best['mu'])
mu_sorted = best['mu'][order]
sigma_sorted = best['sigma'][order]
P_sorted = best['P'][order][:, order]
pi0_sorted = best['pi0'][order]
filtered_sorted = best['filtered'][:, order]
smoothed_sorted = best['smoothed'][:, order]

# K=5 labels — bottom-up by mu
state_labels = ['crisis', 'severe_bear', 'bear', 'recovery', 'bull']
print('=' * 70)
print('K=5 STATE PARAMETERS (sorted by mu)')
print('=' * 70)
print(f'{"State":<14} {"mu (daily)":>12} {"mu (annual)":>12} {"sigma (daily)":>15} {"sigma (annual)":>16} {"occupancy":>11}')
for i, label in enumerate(state_labels):
    occ = smoothed_sorted[:, i].mean()
    print(f'{label:<14} {mu_sorted[i]*100:>11.4f}% {mu_sorted[i]*252*100:>11.2f}% {sigma_sorted[i]*100:>14.4f}% {sigma_sorted[i]*np.sqrt(252)*100:>15.2f}% {occ*100:>10.2f}%')
print()

# ---- BIC / AIC ----
T_obs = len(r)
n_params_k5 = N_STATES * 2 + N_STATES * (N_STATES - 1)
ll = best['log_likelihood']
bic_k5 = -2 * ll + n_params_k5 * np.log(T_obs)
aic_k5 = -2 * ll + 2 * n_params_k5

# Load K=4 and K=3 for full ladder comparison
k3_path = DATA / 'ms3_fixed_results.json'
k4_path = DATA / 'm2b_k4_diagnostic' / 'ms4_fixed_results.json'

bic_k3 = aic_k3 = ll_k3 = None
bic_k4 = aic_k4 = ll_k4 = None

if k3_path.exists():
    with open(k3_path) as f:
        k3 = json.load(f)
    bic_k3 = k3.get('bic')
    aic_k3 = k3.get('aic')
    ll_k3 = k3.get('log_likelihood')
if k4_path.exists():
    with open(k4_path) as f:
        k4 = json.load(f)
    bic_k4 = k4.get('bic')
    aic_k4 = k4.get('aic')
    ll_k4 = k4.get('log_likelihood')

print('=' * 70)
print('MODEL COMPARISON LADDER: K=3 vs K=4 vs K=5 (Fixed Transition)')
print('=' * 70)
print(f'{"Metric":<20} {"K=3":>15} {"K=4":>15} {"K=5":>15}')
def fmt(v): return f'{v:.2f}' if v is not None else 'N/A'
print(f'{"Log-Likelihood":<20} {fmt(ll_k3):>15} {fmt(ll_k4):>15} {ll:>15.2f}')
print(f'{"BIC":<20} {fmt(bic_k3):>15} {fmt(bic_k4):>15} {bic_k5:>15.2f}')
print(f'{"AIC":<20} {fmt(aic_k3):>15} {fmt(aic_k4):>15} {aic_k5:>15.2f}')
print(f'{"# Params":<20} {"12":>15} {"20":>15} {n_params_k5:>15}')
print()

# Which K wins on BIC?
bic_candidates = {3: bic_k3, 4: bic_k4, 5: bic_k5}
bic_candidates = {k: v for k, v in bic_candidates.items() if v is not None}
winner_K = min(bic_candidates, key=bic_candidates.get)
print(f'BIC prefers: K={winner_K}')
print()

# ---- Decision criteria ----
print('=' * 70)
print('DECISION CRITERIA')
print('=' * 70)

occupancies = [smoothed_sorted[:, i].mean() for i in range(N_STATES)]
min_occ = min(occupancies)
crit1_pass = min_occ >= MIN_OCCUPANCY
print(f'C1: All states have occupancy >= {MIN_OCCUPANCY*100:.0f}%')
print(f'    Min occupancy: {min_occ*100:.2f}%  --> {"PASS" if crit1_pass else "FAIL"}')

mu_diffs = np.diff(mu_sorted)
min_mu_diff = mu_diffs.min()
crit2_pass = min_mu_diff >= MIN_MU_SEPARATION
print(f'C2: Adjacent state mu separation >= {MIN_MU_SEPARATION*100:.4f}% (daily)')
print(f'    Min separation: {min_mu_diff*100:.4f}%  --> {"PASS" if crit2_pass else "FAIL"}')

sigma_separated = all((max(s1, s2) / min(s1, s2)) >= MIN_SIGMA_RATIO for s1, s2 in zip(sigma_sorted[:-1], sigma_sorted[1:]))
print(f'C3: Adjacent state sigma ratio >= {MIN_SIGMA_RATIO}')
print(f'    Sigmas: {[f"{s*100:.4f}%" for s in sigma_sorted]}  --> {"PASS" if sigma_separated else "FAIL"}')
print()

bic_pass_vs_k4 = (bic_k4 is None) or (bic_k5 < bic_k4)
print(f'C4: BIC prefers K=5 over K=4')
print(f'    BIC K=4={fmt(bic_k4)}, K=5={bic_k5:.2f}  --> {"PASS" if bic_pass_vs_k4 else "FAIL"}')
print()

all_pass = crit1_pass and crit2_pass and sigma_separated and bic_pass_vs_k4
recommendation = 'PROCEED with full 5-state TVTP refit' if all_pass else 'STAY at K=4 (or K=3 if K=4 also fails); 5-state not supported by data'
print('=' * 70)
print(f'OVERALL: {"ALL CRITERIA PASS" if all_pass else "ONE OR MORE CRITERIA FAIL"}')
print(f'RECOMMENDATION: {recommendation}')
print('=' * 70)

# ---- Save outputs ----
results = {
    'n_states': N_STATES,
    'n_restarts': N_RESTARTS,
    'best_seed': int(best['seed']),
    'best_restart_idx': int(best['restart_idx']),
    'log_likelihood': float(ll),
    'bic': float(bic_k5),
    'aic': float(aic_k5),
    'n_params': int(n_params_k5),
    'n_iter_best': int(best['n_iter']),
    'converged_best': bool(best['converged']),
    'state_labels': state_labels,
    'mu_daily': mu_sorted.tolist(),
    'mu_annual': (mu_sorted * 252).tolist(),
    'sigma_daily': sigma_sorted.tolist(),
    'sigma_annual': (sigma_sorted * np.sqrt(252)).tolist(),
    'transition_matrix': P_sorted.tolist(),
    'pi0': pi0_sorted.tolist(),
    'occupancy': occupancies,
    'observation_count': int(T_obs),
}
with open(OUT / 'ms5_fixed_results.json', 'w') as f:
    json.dump(results, f, indent=2)

probs_df = pd.DataFrame(
    np.hstack([filtered_sorted, smoothed_sorted]),
    index=returns.index,
    columns=[f'filtered_{l}' for l in state_labels] + [f'smoothed_{l}' for l in state_labels]
)
probs_df.to_parquet(OUT / 'ms5_fixed_probs.parquet')

comparison = {
    'k3': {'log_likelihood': ll_k3, 'bic': bic_k3, 'aic': aic_k3, 'n_params': 12},
    'k4': {'log_likelihood': ll_k4, 'bic': bic_k4, 'aic': aic_k4, 'n_params': 20},
    'k5': {'log_likelihood': float(ll), 'bic': float(bic_k5), 'aic': float(aic_k5), 'n_params': int(n_params_k5)},
    'bic_winner': f'K={winner_K}',
    'll_improvement_k4_to_k5': float(ll - ll_k4) if ll_k4 else None,
    'bic_improvement_k4_to_k5': float(bic_k4 - bic_k5) if bic_k4 else None,
}
with open(OUT / 'k3_vs_k4_vs_k5_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

conv_log = {
    'restarts': [
        {
            'idx': int(rs['restart_idx']),
            'seed': int(rs['seed']),
            'final_ll': float(rs['log_likelihood']),
            'n_iter': int(rs['n_iter']),
            'converged': bool(rs['converged']),
            'll_history': [float(x) for x in rs['ll_history']],
        }
        for rs in all_restarts
    ]
}
with open(OUT / 'em_convergence_log.json', 'w') as f:
    json.dump(conv_log, f, indent=2)

with open(OUT / 'all_restarts.pkl', 'wb') as f:
    pickle.dump(all_restarts, f)

decision = {
    'criteria': {
        'C1_min_occupancy': {'threshold': MIN_OCCUPANCY, 'observed': float(min_occ), 'pass': bool(crit1_pass)},
        'C2_min_mu_separation': {'threshold': MIN_MU_SEPARATION, 'observed': float(min_mu_diff), 'pass': bool(crit2_pass)},
        'C3_sigma_separation': {'threshold': MIN_SIGMA_RATIO, 'observed': [float(s) for s in sigma_sorted], 'pass': bool(sigma_separated)},
        'C4_bic_prefers_k5_over_k4': {'k4_bic': bic_k4, 'k5_bic': float(bic_k5), 'pass': bool(bic_pass_vs_k4)},
    },
    'all_pass': bool(all_pass),
    'bic_winner_overall': f'K={winner_K}',
    'recommendation': recommendation,
}
with open(OUT / 'decision.json', 'w') as f:
    json.dump(decision, f, indent=2)

print()
print('=' * 70)
print('SAVED:')
for fn in ['ms5_fixed_results.json', 'ms5_fixed_probs.parquet', 'k3_vs_k4_vs_k5_comparison.json',
           'em_convergence_log.json', 'all_restarts.pkl', 'decision.json']:
    print(f'  {OUT / fn}')
print('=' * 70)
