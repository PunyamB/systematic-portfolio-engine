"""
M2b: Fixed-Transition Markov-Switching with K=4 states (Diagnostic)

Purpose: Test whether equity returns support a 4-state regime structure.
This is a cheap diagnostic before committing to a full 4-state TVTP refit (which would take ~6 hours).

If K=4 wins on BIC AND each state has meaningful occupancy AND state parameters are well-separated,
then 4-state TVTP is justified. Otherwise, the recovery/bear distinction is a labeling convention,
not a statistical regime, and we abandon 4-state.

Outputs saved to regime_switching/data/m2b_k4_diagnostic/:
  - ms4_fixed_results.json   : Parameters, BIC, AIC, log-likelihood, occupancy
  - ms4_fixed_probs.parquet  : Filtered + smoothed probs (4 states)
  - k3_vs_k4_comparison.json : Side-by-side BIC/AIC/LL/occupancy
  - em_convergence_log.json  : Per-restart LL traces
  - all_restarts.pkl         : All restart results for diagnostics
  - decision.json            : Pass/fail on the 3 criteria + recommendation
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.special import logsumexp

# ---- Configuration ----
N_STATES = 4
N_RESTARTS = 12
EM_MAX_ITER = 500
EM_TOL = 1e-6
FILTER_CLAMP = 1e-10
SEED = 42

# Decision thresholds
MIN_OCCUPANCY = 0.05      # Each state must hold >= 5% of sample
MIN_MU_SEPARATION = 0.0003  # Daily mean separation between adjacent states (~7.5% annualized)
MIN_SIGMA_RATIO = 1.15    # Adjacent state sigmas must differ by >= 15%

# ---- Paths ----
DATA = Path('regime_switching/data')
OUT = DATA / 'm2b_k4_diagnostic'
OUT.mkdir(exist_ok=True)

print('=' * 70)
print('M2b: K=4 FIXED-TRANSITION MS DIAGNOSTIC')
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
    """Forward filter. Returns log-likelihood and filtered probabilities."""
    K = len(mu)
    T = len(r)
    log_alpha = np.zeros((T, K))
    
    # t=0
    log_emit = norm.logpdf(r[0], loc=mu, scale=sigma)
    log_alpha[0] = np.log(np.maximum(pi0, FILTER_CLAMP)) + log_emit
    log_norm = logsumexp(log_alpha[0])
    log_alpha[0] -= log_norm
    log_lik = log_norm
    
    log_P = np.log(np.maximum(P, FILTER_CLAMP))
    
    for t in range(1, T):
        # Predict: log P(S_t | y_{1:t-1}) = logsumexp over previous states
        log_pred = logsumexp(log_alpha[t-1][:, None] + log_P, axis=0)
        log_emit = norm.logpdf(r[t], loc=mu, scale=sigma)
        log_alpha[t] = log_pred + log_emit
        log_norm = logsumexp(log_alpha[t])
        log_alpha[t] -= log_norm
        log_lik += log_norm
    
    return log_lik, np.exp(log_alpha)

def kim_smoother(filtered, P):
    """Backward smoother. Returns smoothed probs and joint smoothed P(S_t, S_{t-1})."""
    T, K = filtered.shape
    smoothed = np.zeros_like(filtered)
    smoothed[-1] = filtered[-1]
    
    # Joint smoothed for transition update
    xi = np.zeros((T-1, K, K))  # xi[t, i, j] = P(S_t=i, S_{t+1}=j | all data)
    
    for t in range(T-2, -1, -1):
        # Predicted next-state distribution
        pred = filtered[t] @ P  # shape (K,)
        pred = np.maximum(pred, FILTER_CLAMP)
        
        # Smoothed via Kim's formula
        smoothed[t] = filtered[t] * (P @ (smoothed[t+1] / pred))
        smoothed[t] /= smoothed[t].sum()
        
        # Joint
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = filtered[t, i] * P[i, j] * smoothed[t+1, j] / pred[j]
        # Normalize
        xi[t] /= xi[t].sum()
    
    return smoothed, xi

def em_fit(r, K, n_iter, tol, seed):
    """EM for fixed-transition MS. Returns dict with params, LL, convergence trace."""
    rng = np.random.default_rng(seed)
    
    # --- Initialize ---
    # Sort returns into K quantile buckets to seed mu/sigma
    quantiles = np.quantile(r, np.linspace(0.1, 0.9, K))
    mu = np.sort(quantiles + rng.normal(0, 0.0003, K))  # ascending mu
    sigma = np.abs(rng.normal(r.std(), r.std()*0.3, K))
    sigma = np.sort(sigma)  # ascending sigma usually correlates with magnitude
    # We don't enforce mu/sigma joint ordering; let EM sort it out
    
    # Initial transition matrix: persistent (diag ~ 0.95)
    P = np.full((K, K), (1 - 0.95) / (K-1))
    np.fill_diagonal(P, 0.95)
    pi0 = np.ones(K) / K
    
    ll_history = []
    prev_ll = -np.inf
    converged = False
    
    for it in range(n_iter):
        # E-step
        ll, filtered = hamilton_filter(r, mu, sigma, P, pi0)
        smoothed, xi = kim_smoother(filtered, P)
        ll_history.append(ll)
        
        # M-step: emissions (weighted MLE)
        gamma = smoothed  # shape (T, K)
        Nk = gamma.sum(axis=0)  # state occupancy
        mu = (gamma * r[:, None]).sum(axis=0) / Nk
        # Variance: weighted MSE
        diff_sq = (r[:, None] - mu) ** 2
        sigma = np.sqrt((gamma * diff_sq).sum(axis=0) / Nk)
        sigma = np.maximum(sigma, 1e-5)  # floor
        
        # M-step: transitions (closed form for fixed-TP)
        Nij = xi.sum(axis=0)  # shape (K, K)
        P = Nij / Nij.sum(axis=1, keepdims=True)
        P = np.maximum(P, FILTER_CLAMP)
        P = P / P.sum(axis=1, keepdims=True)
        
        pi0 = smoothed[0]
        
        # Convergence
        if it > 0 and abs(ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol:
            converged = True
            break
        prev_ll = ll
    
    # Final LL with final params
    ll_final, filtered_final = hamilton_filter(r, mu, sigma, P, pi0)
    smoothed_final, _ = kim_smoother(filtered_final, P)
    
    return {
        'mu': mu, 'sigma': sigma, 'P': P, 'pi0': pi0,
        'log_likelihood': ll_final,
        'filtered': filtered_final,
        'smoothed': smoothed_final,
        'n_iter': it + 1,
        'converged': converged,
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

# Pick best
best = max(all_restarts, key=lambda x: x['log_likelihood'])
print()
print(f'Best LL: {best["log_likelihood"]:.2f} (restart {best["restart_idx"]+1}, seed {best["seed"]})')
print()

# ---- Sort states by mu (ascending: crisis < bear < recovery < bull) ----
order = np.argsort(best['mu'])
mu_sorted = best['mu'][order]
sigma_sorted = best['sigma'][order]
P_sorted = best['P'][order][:, order]
pi0_sorted = best['pi0'][order]
filtered_sorted = best['filtered'][:, order]
smoothed_sorted = best['smoothed'][:, order]

state_labels = ['crisis', 'bear', 'recovery', 'bull']  # by ascending mu
print('=' * 70)
print('K=4 STATE PARAMETERS (sorted by mu)')
print('=' * 70)
print(f'{"State":<10} {"mu (daily)":>12} {"mu (annual)":>12} {"sigma (daily)":>15} {"sigma (annual)":>16} {"occupancy":>11}')
for i, label in enumerate(state_labels):
    occ = smoothed_sorted[:, i].mean()
    print(f'{label:<10} {mu_sorted[i]*100:>11.4f}% {mu_sorted[i]*252*100:>11.2f}% {sigma_sorted[i]*100:>14.4f}% {sigma_sorted[i]*np.sqrt(252)*100:>15.2f}% {occ*100:>10.2f}%')
print()

# ---- BIC / AIC ----
T_obs = len(r)
n_params_k4 = N_STATES * 2 + N_STATES * (N_STATES - 1)  # 2 emission per state + (K-1) free transitions per row
ll = best['log_likelihood']
bic_k4 = -2 * ll + n_params_k4 * np.log(T_obs)
aic_k4 = -2 * ll + 2 * n_params_k4

# Load K=3 fixed for comparison
k3_path = DATA / 'ms3_fixed_results.json'
if k3_path.exists():
    with open(k3_path) as f:
        k3 = json.load(f)
    bic_k3 = k3.get('bic')
    aic_k3 = k3.get('aic')
    ll_k3 = k3.get('log_likelihood')
else:
    bic_k3 = None
    aic_k3 = None
    ll_k3 = None
    print('WARNING: ms3_fixed_results.json not found; cannot compare to K=3')

print('=' * 70)
print('MODEL COMPARISON: K=3 vs K=4 (Fixed Transition)')
print('=' * 70)
print(f'{"Metric":<20} {"K=3":>15} {"K=4":>15} {"Delta":>15}')
print(f'{"Log-Likelihood":<20} {ll_k3 if ll_k3 else "N/A":>15} {ll:>15.2f} {(ll - ll_k3) if ll_k3 else "N/A":>15}')
print(f'{"BIC":<20} {bic_k3 if bic_k3 else "N/A":>15} {bic_k4:>15.2f} {(bic_k4 - bic_k3) if bic_k3 else "N/A":>15}')
print(f'{"AIC":<20} {aic_k3 if aic_k3 else "N/A":>15} {aic_k4:>15.2f} {(aic_k4 - aic_k3) if aic_k3 else "N/A":>15}')
print(f'{"# Params":<20} {"12":>15} {n_params_k4:>15} {"":>15}')
print()
if bic_k3 is not None:
    bic_winner = 'K=4' if bic_k4 < bic_k3 else 'K=3'
    print(f'BIC prefers: {bic_winner}')
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

sigma_ratios = sigma_sorted[1:] / sigma_sorted[:-1]
min_sigma_ratio = min(sigma_ratios.min(), (1/sigma_ratios).min())
crit3_pass = (sigma_ratios.max() / sigma_ratios.min()) >= 1.0  # always true; check separation differently
# Better: check that adjacent sigmas differ by >= MIN_SIGMA_RATIO factor
sigma_separated = all((max(s1, s2) / min(s1, s2)) >= MIN_SIGMA_RATIO for s1, s2 in zip(sigma_sorted[:-1], sigma_sorted[1:]))
print(f'C3: Adjacent state sigma ratio >= {MIN_SIGMA_RATIO}')
print(f'    Sigmas: {[f"{s*100:.4f}%" for s in sigma_sorted]}  --> {"PASS" if sigma_separated else "FAIL"}')
print()

bic_pass = (bic_k3 is None) or (bic_k4 < bic_k3)
print(f'C4 (overall): BIC prefers K=4')
print(f'    BIC K=3={bic_k3}, K=4={bic_k4:.2f}  --> {"PASS" if bic_pass else "FAIL"}')
print()

all_pass = crit1_pass and crit2_pass and sigma_separated and bic_pass
recommendation = 'PROCEED with full 4-state TVTP refit' if all_pass else 'ABANDON 4-state idea; 3-state is the right specification'
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
    'bic': float(bic_k4),
    'aic': float(aic_k4),
    'n_params': int(n_params_k4),
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
with open(OUT / 'ms4_fixed_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Filtered + smoothed probs
probs_df = pd.DataFrame(
    np.hstack([filtered_sorted, smoothed_sorted]),
    index=returns.index,
    columns=[f'filtered_{l}' for l in state_labels] + [f'smoothed_{l}' for l in state_labels]
)
probs_df.to_parquet(OUT / 'ms4_fixed_probs.parquet')

# Comparison
comparison = {
    'k3': {'log_likelihood': ll_k3, 'bic': bic_k3, 'aic': aic_k3, 'n_params': 12},
    'k4': {'log_likelihood': float(ll), 'bic': float(bic_k4), 'aic': float(aic_k4), 'n_params': int(n_params_k4)},
    'bic_winner': 'K=4' if (bic_k3 and bic_k4 < bic_k3) else ('K=3' if bic_k3 else 'unknown'),
    'll_improvement': float(ll - ll_k3) if ll_k3 else None,
}
with open(OUT / 'k3_vs_k4_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

# Convergence log
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

# Pickle all restarts (for any later re-analysis)
with open(OUT / 'all_restarts.pkl', 'wb') as f:
    pickle.dump(all_restarts, f)

# Decision
decision = {
    'criteria': {
        'C1_min_occupancy': {'threshold': MIN_OCCUPANCY, 'observed': float(min_occ), 'pass': bool(crit1_pass)},
        'C2_min_mu_separation': {'threshold': MIN_MU_SEPARATION, 'observed': float(min_mu_diff), 'pass': bool(crit2_pass)},
        'C3_sigma_separation': {'threshold': MIN_SIGMA_RATIO, 'observed': [float(s) for s in sigma_sorted], 'pass': bool(sigma_separated)},
        'C4_bic_prefers_k4': {'k3_bic': bic_k3, 'k4_bic': float(bic_k4), 'pass': bool(bic_pass)},
    },
    'all_pass': bool(all_pass),
    'recommendation': recommendation,
}
with open(OUT / 'decision.json', 'w') as f:
    json.dump(decision, f, indent=2)

print()
print('=' * 70)
print('SAVED:')
print(f'  {OUT / "ms4_fixed_results.json"}')
print(f'  {OUT / "ms4_fixed_probs.parquet"}')
print(f'  {OUT / "k3_vs_k4_comparison.json"}')
print(f'  {OUT / "em_convergence_log.json"}')
print(f'  {OUT / "all_restarts.pkl"}')
print(f'  {OUT / "decision.json"}')
print('=' * 70)
