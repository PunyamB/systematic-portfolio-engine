"""
EXP010: Factor Trend Analysis — Statistical Diagnostic of TVTP Regime Detection

Purpose: Test whether TVTP's regime probabilities provide statistically meaningful
predictive information about market stress, beyond what the underlying macro factors
already provide.

Methodology:
  1. Detect factor trend episodes via 90/120-day signed changes, sensitivity over
     thresholds [1%, 2%, 3%, 4%, 5%, 7%, 10%, 13%] + GPD-justified threshold.
  2. For each episode, measure TVTP P_crisis lead time relative to factor peak.
  3. ROC analysis of P_crisis as a stress predictor, threshold-independent (AUC).
  4. Youden's J optimal threshold + sensitivity at [0.3, 0.5, 0.7].
  5. Bootstrap 95% CIs on lead times.
  6. Granger causality: do TVTP probs predict factors beyond their own lags?
  7. Information value test: forward NAV drawdown ~ factors vs factors + TVTP.
  8. Rule-based baseline comparison on 2009-2026 overlap.

All outputs saved to regime_switching/data/factor_trend_analysis/.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import genpareto
from sklearn.metrics import roc_curve, auc
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA = Path('regime_switching/data')
OUT = DATA / 'factor_trend_analysis'
OUT.mkdir(exist_ok=True)
WF = DATA / 'window_fits'

LOOKBACK_DAYS = [90, 120]                         # Trend window lengths
EPISODE_THRESHOLDS_PCT = [1, 2, 3, 4, 5, 7, 10, 13]   # Sensitivity range (top-pct of |change|)
CRISIS_THRESHOLDS = [0.3, 0.5, 0.7]               # P_crisis threshold sensitivity
LOOKAHEAD_DAYS = [30, 60, 90]                     # Forward windows for after-period
N_BOOTSTRAP = 1000                                # Bootstrap iterations for CIs
BLOCK_LEN = 30                                    # Block-bootstrap length (handles autocorr)
SEED = 42
GRANGER_MAX_LAG = 5                               # Lags for Granger test
EPISODE_MIN_SEPARATION_DAYS = 30                  # Merge episodes within this gap

# Factor-specific stress directions (sign of "bad" change)
# +1 = stress when factor RISES, -1 = stress when factor FALLS
STRESS_DIRECTION = {
    'vix_level': +1,           # VIX rising = stress
    'credit_spread': +1,       # Credit widening = stress
    'yield_curve_slope': -1,   # Curve flattening / inverting = stress
}

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print('=' * 80)
print('EXP010: FACTOR TREND ANALYSIS')
print('=' * 80)
print(f'Lookbacks: {LOOKBACK_DAYS} days')
print(f'Episode thresholds: top {EPISODE_THRESHOLDS_PCT}% (sensitivity)')
print(f'Crisis thresholds: {CRISIS_THRESHOLDS}')
print(f'Bootstrap iterations: {N_BOOTSTRAP} (block length {BLOCK_LEN})')
print()

# Daily covariates (FRED factors)
covariates = pd.read_parquet(DATA / 'covariates_raw.parquet')
covariates.index = pd.to_datetime(covariates.index)
print(f'Covariates: {covariates.shape[0]} daily obs, {covariates.index[0].date()} to {covariates.index[-1].date()}')
print(f'Factors: {list(covariates.columns)}')

# Returns
returns = pd.read_parquet(DATA / 'returns.parquet')
returns.index = pd.to_datetime(returns.index)

# Stitch TVTP probabilities from per-window fits (causal: each window's test period only)
print('\nStitching TVTP probabilities from window fits...')
import json as _json
with open(DATA / 'integration_metrics.json') as f:
    metrics = _json.load(f)

tvtp_pieces = []
nav_pieces = []
for p in metrics['per_window']:
    wid = p['window_id']
    win_dir = WF / f'window_{wid}'
    probs = pd.read_parquet(win_dir / 'filtered_probs.parquet')
    nav = pd.read_parquet(win_dir / 'daily_nav.parquet')
    probs.index = pd.to_datetime(probs.index)
    nav.index = pd.to_datetime(nav.index)
    test_period = (probs.index >= nav.index.min()) & (probs.index <= nav.index.max())
    tvtp_pieces.append(probs.loc[test_period].copy())
    nav_pieces.append(nav.copy())

tvtp = pd.concat(tvtp_pieces).sort_index()
tvtp = tvtp[~tvtp.index.duplicated(keep='first')]
nav = pd.concat(nav_pieces).sort_index()
nav = nav[~nav.index.duplicated(keep='first')]

# Standardize column names
if 'p_bull' not in tvtp.columns and 'filtered_bull' in tvtp.columns:
    tvtp = tvtp.rename(columns={'filtered_bull': 'p_bull', 'filtered_bear': 'p_bear', 'filtered_crisis': 'p_crisis'})
print(f'TVTP probs: {tvtp.shape[0]} obs, {tvtp.index[0].date()} to {tvtp.index[-1].date()}')
print(f'NAV: {nav.shape[0]} obs, {nav.index[0].date()} to {nav.index[-1].date()}')

# Rule-based regime (sparse monthly)
rb_path = Path('D:/Projects/SystematicPortfolioEngine/data/backtest/regime_history.parquet')
rb = pd.read_parquet(rb_path)
rb['date'] = pd.to_datetime(rb['date'])
rb = rb.sort_values('date').set_index('date')
print(f'Rule-based regime: {rb.shape[0]} monthly obs, {rb.index[0].date()} to {rb.index[-1].date()}')

# ============================================================================
# 2. COMPUTE FACTOR CHANGE DISTRIBUTIONS
# ============================================================================
print('\n' + '=' * 80)
print('STEP 1: COMPUTING FACTOR CHANGE DISTRIBUTIONS')
print('=' * 80)

changes = {}  # {(factor, lookback): pd.Series of signed changes}
for factor in covariates.columns:
    series = covariates[factor].dropna()
    for lookback in LOOKBACK_DAYS:
        signed_change = series - series.shift(lookback)
        signed_change = signed_change.dropna()
        changes[(factor, lookback)] = signed_change

# Save distribution stats
dist_stats = {}
change_records = []
for (factor, lookback), s in changes.items():
    dist_stats[f'{factor}_{lookback}d'] = {
        'n_obs': len(s),
        'mean': float(s.mean()),
        'std': float(s.std()),
        'skew': float(stats.skew(s)),
        'kurtosis': float(stats.kurtosis(s)),
        'percentiles': {
            '1': float(s.quantile(0.01)),
            '5': float(s.quantile(0.05)),
            '25': float(s.quantile(0.25)),
            '50': float(s.quantile(0.5)),
            '75': float(s.quantile(0.75)),
            '95': float(s.quantile(0.95)),
            '99': float(s.quantile(0.99)),
        },
        'abs_percentiles': {
            str(p): float(s.abs().quantile(1 - p/100)) for p in EPISODE_THRESHOLDS_PCT
        },
    }
    for date, val in s.items():
        change_records.append({
            'date': date, 'factor': factor, 'lookback': lookback,
            'signed_change': val, 'abs_change': abs(val),
        })

changes_df = pd.DataFrame(change_records)
changes_df.to_parquet(OUT / 'factor_change_distributions.parquet')
with open(OUT / 'factor_change_dist_stats.json', 'w') as f:
    json.dump(dist_stats, f, indent=2)
print(f'Saved {len(changes_df)} change observations across {len(changes)} (factor, lookback) pairs')

# ============================================================================
# 3. GPD-JUSTIFIED THRESHOLD PER FACTOR
# ============================================================================
print('\n' + '=' * 80)
print('STEP 2: GPD-JUSTIFIED THRESHOLDS')
print('=' * 80)

def gpd_justified_threshold(s_abs, candidate_quantiles=np.linspace(0.85, 0.99, 30)):
    """
    Find threshold u where GPD fits well via parameter stability check.
    For each candidate u, fit GPD to exceedances and look for the lowest u where
    shape parameter (xi) is stable in subsequent thresholds.
    Returns dict with threshold, xi, sigma, and equivalent percentile.
    """
    s_abs = s_abs.dropna().values
    results = []
    for q in candidate_quantiles:
        u = np.quantile(s_abs, q)
        excesses = s_abs[s_abs > u] - u
        if len(excesses) < 30:
            continue
        try:
            xi, _, sigma = genpareto.fit(excesses, floc=0)
            ks_stat, ks_p = stats.kstest(excesses, 'genpareto', args=(xi, 0, sigma))
            results.append({
                'quantile': q, 'threshold': u, 'n_exceedances': len(excesses),
                'xi': xi, 'sigma': sigma, 'ks_p': ks_p,
            })
        except Exception:
            continue
    if not results:
        return None
    # Pick lowest threshold where KS p > 0.10 (good fit)
    for r in results:
        if r['ks_p'] > 0.10 and r['n_exceedances'] >= 50:
            return r
    # Fallback: lowest threshold with valid fit
    return results[0] if results else None

gpd_thresholds = {}
for (factor, lookback), s in changes.items():
    # GPD on |change|, but separated by direction
    stress_dir = STRESS_DIRECTION[factor]
    s_stress = s if stress_dir > 0 else -s
    s_stress = s_stress[s_stress > 0]  # Only stress-direction exceedances
    
    result = gpd_justified_threshold(s_stress)
    key = f'{factor}_{lookback}d'
    if result is None:
        gpd_thresholds[key] = {'status': 'failed', 'fallback_pct': 5}
        print(f'{key}: GPD fit failed, falling back to 5%')
        continue
    pct = (1 - result['quantile']) * 100
    gpd_thresholds[key] = {
        'status': 'success',
        'threshold_quantile': float(result['quantile']),
        'threshold_value': float(result['threshold']),
        'equivalent_top_pct': float(pct),
        'xi': float(result['xi']),
        'sigma': float(result['sigma']),
        'n_exceedances': int(result['n_exceedances']),
        'ks_p': float(result['ks_p']),
    }
    print(f'{key}: GPD threshold at {pct:.1f}%-tail (xi={result["xi"]:.3f}, n={result["n_exceedances"]}, KS p={result["ks_p"]:.3f})')

with open(OUT / 'gpd_thresholds.json', 'w') as f:
    json.dump(gpd_thresholds, f, indent=2)

# ============================================================================
# 4. EPISODE DETECTION (across all thresholds)
# ============================================================================
print('\n' + '=' * 80)
print('STEP 3: EPISODE DETECTION')
print('=' * 80)

def detect_episodes(s_signed, threshold_value, stress_direction, min_gap_days=30):
    """
    Detect episodes where signed change exceeds threshold in stress direction.
    Returns list of episode dicts: start, end, peak_date, peak_value, factor_at_peak.
    Merges episodes separated by < min_gap_days.
    """
    if stress_direction > 0:
        in_stress = s_signed >= threshold_value
    else:
        in_stress = s_signed <= -threshold_value
    
    if not in_stress.any():
        return []
    
    # Find continuous stress runs
    in_stress = in_stress.astype(int)
    diff = in_stress.diff().fillna(in_stress.iloc[0])
    starts = in_stress.index[diff == 1].tolist()
    ends_idx = in_stress.index[diff == -1].tolist()
    
    if len(starts) == 0:
        return []
    if in_stress.iloc[-1] == 1:
        ends_idx.append(in_stress.index[-1])
    if len(ends_idx) > len(starts):
        starts = [in_stress.index[0]] + starts
    
    raw_episodes = list(zip(starts, ends_idx))
    
    # Merge close episodes
    merged = []
    for start, end in raw_episodes:
        if merged and (start - merged[-1][1]).days < min_gap_days:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    
    # Characterize each episode
    episodes = []
    for start, end in merged:
        window = s_signed.loc[start:end]
        if stress_direction > 0:
            peak_val = window.max()
            peak_date = window.idxmax()
        else:
            peak_val = window.min()
            peak_date = window.idxmin()
        episodes.append({
            'start_date': start,
            'peak_date': peak_date,
            'end_date': end,
            'peak_signed_change': float(peak_val),
            'duration_days': (end - start).days,
        })
    return episodes

# Detect at all (factor, lookback, threshold) combos
all_episodes = []
threshold_sources = []  # what gave us this threshold (pct or 'gpd')

for (factor, lookback), s_signed in changes.items():
    s_abs = s_signed.abs()
    stress_dir = STRESS_DIRECTION[factor]
    
    # Sensitivity thresholds
    for pct in EPISODE_THRESHOLDS_PCT:
        threshold_val = s_abs.quantile(1 - pct/100)
        eps = detect_episodes(s_signed, threshold_val, stress_dir)
        for e in eps:
            e.update({
                'factor': factor, 'lookback': lookback,
                'threshold_source': f'pct_{pct}',
                'threshold_pct': pct,
                'threshold_value': float(threshold_val),
                'stress_direction': stress_dir,
            })
            all_episodes.append(e)
    
    # GPD-justified threshold
    gpd_key = f'{factor}_{lookback}d'
    if gpd_thresholds[gpd_key]['status'] == 'success':
        threshold_val = gpd_thresholds[gpd_key]['threshold_value']
        if stress_dir < 0:
            threshold_val = -threshold_val if False else threshold_val  # threshold magnitude
        eps = detect_episodes(s_signed, threshold_val, stress_dir)
        for e in eps:
            e.update({
                'factor': factor, 'lookback': lookback,
                'threshold_source': 'gpd',
                'threshold_pct': gpd_thresholds[gpd_key]['equivalent_top_pct'],
                'threshold_value': float(threshold_val),
                'stress_direction': stress_dir,
            })
            all_episodes.append(e)

episodes_df = pd.DataFrame(all_episodes)
episodes_df.to_parquet(OUT / 'episodes_full.parquet')
print(f'Detected {len(episodes_df)} total episodes across all (factor, lookback, threshold) combos')
print()
print('Episode counts by (factor, lookback) at GPD threshold:')
gpd_eps = episodes_df[episodes_df['threshold_source'] == 'gpd']
for (f, lb), g in gpd_eps.groupby(['factor', 'lookback']):
    print(f'  {f} ({lb}d): {len(g)} episodes')

# ============================================================================
# 5. LEAD TIME ANALYSIS
# ============================================================================
print('\n' + '=' * 80)
print('STEP 4: LEAD TIME ANALYSIS (P_crisis vs factor peak)')
print('=' * 80)

def lead_time_for_episode(episode, tvtp_series, threshold):
    """
    Lead time = days between (TVTP P_crisis first crossing threshold) and (factor peak).
    Positive = TVTP fired before peak. Negative = TVTP fired after peak.
    Returns None if TVTP never crossed threshold in the relevant pre-peak window.
    
    We look back up to 180 days before peak for the first P_crisis > threshold.
    """
    peak_date = episode['peak_date']
    lookback_start = peak_date - pd.Timedelta(days=180)
    
    pre_peak = tvtp_series.loc[lookback_start:peak_date]
    crossings = pre_peak[pre_peak > threshold]
    
    if len(crossings) == 0:
        return None  # Never fired
    
    first_cross = crossings.index[0]
    lead_days = (peak_date - first_cross).days
    return lead_days

lead_records = []
for _, ep in episodes_df.iterrows():
    if ep['peak_date'] not in tvtp.index and tvtp.index[tvtp.index <= ep['peak_date']].empty:
        continue
    for crisis_thr in CRISIS_THRESHOLDS:
        lead = lead_time_for_episode(ep, tvtp['p_crisis'], crisis_thr)
        lead_records.append({
            **ep.to_dict(),
            'crisis_threshold': crisis_thr,
            'lead_days': lead,
            'fired': lead is not None,
        })

leads_df = pd.DataFrame(lead_records)
leads_df.to_parquet(OUT / 'episode_lead_times.parquet')

# Summary by threshold combo
print('\nLead time summary (median, IQR) by episode threshold and crisis threshold:')
print(f'{"Episode thr":<14} {"Crisis thr":<11} {"N":<5} {"Fired%":<8} {"Median lead":<12} {"Q25 lead":<10} {"Q75 lead":<10}')
print('-' * 75)
for ep_src in sorted(leads_df['threshold_source'].unique()):
    for ct in CRISIS_THRESHOLDS:
        sub = leads_df[(leads_df['threshold_source'] == ep_src) & (leads_df['crisis_threshold'] == ct)]
        if len(sub) == 0:
            continue
        fired = sub['fired'].sum()
        if fired > 0:
            leads = sub.loc[sub['fired'], 'lead_days']
            print(f'{ep_src:<14} {ct:<11} {len(sub):<5} {fired/len(sub)*100:<7.1f}% {leads.median():<11.1f} {leads.quantile(0.25):<9.1f} {leads.quantile(0.75):<9.1f}')

# ============================================================================
# 6. ROC ANALYSIS — threshold-independent predictive value
# ============================================================================
print('\n' + '=' * 80)
print('STEP 5: ROC ANALYSIS (P_crisis as stress predictor)')
print('=' * 80)

# Build aligned dataset: for each TVTP date, is there a "stress episode" starting within next 60 days?
# Use GPD episodes at 90d lookback for VIX (canonical stress).
canonical_eps = episodes_df[
    (episodes_df['threshold_source'] == 'gpd') &
    (episodes_df['lookback'] == 90)
]

# Label each TVTP date: 1 if any episode starts within next 60d
labels = pd.Series(0, index=tvtp.index)
for _, ep in canonical_eps.iterrows():
    start = ep['start_date']
    label_window_start = start - pd.Timedelta(days=60)
    label_window_end = start
    mask = (labels.index >= label_window_start) & (labels.index <= label_window_end)
    labels.loc[mask] = 1

aligned = pd.DataFrame({'p_crisis': tvtp['p_crisis'], 'label': labels}).dropna()
print(f'ROC sample: {len(aligned)} dates, positive rate = {aligned["label"].mean()*100:.2f}%')

if aligned['label'].sum() > 0 and aligned['label'].sum() < len(aligned):
    fpr, tpr, thr = roc_curve(aligned['label'], aligned['p_crisis'])
    roc_auc = auc(fpr, tpr)
    
    # Youden's J
    j = tpr - fpr
    j_idx = j.argmax()
    youden_threshold = thr[j_idx]
    youden_tpr = tpr[j_idx]
    youden_fpr = fpr[j_idx]
    
    print(f'\nROC AUC: {roc_auc:.4f}')
    print(f"Youden's J optimal threshold: {youden_threshold:.4f}")
    print(f'  TPR at Youden J: {youden_tpr:.4f}')
    print(f'  FPR at Youden J: {youden_fpr:.4f}')
    
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thr})
    roc_df.to_parquet(OUT / 'roc_curves.parquet')
    
    roc_summary = {
        'auc': float(roc_auc),
        'youden_threshold': float(youden_threshold),
        'youden_tpr': float(youden_tpr),
        'youden_fpr': float(youden_fpr),
        'positive_rate': float(aligned['label'].mean()),
        'n_obs': int(len(aligned)),
    }
else:
    print('Insufficient label variation for ROC; skipping')
    roc_summary = {'status': 'failed', 'reason': 'insufficient label variation'}

with open(OUT / 'roc_summary.json', 'w') as f:
    json.dump(roc_summary, f, indent=2)

# ============================================================================
# 7. BLOCK BOOTSTRAP — confidence intervals
# ============================================================================
print('\n' + '=' * 80)
print('STEP 6: BLOCK BOOTSTRAP CIs FOR LEAD TIMES')
print('=' * 80)

def block_bootstrap_lead_times(leads, n_iter=N_BOOTSTRAP, block_len=BLOCK_LEN, seed=SEED):
    """Block bootstrap to handle autocorrelation in lead time series."""
    rng = np.random.default_rng(seed)
    leads = np.array(leads)
    n = len(leads)
    if n == 0:
        return None
    n_blocks = max(1, n // block_len)
    medians = []
    means = []
    for _ in range(n_iter):
        block_starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        sample = np.concatenate([leads[s:s+block_len] for s in block_starts])
        sample = sample[:n]
        medians.append(np.median(sample))
        means.append(np.mean(sample))
    return {
        'median': float(np.median(medians)),
        'median_ci_low': float(np.percentile(medians, 2.5)),
        'median_ci_high': float(np.percentile(medians, 97.5)),
        'mean': float(np.median(means)),
        'mean_ci_low': float(np.percentile(means, 2.5)),
        'mean_ci_high': float(np.percentile(means, 97.5)),
    }

bootstrap_results = {}
for (ep_src, ct), grp in leads_df[leads_df['fired']].groupby(['threshold_source', 'crisis_threshold']):
    leads = grp['lead_days'].values
    boot = block_bootstrap_lead_times(leads)
    if boot is not None:
        bootstrap_results[f'{ep_src}_crisis_{ct}'] = {
            'n_episodes': len(leads),
            **boot,
        }

with open(OUT / 'bootstrap_results.json', 'w') as f:
    json.dump(bootstrap_results, f, indent=2)

print(f'\n95% CIs for median lead time (selected thresholds):')
print(f'{"Combo":<35} {"N":<5} {"Median":<10} {"CI Low":<10} {"CI High":<10}')
for k, v in bootstrap_results.items():
    if 'gpd' in k or 'pct_5' in k:
        print(f'{k:<35} {v["n_episodes"]:<5} {v["median"]:<10.1f} {v["median_ci_low"]:<10.1f} {v["median_ci_high"]:<10.1f}')

# ============================================================================
# 8. GRANGER CAUSALITY
# ============================================================================
print('\n' + '=' * 80)
print('STEP 7: GRANGER CAUSALITY (TVTP <-> factors)')
print('=' * 80)

# Daily aligned series
common_idx = tvtp.index.intersection(covariates.index)
gc_data = pd.DataFrame({
    'p_crisis': tvtp.loc[common_idx, 'p_crisis'].values,
    'p_bear': tvtp.loc[common_idx, 'p_bear'].values,
    'vix': covariates.loc[common_idx, 'vix_level'].values,
    'credit': covariates.loc[common_idx, 'credit_spread'].values,
    'curve': covariates.loc[common_idx, 'yield_curve_slope'].values,
}).dropna()

# First-difference for stationarity
gc_data_diff = gc_data.diff().dropna()

granger_results = {}
test_pairs = [
    ('p_crisis_predicts_vix', ['vix', 'p_crisis']),
    ('vix_predicts_p_crisis', ['p_crisis', 'vix']),
    ('p_crisis_predicts_credit', ['credit', 'p_crisis']),
    ('credit_predicts_p_crisis', ['p_crisis', 'credit']),
    ('p_crisis_predicts_curve', ['curve', 'p_crisis']),
    ('curve_predicts_p_crisis', ['p_crisis', 'curve']),
]
for name, cols in test_pairs:
    try:
        # statsmodels grangercausalitytests: cols[0] is "caused", cols[1] is "cause"
        # Tests if cols[1] Granger-causes cols[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = grangercausalitytests(gc_data_diff[cols].values, maxlag=GRANGER_MAX_LAG, verbose=False)
        # Extract F-test p-values per lag
        p_values = {lag: float(res[lag][0]['ssr_ftest'][1]) for lag in res}
        min_p = min(p_values.values())
        granger_results[name] = {
            'p_values_by_lag': p_values,
            'min_p_value': min_p,
            'best_lag': int(min(p_values, key=p_values.get)),
            'significant_at_5pct': bool(min_p < 0.05),
        }
        print(f'{name}: min p-value = {min_p:.4f} at lag {granger_results[name]["best_lag"]} {"(significant)" if min_p < 0.05 else ""}')
    except Exception as e:
        granger_results[name] = {'status': 'failed', 'error': str(e)}
        print(f'{name}: FAILED ({e})')

with open(OUT / 'granger_test_results.json', 'w') as f:
    json.dump(granger_results, f, indent=2)

# ============================================================================
# 9. INFORMATION VALUE — forward NAV drawdown regression
# ============================================================================
print('\n' + '=' * 80)
print('STEP 8: INFORMATION VALUE (forward drawdown regression)')
print('=' * 80)

# Compute forward 30/60/90 day max drawdown from each date
nav_series = nav['nav']
forward_dd = {}
for ld in LOOKAHEAD_DAYS:
    fdd = []
    for date in nav_series.index:
        future = nav_series.loc[date:date + pd.Timedelta(days=ld)]
        if len(future) < 2:
            fdd.append(np.nan)
            continue
        cmax = future.cummax()
        dd = (future / cmax - 1).min()
        fdd.append(dd)
    forward_dd[ld] = pd.Series(fdd, index=nav_series.index)

info_value_results = {}
for ld in LOOKAHEAD_DAYS:
    common = tvtp.index.intersection(covariates.index).intersection(forward_dd[ld].dropna().index)
    df = pd.DataFrame({
        'fwd_dd': forward_dd[ld].loc[common].values,
        'vix': covariates.loc[common, 'vix_level'].values,
        'credit': covariates.loc[common, 'credit_spread'].values,
        'curve': covariates.loc[common, 'yield_curve_slope'].values,
        'p_crisis': tvtp.loc[common, 'p_crisis'].values,
        'p_bear': tvtp.loc[common, 'p_bear'].values,
    }).dropna()
    
    # Model 1: factors only
    X1 = sm.add_constant(df[['vix', 'credit', 'curve']])
    m1 = sm.OLS(df['fwd_dd'], X1).fit(cov_type='HAC', cov_kwds={'maxlags': BLOCK_LEN})
    
    # Model 2: factors + TVTP
    X2 = sm.add_constant(df[['vix', 'credit', 'curve', 'p_crisis', 'p_bear']])
    m2 = sm.OLS(df['fwd_dd'], X2).fit(cov_type='HAC', cov_kwds={'maxlags': BLOCK_LEN})
    
    # Likelihood ratio test (TVTP coefficients jointly zero?)
    lr_stat = 2 * (m2.llf - m1.llf)
    lr_df = 2  # two extra params
    lr_p = 1 - stats.chi2.cdf(lr_stat, lr_df)
    
    info_value_results[f'fwd_{ld}d'] = {
        'n_obs': len(df),
        'model_factors_only': {
            'r_squared': float(m1.rsquared),
            'adj_r_squared': float(m1.rsquared_adj),
            'aic': float(m1.aic),
            'bic': float(m1.bic),
        },
        'model_factors_plus_tvtp': {
            'r_squared': float(m2.rsquared),
            'adj_r_squared': float(m2.rsquared_adj),
            'aic': float(m2.aic),
            'bic': float(m2.bic),
            'p_crisis_coef': float(m2.params['p_crisis']),
            'p_crisis_pvalue': float(m2.pvalues['p_crisis']),
            'p_bear_coef': float(m2.params['p_bear']),
            'p_bear_pvalue': float(m2.pvalues['p_bear']),
        },
        'r_squared_improvement': float(m2.rsquared - m1.rsquared),
        'likelihood_ratio_test': {
            'lr_statistic': float(lr_stat),
            'df': lr_df,
            'p_value': float(lr_p),
            'significant_at_5pct': bool(lr_p < 0.05),
        },
    }
    print(f'\nForward {ld}d drawdown regression (N={len(df)}):')
    print(f'  R^2 factors only: {m1.rsquared:.4f}')
    print(f'  R^2 factors + TVTP: {m2.rsquared:.4f}')
    print(f'  Improvement: +{(m2.rsquared - m1.rsquared)*100:.2f}pp')
    print(f'  LR test p-value: {lr_p:.4f} {"(significant)" if lr_p < 0.05 else ""}')

with open(OUT / 'forward_drawdown_regression.json', 'w') as f:
    json.dump(info_value_results, f, indent=2)

# ============================================================================
# 10. RULE-BASED BASELINE COMPARISON
# ============================================================================
print('\n' + '=' * 80)
print('STEP 9: RULE-BASED BASELINE COMPARISON (2009-2026)')
print('=' * 80)

# Build daily RB regime forward-filled
rb_daily = rb['composite'].reindex(pd.date_range(rb.index.min(), rb.index.max(), freq='D'), method='ffill')
rb_daily = rb_daily.reindex(tvtp.index, method='ffill').dropna()

# Lead time for RB: when did RB first switch to 'crisis' before each episode?
def rb_lead_time(episode, rb_series, target_state='crisis'):
    peak = episode['peak_date']
    if peak not in rb_series.index and rb_series.index[rb_series.index <= peak].empty:
        return None
    lookback_start = peak - pd.Timedelta(days=180)
    pre = rb_series.loc[lookback_start:peak]
    matches = pre[pre == target_state]
    if len(matches) == 0:
        return None
    return (peak - matches.index[0]).days

rb_records = []
canonical_eps_in_rb = canonical_eps[canonical_eps['peak_date'] >= rb.index.min()]
for _, ep in canonical_eps_in_rb.iterrows():
    rb_lead = rb_lead_time(ep, rb_daily, 'crisis')
    # Match TVTP at threshold 0.5 for fair comparison
    tvtp_lead = lead_time_for_episode(ep, tvtp['p_crisis'], 0.5)
    rb_records.append({
        'factor': ep['factor'],
        'peak_date': ep['peak_date'],
        'rb_lead_days': rb_lead,
        'tvtp_lead_days_at_0.5': tvtp_lead,
        'rb_fired': rb_lead is not None,
        'tvtp_fired': tvtp_lead is not None,
    })

rb_compare_df = pd.DataFrame(rb_records)
rb_summary = {
    'n_episodes_compared': len(rb_compare_df),
    'rb_fire_rate': float(rb_compare_df['rb_fired'].mean()) if len(rb_compare_df) else None,
    'tvtp_fire_rate': float(rb_compare_df['tvtp_fired'].mean()) if len(rb_compare_df) else None,
    'rb_median_lead': float(rb_compare_df.loc[rb_compare_df['rb_fired'], 'rb_lead_days'].median()) if rb_compare_df['rb_fired'].any() else None,
    'tvtp_median_lead': float(rb_compare_df.loc[rb_compare_df['tvtp_fired'], 'tvtp_lead_days_at_0.5'].median()) if rb_compare_df['tvtp_fired'].any() else None,
    'both_fired_n': int((rb_compare_df['rb_fired'] & rb_compare_df['tvtp_fired']).sum()),
}
both = rb_compare_df[rb_compare_df['rb_fired'] & rb_compare_df['tvtp_fired']]
if len(both) >= 5:
    diff = both['tvtp_lead_days_at_0.5'] - both['rb_lead_days']
    t_stat, t_p = stats.ttest_1samp(diff, 0)
    w_stat, w_p = stats.wilcoxon(both['tvtp_lead_days_at_0.5'], both['rb_lead_days'])
    rb_summary['paired_test'] = {
        'mean_diff_days': float(diff.mean()),
        'median_diff_days': float(diff.median()),
        't_statistic': float(t_stat),
        't_p_value': float(t_p),
        'wilcoxon_p_value': float(w_p),
    }
    print(f'\nPaired comparison (N={len(both)} episodes both detected):')
    print(f'  Mean lead diff (TVTP - RB): {diff.mean():+.1f} days')
    print(f'  Median lead diff: {diff.median():+.1f} days')
    print(f'  Paired t-test p-value: {t_p:.4f}')
    print(f'  Wilcoxon p-value: {w_p:.4f}')

with open(OUT / 'rule_based_baseline.json', 'w') as f:
    json.dump(rb_summary, f, indent=2)
rb_compare_df.to_parquet(OUT / 'rb_baseline_episode_comparison.parquet')

# ============================================================================
# 11. THRESHOLD SENSITIVITY MASTER TABLE
# ============================================================================
print('\n' + '=' * 80)
print('STEP 10: THRESHOLD SENSITIVITY MASTER TABLE')
print('=' * 80)

sensitivity = {}
for ep_src in sorted(leads_df['threshold_source'].unique()):
    sensitivity[ep_src] = {}
    for ct in CRISIS_THRESHOLDS:
        sub = leads_df[(leads_df['threshold_source'] == ep_src) & (leads_df['crisis_threshold'] == ct)]
        fired = sub[sub['fired']]
        if len(fired) > 0:
            boot_key = f'{ep_src}_crisis_{ct}'
            boot = bootstrap_results.get(boot_key, {})
            sensitivity[ep_src][f'crisis_{ct}'] = {
                'n_episodes': int(len(sub)),
                'n_fired': int(len(fired)),
                'fire_rate': float(len(fired) / len(sub)),
                'median_lead_days': float(fired['lead_days'].median()),
                'mean_lead_days': float(fired['lead_days'].mean()),
                'q25_lead_days': float(fired['lead_days'].quantile(0.25)),
                'q75_lead_days': float(fired['lead_days'].quantile(0.75)),
                'bootstrap_median_ci': [boot.get('median_ci_low'), boot.get('median_ci_high')],
            }
        else:
            sensitivity[ep_src][f'crisis_{ct}'] = {'n_episodes': int(len(sub)), 'n_fired': 0}

with open(OUT / 'threshold_sensitivity.json', 'w') as f:
    json.dump(sensitivity, f, indent=2)

# ============================================================================
# 12. DECISION SUMMARY (markdown)
# ============================================================================
print('\n' + '=' * 80)
print('STEP 11: WRITING DECISION SUMMARY')
print('=' * 80)

summary_md = f"""# EXP010: Factor Trend Analysis — Decision Summary

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Methodology Recap

- **Episode detection:** Top-pct of 90/120-day signed factor changes, sensitivity over [1,2,3,4,5,7,10,13]% + GPD-justified per-factor threshold
- **Crisis threshold sensitivity:** P_crisis at [0.3, 0.5, 0.7]
- **Statistical tests:** ROC AUC, Youden's J, block bootstrap 95% CI, Granger causality, forward-drawdown regression, paired comparison vs rule-based baseline

## Headline Results

### ROC AUC of P_crisis as stress predictor
- **AUC = {roc_summary.get('auc', 'N/A'):.4f}** (60-day forward window, GPD episodes)
- Youden's J optimal threshold: {roc_summary.get('youden_threshold', 'N/A'):.4f}
- Reference: AUC=0.5 is random, AUC>0.7 is "useful", AUC>0.8 is "strong"

### Forward drawdown regression (information value)
"""
for ld in LOOKAHEAD_DAYS:
    r = info_value_results[f'fwd_{ld}d']
    summary_md += f"\n- **{ld}d forward:** R^2 factors only = {r['model_factors_only']['r_squared']:.4f}, +TVTP = {r['model_factors_plus_tvtp']['r_squared']:.4f} (+{r['r_squared_improvement']*100:.2f}pp), LR p-value = {r['likelihood_ratio_test']['p_value']:.4f}"

summary_md += "\n\n### Granger causality (TVTP vs factors, first-differenced)\n"
for name, r in granger_results.items():
    if 'min_p_value' in r:
        sig = '***' if r['min_p_value'] < 0.01 else ('**' if r['min_p_value'] < 0.05 else '')
        summary_md += f"- {name}: p={r['min_p_value']:.4f} (lag {r['best_lag']}) {sig}\n"

summary_md += "\n### Rule-based baseline comparison\n"
if 'paired_test' in rb_summary:
    pt = rb_summary['paired_test']
    summary_md += f"""- N episodes (both fired): {rb_summary['both_fired_n']}
- TVTP median lead: {rb_summary['tvtp_median_lead']:.1f} days
- RB median lead: {rb_summary['rb_median_lead']:.1f} days
- Mean diff (TVTP - RB): {pt['mean_diff_days']:+.1f} days
- Paired t p-value: {pt['t_p_value']:.4f}
- Wilcoxon p-value: {pt['wilcoxon_p_value']:.4f}
"""

summary_md += "\n## Files Saved\n"
for p in sorted(OUT.glob('*')):
    summary_md += f"- `{p.name}`\n"

with open(OUT / 'decision_summary.md', 'w', encoding='utf-8') as f:
    f.write(summary_md)

print('\n' + '=' * 80)
print('ALL DONE')
print('=' * 80)
print(f'Output directory: {OUT}')
print('Files saved:')
for p in sorted(OUT.glob('*')):
    sz = p.stat().st_size
    sz_str = f'{sz/1024:.1f}KB' if sz < 1e6 else f'{sz/1e6:.1f}MB'
    print(f'  {p.name:<50} {sz_str}')
