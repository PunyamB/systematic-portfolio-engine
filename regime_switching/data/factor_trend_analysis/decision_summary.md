# EXP010: Factor Trend Analysis - Decision Summary

Generated: 2026-05-04 14:10

## Methodology

- Episode detection: top-pct of 90/120-day signed factor changes, sensitivity over [1,2,3,4,5,7,10,13]% + GPD-justified per-factor threshold
- Crisis threshold sensitivity: P_crisis at [0.3, 0.5, 0.7]
- Statistical tests: ROC AUC, Youden's J, block bootstrap 95% CI, Granger causality, forward-drawdown regression, paired comparison vs rule-based baseline

## Headline Results

### ROC AUC of P_crisis as stress predictor
- AUC = 0.5983
- Youden's J optimal threshold: 0.0052
- TPR at Youden J: 0.5040
- FPR at Youden J: 0.3352

### Forward drawdown regression (information value)
- **30d forward**: R^2 factors only = 0.0182, +TVTP = 0.0183 (+0.02pp), LR p = 0.6457
- **60d forward**: R^2 factors only = 0.0193, +TVTP = 0.0204 (+0.10pp), LR p = 0.0624
- **90d forward**: R^2 factors only = 0.0235, +TVTP = 0.0251 (+0.16pp), LR p = 0.0128

### Granger causality (first-differenced daily series)
- p_crisis_predicts_vix: p=0.0655 (lag 4) 
- vix_predicts_p_crisis: p=0.0000 (lag 5) ***
- p_crisis_predicts_credit: p=0.0003 (lag 3) ***
- credit_predicts_p_crisis: p=0.0000 (lag 5) ***
- p_crisis_predicts_curve: p=0.0964 (lag 3) 
- curve_predicts_p_crisis: p=0.0081 (lag 2) ***

### Rule-based baseline comparison (2009-2026)
- N episodes (both fired): 9
- TVTP median lead (at 0.5): 38.5 days
- RB median lead: 55.0 days
- Mean diff (TVTP - RB): +27.0 days
- Paired t p-value: 0.0394
- Wilcoxon p-value: 0.0547

## Files Saved
- `bootstrap_results.json`
- `episode_lead_times.parquet`
- `episodes_full.parquet`
- `factor_change_dist_stats.json`
- `factor_change_distributions.parquet`
- `forward_drawdown_regression.json`
- `gpd_thresholds.json`
- `granger_test_results.json`
- `rb_baseline_episode_comparison.parquet`
- `roc_curves.parquet`
- `roc_summary.json`
- `rule_based_baseline.json`
- `threshold_sensitivity.json`
