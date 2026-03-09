import json
with open('walk_forward_log.json') as f:
    log = json.load(f)
for w in log['windows']:
    params = w.get('calibrated_params') or {}
    lam = params.get('lambda')
    ra = params.get('risk_aversion')
    print("Window", w['window_id'], ": lambda=", lam, "| risk_aversion=", ra)
holdout = log['final_holdout']
params = holdout.get('calibrated_params') or {}
print("Holdout: lambda=", params.get('lambda'), "| risk_aversion=", params.get('risk_aversion'))