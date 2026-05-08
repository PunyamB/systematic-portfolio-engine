"""
Step 7: K=3 (B2) vs K=4 (M3b/M6b) walk-forward comparison.

Reads per-window saved data from both runs and produces:
- Per-window metrics (CAGR, Sharpe, MaxDD)
- Win counts on each metric
- Aggregate stitched comparison
- Integration criteria C1/C2/C3 for K=4
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

K3_DIR = Path("regime_switching/data/window_fits")
K4_DIR = Path("regime_switching/data/k4_extension_constrained/window_fits")
HARD_LOG = Path("D:/Projects/StrategyResearchLab/experiments/exp006_extended_wf/walk_forward_log.json")
OUT_DIR = Path("regime_switching/data/k4_extension_constrained")

# Window IDs in order
WINDOW_IDS = list(range(1, 18)) + ["holdout"]


def compute_window_metrics(nav_series):
    """Return CAGR, Sharpe, MaxDD for a NAV series."""
    nav = nav_series.dropna()
    if len(nav) < 2:
        return None
    rets = nav.pct_change().dropna()
    n_y = len(rets) / 252
    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    cagr = (1 + total_ret) ** (1/n_y) - 1 if n_y > 0 else 0
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / ann_vol if ann_vol > 0 else 0
    max_dd = float(((nav / nav.cummax()) - 1).min())
    return {
        "cagr": cagr * 100,
        "sharpe": sharpe,
        "max_dd": max_dd * 100,
        "final_nav": nav.iloc[-1],
        "n_days": len(nav),
    }


def load_window_nav(base_dir, wid):
    p = base_dir / f"window_{wid}" / "daily_nav.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df["nav"]


# ============================================================
# PER-WINDOW COMPARISON
# ============================================================

print("=" * 90)
print(f"{'Window':<8} | {'K=3 CAGR':<10} {'K=4 CAGR':<10} {'Diff':<8} | {'K=3 Shp':<8} {'K=4 Shp':<8} {'Diff':<8} | {'K=3 DD':<8} {'K=4 DD':<8}")
print("=" * 90)

per_window_results = []

for wid in WINDOW_IDS:
    nav_k3 = load_window_nav(K3_DIR, wid)
    nav_k4 = load_window_nav(K4_DIR, wid)

    if nav_k3 is None or nav_k4 is None:
        print(f"{str(wid):<8} | MISSING DATA")
        continue

    m_k3 = compute_window_metrics(nav_k3)
    m_k4 = compute_window_metrics(nav_k4)

    cagr_diff = m_k4["cagr"] - m_k3["cagr"]
    sharpe_diff = m_k4["sharpe"] - m_k3["sharpe"]

    print(f"{str(wid):<8} | "
          f"{m_k3['cagr']:>8.2f}%  {m_k4['cagr']:>8.2f}%  {cagr_diff:>+6.2f}% | "
          f"{m_k3['sharpe']:>7.3f}  {m_k4['sharpe']:>7.3f}  {sharpe_diff:>+6.3f} | "
          f"{m_k3['max_dd']:>7.2f}% {m_k4['max_dd']:>7.2f}%")

    per_window_results.append({
        "window": str(wid),
        "k3_cagr": m_k3["cagr"], "k4_cagr": m_k4["cagr"],
        "k3_sharpe": m_k3["sharpe"], "k4_sharpe": m_k4["sharpe"],
        "k3_max_dd": m_k3["max_dd"], "k4_max_dd": m_k4["max_dd"],
        "k4_minus_k3_cagr": cagr_diff,
        "k4_minus_k3_sharpe": sharpe_diff,
    })

print("=" * 90)

# ============================================================
# WIN COUNTS
# ============================================================

n = len(per_window_results)
k4_cagr_wins = sum(1 for r in per_window_results if r["k4_cagr"] > r["k3_cagr"])
k4_sharpe_wins = sum(1 for r in per_window_results if r["k4_sharpe"] > r["k3_sharpe"])
k4_dd_wins = sum(1 for r in per_window_results if r["k4_max_dd"] > r["k3_max_dd"])  # less negative = better

print(f"\nWin counts (out of {n} windows):")
print(f"  K=4 CAGR wins:   {k4_cagr_wins}/{n}")
print(f"  K=4 Sharpe wins: {k4_sharpe_wins}/{n}")
print(f"  K=4 MaxDD wins:  {k4_dd_wins}/{n} (less negative = better)")

mean_cagr_diff = np.mean([r["k4_minus_k3_cagr"] for r in per_window_results])
mean_sharpe_diff = np.mean([r["k4_minus_k3_sharpe"] for r in per_window_results])
print(f"\nAverage K=4 - K=3:")
print(f"  CAGR:   {mean_cagr_diff:+.3f}%")
print(f"  Sharpe: {mean_sharpe_diff:+.4f}")

# ============================================================
# STITCHED AGGREGATE COMPARISON
# ============================================================

def stitch_navs(base_dir):
    INITIAL_CAPITAL = 1_000_000.0
    pieces = []
    running_nav = INITIAL_CAPITAL
    sorted_wins = [w for w in range(1, 18)]
    for wid in sorted_wins:
        nav = load_window_nav(base_dir, wid)
        if nav is None: continue
        scale = running_nav / nav.iloc[0]
        nav = nav * scale
        running_nav = nav.iloc[-1]
        pieces.append(nav)
    nav_holdout = load_window_nav(base_dir, "holdout")
    if nav_holdout is not None:
        scale = running_nav / nav_holdout.iloc[0]
        pieces.append(nav_holdout * scale)
    if not pieces:
        return None
    stitched = pd.concat(pieces).sort_index()
    return stitched[~stitched.index.duplicated(keep="last")]


print("\n" + "=" * 60)
print("STITCHED AGGREGATE")
print("=" * 60)

s_k3 = stitch_navs(K3_DIR)
s_k4 = stitch_navs(K4_DIR)

agg_k3 = compute_window_metrics(s_k3)
agg_k4 = compute_window_metrics(s_k4)

print(f"{'Metric':<12} {'K=3':<12} {'K=4':<12} {'Diff':<10}")
print("-" * 50)
print(f"{'CAGR':<12} {agg_k3['cagr']:>9.2f}%   {agg_k4['cagr']:>9.2f}%   {agg_k4['cagr'] - agg_k3['cagr']:>+7.2f}%")
print(f"{'Sharpe':<12} {agg_k3['sharpe']:>10.3f}   {agg_k4['sharpe']:>10.3f}   {agg_k4['sharpe'] - agg_k3['sharpe']:>+8.3f}")
print(f"{'MaxDD':<12} {agg_k3['max_dd']:>9.2f}%   {agg_k4['max_dd']:>9.2f}%   {agg_k4['max_dd'] - agg_k3['max_dd']:>+7.2f}%")
print(f"{'Final NAV':<12} ${agg_k3['final_nav']:>10,.0f}  ${agg_k4['final_nav']:>10,.0f}")

# ============================================================
# INTEGRATION CRITERIA C1/C2/C3 FOR K=4
# ============================================================

print("\n" + "=" * 60)
print("K=4 INTEGRATION CRITERIA (vs Hard regime baseline)")
print("=" * 60)

# Hard regime metrics (from EXP006 log)
with open(HARD_LOG) as f:
    hard_log = json.load(f)

# Read EXP006 stitched if available, otherwise compute from per-window
hard_dir = Path("D:/Projects/StrategyResearchLab/experiments/exp006_extended_wf/data/wf_results")
print(f"\nHard regime baseline (EXP006):")

# Try the metrics file
hard_metrics_path = Path("D:/Projects/StrategyResearchLab/experiments/exp006_extended_wf/data/exp006_metrics.json")
if hard_metrics_path.exists():
    with open(hard_metrics_path) as f:
        hard_metrics = json.load(f)
    print(f"  CAGR:   {hard_metrics.get('cagr')}%")
    print(f"  Sharpe: {hard_metrics.get('sharpe')}")
    print(f"  MaxDD:  {hard_metrics.get('max_drawdown')}%")
    hard_cagr = hard_metrics.get("cagr")
    hard_sharpe = hard_metrics.get("sharpe")
else:
    print("  (Hard metrics file not found, using B2 published numbers)")
    hard_cagr = 23.56
    hard_sharpe = 1.309

# C3: walk-forward improvement vs hard
c3_cagr_diff = agg_k4["cagr"] - hard_cagr
c3_sharpe_diff = agg_k4["sharpe"] - hard_sharpe

print(f"\nC3: Walk-forward improvement (K=4 soft vs hard)")
print(f"  CAGR:   {c3_cagr_diff:+.2f}% ({'PASS' if c3_cagr_diff > 0 else 'FAIL'})")
print(f"  Sharpe: {c3_sharpe_diff:+.3f} ({'PASS' if c3_sharpe_diff > 0 else 'FAIL'})")
c3_pass = c3_cagr_diff > 0 and c3_sharpe_diff > 0
print(f"  C3 OVERALL: {'PASS' if c3_pass else 'FAIL'}")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "per_window": per_window_results,
    "win_counts": {
        "n_windows": n,
        "k4_cagr_wins": k4_cagr_wins,
        "k4_sharpe_wins": k4_sharpe_wins,
        "k4_dd_wins": k4_dd_wins,
        "mean_cagr_diff": float(mean_cagr_diff),
        "mean_sharpe_diff": float(mean_sharpe_diff),
    },
    "aggregate": {
        "k3_cagr": agg_k3["cagr"], "k3_sharpe": agg_k3["sharpe"], "k3_max_dd": agg_k3["max_dd"],
        "k4_cagr": agg_k4["cagr"], "k4_sharpe": agg_k4["sharpe"], "k4_max_dd": agg_k4["max_dd"],
        "k3_final_nav": agg_k3["final_nav"], "k4_final_nav": agg_k4["final_nav"],
        "hard_cagr": hard_cagr, "hard_sharpe": hard_sharpe,
    },
    "c3_evaluation": {
        "cagr_diff_vs_hard": c3_cagr_diff,
        "sharpe_diff_vs_hard": c3_sharpe_diff,
        "pass": c3_pass,
    },
}

with open(OUT_DIR / "k3_vs_k4_comparison.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nResults saved to: {OUT_DIR / 'k3_vs_k4_comparison.json'}")
