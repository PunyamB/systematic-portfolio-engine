# notebooks/wf_layer3_stitch.py
# WALK-FORWARD LAYER 3 — Stitch + Final Analytics
#
# Run ONLY after all 9 walk-forward windows AND the final holdout are complete.
# Stitches all test-window NAV series into one continuous out-of-sample series.
# Computes institutional metrics on the stitched series vs SPY.
# Ranks all 4 strategies and prints live recommendation.
# Generates charts.
#
# Run: python notebooks/wf_layer3_stitch.py

import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WF_RESULTS = Path("data/backtest/wf_results")
RESULTS_DIR = Path("data/backtest/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE   = Path("walk_forward_log.json")

STRATEGIES = [
    ("mv",  "monthly",   "MV-Monthly"),
    ("mv",  "quarterly", "MV-Quarterly"),
    ("bl",  "monthly",   "BL-Monthly"),
    ("bl",  "quarterly", "BL-Quarterly"),
]

COLORS = {
    "mv_monthly":   "#2196F3",
    "mv_quarterly": "#FF9800",
    "bl_monthly":   "#9C27B0",
    "bl_quarterly": "#F44336",
}

print("[wf_layer3] ================================================")
print("[wf_layer3] Walk-Forward Stitch + Analytics")
print("[wf_layer3] ================================================\n")


# ============================================================
# LOAD LOG
# ============================================================

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

completed = [w for w in log["windows"] if w["status"] == "complete"]
print(f"[wf_layer3] Completed walk-forward windows: {len(completed)}/9")

holdout_complete = log["final_holdout"]["status"] == "complete"
print(f"[wf_layer3] Final holdout complete: {holdout_complete}")

if len(completed) == 0:
    print("[wf_layer3] No completed windows found. Run all walk-forward windows first.")
    sys.exit(1)


# ============================================================
# STITCH NAV SERIES
# ============================================================

def stitch_navs(strategy: str, frequency: str) -> pd.Series:
    """
    Stitches test-window NAV series into one continuous series.
    Each window's NAV is rebased to continue from the previous window's
    final NAV value, maintaining continuity of compounding.
    """
    all_pieces = []
    running_nav = 1_000_000.0

    for w in sorted(completed, key=lambda x: x["window_id"]):
        wid      = w["window_id"]
        file_key = f"window_{wid}_{strategy}_{frequency}"
        path     = WF_RESULTS / f"nav_{file_key}.parquet"

        if not path.exists():
            print(f"[wf_layer3] WARNING: {path} not found, skipping window {wid}")
            continue

        nav = pd.read_parquet(path)["nav"]
        nav.index = pd.to_datetime(nav.index)

        # Rebase: scale so this window starts at running_nav
        scale = running_nav / nav.iloc[0]
        nav   = nav * scale

        running_nav = nav.iloc[-1]
        all_pieces.append(nav)

    # Append holdout if complete
    if holdout_complete:
        file_key = f"window_holdout_{strategy}_{frequency}"
        path     = WF_RESULTS / f"nav_{file_key}.parquet"
        if path.exists():
            nav   = pd.read_parquet(path)["nav"]
            nav.index = pd.to_datetime(nav.index)
            scale = running_nav / nav.iloc[0]
            nav   = nav * scale
            all_pieces.append(nav)

    if not all_pieces:
        return pd.Series(dtype=float)

    stitched = pd.concat(all_pieces).sort_index()
    # Remove duplicate dates at window boundaries
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    return stitched


print("[wf_layer3] Stitching NAV series...")
stitched_navs = {}
for strategy, frequency, label in STRATEGIES:
    key = f"{strategy}_{frequency}"
    nav = stitch_navs(strategy, frequency)
    if not nav.empty:
        stitched_navs[key] = nav
        print(f"[wf_layer3] {label}: {len(nav)} days | "
              f"{nav.index[0].date()} to {nav.index[-1].date()}")

if not stitched_navs:
    print("[wf_layer3] No stitched NAV series available.")
    sys.exit(1)


# ============================================================
# SPY BENCHMARK
# ============================================================

spy_path = Path("data/raw/prices.parquet")
spy_df   = pd.read_parquet(spy_path) if spy_path.exists() else pd.DataFrame()
if not spy_df.empty:
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    spy = spy_df[spy_df["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
else:
    spy = pd.Series(dtype=float)

if spy.empty:
    from data.fetchers.fmp_fetcher import get_daily_prices
    from datetime import date as date_
    spy_raw = get_daily_prices("SPY", "2009-01-01", date_.today().strftime("%Y-%m-%d"))
    if not spy_raw.empty:
        spy_raw["date"] = pd.to_datetime(spy_raw["date"])
        spy = spy_raw.set_index("date")["close"].sort_index()
    else:
        print("[wf_layer3] ERROR: Could not load SPY.")
        sys.exit(1)

start_date = min(nav.index[0] for nav in stitched_navs.values())
end_date   = max(nav.index[-1] for nav in stitched_navs.values())
spy_window = spy[(spy.index >= start_date) & (spy.index <= end_date)].dropna()
print(f"[wf_layer3] SPY: {len(spy_window)} days | {spy_window.index[0].date()} to {spy_window.index[-1].date()}\n")


# ============================================================
# METRICS
# ============================================================

LOWER_IS_BETTER = {"ann_vol", "max_drawdown", "max_dd_days",
                   "tracking_error", "down_capture", "beta"}

def compute_metrics(nav: pd.Series, spy: pd.Series, label: str) -> dict:
    common    = nav.index.intersection(spy.index)
    nav       = nav.reindex(common).dropna()
    bench     = spy.reindex(common).dropna()
    bench_nav = (bench / bench.iloc[0]) * nav.iloc[0]

    pr  = nav.pct_change().dropna()
    br  = bench_nav.pct_change().dropna()
    cr  = pr.index.intersection(br.index)
    pr, br = pr.reindex(cr).dropna(), br.reindex(cr).dropna()
    ex  = pr - br
    n_y = len(pr) / 252

    total_ret  = (nav.iloc[-1] / nav.iloc[0]) - 1
    cagr       = (1 + total_ret) ** (1 / n_y) - 1
    bench_cagr = ((bench_nav.iloc[-1] / bench_nav.iloc[0]) ** (1 / n_y)) - 1
    ann_vol    = pr.std() * np.sqrt(252)
    down_vol   = pr[pr < 0].std() * np.sqrt(252)
    sharpe     = cagr / ann_vol if ann_vol > 0 else 0
    sortino    = cagr / down_vol if down_vol > 0 else 0

    roll_max   = nav.cummax()
    dd         = (nav / roll_max) - 1
    max_dd     = dd.min()
    calmar     = cagr / abs(max_dd) if max_dd != 0 else 0
    dd_end     = dd.idxmin()
    dd_start   = nav[:dd_end].idxmax()
    dd_days    = (dd_end - dd_start).days

    te         = ex.std() * np.sqrt(252)
    ir         = (ex.mean() * 252) / te if te > 0 else 0
    beta       = pr.cov(br) / br.var() if br.var() > 0 else 1.0
    alpha      = cagr - beta * bench_cagr

    up_m   = br[br > 0]
    dn_m   = br[br < 0]
    up_cap = (pr.reindex(up_m.index).mean() / up_m.mean() * 100) if len(up_m) > 0 else 0
    dn_cap = (pr.reindex(dn_m.index).mean() / dn_m.mean() * 100) if len(dn_m) > 0 else 0

    monthly_ex = (1 + ex).resample("ME").prod() - 1
    hit_rate   = (monthly_ex > 0).mean() if len(monthly_ex) > 0 else 0

    return {
        "label":          label,
        "cagr":           round(cagr * 100, 2),
        "bench_cagr":     round(bench_cagr * 100, 2),
        "total_return":   round(total_ret * 100, 2),
        "ann_vol":        round(ann_vol * 100, 2),
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "calmar":         round(calmar, 3),
        "max_drawdown":   round(max_dd * 100, 2),
        "max_dd_days":    dd_days,
        "tracking_error": round(te * 100, 2),
        "info_ratio":     round(ir, 3),
        "alpha":          round(alpha * 100, 2),
        "beta":           round(beta, 3),
        "up_capture":     round(up_cap, 1),
        "down_capture":   round(dn_cap, 1),
        "hit_rate":       round(hit_rate * 100, 1),
        "annual_port":    (1 + pr).resample("YE").prod() - 1,
        "annual_bench":   (1 + br).resample("YE").prod() - 1,
    }


STRATEGY_MAP = {f"{s}_{f}": lbl for s, f, lbl in STRATEGIES}
all_metrics  = {k: compute_metrics(nav, spy_window, STRATEGY_MAP[k])
                for k, nav in stitched_navs.items()}


# ============================================================
# PRINT METRICS TABLE
# ============================================================

METRIC_ROWS = [
    ("CAGR (%)",               "cagr"),
    ("Benchmark CAGR (%)",     "bench_cagr"),
    ("Total Return (%)",       "total_return"),
    ("Ann. Volatility (%)",    "ann_vol"),
    ("Sharpe Ratio",           "sharpe"),
    ("Sortino Ratio",          "sortino"),
    ("Calmar Ratio",           "calmar"),
    ("Max Drawdown (%)",       "max_drawdown"),
    ("Max DD Duration (days)", "max_dd_days"),
    ("Tracking Error (%)",     "tracking_error"),
    ("Information Ratio",      "info_ratio"),
    ("Alpha (%)",              "alpha"),
    ("Beta",                   "beta"),
    ("Up Capture (%)",         "up_capture"),
    ("Down Capture (%)",       "down_capture"),
    ("Hit Rate (%)",           "hit_rate"),
]

keys = list(all_metrics.keys())

print("=" * 85)
print("WALK-FORWARD OUT-OF-SAMPLE PERFORMANCE SUMMARY")
print(f"Period: {start_date.date()} to {end_date.date()} (stitched test windows)")
print("=" * 85)
header = f"{'Metric':<30}" + "".join(f"{STRATEGY_MAP[k]:>14}" for k in keys)
print(header)
print("-" * 85)

per_metric_winner = {}
for lbl, key in METRIC_ROWS:
    vals   = {k: all_metrics[k][key] for k in keys}
    best_k = min(vals, key=vals.get) if key in LOWER_IS_BETTER else max(vals, key=vals.get)
    per_metric_winner[key] = best_k
    row = f"{lbl:<30}"
    for k in keys:
        tag = " *" if k == best_k else "  "
        row += f"{str(vals[k])+tag:>14}"
    print(row)

print("=" * 85)
print("  * = best in class")


# ============================================================
# RANKING
# ============================================================

RANK_WEIGHTS = {"sharpe": 3, "info_ratio": 3, "calmar": 2,
                "max_drawdown": 2, "hit_rate": 1, "alpha": 1}

composite_scores = {k: 0 for k in keys}
for metric, weight in RANK_WEIGHTS.items():
    vals = [(k, all_metrics[k][metric]) for k in keys]
    for rank, (k, _) in enumerate(
        sorted(vals, key=lambda x: x[1], reverse=(metric not in LOWER_IS_BETTER))
    ):
        composite_scores[k] += (len(keys) - rank) * weight

ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
winner = ranked[0][0]

print("\n" + "=" * 85)
print("STRATEGY RANKING (walk-forward out-of-sample)")
print("=" * 85)
print(f"{'Rank':<6}{'Strategy':<20}{'Score':<10}{'Sharpe':<10}{'IR':<10}{'Calmar':<10}{'MaxDD%':<10}")
print("-" * 85)
for rank, (k, score) in enumerate(ranked, 1):
    m = all_metrics[k]
    print(f"{rank:<6}{STRATEGY_MAP[k]:<20}{score:<10}"
          f"{m['sharpe']:<10}{m['info_ratio']:<10}"
          f"{m['calmar']:<10}{m['max_drawdown']:<10}")

print("=" * 85)
print(f"\nRECOMMENDED FOR LIVE: {STRATEGY_MAP[winner]}")
print(f"Composite score: {composite_scores[winner]} pts | "
      f"Sharpe: {all_metrics[winner]['sharpe']} | "
      f"IR: {all_metrics[winner]['info_ratio']} | "
      f"Alpha: {all_metrics[winner]['alpha']}%")
print("=" * 85)


# ============================================================
# CALENDAR YEAR RETURNS
# ============================================================

print("\nCALENDAR YEAR RETURNS (%) — Out-of-Sample Only")
print("-" * 85)
all_years_raw = sorted(set(yr for k in keys for yr in all_metrics[k]["annual_port"].index))
seen_years = set()
all_years = []
for yr in all_years_raw:
    if yr.year not in seen_years:
        seen_years.add(yr.year)
        all_years.append(yr)

yr_header = f"{'Year':<8}" + "".join(f"{STRATEGY_MAP[k]:>14}" for k in keys) + f"{'SPY':>10}"
print(yr_header)
print("-" * 85)
for yr in all_years:
    row = f"{yr.year:<8}"
    for k in keys:
        val = all_metrics[k]["annual_port"].get(yr)
        row += f"{round(val*100,1) if val is not None else 'N/A':>14}"
    spy_val = all_metrics[keys[0]]["annual_bench"].get(yr)
    row += f"{round(spy_val*100,1) if spy_val is not None else 'N/A':>10}"
    print(row)
print("=" * 85)

# ============================================================
# SAVE
# ============================================================

metrics_out = pd.DataFrame([
    {"strategy": k, **{mk: mv for mk, mv in m.items()
                       if mk not in ["annual_port", "annual_bench", "label"]}}
    for k, m in all_metrics.items()
])
metrics_out["composite_score"]  = metrics_out["strategy"].map(composite_scores)
metrics_out["rank"]             = metrics_out["composite_score"].rank(ascending=False).astype(int)
metrics_out["recommended_live"] = metrics_out["strategy"] == winner
metrics_out.to_csv(RESULTS_DIR / "wf_metrics_summary.csv", index=False)
print(f"\n[wf_layer3] Metrics saved: {RESULTS_DIR / 'wf_metrics_summary.csv'}")

# Save parameter log summary
param_summary = []
for w in log["windows"]:
    if w.get("calibrated_params"):
        param_summary.append({
            "window_id":    w["window_id"],
            "train_end":    w["train_end"],
            "test_year":    w["test_start"][:4],
            "lambda":       w["calibrated_params"].get("lambda"),
            "risk_aversion": w["calibrated_params"].get("risk_aversion"),
            "ic_weighted":  w.get("ic_weighted_signals", False),
        })
if param_summary:
    pd.DataFrame(param_summary).to_csv(RESULTS_DIR / "wf_parameter_log.csv", index=False)
    print(f"[wf_layer3] Parameter log saved: {RESULTS_DIR / 'wf_parameter_log.csv'}")


# ============================================================
# PLOTS
# ============================================================

print("[wf_layer3] Generating plots...")
fig, axes = plt.subplots(3, 1, figsize=(16, 18))
fig.suptitle("Walk-Forward Out-of-Sample Backtest — 4 Strategies",
             fontsize=14, fontweight="bold")

spy_norm = spy_window / spy_window.iloc[0]

# Plot 1: Cumulative return
ax1 = axes[0]
for k, nav in stitched_navs.items():
    m   = all_metrics[k]
    tag = " [LIVE PICK]" if k == winner else ""
    lbl = f"{STRATEGY_MAP[k]}{tag} | Sharpe={m['sharpe']} | IR={m['info_ratio']} | CAGR={m['cagr']}%"
    ax1.plot(nav.index, nav / nav.iloc[0], label=lbl,
             color=COLORS[k], linewidth=2.2 if k == winner else 1.5,
             linestyle="-" if k == winner else "--")
ax1.plot(spy_norm.index, spy_norm,
         label=f"SPY | CAGR={all_metrics[keys[0]]['bench_cagr']}%",
         color="#4CAF50", linewidth=1.5, linestyle=":")
ax1.set_title("Cumulative Return — Walk-Forward Out-of-Sample", fontsize=11)
ax1.set_ylabel("Growth of $1M")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Plot 2: Drawdown
ax2 = axes[1]
for k, nav in stitched_navs.items():
    dd = (nav / nav.cummax()) - 1
    ax2.fill_between(dd.index, dd * 100, 0, alpha=0.4, color=COLORS[k],
                     label=f"{STRATEGY_MAP[k]} (max={all_metrics[k]['max_drawdown']}%)")
spy_dd = (spy_window / spy_window.cummax()) - 1
ax2.fill_between(spy_dd.index, spy_dd * 100, 0, alpha=0.2, color="#4CAF50", label="SPY")
ax2.set_title("Drawdown (%)", fontsize=11)
ax2.set_ylabel("Drawdown (%)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Plot 3: Rolling 12-month IR
ax3 = axes[2]
for k, nav in stitched_navs.items():
    pr     = nav.pct_change().dropna()
    br     = spy_window.pct_change().dropna()
    common = pr.index.intersection(br.index)
    ex     = pr.reindex(common) - br.reindex(common)
    roll_ir = (ex.rolling(252).mean() * 252) / (ex.rolling(252).std() * np.sqrt(252))
    ax3.plot(roll_ir.index, roll_ir, label=STRATEGY_MAP[k],
             color=COLORS[k], linewidth=2.2 if k == winner else 1.5,
             linestyle="-" if k == winner else "--")
ax3.axhline(0,    color="black", linewidth=0.8, linestyle="--")
ax3.axhline(0.5,  color="gray",  linewidth=0.8, linestyle=":", alpha=0.7, label="IR=0.5")
ax3.axhline(0.75, color="gray",  linewidth=0.8, linestyle=":", alpha=0.4, label="IR=0.75")
ax3.set_title("Rolling 12-Month Information Ratio — Out-of-Sample", fontsize=11)
ax3.set_ylabel("Information Ratio")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plot_path = RESULTS_DIR / "wf_backtest_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"[wf_layer3] Plot saved: {plot_path}")

print(f"\n[wf_layer3] ================================================")
print(f"[wf_layer3] Walk-forward analytics complete.")
print(f"[wf_layer3] Recommended for live: {STRATEGY_MAP[winner]}")
print(f"[wf_layer3] ================================================")