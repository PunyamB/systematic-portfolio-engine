# notebooks/backtest_layer3_analytics.py
# LAYER 3 — Analytics and Strategy Ranking (instant, re-run anytime)
#
# Reads all 4 NAV series from Layer 2 and produces:
#   - Full metrics table (all 4 strategies side by side)
#   - Per-metric winner
#   - Composite ranking score across all key metrics
#   - Live recommendation with reasoning
#   - Calendar year returns table
#   - Three charts: cumulative return, drawdown, rolling 12-month IR
#   - Saves metrics_summary.csv and backtest_results.png
#
# Run: python notebooks/backtest_layer3_analytics.py

import sys
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

BACKTEST_START  = pd.Timestamp("2009-01-01")
BACKTEST_END    = pd.Timestamp("2026-01-01")
INITIAL_CAPITAL = 1_000_000.0

RESULTS_DIR  = Path("data/backtest/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_LABELS = {
    "mv_monthly":    "MV-Monthly",
    "mv_quarterly":  "MV-Quarterly",
    "bl_monthly":    "BL-Monthly",
    "bl_quarterly":  "BL-Quarterly",
}

COLORS = {
    "mv_monthly":   "#2196F3",
    "mv_quarterly": "#FF9800",
    "bl_monthly":   "#9C27B0",
    "bl_quarterly": "#F44336",
}

print("[layer3] Loading results...")

# ============================================================
# LOAD NAV SERIES
# ============================================================

navs = {}
for key in STRATEGY_LABELS:
    path = RESULTS_DIR / f"nav_{key}.parquet"
    if not path.exists():
        print(f"[layer3] WARNING: {path} not found — run Layer 2 first")
        continue
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    navs[key] = df["nav"].sort_index()

if not navs:
    print("[layer3] No results found. Run backtest_layer2_simulate.py first.")
    sys.exit(1)

print(f"[layer3] Strategies loaded: {list(navs.keys())}")


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
    print("[layer3] SPY not in raw prices — fetching from FMP...")
    from data.fetchers.fmp_fetcher import get_daily_prices
    from datetime import date as date_
    spy_raw = get_daily_prices("SPY", "2009-01-01", date_.today().strftime("%Y-%m-%d"))
    if not spy_raw.empty:
        spy_raw["date"] = pd.to_datetime(spy_raw["date"])
        spy = spy_raw.set_index("date")["close"].sort_index()
    else:
        print("[layer3] ERROR: Could not load SPY. Benchmark comparison unavailable.")
        sys.exit(1)

spy_window = spy[(spy.index >= BACKTEST_START) & (spy.index < BACKTEST_END)].dropna()
print(f"[layer3] SPY: {len(spy_window)} days\n")


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_metrics(nav: pd.Series, spy: pd.Series, label: str) -> dict:
    nav   = nav.sort_index().dropna()
    bench = spy.sort_index().dropna()

    common   = nav.index.intersection(bench.index)
    nav      = nav.reindex(common).dropna()
    bench    = bench.reindex(common).dropna()
    bench_nav = (bench / bench.iloc[0]) * nav.iloc[0]

    pr  = nav.pct_change().dropna()
    br  = bench_nav.pct_change().dropna()
    common_r = pr.index.intersection(br.index)
    pr       = pr.reindex(common_r).dropna()
    br       = br.reindex(common_r).dropna()
    ex       = pr - br
    n_y      = len(pr) / 252

    total_ret   = (nav.iloc[-1] / nav.iloc[0]) - 1
    cagr        = (1 + total_ret) ** (1 / n_y) - 1
    bench_total = (bench_nav.iloc[-1] / bench_nav.iloc[0]) - 1
    bench_cagr  = (1 + bench_total) ** (1 / n_y) - 1

    ann_vol  = pr.std() * np.sqrt(252)
    down_vol = pr[pr < 0].std() * np.sqrt(252)
    sharpe   = cagr / ann_vol if ann_vol > 0 else 0
    sortino  = cagr / down_vol if down_vol > 0 else 0

    roll_max = nav.cummax()
    dd       = (nav / roll_max) - 1
    max_dd   = dd.min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0

    dd_end   = dd.idxmin()
    dd_start = nav[:dd_end].idxmax()
    dd_days  = (dd_end - dd_start).days

    te         = ex.std() * np.sqrt(252)
    info_ratio = (ex.mean() * 252) / te if te > 0 else 0
    beta       = pr.cov(br) / br.var() if br.var() > 0 else 1.0
    alpha      = cagr - beta * bench_cagr

    up_m   = br[br > 0]
    dn_m   = br[br < 0]
    up_cap = (pr.reindex(up_m.index).mean() / up_m.mean() * 100) if len(up_m) > 0 else 0
    dn_cap = (pr.reindex(dn_m.index).mean() / dn_m.mean() * 100) if len(dn_m) > 0 else 0

    monthly_ex = (1 + ex).resample("ME").prod() - 1
    hit_rate   = (monthly_ex > 0).mean() if len(monthly_ex) > 0 else 0

    annual_port  = (1 + pr).resample("YE").prod() - 1
    annual_bench = (1 + br).resample("YE").prod() - 1

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
        "info_ratio":     round(info_ratio, 3),
        "alpha":          round(alpha * 100, 2),
        "beta":           round(beta, 3),
        "up_capture":     round(up_cap, 1),
        "down_capture":   round(dn_cap, 1),
        "hit_rate":       round(hit_rate * 100, 1),
        "annual_port":    annual_port,
        "annual_bench":   annual_bench,
    }


all_metrics = {k: compute_metrics(nav, spy_window, STRATEGY_LABELS[k])
               for k, nav in navs.items()}


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

print("=" * 80)
print("PERFORMANCE SUMMARY  (2009-01-01 to 2026-01-01)")
print("=" * 80)
header = f"{'Metric':<30}" + "".join(f"{STRATEGY_LABELS[k]:>12}" for k in keys)
print(header)
print("-" * 80)

# Track per-metric winners (higher is better for most, lower for drawdown/vol/beta)
LOWER_IS_BETTER = {"ann_vol", "max_drawdown", "max_dd_days", "tracking_error", "down_capture", "beta"}

per_metric_winner = {}
for lbl, key in METRIC_ROWS:
    vals    = {k: all_metrics[k][key] for k in keys}
    best_k  = min(vals, key=vals.get) if key in LOWER_IS_BETTER else max(vals, key=vals.get)
    per_metric_winner[key] = best_k
    row = f"{lbl:<30}"
    for k in keys:
        v   = vals[k]
        tag = " *" if k == best_k else "  "
        row += f"{str(v)+tag:>12}"
    print(row)

print("=" * 80)
print("  * = best in class for this metric")


# ============================================================
# RANKING AND RECOMMENDATION
# ============================================================

# Score each strategy: 4 points for best, 3 for second, etc.
# Weighted: Sharpe x3, IR x3, Calmar x2, Max DD x2, Hit Rate x1, Alpha x1
RANK_WEIGHTS = {
    "sharpe":       3,
    "info_ratio":   3,
    "calmar":       2,
    "max_drawdown": 2,
    "hit_rate":     1,
    "alpha":        1,
}

composite_scores = {k: 0 for k in keys}

for metric, weight in RANK_WEIGHTS.items():
    vals   = [(k, all_metrics[k][metric]) for k in keys]
    sorted_vals = sorted(vals, key=lambda x: x[1],
                         reverse=(metric not in LOWER_IS_BETTER))
    for rank, (k, _) in enumerate(sorted_vals):
        points = (len(keys) - rank) * weight
        composite_scores[k] += points

ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
winner = ranked[0][0]

print("\n" + "=" * 80)
print("STRATEGY RANKING")
print("=" * 80)
print(f"{'Rank':<6}{'Strategy':<20}{'Composite Score':<20}{'Sharpe':<12}{'IR':<12}{'Calmar':<12}")
print("-" * 80)
for rank, (k, score) in enumerate(ranked, 1):
    m = all_metrics[k]
    print(f"{rank:<6}{STRATEGY_LABELS[k]:<20}{score:<20}{m['sharpe']:<12}{m['info_ratio']:<12}{m['calmar']:<12}")

print("=" * 80)
print(f"\nRECOMMENDED FOR LIVE: {STRATEGY_LABELS[winner]}")
print(f"Reason: Highest composite score ({composite_scores[winner]} pts) across "
      f"Sharpe, Information Ratio, Calmar, Max Drawdown, Hit Rate, and Alpha.")
print(f"\nSharpe: {all_metrics[winner]['sharpe']} | "
      f"IR: {all_metrics[winner]['info_ratio']} | "
      f"Calmar: {all_metrics[winner]['calmar']} | "
      f"Max DD: {all_metrics[winner]['max_drawdown']}% | "
      f"Alpha: {all_metrics[winner]['alpha']}%")
print("=" * 80)


# ============================================================
# CALENDAR YEAR RETURNS
# ============================================================

print("\nCALENDAR YEAR RETURNS (%)")
print("-" * 80)
all_years = sorted(set(yr for k in keys for yr in all_metrics[k]["annual_port"].index))
yr_header = f"{'Year':<8}" + "".join(f"{STRATEGY_LABELS[k]:>14}" for k in keys) + f"{'SPY':>10}"
print(yr_header)
print("-" * 80)
for yr in all_years:
    row = f"{yr.year:<8}"
    for k in keys:
        val = all_metrics[k]["annual_port"].get(yr)
        row += f"{round(val*100, 1) if val is not None else 'N/A':>14}"
    spy_val = all_metrics[keys[0]]["annual_bench"].get(yr)
    row += f"{round(spy_val*100, 1) if spy_val is not None else 'N/A':>10}"
    print(row)
print("=" * 80)


# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics_out = pd.DataFrame([
    {"strategy": k, **{mk: mv for mk, mv in m.items()
                       if mk not in ["annual_port", "annual_bench", "label"]}}
    for k, m in all_metrics.items()
])
metrics_out["composite_score"] = metrics_out["strategy"].map(composite_scores)
metrics_out["rank"]            = metrics_out["composite_score"].rank(ascending=False).astype(int)
metrics_out["recommended_live"] = metrics_out["strategy"] == winner
metrics_out.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
metrics_out.to_parquet(RESULTS_DIR / "metrics_summary.parquet", index=False)
print(f"\n[layer3] Metrics saved: {RESULTS_DIR / 'metrics_summary.csv'}")


# ============================================================
# PLOTS
# ============================================================

print("[layer3] Generating plots...")

fig, axes = plt.subplots(3, 1, figsize=(16, 18))
fig.suptitle("SystematicPortfolioEngine — 4-Strategy Backtest (2009–2026)",
             fontsize=14, fontweight="bold")

first_nav_date = min(nav.index[0] for nav in navs.values())
spy_aligned    = spy_window[spy_window.index >= first_nav_date]
spy_norm       = spy_aligned / spy_aligned.iloc[0]

# Plot 1: Cumulative return
ax1 = axes[0]
for k, nav in navs.items():
    m   = all_metrics[k]
    tag = " [RECOMMENDED]" if k == winner else ""
    lbl = f"{STRATEGY_LABELS[k]}{tag} | Sharpe={m['sharpe']} | IR={m['info_ratio']} | CAGR={m['cagr']}%"
    ax1.plot(nav.index, nav / nav.iloc[0], label=lbl,
             color=COLORS[k], linewidth=2.0 if k == winner else 1.5,
             linestyle="-" if k == winner else "--")
ax1.plot(spy_norm.index, spy_norm,
         label=f"SPY | CAGR={all_metrics[keys[0]]['bench_cagr']}%",
         color="#4CAF50", linewidth=1.5, linestyle=":")
ax1.set_title("Cumulative Return (Growth of $1)", fontsize=11)
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Plot 2: Drawdown
ax2 = axes[1]
for k, nav in navs.items():
    dd = (nav / nav.cummax()) - 1
    ax2.fill_between(dd.index, dd * 100, 0, alpha=0.4, color=COLORS[k],
                     label=f"{STRATEGY_LABELS[k]} (max={all_metrics[k]['max_drawdown']}%)")
spy_dd = (spy_aligned / spy_aligned.cummax()) - 1
ax2.fill_between(spy_dd.index, spy_dd * 100, 0, alpha=0.2, color="#4CAF50", label="SPY")
ax2.set_title("Drawdown (%)", fontsize=11)
ax2.set_ylabel("Drawdown (%)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Plot 3: Rolling 12-month IR
ax3 = axes[2]
for k, nav in navs.items():
    pr     = nav.pct_change().dropna()
    br     = spy_aligned.pct_change().dropna()
    common = pr.index.intersection(br.index)
    ex     = pr.reindex(common) - br.reindex(common)
    roll_ir = (ex.rolling(252).mean() * 252) / (ex.rolling(252).std() * np.sqrt(252))
    ax3.plot(roll_ir.index, roll_ir, label=STRATEGY_LABELS[k],
             color=COLORS[k], linewidth=2.0 if k == winner else 1.5,
             linestyle="-" if k == winner else "--")
ax3.axhline(0,    color="black", linewidth=0.8, linestyle="--")
ax3.axhline(0.5,  color="gray",  linewidth=0.8, linestyle=":", alpha=0.7, label="IR=0.5")
ax3.axhline(0.75, color="gray",  linewidth=0.8, linestyle=":", alpha=0.4, label="IR=0.75")
ax3.set_title("Rolling 12-Month Information Ratio", fontsize=11)
ax3.set_ylabel("Information Ratio")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plot_path = RESULTS_DIR / "backtest_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"[layer3] Plot saved: {plot_path}")

print("\n[layer3] ================================================")
print("[layer3] Analytics complete.")
print(f"[layer3] Recommended for live: {STRATEGY_LABELS[winner]}")
print(f"[layer3] Results in: {RESULTS_DIR}")
print("[layer3] ================================================")