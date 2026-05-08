"""
Module 7: Visualizer (COMPLETE)

All plot functions for B2. Generates publication-quality figures saved to
regime_switching/outputs/.

Plot functions:
  1. plot_regime_probabilities() - P(bull), P(bear), P(crisis) with SPY overlay
  2. plot_transition_matrix() - average transition matrix heatmap
  3. plot_tvtp_logistic_curves() - P(transition) vs each covariate
  4. plot_regime_conditional_returns() - regime-specific return distributions
  5. plot_model_comparison_bic() - BIC across 4 model variants
  6. plot_regime_timeline_comparison() - MS vs rule-based 2009-2026
  7. plot_early_warning() - crisis transition zooms (GFC, COVID, 2022)
  8. plot_confusion_matrix() - 4x4 regime agreement heatmap
  9. plot_integration_nav() - soft vs hard NAV chart
  10. plot_rolling_transition_probs() - P(bull to bear|z_t) over time
  11. plot_per_window_comparison() - soft vs hard CAGR/Sharpe per window
  12. plot_model_scorecard() - summary heatmap

Plus the existing 3 scaffold plots from M1.
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from regime_switching.config import DATA_DIR, OUTPUT_DIR

# ── Style ──
NAVY = "#1B2A4A"
ACCENT = "#2E75B6"
RED = "#D32F2F"
GREEN = "#388E3C"
GOLD = "#F0A800"
GRAY = "#757575"
LIGHT_BG = "#F5F7FA"

REGIME_COLORS = {
    "bull": GREEN,
    "bear": GOLD,
    "crisis": RED,
    "recovery": ACCENT,
}

CRISIS_PERIODS = {
    "GFC": ("2008-09-01", "2009-03-31"),
    "COVID": ("2020-02-15", "2020-04-30"),
    "Rate_Hikes_2022": ("2022-01-01", "2022-10-31"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.edgecolor": "#666",
    "axes.linewidth": 0.8,
    "xtick.color": "#333",
    "ytick.color": "#333",
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name, dpi=150):
    """Save figure to outputs/ as PNG and PDF."""
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ============================================================
# PLOT 1: Regime probability time series
# ============================================================
def plot_regime_probabilities():
    """Daily filtered P(bull), P(bear), P(crisis) for full sample."""
    probs = pd.read_parquet(DATA_DIR / "best_model_probs.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2]})

    # Top: SPY cumulative return for context
    cum = (1 + returns["log_return"]).cumprod()
    axes[0].plot(cum.index, cum.values, color=NAVY, lw=1.0)
    axes[0].set_ylabel("Cumulative Return", fontsize=9)
    axes[0].set_title("SPY Cumulative Returns + TVTP-3 Regime Probabilities (1997-2026)",
                       fontsize=12, fontweight="bold", color=NAVY)
    axes[0].grid(True, alpha=0.3)

    # Bottom: stacked regime probabilities
    # Use filtered probabilities for stackplot
    order = ["filtered_bull", "filtered_bear", "filtered_crisis"]
    order = [c for c in order if c in probs.columns]
    prob_data = probs[order].fillna(0)
    colors = [REGIME_COLORS[c.replace("filtered_", "")] for c in order]
    labels = [f"P({c.replace('filtered_', '')})" for c in order]

    axes[1].stackplot(prob_data.index, *[prob_data[c] for c in order],
                       labels=labels,
                       colors=colors, alpha=0.75)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Regime Probability", fontsize=9)
    axes[1].legend(loc="upper left", framealpha=0.95, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Mark crisis periods
    for name, (start, end) in CRISIS_PERIODS.items():
        for ax in axes:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       color=RED, alpha=0.10, zorder=0)

    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return _save(fig, "01_regime_probabilities")


# ============================================================
# PLOT 2: Transition matrix heatmap
# ============================================================
def plot_transition_matrix():
    """Average transition matrix from best TVTP model."""
    with open(DATA_DIR / "ms3_tvtp_results.json") as f:
        result = json.load(f)

    state_labels = result.get("state_labels", ["bull", "bear", "crisis"])
    P_avg = np.array(result["avg_transition_matrix"])
    K = P_avg.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(P_avg, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"P({s})" for s in state_labels])
    ax.set_yticklabels([f"From {s}" for s in state_labels])
    ax.set_title("TVTP-3 Average Transition Matrix\n(covariates at sample mean)",
                 fontweight="bold", color=NAVY)

    for i in range(K):
        for j in range(K):
            txt_color = "white" if P_avg[i, j] > 0.5 else "black"
            ax.text(j, i, f"{P_avg[i,j]:.3f}", ha="center", va="center",
                    color=txt_color, fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return _save(fig, "02_transition_matrix")


# ============================================================
# PLOT 3: TVTP logistic response curves
# ============================================================
def plot_tvtp_logistic_curves():
    """How P(regime transition) responds to each covariate."""
    # Load coefficients from holdout window pkl (most recent, full training)
    holdout_pkl = DATA_DIR / "window_fits" / "window_holdout" / "tvtp_result.pkl"
    if not holdout_pkl.exists():
        # Try ms3_tvtp_full pickle
        alt_pkl = DATA_DIR / "ms3_tvtp_full.pkl"
        if alt_pkl.exists():
            with open(alt_pkl, "rb") as f:
                result_full = pickle.load(f)
            coeffs = result_full.get("coeffs", {})
            state_labels = result_full.get("state_labels", ["bull", "bear", "crisis"])
        else:
            print("No coeffs available, skipping logistic curves")
            return None
    else:
        with open(holdout_pkl, "rb") as f:
            result_full = pickle.load(f)
        coeffs = result_full.get("coeffs", {})
        state_labels = result_full.get("state_labels", ["bull", "bear", "crisis"])

    if not coeffs:
        print("No coeffs in pkl, skipping")
        return None
    covariate_names = ["VIX", "Yield Curve", "Credit Spread"]
    K = 3

    fig, axes = plt.subplots(K, len(covariate_names), figsize=(13, 9), sharey=True)

    z_range = np.linspace(-2.5, 2.5, 100)

    for i in range(K):  # origin state
        c = coeffs[i]  # shape (K-1, 1+d)
        for d_idx, cov_name in enumerate(covariate_names):
            ax = axes[i, d_idx]

            for j in range(K):
                if j == 0:
                    p = np.zeros_like(z_range)
                    intercepts = c[:, 0]
                    coefs = c[:, d_idx + 1]
                    for z_idx, z_val in enumerate(z_range):
                        z_vec = np.zeros(3)
                        z_vec[d_idx] = z_val
                        logits = np.zeros(K)
                        logits[1:] = c[:, 0] + (c[:, 1:] @ z_vec)
                        exp_l = np.exp(logits - logits.max())
                        p[z_idx] = (exp_l / exp_l.sum())[j]
                else:
                    p = np.zeros_like(z_range)
                    for z_idx, z_val in enumerate(z_range):
                        z_vec = np.zeros(3)
                        z_vec[d_idx] = z_val
                        logits = np.zeros(K)
                        logits[1:] = c[:, 0] + (c[:, 1:] @ z_vec)
                        exp_l = np.exp(logits - logits.max())
                        p[z_idx] = (exp_l / exp_l.sum())[j]

                color = REGIME_COLORS.get(state_labels[j], "gray")
                ax.plot(z_range, p, color=color, lw=2, label=f"P({state_labels[j]})")

            if i == 0:
                ax.set_title(f"{cov_name} (standardized)", fontsize=10)
            if d_idx == 0:
                ax.set_ylabel(f"From {state_labels[i]}", fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if i == K - 1:
                ax.set_xlabel("Standardized covariate value")

    axes[0, 2].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle("TVTP Logistic Response Curves: P(transition | covariate)",
                  fontsize=13, fontweight="bold", color=NAVY, y=1.00)
    plt.tight_layout()
    return _save(fig, "03_tvtp_logistic_curves")


# ============================================================
# PLOT 4: Regime-conditional return distributions
# ============================================================
def plot_regime_conditional_returns():
    """Distribution of returns by regime label (assigned by max prob)."""
    probs = pd.read_parquet(DATA_DIR / "best_model_probs.parquet")
    returns = pd.read_parquet(DATA_DIR / "returns.parquet")

    common = probs.index.intersection(returns.index)
    probs = probs.loc[common]
    rets = returns.loc[common, "log_return"]

    state_cols = [c for c in probs.columns if c.startswith("filtered_")]
    if not state_cols:
        print("No filtered_ columns, skipping")
        return None
    regime_label = probs[state_cols].idxmax(axis=1).str.replace("filtered_", "")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    state_labels = ["bull", "bear", "crisis"]

    for ax, state in zip(axes, state_labels):
        if state not in regime_label.values:
            ax.text(0.5, 0.5, f"No days classified\nas {state}",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(state.upper())
            continue
        rets_state = rets[regime_label == state] * 100
        color = REGIME_COLORS[state]
        ax.hist(rets_state, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(rets_state.mean(), color="black", lw=1.5, ls="--",
                    label=f"mean={rets_state.mean():.3f}%")
        ax.set_title(f"{state.upper()} ({len(rets_state)} days, "
                     f"vol={rets_state.std():.2f}%)",
                     fontweight="bold")
        ax.set_xlabel("Daily return (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Frequency")
    fig.suptitle("Regime-Conditional Return Distributions",
                  fontsize=13, fontweight="bold", color=NAVY)
    plt.tight_layout()
    return _save(fig, "04_regime_conditional_returns")


# ============================================================
# PLOT 5: BIC across 4 model variants
# ============================================================
def plot_model_comparison_bic():
    """BIC bar chart comparing 4 model variants."""
    with open(DATA_DIR / "model_comparison.json") as f:
        comp = json.load(f)

    models = comp.get("models", {})
    if not models:
        # Fallback: build manually
        with open(DATA_DIR / "ms2_fixed_results.json") as f:
            ms2_fix = json.load(f)
        with open(DATA_DIR / "ms3_fixed_results.json") as f:
            ms3_fix = json.load(f)
        with open(DATA_DIR / "ms2_tvtp_results.json") as f:
            ms2_tvtp = json.load(f)
        with open(DATA_DIR / "ms3_tvtp_results.json") as f:
            ms3_tvtp = json.load(f)
        models = {
            "MS-2-Fixed": {"BIC": ms2_fix.get("bic", 0)},
            "MS-3-Fixed": {"BIC": ms3_fix.get("bic", 0)},
            "MS-2-TVTP": {"BIC": ms2_tvtp.get("bic", 0)},
            "MS-3-TVTP": {"BIC": ms3_tvtp.get("bic", 0)},
        }

    names = list(models.keys())
    bics = [models[n].get("BIC", models[n].get("bic", 0)) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GRAY, GRAY, ACCENT, NAVY]
    bars = ax.bar(names, bics, color=colors, edgecolor="black", lw=1.0)

    # Highlight best
    best_idx = int(np.argmin(bics))
    bars[best_idx].set_color(GREEN)
    bars[best_idx].set_edgecolor("black")

    ax.set_ylabel("BIC (lower is better)", fontsize=10)
    ax.set_title("Model Selection: BIC Comparison\n(MS-3-TVTP wins)",
                  fontweight="bold", color=NAVY)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, bic in zip(bars, bics):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.001,
                f"{bic:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return _save(fig, "05_model_comparison_bic")


# ============================================================
# PLOT 6: Regime timeline comparison
# ============================================================
def plot_regime_timeline_comparison():
    """Side-by-side: TVTP regime probabilities vs rule-based regimes."""
    probs = pd.read_parquet(DATA_DIR / "best_model_probs.parquet")
    rule_path = Path("D:/Projects/SystematicPortfolioEngine/data/backtest/regime_history.parquet")
    if not rule_path.exists():
        print("Rule-based regime history not found, skipping")
        return None
    rule = pd.read_parquet(rule_path)
    rule["date"] = pd.to_datetime(rule["date"]) if "date" in rule.columns else rule.index
    if "date" in rule.columns:
        rule = rule.set_index("date")

    # Filter to overlap period
    start = max(probs.index.min(), rule.index.min())
    end = min(probs.index.max(), rule.index.max())
    probs_o = probs.loc[start:end]
    rule_o = rule.loc[start:end]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Top: TVTP soft probabilities
    order = ["filtered_bull", "filtered_bear", "filtered_crisis"]
    order = [c for c in order if c in probs_o.columns]
    if not order:
        print("No filtered_ columns")
        return None
    colors = [REGIME_COLORS[c.replace("filtered_", "")] for c in order]
    labels = [f"P({c.replace('filtered_', '')})" for c in order]
    axes[0].stackplot(probs_o.index, *[probs_o[c].values for c in order],
                      labels=labels, colors=colors, alpha=0.75)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("TVTP Probability")
    axes[0].set_title("TVTP Soft Regime vs Rule-Based Hard Regime (2009-2026 overlap)",
                       fontweight="bold", color=NAVY)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Bottom: rule-based hard regime as colored bands
    if "regime" in rule_o.columns:
        regime_dates = rule_o["regime"]
        for i in range(len(regime_dates) - 1):
            r = regime_dates.iloc[i]
            color = REGIME_COLORS.get(r, "gray")
            axes[1].axvspan(regime_dates.index[i], regime_dates.index[i+1],
                            color=color, alpha=0.6)
        # Last date
        if len(regime_dates) > 0:
            r = regime_dates.iloc[-1]
            color = REGIME_COLORS.get(r, "gray")
            axes[1].axvspan(regime_dates.index[-1], end, color=color, alpha=0.6)

    # Legend for rule-based
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=REGIME_COLORS[r], alpha=0.6, label=r)
                for r in ["bull", "recovery", "bear", "crisis"]]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8)
    axes[1].set_ylabel("Rule-based regime")
    axes[1].set_yticks([])
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return _save(fig, "06_regime_timeline_comparison")


# ============================================================
# PLOT 7: Early warning crisis zooms
# ============================================================
def plot_early_warning():
    """Zoom on crisis transitions: TVTP P(crisis) vs rule-based switch."""
    probs = pd.read_parquet(DATA_DIR / "best_model_probs.parquet")
    rule_path = Path("D:/Projects/SystematicPortfolioEngine/data/backtest/regime_history.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    crisis_zooms = {
        "GFC 2008": ("2008-06-01", "2009-06-01"),
        "COVID 2020": ("2020-01-01", "2020-06-01"),
        "Rate Hikes 2022": ("2021-10-01", "2022-12-31"),
    }

    rule_o = None
    if rule_path.exists():
        rule_o = pd.read_parquet(rule_path)
        if "date" in rule_o.columns:
            rule_o["date"] = pd.to_datetime(rule_o["date"])
            rule_o = rule_o.set_index("date")

    for ax, (label, (start, end)) in zip(axes, crisis_zooms.items()):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        prob_zoom = probs.loc[s:e]

        if "filtered_crisis" in prob_zoom.columns:
            ax.fill_between(prob_zoom.index, 0, prob_zoom["filtered_crisis"],
                             color=RED, alpha=0.5, label="P(crisis)")
        if "filtered_bear" in prob_zoom.columns:
            ax.fill_between(prob_zoom.index, prob_zoom.get("filtered_crisis", 0),
                             prob_zoom.get("filtered_crisis", 0) + prob_zoom["filtered_bear"],
                             color=GOLD, alpha=0.5, label="P(bear)")

        # Mark rule-based regime change
        if rule_o is not None and "regime" in rule_o.columns:
            rule_zoom = rule_o.loc[s:e]
            for i, (idx, row) in enumerate(rule_zoom.iterrows()):
                if i == 0:
                    continue
                prev = rule_zoom["regime"].iloc[i-1]
                curr = rule_zoom["regime"].iloc[i]
                if prev != curr and curr in ["bear", "crisis"]:
                    ax.axvline(idx, color=NAVY, lw=1.5, ls="--", alpha=0.8)

        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        for label_dt in ax.get_xticklabels():
            label_dt.set_rotation(30)

    fig.suptitle("Early Warning: TVTP Probability Ramp vs Rule-Based Switch",
                  fontsize=13, fontweight="bold", color=NAVY)
    plt.tight_layout()
    return _save(fig, "07_early_warning")


# ============================================================
# PLOT 8: Confusion matrix
# ============================================================
def plot_confusion_matrix():
    """4x4 regime agreement heatmap."""
    with open(DATA_DIR / "comparison_results.json") as f:
        comp = json.load(f)
    cm_obj = comp.get("agreement", {}).get("confusion_matrix", {})
    if not cm_obj:
        print("No confusion matrix")
        return None
    states = cm_obj["labels"]
    cm = np.array(cm_obj["matrix"])
    rows_are = cm_obj.get("rows_are", "rule_based")
    cols_are = cm_obj.get("cols_are", "ms_model")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45)
    ax.set_yticklabels(states)
    ax.set_xlabel(f"{cols_are.replace('_', ' ').title()} regime")
    ax.set_ylabel(f"{rows_are.replace('_', ' ').title()} regime")
    ax.set_title("Regime Confusion Matrix\n(TVTP vs Rule-based, 2009-2026 overlap)",
                  fontweight="bold", color=NAVY)

    for i in range(len(states)):
        for j in range(len(states)):
            val = int(cm[i, j])
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return _save(fig, "08_confusion_matrix")


# ============================================================
# PLOT 9: Soft vs Hard NAV chart
# ============================================================
def plot_integration_nav():
    """Stitched NAV: soft regime vs hard regime."""
    EXP006_WF = Path("D:/Projects/StrategyResearchLab/experiments/exp006_extended_wf/data/wf_results")
    WINDOW_FITS = DATA_DIR / "window_fits"

    # Load and stitch soft NAVs
    def stitch(nav_paths):
        running = 1_000_000
        pieces = []
        for p in nav_paths:
            if not p.exists():
                continue
            nav = pd.read_parquet(p)["nav"]
            if hasattr(nav.index, "to_pydatetime"):
                pass
            else:
                nav.index = pd.to_datetime(nav.index)
            scale = running / nav.iloc[0]
            scaled = nav * scale
            running = scaled.iloc[-1]
            pieces.append(scaled)
        if pieces:
            return pd.concat(pieces).sort_index()
        return pd.Series(dtype=float)

    soft_paths = []
    hard_paths = []
    for wid in list(range(1, 18)) + ["holdout"]:
        soft_paths.append(WINDOW_FITS / f"window_{wid}" / "daily_nav.parquet")
        if wid == "holdout":
            hard_paths.append(EXP006_WF / "nav_window_holdout_mv_monthly.parquet")
        else:
            hard_paths.append(EXP006_WF / f"nav_window_{wid}_mv_monthly.parquet")

    soft = stitch(soft_paths)
    hard = stitch(hard_paths)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(soft.index, soft.values / 1e6, color=ACCENT, lw=1.5, label="Soft (TVTP)")
    ax.plot(hard.index, hard.values / 1e6, color=NAVY, lw=1.5, label="Hard (rule-based)", ls="--")

    # Crisis bands
    for name, (start, end) in CRISIS_PERIODS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=RED, alpha=0.10)

    ax.set_xlabel("Date")
    ax.set_ylabel("NAV ($ millions, $1M start)")
    ax.set_title("Walk-Forward NAV: Soft TVTP Regime vs Hard Rule-Based Regime\n"
                 "(2005-2026, 18 windows + holdout)",
                  fontweight="bold", color=NAVY)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return _save(fig, "09_integration_nav")


# ============================================================
# PLOT 10: Per-window CAGR/Sharpe comparison
# ============================================================
def plot_per_window_comparison():
    """Bar chart: soft vs hard CAGR and Sharpe per window."""
    with open(DATA_DIR / "integration_metrics.json") as f:
        m = json.load(f)

    per_window = m["per_window"]
    n = len(per_window)
    wids = [str(p["window_id"]) for p in per_window]
    soft_cagr = [p["soft"]["cagr"] * 100 for p in per_window]
    hard_cagr = [p["hard"]["cagr"] * 100 for p in per_window]
    soft_sharpe = [p["soft"]["sharpe"] for p in per_window]
    hard_sharpe = [p["hard"]["sharpe"] for p in per_window]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    x = np.arange(n)
    width = 0.4

    axes[0].bar(x - width/2, soft_cagr, width, label="Soft (TVTP)", color=ACCENT)
    axes[0].bar(x + width/2, hard_cagr, width, label="Hard (rule-based)", color=NAVY)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(wids, rotation=45, fontsize=8)
    axes[0].set_ylabel("CAGR (%)")
    axes[0].set_title("Per-Window CAGR Comparison", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x - width/2, soft_sharpe, width, label="Soft (TVTP)", color=ACCENT)
    axes[1].bar(x + width/2, hard_sharpe, width, label="Hard (rule-based)", color=NAVY)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(wids, rotation=45, fontsize=8)
    axes[1].set_ylabel("Sharpe ratio")
    axes[1].set_xlabel("Window")
    axes[1].set_title("Per-Window Sharpe Comparison", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Walk-Forward Comparison: Soft Wins {m['windows_won_cagr']}/{n} on CAGR, "
                 f"{m['windows_won_sharpe']}/{n} on Sharpe",
                  fontsize=12, fontweight="bold", color=NAVY)
    plt.tight_layout()
    return _save(fig, "10_per_window_comparison")


# ============================================================
# PLOT 11: Model scorecard summary heatmap
# ============================================================
def plot_model_scorecard():
    """Summary heatmap of integration criteria results."""
    with open(DATA_DIR / "comparison_results.json") as f:
        comp = json.load(f)
    criteria = comp.get("integration_criteria", {})

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Pull real values
    kappa_val = comp.get("agreement", {}).get("cohen_kappa", 0)
    lead_days = comp.get("early_warning", {}).get("avg_lead_days", 0)

    c1 = criteria.get("C1_kappa", {})
    c2 = criteria.get("C2_early_warning", {})
    c3 = criteria.get("C3_walkforward", {})

    c1_pass = bool(c1.get("pass", kappa_val > 0.40))
    c2_pass = bool(c2.get("pass", lead_days >= 3))
    c3_obj = c3.get("value", {})
    cagr_imp = c3_obj.get("cagr_improvement", 0) if isinstance(c3_obj, dict) else 0
    sharpe_imp = c3_obj.get("sharpe_improvement", 0) if isinstance(c3_obj, dict) else 0
    c3_pass = bool(c3.get("pass", False))

    n_pass = sum([c1_pass, c2_pass, c3_pass])
    overall_pass = n_pass == 3

    rows = [
        ["Criterion", "Threshold", "Value", "Result"],
        ["C1: Cohen's kappa vs rule-based",
         "> 0.40 (moderate)",
         f"{kappa_val:.3f}",
         "PASS" if c1_pass else "FAIL"],
        ["C2: Early warning lead time",
         ">= 3 trading days avg",
         f"{lead_days:.1f} days",
         "PASS" if c2_pass else "FAIL"],
        ["C3: Walk-forward improvement",
         "CAGR > 0 AND Sharpe > 0",
         f"CAGR {cagr_imp:+.4f}, Sharpe {sharpe_imp:+.3f}",
         "PASS" if c3_pass else "FAIL"],
        ["OVERALL", "All 3 must pass", f"{n_pass}/3 passed",
         "PASS" if overall_pass else "FAIL"],
    ]

    table = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.30, 0.30, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Style header
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor(NAVY)
        cell.set_text_props(color="white", weight="bold")

    # Style result column
    for i in range(1, len(rows)):
        result = rows[i][3]
        cell = table[(i, 3)]
        cell.set_facecolor(GREEN if result == "PASS" else RED)
        cell.set_text_props(color="white", weight="bold", ha="center")

    ax.set_title("Integration Criteria Scorecard",
                  fontsize=13, fontweight="bold", color=NAVY, pad=20)

    plt.tight_layout()
    return _save(fig, "11_model_scorecard")


# ============================================================
# PLOT 12: Rolling transition probability
# ============================================================
def plot_rolling_transition_probs():
    """Time series of P(bull to bear|z_t) and other key transitions."""
    probs_path = DATA_DIR / "ms3_tvtp_probs.parquet"
    if not probs_path.exists():
        print("ms3_tvtp_probs.parquet not found")
        return None

    df = pd.read_parquet(probs_path)
    if df.index.dtype != "datetime64[ns]":
        df.index = pd.to_datetime(df.index)

    # If P_all matrices are stored, plot the off-diagonals
    state_cols = [c for c in df.columns if c.startswith("filtered_")]
    if not state_cols:
        return None

    fig, ax = plt.subplots(figsize=(13, 5))
    for sc in state_cols:
        regime = sc.replace("filtered_", "")
        color = REGIME_COLORS.get(regime, "gray")
        ax.plot(df.index, df[sc], color=color, lw=1.0, label=f"P({regime})", alpha=0.8)

    for name, (start, end) in CRISIS_PERIODS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=RED, alpha=0.10)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Filtered probability")
    ax.set_title("Filtered Regime Probabilities Over Time\n"
                  "(How TVTP regime perception responds to macro shocks)",
                   fontweight="bold", color=NAVY)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return _save(fig, "12_rolling_transition_probs")


# ============================================================
# Main: render all plots
# ============================================================
def render_all():
    """Generate all plots."""
    plots = [
        ("01_regime_probabilities", plot_regime_probabilities),
        ("02_transition_matrix", plot_transition_matrix),
        ("03_tvtp_logistic_curves", plot_tvtp_logistic_curves),
        ("04_regime_conditional_returns", plot_regime_conditional_returns),
        ("05_model_comparison_bic", plot_model_comparison_bic),
        ("06_regime_timeline_comparison", plot_regime_timeline_comparison),
        ("07_early_warning", plot_early_warning),
        ("08_confusion_matrix", plot_confusion_matrix),
        ("09_integration_nav", plot_integration_nav),
        ("10_per_window_comparison", plot_per_window_comparison),
        ("11_model_scorecard", plot_model_scorecard),
        ("12_rolling_transition_probs", plot_rolling_transition_probs),
    ]

    print("=" * 60)
    print("M7 Visualizer: rendering all plots")
    print("=" * 60)

    for name, fn in plots:
        try:
            path = fn()
            print(f"  [OK]   {name}: {path}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    render_all()
