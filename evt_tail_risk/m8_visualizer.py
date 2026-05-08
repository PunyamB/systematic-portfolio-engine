"""
Module 8: Visualizer
====================
Centralized plotting module. All EVT plots are generated here.
Modules 2-7 call these functions rather than containing their own plotting code.

Current functions:
  M2: plot_loss_histogram, plot_qq_normal, plot_rolling_vol,
      plot_drawdown, plot_acf_volatility, plot_tail_probability
  M3: plot_mrl, plot_parameter_stability

Usage:
    from evt_tail_risk.m8_visualizer import plot_loss_histogram, plot_mrl
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from evt_tail_risk import config


def _apply_style():
    """Apply consistent EVT plot styling."""
    plt.rcParams.update(config.PLOT_STYLE)


def _save_fig(fig, name, dpi=150):
    """Save figure to outputs/."""
    import os
    path = os.path.join(config.OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── M2 Plots ────────────────────────────────────────────────────

def plot_loss_histogram(losses, name="SPY", bins=100, save=True):
    """
    Histogram of loss series with fitted normal overlay.
    Shows visual departure from Gaussian, especially in tails.
    """
    _apply_style()
    c = config.COLORS

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(losses, bins=bins, density=True, alpha=0.6,
            color=c["accent"], edgecolor="white", linewidth=0.3, label="Empirical")

    mu, sigma = losses.mean(), losses.std()
    x = np.linspace(losses.min(), losses.max(), 300)
    ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), color=c["red"],
            linewidth=2, label=f"Normal(mu={mu:.4f}, sigma={sigma:.4f})")

    ax.set_title(f"{name} Daily Loss Distribution", fontsize=14, color=c["navy"])
    ax.set_xlabel("Loss (negative log return)")
    ax.set_ylabel("Density")
    ax.legend()

    if save:
        return _save_fig(fig, f"loss_histogram_{name.lower()}")
    return fig


def plot_qq_normal(losses, name="SPY", save=True):
    """
    QQ plot of losses against normal distribution.
    Curvature in tails confirms heavy-tail behavior.
    """
    _apply_style()
    c = config.COLORS

    fig, ax = plt.subplots(figsize=(8, 8))
    res = sp_stats.probplot(losses, dist="norm", plot=None)
    theoretical, ordered = res[0]

    ax.scatter(theoretical, ordered, s=8, alpha=0.5, color=c["accent"], label="Data")

    lims = [min(theoretical.min(), ordered.min()), max(theoretical.max(), ordered.max())]
    ax.plot(lims, lims, color=c["red"], linewidth=1.5, linestyle="--", label="Normal reference")

    ax.set_title(f"{name} QQ Plot vs Normal", fontsize=14, color=c["navy"])
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Sample Quantiles")
    ax.legend()

    if save:
        return _save_fig(fig, f"qq_normal_{name.lower()}")
    return fig


def plot_rolling_vol(losses, dates, name="SPY", window=252, save=True):
    """Rolling 252-day volatility with crisis period bands."""
    _apply_style()
    c = config.COLORS

    series = pd.Series(losses, index=pd.to_datetime(dates))
    rolling_vol = series.rolling(window).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_vol.index, rolling_vol.values, color=c["navy"], linewidth=1)

    for crisis_name, (start, end) in config.CRISIS_PERIODS.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        if start_dt >= rolling_vol.index.min() and start_dt <= rolling_vol.index.max():
            ax.axvspan(start_dt, end_dt, alpha=0.15, color=c["red"], label=crisis_name)

    ax.set_title(f"{name} Rolling {window}-Day Annualized Volatility", fontsize=14, color=c["navy"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    if save:
        return _save_fig(fig, f"rolling_vol_{name.lower()}")
    return fig


def plot_drawdown(losses, dates, name="SPY", save=True):
    """Drawdown time series with crisis annotations."""
    _apply_style()
    c = config.COLORS

    log_returns = -pd.Series(losses, index=pd.to_datetime(dates))
    cum_returns = log_returns.cumsum()
    running_max = cum_returns.cummax()
    drawdown = cum_returns - running_max

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color=c["red"], alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color=c["red"], linewidth=0.5)

    for crisis_name, (start, end) in config.CRISIS_PERIODS.items():
        start_dt = pd.Timestamp(start)
        if start_dt >= drawdown.index.min() and start_dt <= drawdown.index.max():
            ax.axvline(start_dt, color=c["gray"], linewidth=0.8, linestyle="--", alpha=0.7)
            ax.text(start_dt, ax.get_ylim()[0] * 0.9, f" {crisis_name}",
                    fontsize=8, color=c["gray"], rotation=90, va="bottom")

    ax.set_title(f"{name} Drawdown (Log Returns)", fontsize=14, color=c["navy"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")

    if save:
        return _save_fig(fig, f"drawdown_{name.lower()}")
    return fig


def plot_acf_volatility(losses, name="SPY", max_lags=50, save=True):
    """ACF of |returns| and returns^2. Shows volatility clustering."""
    _apply_style()
    c = config.COLORS
    from statsmodels.tsa.stattools import acf

    abs_losses = np.abs(losses)
    sq_losses = losses ** 2

    acf_abs = acf(abs_losses, nlags=max_lags, fft=True)
    acf_sq = acf(sq_losses, nlags=max_lags, fft=True)

    n = len(losses)
    ci = 1.96 / np.sqrt(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    lags = np.arange(max_lags + 1)
    ax1.bar(lags, acf_abs, width=0.6, color=c["accent"], alpha=0.7)
    ax1.axhline(ci, color=c["red"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.axhline(-ci, color=c["red"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title(f"{name} ACF of |Returns|", fontsize=13, color=c["navy"])
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("Autocorrelation")

    ax2.bar(lags, acf_sq, width=0.6, color=c["accent"], alpha=0.7)
    ax2.axhline(ci, color=c["red"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axhline(-ci, color=c["red"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title(f"{name} ACF of Returns\u00b2", fontsize=13, color=c["navy"])
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")

    fig.tight_layout()

    if save:
        return _save_fig(fig, f"acf_volatility_{name.lower()}")
    return fig


def plot_tail_probability(losses, name="SPY", save=True):
    """Log-scale empirical tail plot. P(Loss > x) on log-log axes."""
    _apply_style()
    c = config.COLORS

    pos_losses = np.sort(losses[losses > 0])[::-1]
    n = len(pos_losses)
    probs = np.arange(1, n + 1) / (n + 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(pos_losses, probs, s=4, alpha=0.4, color=c["accent"], label="Empirical tail")

    mu, sigma = losses.mean(), losses.std()
    x_grid = np.linspace(pos_losses.min(), pos_losses.max(), 200)
    normal_tail = 1 - sp_stats.norm.cdf(x_grid, mu, sigma)
    ax.plot(x_grid, normal_tail, color=c["red"], linewidth=1.5,
            linestyle="--", label="Normal tail")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{name} Tail Probability P(Loss > x)", fontsize=14, color=c["navy"])
    ax.set_xlabel("Loss threshold x (log scale)")
    ax.set_ylabel("P(Loss > x) (log scale)")
    ax.legend()

    if save:
        return _save_fig(fig, f"tail_probability_{name.lower()}")
    return fig


# ── M3 Plots ────────────────────────────────────────────────────

def plot_mrl(thresholds, mrl_values, mrl_ci, chosen_u=None, name="SPY", save=True):
    """
    Mean Residual Life plot.
    E[X - u | X > u] vs u with 95% confidence bands.
    If GPD valid above u, should be approximately linear.
    Vertical line marks chosen threshold.
    """
    _apply_style()
    c = config.COLORS

    fig, ax = plt.subplots(figsize=(12, 6))

    valid = ~np.isnan(mrl_values)
    t = thresholds[valid]
    m = mrl_values[valid]
    ci = mrl_ci[valid]

    ax.plot(t, m, color=c["navy"], linewidth=1.5, label="Mean excess")
    ax.fill_between(t, m - ci, m + ci, alpha=0.2, color=c["accent"], label="95% CI")

    if chosen_u is not None:
        ax.axvline(chosen_u, color=c["red"], linewidth=2, linestyle="--",
                   label=f"Chosen u = {chosen_u:.5f}")

    ax.set_title(f"{name} Mean Residual Life Plot", fontsize=14, color=c["navy"])
    ax.set_xlabel("Threshold u")
    ax.set_ylabel("Mean Excess E[X - u | X > u]")
    ax.legend(fontsize=9)

    if save:
        return _save_fig(fig, f"mrl_{name.lower()}")
    return fig


def plot_parameter_stability(thresholds, xi_values, xi_ci,
                              sigma_star_values, sigma_star_ci,
                              chosen_u=None, name="SPY", save=True):
    """
    Parameter stability plot: GPD shape (xi) and reparameterized scale
    (sigma* = sigma - xi*u) vs threshold. Both should stabilize at
    the correct threshold.
    """
    _apply_style()
    c = config.COLORS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    valid_xi = ~np.isnan(xi_values)
    valid_ss = ~np.isnan(sigma_star_values)

    # Shape (xi) plot
    t_xi = thresholds[valid_xi]
    xi = xi_values[valid_xi]
    ci_xi = xi_ci[valid_xi]

    ax1.plot(t_xi, xi, color=c["navy"], linewidth=1.5, label="Shape (xi)")
    ax1.fill_between(t_xi, xi - ci_xi, xi + ci_xi, alpha=0.2, color=c["accent"])
    if chosen_u is not None:
        ax1.axvline(chosen_u, color=c["red"], linewidth=2, linestyle="--",
                    label=f"Chosen u = {chosen_u:.5f}")
    ax1.set_ylabel("Shape parameter (xi)")
    ax1.set_title(f"{name} GPD Parameter Stability", fontsize=14, color=c["navy"])
    ax1.legend(fontsize=9)
    ax1.axhline(0, color=c["gray"], linewidth=0.5, linestyle=":")

    # Reparameterized scale (sigma*) plot
    t_ss = thresholds[valid_ss]
    ss = sigma_star_values[valid_ss]
    ci_ss = sigma_star_ci[valid_ss]

    ax2.plot(t_ss, ss, color=c["navy"], linewidth=1.5, label="Scale (sigma*)")
    ax2.fill_between(t_ss, ss - ci_ss, ss + ci_ss, alpha=0.2, color=c["accent"])
    if chosen_u is not None:
        ax2.axvline(chosen_u, color=c["red"], linewidth=2, linestyle="--")
    ax2.set_xlabel("Threshold u")
    ax2.set_ylabel("Reparameterized scale (sigma - xi*u)")
    ax2.legend(fontsize=9)

    fig.tight_layout()

    if save:
        return _save_fig(fig, f"param_stability_{name.lower()}")
    return fig


# ── M4 Plots ────────────────────────────────────────────────────

def plot_gpd_4panel(exceedances, xi, sigma, threshold, n_total,
                    name="SPY", save=True):
    """
    4-panel GPD diagnostic plot:
      1. QQ plot (empirical vs GPD quantiles)
      2. PP plot (empirical vs GPD probabilities)
      3. Return level plot (N-year return levels with CIs)
      4. Density plot (empirical histogram vs fitted GPD)
    """
    _apply_style()
    c = config.COLORS

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    n_exc = len(exceedances)
    sorted_exc = np.sort(exceedances)

    # ── Panel 1: QQ Plot ────────────────────────────────────────
    ax = axes[0, 0]
    empirical_q = sorted_exc
    p_values = (np.arange(1, n_exc + 1) - 0.5) / n_exc
    theoretical_q = sp_stats.genpareto.ppf(p_values, c=xi, loc=0, scale=sigma)

    ax.scatter(theoretical_q, empirical_q, s=12, alpha=0.6, color=c["accent"])
    lims = [0, max(theoretical_q.max(), empirical_q.max()) * 1.05]
    ax.plot(lims, lims, color=c["red"], linewidth=1.5, linestyle="--")
    ax.set_title("QQ Plot: Empirical vs GPD", fontsize=12, color=c["navy"])
    ax.set_xlabel("GPD Theoretical Quantiles")
    ax.set_ylabel("Empirical Quantiles")

    # ── Panel 2: PP Plot ────────────────────────────────────────
    ax = axes[0, 1]
    empirical_p = p_values
    theoretical_p = sp_stats.genpareto.cdf(sorted_exc, c=xi, loc=0, scale=sigma)

    ax.scatter(theoretical_p, empirical_p, s=12, alpha=0.6, color=c["accent"])
    ax.plot([0, 1], [0, 1], color=c["red"], linewidth=1.5, linestyle="--")
    ax.set_title("PP Plot: Empirical vs GPD", fontsize=12, color=c["navy"])
    ax.set_xlabel("GPD Theoretical Probability")
    ax.set_ylabel("Empirical Probability")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # ── Panel 3: Return Level Plot ──────────────────────────────
    ax = axes[1, 0]

    # Return periods in years (assuming 252 trading days)
    return_periods = np.array([1, 2, 5, 10, 20, 50, 100])
    return_periods_days = return_periods * 252

    # Exceedance rate
    zeta = n_exc / n_total

    # Return levels: x_m = u + (sigma/xi) * [(m * zeta)^xi - 1]
    return_levels = []
    rl_ci_lo = []
    rl_ci_hi = []

    se_xi = (1 + xi) / np.sqrt(n_exc) if (1 + xi) > 0 else 0.1
    se_sigma = sigma * np.sqrt(2 * (1 + xi)) / np.sqrt(n_exc) if (1 + xi) > 0 else 0.01

    for m in return_periods_days:
        if xi != 0:
            rl = threshold + (sigma / xi) * ((m * zeta) ** xi - 1)
        else:
            rl = threshold + sigma * np.log(m * zeta)
        return_levels.append(rl)

        # Approximate CI via delta method (simplified)
        # Perturbation approach
        delta = 0.02
        rl_up = threshold + ((sigma + delta * se_sigma) / (xi + delta * se_xi)) * ((m * zeta) ** (xi + delta * se_xi) - 1) if (xi + delta * se_xi) != 0 else rl
        rl_dn = threshold + ((sigma - delta * se_sigma) / (xi - delta * se_xi)) * ((m * zeta) ** (xi - delta * se_xi) - 1) if (xi - delta * se_xi) != 0 else rl

        spread = abs(rl_up - rl_dn) / (2 * delta) * 1.96
        rl_ci_lo.append(rl - spread)
        rl_ci_hi.append(rl + spread)

    return_levels = np.array(return_levels)
    rl_ci_lo = np.array(rl_ci_lo)
    rl_ci_hi = np.array(rl_ci_hi)

    ax.plot(return_periods, return_levels, color=c["navy"], linewidth=2,
            marker="o", markersize=6, label="Return level")
    ax.fill_between(return_periods, rl_ci_lo, rl_ci_hi,
                    alpha=0.2, color=c["accent"], label="95% CI")
    ax.set_xscale("log")
    ax.set_title("Return Level Plot", fontsize=12, color=c["navy"])
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level (loss magnitude)")
    ax.legend(fontsize=9)

    # Annotate key return levels
    for rp, rl in zip(return_periods, return_levels):
        if rp in [1, 10, 100]:
            ax.annotate(f"{rl:.4f}", (rp, rl), textcoords="offset points",
                       xytext=(10, 5), fontsize=8, color=c["gray"])

    # ── Panel 4: Density Plot ───────────────────────────────────
    ax = axes[1, 1]

    ax.hist(exceedances, bins=40, density=True, alpha=0.5,
            color=c["accent"], edgecolor="white", linewidth=0.3, label="Empirical")

    x_grid = np.linspace(0, exceedances.max() * 1.1, 200)
    gpd_pdf = sp_stats.genpareto.pdf(x_grid, c=xi, loc=0, scale=sigma)
    ax.plot(x_grid, gpd_pdf, color=c["red"], linewidth=2, label=f"GPD(xi={xi:.3f}, sigma={sigma:.5f})")

    ax.set_title("Density: Empirical vs Fitted GPD", fontsize=12, color=c["navy"])
    ax.set_xlabel("Exceedance (loss - threshold)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    fig.suptitle(f"{name} GPD Diagnostic — u={threshold:.5f}, n={n_exc}",
                 fontsize=14, color=c["navy"], fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        return _save_fig(fig, f"gpd_4panel_{name.lower()}")
    return fig


# ── M7 Plots ────────────────────────────────────────────────────

def plot_rolling_tail_index(rolling_df, name="SPY", save=True):
    """
    Rolling tail index (xi) over time with crisis annotations.
    Shows how tail heaviness evolves — key for dynamic monitoring.
    """
    _apply_style()
    c = config.COLORS

    dates = pd.to_datetime(rolling_df["date"])
    xi = rolling_df["xi"].values

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, xi, color=c["navy"], linewidth=1, label="Rolling xi")
    ax.axhline(0, color=c["gray"], linewidth=0.8, linestyle=":")

    # Crisis shading
    for crisis_name, (start, end) in config.CRISIS_PERIODS.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        if start_dt >= dates.min() and start_dt <= dates.max():
            ax.axvspan(start_dt, end_dt, alpha=0.15, color=c["red"], label=crisis_name)

    ax.set_title(f"{name} Rolling Tail Index (xi) — {config.ROLLING_WINDOW}-Day Window",
                 fontsize=14, color=c["navy"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Shape parameter (xi)")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    if save:
        return _save_fig(fig, f"rolling_tail_index_{name.lower()}")
    return fig


def plot_backtest_violations(merged_df, backtest_results, name="SPY",
                              level="99.0%", save=True):
    """
    VaR violations timeline: realized losses vs predicted VaR for each model.
    Shows where each model's VaR gets breached.
    """
    _apply_style()
    c = config.COLORS

    dates = pd.to_datetime(merged_df["date"])
    realized = merged_df["realized_loss"].values

    # Map level to column suffix
    p_map = {"95.0%": "0950", "99.0%": "0990", "99.5%": "0995"}
    pct = p_map.get(level, "0990")

    models = {
        "Gaussian": ("gauss", c["gray"]),
        "Hist Sim": ("hs", c["green"]),
        "Cornish-Fisher": ("cf", c["orange"]),
        "FHS": ("fhs", c["accent"]),
        "EVT-GPD": ("evt", c["red"]),
    }

    fig, axes = plt.subplots(len(models), 1, figsize=(14, 3 * len(models)), sharex=True)

    for idx, (model_name, (prefix, color)) in enumerate(models.items()):
        ax = axes[idx]
        var_col = f"{prefix}_var_{pct}"
        if var_col not in merged_df.columns:
            ax.text(0.5, 0.5, f"{model_name}: no data", transform=ax.transAxes,
                    ha="center", fontsize=12)
            continue

        var_vals = merged_df[var_col].values
        violations = realized > var_vals

        ax.plot(dates, realized, color=c["gray"], linewidth=0.3, alpha=0.5)
        ax.plot(dates, var_vals, color=color, linewidth=0.8, label=f"{model_name} VaR")
        ax.scatter(dates[violations], realized[violations], color=c["red"],
                   s=8, alpha=0.7, zorder=5, label=f"Violations ({violations.sum()})")

        ax.set_ylabel("Loss")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{model_name}", fontsize=10, color=c["navy"], loc="left")

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{name} VaR Backtest at {level}", fontsize=14,
                 color=c["navy"], fontweight="bold")
    fig.tight_layout()

    if save:
        return _save_fig(fig, f"backtest_violations_{name.lower()}_{pct}")
    return fig


def plot_model_scorecard(backtest_results, name="SPY", save=True):
    """
    Summary heatmap of backtest p-values across all models and levels.
    Green = pass (p > 0.05), red = fail.
    """
    _apply_style()
    c = config.COLORS

    models = list(next(iter(backtest_results.values())).keys())
    levels = list(backtest_results.keys())

    # Build matrix: rows = models, cols = levels, values = Kupiec p-value
    kupiec_matrix = np.zeros((len(models), len(levels)))
    christ_matrix = np.zeros((len(models), len(levels)))
    viol_matrix = np.zeros((len(models), len(levels)))

    for j, level in enumerate(levels):
        for i, model in enumerate(models):
            bt = backtest_results[level].get(model, {})
            kupiec_matrix[i, j] = bt.get("kupiec", {}).get("p_value", np.nan)
            christ_matrix[i, j] = bt.get("christoffersen", {}).get("p_value", np.nan)
            viol_matrix[i, j] = bt.get("violation_rate", np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Kupiec heatmap
    im1 = ax1.imshow(kupiec_matrix, cmap="RdYlGn", vmin=0, vmax=0.5, aspect="auto")
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels(levels, fontsize=10)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=10)
    ax1.set_title("Kupiec Test p-values", fontsize=13, color=c["navy"])

    for i in range(len(models)):
        for j in range(len(levels)):
            val = kupiec_matrix[i, j]
            vr = viol_matrix[i, j]
            text = f"{val:.3f}\n({vr:.3%})" if not np.isnan(val) else "N/A"
            color = "white" if val < 0.1 else "black"
            ax1.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    # Christoffersen heatmap
    im2 = ax2.imshow(christ_matrix, cmap="RdYlGn", vmin=0, vmax=0.5, aspect="auto")
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels(levels, fontsize=10)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models, fontsize=10)
    ax2.set_title("Christoffersen Test p-values", fontsize=13, color=c["navy"])

    for i in range(len(models)):
        for j in range(len(levels)):
            val = christ_matrix[i, j]
            text = f"{val:.3f}" if not np.isnan(val) else "N/A"
            color = "white" if val < 0.1 else "black"
            ax2.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im1, ax=ax1, shrink=0.8, label="p-value")
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="p-value")

    fig.suptitle(f"{name} Model Scorecard — VaR Backtest",
                 fontsize=14, color=c["navy"], fontweight="bold")
    fig.tight_layout()

    if save:
        return _save_fig(fig, f"model_scorecard_{name.lower()}")
    return fig


if __name__ == "__main__":
    from evt_tail_risk.m1_data_loader import load_spy_losses
    import os

    print("Generating all plots...")

    spy = load_spy_losses(save=False)
    mer = pd.read_parquet(os.path.join(config.DATA_DIR, "meridian_losses.parquet"))

    for name, losses, dates in [
        ("SPY", spy["loss"].values, spy["date"].values),
        ("Meridian", mer["loss"].values, mer["date"].values),
    ]:
        print(f"\n{name}:")
        plot_loss_histogram(losses, name=name)
        plot_qq_normal(losses, name=name)
        plot_rolling_vol(losses, dates, name=name)
        plot_drawdown(losses, dates, name=name)
        plot_acf_volatility(losses, name=name)
        plot_tail_probability(losses, name=name)

    print("\nAll plots complete.")
