"""
EVT Tail Risk — Configuration
==============================
Single source of truth for all EVT-specific parameters.
"""

# ── Rolling / Window ────────────────────────────────────────────
ROLLING_WINDOW = 500            # Trading days for rolling estimation (~2 years)
COV_LOOKBACK = 252              # Not used directly in EVT, kept for reference

# ── Confidence Levels ───────────────────────────────────────────
CONFIDENCE_LEVELS = [0.95, 0.99, 0.995]

# ── Threshold Selection (Module 3) ──────────────────────────────
THRESHOLD_QUANTILE = 0.95       # Default threshold as quantile (fallback)
MIN_EXCEEDANCES = 50            # Minimum exceedances for GPD fit
MAX_EXCEEDANCES = 250           # Upper bound for threshold selection

# ── GPD Estimation (Module 4) ───────────────────────────────────
GPD_SYNTHETIC_N = 10_000        # Sample size for synthetic recovery test
GPD_SYNTHETIC_REPS = 100        # Number of repetitions for coverage test
GPD_SYNTHETIC_XI = 0.25         # True xi for synthetic test
GPD_SYNTHETIC_SIGMA = 1.0       # True sigma for synthetic test

# ── GARCH (Module 6) ───────────────────────────────────────────
GARCH_P = 1                     # GARCH lag order
GARCH_Q = 1                     # ARCH lag order

# ── Rolling Backtest (Module 7) ─────────────────────────────────
ROLLING_REFIT_FREQ = 5          # Refit GPD/GARCH every N days
BACKTEST_START_BUFFER = 500     # Initial fitting window (days)

# ── Crisis Periods ──────────────────────────────────────────────
CRISIS_PERIODS = {
    "Dotcom":       ("2000-03-10", "2002-10-09"),
    "GFC":          ("2007-10-09", "2009-03-09"),
    "EU Debt":      ("2011-07-01", "2011-10-04"),
    "COVID":        ("2020-02-19", "2020-03-23"),
    "Rate Hikes":   ("2022-01-03", "2022-10-12"),
}

# ── Plot Style ──────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.figsize": (12, 6),
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "lines.linewidth": 1.2,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

COLORS = {
    "navy":    "#1B2A4A",
    "accent":  "#2E75B6",
    "red":     "#D32F2F",
    "green":   "#388E3C",
    "orange":  "#F57C00",
    "gray":    "#757575",
    "light":   "#E8F0FE",
}

# ── Data Paths (relative to EVT project root) ──────────────────
import os as _os
_EVT_DIR = _os.path.dirname(_os.path.abspath(__file__))
DATA_DIR = _os.path.join(_EVT_DIR, "data")
OUTPUT_DIR = _os.path.join(_EVT_DIR, "outputs")

# ── Source Data Paths (relative to SPE root) ────────────────────
_SPE_DIR = _os.path.dirname(_EVT_DIR)
BACKTEST_PRICES_PATH = _os.path.join(_SPE_DIR, "data", "backtest", "prices.parquet")
WF_RESULTS_DIR = _os.path.join(_SPE_DIR, "data", "backtest", "wf_results")
REGIME_HISTORY_PATH = _os.path.join(_SPE_DIR, "data", "backtest", "regime_history.parquet")
NAV_HISTORY_PATH = _os.path.join(_SPE_DIR, "data", "processed", "nav_history.parquet")
