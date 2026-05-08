"""
B2: TVTP Markov-Switching Regime Detection
Configuration - single source of truth for all parameters.
"""
from pathlib import Path
import numpy as np

# -- Project paths --
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SPE_ROOT = PROJECT_ROOT.parent
RELABS_ROOT = Path("D:/Projects/StrategyResearchLab")

# -- Data sources --
SPY_PRICES_PATH = SPE_ROOT / "evt_tail_risk" / "data" / "spy_prices.parquet"
MACRO_FEATURES_PATH = RELABS_ROOT / "data" / "raw" / "macro_features.parquet"
BREADTH_PATH = RELABS_ROOT / "data" / "raw" / "breadth.parquet"
REGIME_HISTORY_PATH = SPE_ROOT / "data" / "backtest" / "regime_history.parquet"
EXP006_WF_DIR = RELABS_ROOT / "experiments" / "exp006_extended_wf" / "data" / "wf_results"

# -- Sample period --
SAMPLE_START = "1997-01-02"
SAMPLE_END = "2026-03-19"  # macro_features end date (binding constraint)
COMPARISON_START = "2009-01-01"  # rule-based regime overlap start

# -- Model specification --
N_STATES_OPTIONS = [2, 3]
COVARIATES = ["vix_level", "yield_curve_slope", "credit_spread"]

# -- EM algorithm --
EM_MAX_ITER = 500
EM_TOL = 1e-6
N_RESTARTS = 15
FILTER_CLAMP = 1e-10  # floor for filtered probs to prevent log(0)

# -- TVTP logistic optimization --
LBFGS_MAXITER = 100
LOGISTIC_CLAMP = 20.0  # clamp logistic inputs to [-20, 20]

# -- Synthetic tests --
SYNTHETIC_N = 5000
SYNTHETIC_REPS = 50

# -- Crisis events for annotation --
CRISIS_EVENTS = {
    "Asian Crisis": ("1997-07-01", "1998-01-31"),
    "Dot-Com Bust": ("2000-03-10", "2002-10-09"),
    "GFC": ("2007-10-01", "2009-03-09"),
    "European Debt": ("2011-07-01", "2011-12-31"),
    "Taper Tantrum": ("2013-05-22", "2013-09-30"),
    "COVID": ("2020-02-19", "2020-03-23"),
    "Rate Hikes": ("2022-01-03", "2022-10-12"),
    "Tariff Regime": ("2025-02-01", "2025-04-30"),
}

# -- Integration criteria --
INTEGRATION_CRITERIA = {
    "C1_kappa_threshold": 0.40,
    "C2_early_warning_days": 3,
    "C3_cagr_improvement": 0.0,
    "C3_sharpe_improvement": 0.0,
}

# -- Plotting --
PLOT_STYLE = {
    "navy": "#1B3A5C",
    "accent": "#2E75B6",
    "bull_color": "#2E8B57",
    "bear_color": "#DC143C",
    "crisis_color": "#8B0000",
    "figsize": (14, 6),
    "dpi": 150,
    "font_size": 10,
}
