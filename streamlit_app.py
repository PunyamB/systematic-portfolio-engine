"""
QuantForge — Systematic Equity Backtest Dashboard
Streamlit Cloud entry point: streamlit_app.py

UPDATED: Added EXP005 (15 Signals) as third strategy option.
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantForge | Backtest",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .main { background-color: #080C14; }
  [data-testid="stSidebar"] { background-color: #0C1220; border-right: 1px solid #1E2D45; }

  .qf-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    letter-spacing: 0.2em; color: #4A90D9; text-transform: uppercase; margin-bottom: 2px;
  }
  .qf-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 28px;
    font-weight: 600; color: #E8EEF7; letter-spacing: -0.01em;
  }

  .kpi-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 12px; margin: 18px 0; }
  .kpi-card {
    background: #0C1220; border: 1px solid #1E2D45; border-radius: 6px;
    padding: 14px 16px; position: relative; overflow: hidden;
  }
  .kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #1B6EBE, #00C4B4);
  }
  .kpi-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.12em; color: #4A6080; text-transform: uppercase; margin-bottom: 6px;
  }
  .kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600; color: #E8EEF7; }
  .kpi-value.positive { color: #00C4B4; }
  .kpi-value.negative { color: #E05252; }

  .section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.18em; color: #4A6080; text-transform: uppercase;
    margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #1E2D45;
  }

  .stTabs [data-baseweb="tab-list"] { background-color: #080C14; gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.08em;
    background: #0C1220; border: 1px solid #1E2D45; border-radius: 4px;
    color: #4A6080; padding: 8px 18px;
  }
  .stTabs [aria-selected="true"] { background: #1B6EBE20; border-color: #1B6EBE; color: #4A90D9; }

  .sidebar-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    letter-spacing: 0.2em; color: #4A6080; text-transform: uppercase; margin-bottom: 4px;
  }
  .sidebar-stat { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #8BAEC8; margin-bottom: 10px; }

  .regime-legend {
    display: flex; align-items: center; gap: 20px;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.12em; color: #4A6080; padding: 6px 0;
  }
  .regime-swatch {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 2px; margin-right: 5px; vertical-align: middle;
  }

  .method-card {
    background: #0C1220; border: 1px solid #1E2D45; border-radius: 8px;
    padding: 20px 24px; margin-bottom: 16px;
  }
  .method-card-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    letter-spacing: 0.15em; color: #4A90D9; text-transform: uppercase;
    margin-bottom: 10px;
  }
  .method-card p {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 14px;
    color: #8BAEC8; line-height: 1.7; margin: 0;
  }
  .method-card ul {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 14px;
    color: #8BAEC8; line-height: 1.9; margin: 8px 0 0 0; padding-left: 18px;
  }
  .method-card li span.hl { color: #E8EEF7; font-weight: 500; }
  .method-card li span.teal { color: #00C4B4; font-weight: 500; }
  .method-card li span.amber { color: #D4A017; font-weight: 500; }
  .pipeline-step {
    display: flex; align-items: flex-start; gap: 16px;
    padding: 14px 0; border-bottom: 1px solid #1E2D45;
  }
  .pipeline-step:last-child { border-bottom: none; }
  .step-num {
    font-family: 'IBM Plex Mono', monospace; font-size: 18px;
    font-weight: 600; color: #1B6EBE; min-width: 32px; line-height: 1.4;
  }
  .step-body { flex: 1; }
  .step-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    color: #E8EEF7; font-weight: 600; margin-bottom: 4px;
  }
  .step-desc {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 13px;
    color: #8BAEC8; line-height: 1.6;
  }
  .honesty-box {
    background: #0D1A14; border: 1px solid #1A3828;
    border-left: 3px solid #00C4B4; border-radius: 6px;
    padding: 16px 20px; margin-bottom: 12px;
  }
  .honesty-box p {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 13px;
    color: #8BAEC8; line-height: 1.7; margin: 0;
  }
  .honesty-box strong { color: #E8EEF7; }
</style>
""", unsafe_allow_html=True)

# ── COLORS ────────────────────────────────────────────────────────────────────
C = {
    "strategy":  "#00C4B4", "benchmark": "#E05252", "positive":  "#00C4B4",
    "negative":  "#E05252", "neutral":   "#4A6080", "accent":    "#4A90D9",
    "grid":      "#1E2D45", "bg":        "#080C14", "card":      "#0C1220",
    "text":      "#E8EEF7", "muted":     "#8BAEC8", "purple":    "#7C65D4",
    "amber":     "#D4A017",
}

YAXIS = dict(gridcolor="#1E2D45", zeroline=False, showgrid=True)

PLOTLY_BASE = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
    font=dict(family="IBM Plex Mono", color=C["muted"], size=11),
    xaxis=dict(gridcolor=C["grid"], zeroline=False, showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)

REGIME_COLORS = {
    "bull":     "rgba(0,196,180,0.13)",
    "recovery": "rgba(74,144,217,0.10)",
    "bear":     "rgba(212,160,23,0.13)",
    "crisis":   "rgba(224,82,82,0.14)",
}

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def dpath(*parts):
    return os.path.join(BASE, "data", *parts)


# ── STRATEGY CONFIG ───────────────────────────────────────────────────────────
# EXP005 uses "exp005" as strategy key and "monthly" as fixed frequency
STRATEGY_OPTIONS = {
    "mv": "Mean-Variance (MV)",
    "bl": "Black-Litterman (BL)",
    "exp005": "EXP005 — 15 Signals",
}

SIGNAL_COLS_11 = [
    'momentum_12_1', 'earnings_momentum', 'pe_zscore', 'pb_zscore',
    'ev_ebitda_zscore', 'roe_stability', 'gross_margin_trend',
    'piotroski', 'earnings_accruals', 'short_term_reversal', 'rsi_extremes',
]

SIGNAL_COLS_15 = SIGNAL_COLS_11 + [
    'revenue_growth', 'low_volatility', 'fcf_yield', 'volume_momentum',
]

SIGNAL_LABELS = {
    'momentum_12_1': '12-1 Momentum', 'earnings_momentum': 'Earnings Momentum',
    'pe_zscore': 'P/E Z-Score', 'pb_zscore': 'P/B Z-Score',
    'ev_ebitda_zscore': 'EV/EBITDA Z-Score', 'roe_stability': 'ROE Stability',
    'gross_margin_trend': 'Gross Margin Trend', 'piotroski': 'Piotroski F-Score',
    'earnings_accruals': 'Earnings Accruals', 'short_term_reversal': 'Short-Term Reversal',
    'rsi_extremes': 'RSI Extremes',
    'revenue_growth': 'Revenue Growth (QoQ)', 'low_volatility': 'Low Volatility',
    'fcf_yield': 'FCF Yield', 'volume_momentum': 'Volume Momentum',
}

GROUPS = {
    'Momentum':  ['momentum_12_1', 'earnings_momentum'],
    'Value':     ['pe_zscore', 'pb_zscore', 'ev_ebitda_zscore', 'fcf_yield'],
    'Quality':   ['roe_stability', 'gross_margin_trend', 'piotroski', 'earnings_accruals'],
    'Mean Rev':  ['short_term_reversal', 'rsi_extremes'],
    'Growth':    ['revenue_growth'],
    'Defensive': ['low_volatility'],
    'Sentiment': ['volume_momentum'],
}

GROUP_COLORS = {
    'Momentum': C["strategy"], 'Value': C["accent"],
    'Quality': C["purple"], 'Mean Rev': C["amber"],
    'Growth': '#FF6B6B', 'Defensive': '#4ECDC4', 'Sentiment': '#FFE66D',
}


# ── DATA LOADERS ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_prices():
    return pd.read_parquet(dpath("backtest/prices.parquet"))

@st.cache_data(show_spinner=False)
def load_signals(is_exp005=False):
    fname = "signals_history_exp005.parquet" if is_exp005 else "signals_history.parquet"
    df = pd.read_parquet(dpath(f"backtest/precomputed/{fname}"))
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(show_spinner=False)
def load_fwd_returns(is_exp005=False):
    fname = "forward_returns_exp005.parquet" if is_exp005 else "forward_returns.parquet"
    df = pd.read_parquet(dpath(f"backtest/precomputed/{fname}"))
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(show_spinner=False)
def load_signal_decay():
    df = pd.read_parquet(dpath("processed/signal_decay.parquet"))
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(show_spinner=False)
def load_regime_history():
    p = dpath("backtest/regime_history.parquet")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_wf_nav_stitched(strategy, freq):
    windows = [str(i) for i in range(1, 10)] + ['holdout']
    return_series = []
    for w in windows:
        f = dpath(f"backtest/wf_results/nav_window_{w}_{strategy}_{freq}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        rets = df['nav'].pct_change().dropna()
        return_series.append(rets)
    if not return_series:
        return pd.DataFrame()
    all_returns = pd.concat(return_series).sort_index()
    all_returns = all_returns[~all_returns.index.duplicated(keep='first')]
    nav = (1 + all_returns).cumprod() * 1_000_000
    return pd.DataFrame({'nav': nav})

@st.cache_data(show_spinner=False)
def load_all_wf_portfolios(strategy, freq):
    dfs = []
    for f in sorted(glob.glob(dpath(f"backtest/wf_results/portfolios_window_*_{strategy}_{freq}.parquet"))):
        dfs.append(pd.read_parquet(f))
    return pd.concat(dfs) if dfs else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_all_wf_trades(strategy, freq):
    dfs = []
    for f in sorted(glob.glob(dpath(f"backtest/wf_results/trades_window_*_{strategy}_{freq}.parquet"))):
        dfs.append(pd.read_parquet(f))
    return pd.concat(dfs) if dfs else pd.DataFrame()

def get_spy_nav(index, start_nav):
    try:
        spy_df = pd.read_parquet(dpath("backtest/spy_benchmark.parquet"))
        spy_df.index = pd.to_datetime(spy_df.index)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = [c[0].lower() for c in spy_df.columns]
        spy = spy_df['close'].sort_index()
        spy = spy.reindex(index, method='ffill').dropna()
        if spy.empty:
            return pd.Series(dtype=float)
        return spy / spy.iloc[0] * start_nav
    except Exception:
        return pd.Series(dtype=float)


# ── REGIME OVERLAY HELPER ─────────────────────────────────────────────────────
def add_regime_overlay(fig, regime_df, nav_start, nav_end):
    if regime_df.empty:
        return
    dates      = regime_df["date"].tolist()
    composites = regime_df["composite"].tolist()
    for i, (dt, state) in enumerate(zip(dates, composites)):
        x0 = max(dt, nav_start)
        x1 = min(dates[i + 1] if i + 1 < len(dates) else nav_end, nav_end)
        if x0 >= x1:
            continue
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=REGIME_COLORS.get(state, "rgba(255,255,255,0.03)"),
            line_width=0,
            layer="below",
        )


# ── ANALYTICS ─────────────────────────────────────────────────────────────────
def metrics(nav_series, spy_series=None, rf=0.02):
    nav = nav_series.dropna()
    if len(nav) < 10:
        return {}
    r     = nav.pct_change().dropna()
    ann   = 252
    n_yr  = len(nav) / ann
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret   = (1 + total_ret) ** (1 / n_yr) - 1 if n_yr > 0 else 0
    vol       = r.std() * np.sqrt(ann)
    sharpe    = (ann_ret - rf) / vol if vol > 0 else 0
    down      = r[r < 0].std() * np.sqrt(ann)
    sortino   = (ann_ret - rf) / down if down > 0 else 0
    dd        = (nav / nav.cummax() - 1)
    max_dd    = dd.min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0
    out = dict(total_ret=total_ret, ann_ret=ann_ret, vol=vol,
               sharpe=sharpe, sortino=sortino, max_dd=max_dd, calmar=calmar)
    if spy_series is not None and not spy_series.empty:
        spy_r  = spy_series.pct_change().dropna()
        common = r.index.intersection(spy_r.index)
        if len(common) > 10:
            active  = r.loc[common] - spy_r.loc[common]
            te      = active.std() * np.sqrt(ann)
            cov     = np.cov(r.loc[common], spy_r.loc[common])
            beta    = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1
            spy_ann = (spy_series.iloc[-1] / spy_series.iloc[0]) ** (1 / n_yr) - 1
            alpha   = ann_ret - beta * spy_ann
            out.update(dict(te=te, beta=beta, alpha=alpha))
    return out

def fmt(v, style="pct"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if style == "pct":       return f"{v:+.1%}" if v != 0 else "0.0%"
    if style == "pct_plain": return f"{v:.1%}"
    if style == "x2":        return f"{v:.2f}"
    if style == "x3":        return f"{v:.3f}"
    return str(v)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:20px'>
      <div class='qf-header'>Systematic Equity Engine</div>
      <div class='qf-title'>QuantForge</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Strategy</div>', unsafe_allow_html=True)
    strategy = st.selectbox("", list(STRATEGY_OPTIONS.keys()),
        format_func=lambda x: STRATEGY_OPTIONS[x],
        label_visibility="collapsed")

    is_exp005 = (strategy == "exp005")

    # EXP005 is monthly only — no frequency choice needed
    if is_exp005:
        freq = "monthly"
        st.markdown('<div class="sidebar-label" style="margin-top:12px">Rebalance Frequency</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-stat">Monthly (fixed)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-label" style="margin-top:12px">Rebalance Frequency</div>', unsafe_allow_html=True)
        freq = st.selectbox("", ["monthly", "quarterly"],
            format_func=lambda x: x.capitalize(),
            label_visibility="collapsed")

    n_signals = 15 if is_exp005 else 11
    signals_label = f"{n_signals} Multi-Factor"

    st.markdown("---")
    st.markdown(f"""
    <div class="sidebar-label">Universe</div>
    <div class="sidebar-stat">S&P 500 Constituents</div>
    <div class="sidebar-label">Capital</div>
    <div class="sidebar-stat">$1,000,000 Paper</div>
    <div class="sidebar-label">Benchmark</div>
    <div class="sidebar-stat">SPY (Total Return)</div>
    <div class="sidebar-label">Walk-Forward Windows</div>
    <div class="sidebar-stat">9 OOS + Holdout</div>
    <div class="sidebar-label">Training Window</div>
    <div class="sidebar-stat">Expanding (2009-based)</div>
    <div class="sidebar-label">Max Position Size</div>
    <div class="sidebar-stat">5% per Stock</div>
    <div class="sidebar-label">Signals</div>
    <div class="sidebar-stat">{signals_label}</div>
    """, unsafe_allow_html=True)

    if is_exp005:
        st.markdown("""
        <div style='margin-top:12px; padding:10px; background:#0D1A14; border:1px solid #1A3828; border-radius:6px'>
          <div class="sidebar-label" style="color:#00C4B4">EXP005 Additions</div>
          <div style="font-size:11px; color:#8BAEC8; line-height:1.6">
            + Revenue Growth<br>
            + Low Volatility<br>
            + FCF Yield<br>
            + Volume Momentum<br>
            + VIX threshold 0.95<br>
            + Regime multipliers
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── LOAD CORE DATA ────────────────────────────────────────────────────────────
with st.spinner("Loading backtest data…"):
    wf_nav     = load_wf_nav_stitched(strategy, freq)
    regime_df  = load_regime_history()

if wf_nav.empty:
    st.error(f"No walk-forward NAV data found for {strategy}/{freq}. Check data/backtest/wf_results/")
    st.stop()

primary_nav = wf_nav['nav']
spy_nav     = get_spy_nav(primary_nav.index, primary_nav.iloc[0])

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
strategy_label = STRATEGY_OPTIONS[strategy]
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"""
    <div class='qf-header' style='margin-top:8px'>Backtest Results · {strategy_label} · {freq.capitalize()}</div>
    <div class='qf-title'>Performance Dashboard</div>
    """, unsafe_allow_html=True)
with col_h2:
    start = primary_nav.index[0].strftime("%b %Y")
    end   = primary_nav.index[-1].strftime("%b %Y")
    st.markdown(f"""
    <div style='text-align:right; margin-top:12px'>
      <div class='sidebar-label'>Period</div>
      <div style='font-family:IBM Plex Mono; color:#8BAEC8; font-size:13px'>{start} → {end}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "▲  PERFORMANCE",
    "◈  RISK",
    "◻  PORTFOLIO",
    "◉  SIGNALS",
    "⟳  WALK-FORWARD",
    "≡  METHODOLOGY",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    m = metrics(primary_nav, spy_nav)

    kpi_data = [
        ("Total Return",  fmt(m.get('total_ret'), 'pct'),   m.get('total_ret', 0) > 0),
        ("Ann. Return",   fmt(m.get('ann_ret'), 'pct'),     m.get('ann_ret', 0) > 0),
        ("Volatility",    fmt(m.get('vol'), 'pct_plain'),   None),
        ("Sharpe Ratio",  fmt(m.get('sharpe'), 'x2'),       m.get('sharpe', 0) > 1),
        ("Sortino Ratio", fmt(m.get('sortino'), 'x2'),      m.get('sortino', 0) > 1),
        ("Max Drawdown",  fmt(m.get('max_dd'), 'pct'),      False),
        ("Calmar Ratio",  fmt(m.get('calmar'), 'x2'),       m.get('calmar', 0) > 0.5),
    ]

    kpi_html = '<div class="kpi-grid">'
    for label, value, is_good in kpi_data:
        cls = "positive" if is_good is True else ("negative" if is_good is False and label != "Volatility" else "")
        kpi_html += f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value {cls}">{value}</div></div>'
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Cumulative Growth of $1,000,000</div>', unsafe_allow_html=True)

    show_regime = False
    if not regime_df.empty:
        col_tog, col_leg = st.columns([1, 5])
        with col_tog:
            show_regime = st.toggle("Regime overlay", value=True)
        if show_regime:
            with col_leg:
                st.markdown("""
                <div class="regime-legend">
                  <span><span class="regime-swatch" style="background:rgba(0,196,180,0.45)"></span>BULL</span>
                  <span><span class="regime-swatch" style="background:rgba(74,144,217,0.40)"></span>RECOVERY</span>
                  <span><span class="regime-swatch" style="background:rgba(212,160,23,0.45)"></span>BEAR</span>
                  <span><span class="regime-swatch" style="background:rgba(224,82,82,0.45)"></span>CRISIS</span>
                </div>
                """, unsafe_allow_html=True)

    fig = go.Figure()

    if show_regime and not regime_df.empty:
        add_regime_overlay(fig, regime_df, primary_nav.index[0], primary_nav.index[-1])

    fig.add_trace(go.Scatter(
        x=primary_nav.index, y=primary_nav,
        name=f"Strategy ({strategy_label})",
        line=dict(color=C["strategy"], width=2),
        fill='tozeroy', fillcolor='rgba(0,196,180,0.06)',
    ))
    if not spy_nav.empty:
        fig.add_trace(go.Scatter(
            x=spy_nav.index, y=spy_nav,
            name="SPY Benchmark",
            line=dict(color=C["benchmark"], width=1.5, dash='dot'),
        ))
    fig.update_layout(**PLOTLY_BASE, height=400,
                      yaxis=dict(**YAXIS, tickprefix="$", tickformat=","))
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-label">Annual Returns</div>', unsafe_allow_html=True)
        ann_ret = primary_nav.resample('YE').last().pct_change().dropna()
        ann_ret.index = ann_ret.index.year
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=ann_ret.index, y=ann_ret.values * 100,
            name="Strategy",
            marker_color=[C["positive"] if r > 0 else C["negative"] for r in ann_ret.values],
            text=[f"{r:.1%}" for r in ann_ret.values],
            textposition='outside', textfont=dict(size=10),
        ))
        if not spy_nav.empty:
            spy_ann = spy_nav.resample('YE').last().pct_change().dropna()
            spy_ann.index = spy_ann.index.year
            fig2.add_trace(go.Scatter(
                x=spy_ann.index, y=spy_ann.values * 100,
                mode='lines+markers', name="SPY",
                line=dict(color=C["benchmark"], width=1.5),
                marker=dict(size=5),
            ))
        fig2.update_layout(**PLOTLY_BASE, height=300, yaxis_title="Return (%)")
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-label">Strategy vs Benchmark</div>', unsafe_allow_html=True)
        if not spy_nav.empty:
            sm = metrics(spy_nav)
            rows = [
                ("Total Return",   fmt(m.get('total_ret'), 'pct'),   fmt(sm.get('total_ret'), 'pct')),
                ("Ann. Return",    fmt(m.get('ann_ret'), 'pct'),     fmt(sm.get('ann_ret'), 'pct')),
                ("Volatility",     fmt(m.get('vol'), 'pct_plain'),   fmt(sm.get('vol'), 'pct_plain')),
                ("Sharpe",         fmt(m.get('sharpe'), 'x2'),       fmt(sm.get('sharpe'), 'x2')),
                ("Sortino",        fmt(m.get('sortino'), 'x2'),      fmt(sm.get('sortino'), 'x2')),
                ("Max Drawdown",   fmt(m.get('max_dd'), 'pct'),      fmt(sm.get('max_dd'), 'pct')),
                ("Calmar",         fmt(m.get('calmar'), 'x2'),       fmt(sm.get('calmar'), 'x2')),
                ("Beta vs SPY",    fmt(m.get('beta'), 'x2'),         "1.00"),
                ("Alpha (Ann.)",   fmt(m.get('alpha'), 'pct'),       "—"),
                ("Tracking Error", fmt(m.get('te'), 'pct_plain'),    "—"),
            ]
            cmp_df = pd.DataFrame(rows, columns=["Metric", strategy_label, "SPY"])
            st.dataframe(cmp_df.set_index("Metric"), use_container_width=True, height=340)
        else:
            st.info("SPY benchmark not available.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Drawdown Analysis</div>', unsafe_allow_html=True)
    dd_series = (primary_nav / primary_nav.cummax() - 1) * 100
    fig_dd = go.Figure()

    if show_regime and not regime_df.empty:
        add_regime_overlay(fig_dd, regime_df, primary_nav.index[0], primary_nav.index[-1])

    fig_dd.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series,
        fill='tozeroy', fillcolor='rgba(224,82,82,0.15)',
        line=dict(color=C["negative"], width=1.5), name="Drawdown %",
    ))
    max_dd_idx = dd_series.idxmin()
    fig_dd.add_trace(go.Scatter(
        x=[max_dd_idx], y=[dd_series.min()], mode='markers+text',
        marker=dict(color=C["negative"], size=10, symbol='x'),
        text=[f"  Max DD: {dd_series.min():.1f}%"],
        textfont=dict(color=C["muted"], size=10), textposition='middle right',
        name="Max Drawdown",
    ))
    fig_dd.update_layout(**PLOTLY_BASE, height=280, yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    r = primary_nav.pct_change().dropna()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Rolling 60-Day Sharpe</div>', unsafe_allow_html=True)
        roll_s = (r.rolling(60).mean() / r.rolling(60).std() * np.sqrt(252)).dropna()
        fig_rs = go.Figure()
        fig_rs.add_hline(y=0, line_color=C["grid"], line_width=1)
        fig_rs.add_hline(y=1, line_dash="dot", line_color=C["accent"], line_width=1,
                         annotation_text="Sharpe=1", annotation_font_color=C["accent"])
        fig_rs.add_trace(go.Scatter(
            x=roll_s.index, y=roll_s,
            line=dict(color=C["strategy"], width=1.5),
            fill='tozeroy', fillcolor='rgba(0,196,180,0.08)',
            name="Rolling Sharpe",
        ))
        fig_rs.update_layout(**PLOTLY_BASE, height=270)
        st.plotly_chart(fig_rs, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Rolling 60-Day Volatility</div>', unsafe_allow_html=True)
        roll_v = r.rolling(60).std() * np.sqrt(252) * 100
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(
            x=roll_v.index, y=roll_v,
            line=dict(color=C["strategy"], width=1.5), name="Strategy Vol",
        ))
        if not spy_nav.empty:
            spy_r      = spy_nav.pct_change().dropna()
            spy_roll_v = spy_r.rolling(60).std() * np.sqrt(252) * 100
            fig_rv.add_trace(go.Scatter(
                x=spy_roll_v.index, y=spy_roll_v,
                line=dict(color=C["benchmark"], width=1.2, dash='dot'), name="SPY Vol",
            ))
        fig_rv.update_layout(**PLOTLY_BASE, height=270, yaxis_title="Vol (%)")
        st.plotly_chart(fig_rv, use_container_width=True)

    if not spy_nav.empty:
        st.markdown('<div class="section-label">Tracking Error vs SPY (Rolling 60-Day)</div>', unsafe_allow_html=True)
        spy_r  = spy_nav.pct_change().dropna()
        common = r.index.intersection(spy_r.index)
        if len(common) > 60:
            active = r.loc[common] - spy_r.loc[common]
            te     = active.rolling(60).std() * np.sqrt(252) * 100
            fig_te = go.Figure()
            fig_te.add_hrect(y0=4, y1=6, fillcolor='rgba(212,160,23,0.08)', line_width=0,
                             annotation_text="Target band 4–6%", annotation_position="top left",
                             annotation_font_color=C["amber"])
            fig_te.add_hline(y=4, line_dash="dot", line_color=C["amber"], line_width=1)
            fig_te.add_hline(y=6, line_dash="dot", line_color=C["negative"], line_width=1,
                             annotation_text="Cap 6%", annotation_font_color=C["negative"])
            fig_te.add_trace(go.Scatter(
                x=te.index, y=te, fill='tozeroy', fillcolor='rgba(0,196,180,0.08)',
                line=dict(color=C["strategy"], width=1.5), name="Tracking Error",
            ))
            fig_te.update_layout(**PLOTLY_BASE, height=260, yaxis_title="TE (%)")
            st.plotly_chart(fig_te, use_container_width=True)

    st.markdown('<div class="section-label">Monthly Return Distribution</div>', unsafe_allow_html=True)
    monthly_r = primary_nav.resample('ME').last().pct_change().dropna() * 100
    fig_hist  = go.Figure()
    fig_hist.add_trace(go.Histogram(x=monthly_r, nbinsx=40, marker_color=C["strategy"], opacity=0.7, name="Strategy"))
    if not spy_nav.empty:
        spy_monthly = spy_nav.resample('ME').last().pct_change().dropna() * 100
        fig_hist.add_trace(go.Histogram(x=spy_monthly, nbinsx=40, marker_color=C["benchmark"], opacity=0.5, name="SPY"))
    fig_hist.update_layout(**PLOTLY_BASE, height=250, barmode='overlay',
                           xaxis_title="Monthly Return (%)", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    portfolios = load_all_wf_portfolios(strategy, freq)
    trades_df  = load_all_wf_trades(strategy, freq)

    if not portfolios.empty:
        portfolios['date'] = pd.to_datetime(portfolios['date'])
        active = portfolios[portfolios['target_weight'] > 0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Active Position Count per Rebalance</div>', unsafe_allow_html=True)
            pos_count = active.groupby('date')['ticker'].count()
            fig_pc = go.Figure()
            fig_pc.add_trace(go.Scatter(
                x=pos_count.index, y=pos_count, mode='lines+markers', marker=dict(size=4),
                line=dict(color=C["strategy"]), fill='tozeroy', fillcolor='rgba(0,196,180,0.06)',
                name="Positions",
            ))
            fig_pc.add_hline(y=pos_count.mean(), line_dash="dot", line_color=C["muted"],
                             annotation_text=f"  Avg: {pos_count.mean():.0f}",
                             annotation_font_color=C["muted"])
            fig_pc.update_layout(**PLOTLY_BASE, height=280, yaxis_title="# Positions")
            st.plotly_chart(fig_pc, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Top-10 Holdings Concentration</div>', unsafe_allow_html=True)
            top10 = active.groupby('date').apply(lambda g: g['target_weight'].nlargest(10).sum() * 100)
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=top10.index, y=top10, fill='tozeroy', fillcolor='rgba(74,144,217,0.08)',
                line=dict(color=C["accent"], width=1.5), name="Top-10 Weight",
            ))
            fig_c.update_layout(**PLOTLY_BASE, height=280, yaxis_title="Weight (%)")
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown('<div class="section-label">Composite Score of Holdings vs Universe</div>', unsafe_allow_html=True)
        if 'composite_score' in active.columns:
            try:
                signals_df   = load_signals(is_exp005)
                univ_monthly = signals_df.set_index('date').resample('ME')['composite_score'].mean()
                held_monthly = active.set_index('date').resample('ME')['composite_score'].mean()
                fig_score = go.Figure()
                fig_score.add_trace(go.Scatter(x=univ_monthly.index, y=univ_monthly,
                    line=dict(color=C["neutral"], width=1, dash='dot'), name="Universe Avg"))
                fig_score.add_trace(go.Scatter(x=held_monthly.index, y=held_monthly,
                    line=dict(color=C["strategy"], width=2), name="Holdings Avg"))
                fig_score.update_layout(**PLOTLY_BASE, height=250, yaxis_title="Composite Score")
                st.plotly_chart(fig_score, use_container_width=True)
            except Exception:
                pass

        if not trades_df.empty:
            st.markdown('<div class="section-label">Monthly Portfolio Turnover ($M)</div>', unsafe_allow_html=True)
            trades_df['date']  = pd.to_datetime(trades_df['date'])
            trades_df['value'] = trades_df['shares'] * trades_df['price']
            monthly_to = trades_df.groupby(pd.Grouper(key='date', freq='ME'))['value'].sum() / 1e6
            fig_to = go.Figure()
            fig_to.add_trace(go.Bar(x=monthly_to.index, y=monthly_to,
                marker_color=C["accent"], opacity=0.8, name="Turnover ($M)"))
            fig_to.update_layout(**PLOTLY_BASE, height=250, yaxis_title="$ Millions")
            st.plotly_chart(fig_to, use_container_width=True)

        st.markdown('<div class="section-label">Most Frequently Held Stocks</div>', unsafe_allow_html=True)
        top_held = active['ticker'].value_counts().head(25)
        fig_top  = go.Figure(go.Bar(
            x=top_held.index, y=top_held.values,
            marker=dict(color=top_held.values, colorscale=[[0, '#1B6EBE'], [1, '#00C4B4']], showscale=False),
            text=top_held.values, textposition='outside', name="Appearances",
        ))
        fig_top.update_layout(**PLOTLY_BASE, height=300, yaxis_title="Rebalance Appearances")
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.warning("No portfolio data found for the selected strategy/frequency.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    ACTIVE_SIGNAL_COLS = SIGNAL_COLS_15 if is_exp005 else SIGNAL_COLS_11
    # Filter groups to only show relevant signals
    active_groups = {}
    for g, cols in GROUPS.items():
        active_cols = [c for c in cols if c in ACTIVE_SIGNAL_COLS]
        if active_cols:
            active_groups[g] = active_cols

    @st.cache_data(show_spinner=False)
    def compute_ic(_is_exp005):
        sig    = load_signals(_is_exp005)
        fwd    = load_fwd_returns(_is_exp005)
        merged = sig.merge(fwd[['ticker', 'date', 'fwd_1m']], on=['ticker', 'date'], how='inner')
        signal_cols = SIGNAL_COLS_15 if _is_exp005 else SIGNAL_COLS_11
        results = []
        for col in signal_cols:
            if col not in merged.columns:
                continue
            ic_ts = merged.groupby('date').apply(
                lambda g: g[[col, 'fwd_1m']].dropna().corr(method='spearman').iloc[0, 1]
            ).dropna()
            group = next((g for g, cols in GROUPS.items() if col in cols), 'Other')
            results.append({
                'signal': col, 'Signal': SIGNAL_LABELS.get(col, col), 'Group': group,
                'Mean IC': ic_ts.mean(), 'IC Std': ic_ts.std(),
                'IC IR': ic_ts.mean() / ic_ts.std() if ic_ts.std() > 0 else 0,
                'Hit Rate': (ic_ts > 0).mean(), 'ic_ts': ic_ts,
            })
        return results

    with st.spinner("Computing signal ICs…"):
        try:
            ic_results = compute_ic(is_exp005)
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown('<div class="section-label">Information Coefficient (IC) — Mean Rank Correlation with 1-Month Forward Returns</div>', unsafe_allow_html=True)
                ic_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'ic_ts'} for r in ic_results])
                ic_df = ic_df.sort_values('Mean IC', ascending=True)
                fig_ic = go.Figure()
                for group, color in GROUP_COLORS.items():
                    sub = ic_df[ic_df['Group'] == group]
                    if sub.empty:
                        continue
                    fig_ic.add_trace(go.Bar(
                        x=sub['Mean IC'], y=sub['Signal'], orientation='h',
                        name=group, marker_color=color,
                        text=[f"{v:.4f}" for v in sub['Mean IC']],
                        textposition='outside', textfont=dict(size=10),
                    ))
                fig_ic.add_vline(x=0, line_color=C["grid"], line_width=1)
                ic_layout = {**PLOTLY_BASE, 'margin': dict(l=0, r=80, t=30, b=10)}
                fig_ic.update_layout(**ic_layout, height=450 if is_exp005 else 400,
                                     xaxis_title="Mean IC (Spearman Rank Correlation)",
                                     barmode='relative')
                st.plotly_chart(fig_ic, use_container_width=True)

            with col2:
                st.markdown('<div class="section-label">IC Summary Table</div>', unsafe_allow_html=True)
                tbl = ic_df[['Signal', 'Group', 'Mean IC', 'IC IR', 'Hit Rate']].sort_values('Mean IC', ascending=False).copy()
                tbl['Mean IC']  = tbl['Mean IC'].map(lambda x: f"{x:.4f}")
                tbl['IC IR']    = tbl['IC IR'].map(lambda x: f"{x:.2f}")
                tbl['Hit Rate'] = tbl['Hit Rate'].map(lambda x: f"{x:.1%}")
                st.dataframe(tbl.set_index('Signal'), use_container_width=True, height=450 if is_exp005 else 400)

            st.markdown('<div class="section-label">Rolling IC Over Time — Top Signals</div>', unsafe_allow_html=True)
            top_signals = sorted(ic_results, key=lambda x: abs(x['Mean IC']), reverse=True)[:5]
            fig_ric = go.Figure()
            palette = [C["strategy"], C["accent"], C["purple"], C["amber"], C["negative"]]
            for i, sig_data in enumerate(top_signals):
                ic_ts = sig_data['ic_ts'].rolling(12).mean()
                fig_ric.add_trace(go.Scatter(x=ic_ts.index, y=ic_ts,
                    name=sig_data['Signal'], line=dict(color=palette[i], width=1.5)))
            fig_ric.add_hline(y=0, line_color=C["grid"], line_width=1)
            fig_ric.update_layout(**PLOTLY_BASE, height=280, yaxis_title="12M Rolling Mean IC")
            st.plotly_chart(fig_ric, use_container_width=True)
        except Exception as e:
            st.error(f"IC computation error: {e}")

    if not is_exp005:
        st.markdown('<div class="section-label">Signal Decay — Rank Correlation vs Holding Horizon</div>', unsafe_allow_html=True)
        try:
            decay_df = load_signal_decay().dropna(subset=['rank_correlation'])
            if not decay_df.empty and len(decay_df) > 2:
                fig_decay = go.Figure()
                for sig in decay_df['signal_name'].unique():
                    sub = decay_df[decay_df['signal_name'] == sig].sort_values('date')
                    fig_decay.add_trace(go.Scatter(x=sub['date'], y=sub['rank_correlation'],
                        name=SIGNAL_LABELS.get(sig, sig), mode='lines', line=dict(width=1.5)))
                fig_decay.add_hline(y=0.65, line_dash="dot", line_color=C["amber"],
                                    annotation_text="  Drift-rebalance trigger (0.65)",
                                    annotation_font_color=C["amber"])
                fig_decay.update_layout(**PLOTLY_BASE, height=280, yaxis_title="Rank Correlation")
                st.plotly_chart(fig_decay, use_container_width=True)
            else:
                st.info("Signal decay data is sparse — requires more live pipeline cycles to populate.")
        except Exception as e:
            st.warning(f"Signal decay unavailable: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="section-label">Out-of-Sample Walk-Forward · 9 Windows + Holdout · {strategy_label}</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def per_window_metrics(strat, fr):
        rows = []
        for w in [str(i) for i in range(1, 10)] + ['holdout']:
            f = dpath(f"backtest/wf_results/nav_window_{w}_{strat}_{fr}.parquet")
            if not os.path.exists(f):
                continue
            try:
                df = pd.read_parquet(f)
                df.index = pd.to_datetime(df.index)
                m2 = metrics(df['nav'])
                rows.append({
                    'Window': f"W{w}" if w != 'holdout' else 'Holdout',
                    'Start':  df.index[0].strftime('%Y-%m'),
                    'End':    df.index[-1].strftime('%Y-%m'),
                    **m2,
                })
            except Exception:
                pass
        return pd.DataFrame(rows)

    wm_df = per_window_metrics(strategy, freq)

    if not wm_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Sharpe per Window</div>', unsafe_allow_html=True)
            fig_ws = go.Figure(go.Bar(
                x=wm_df['Window'], y=wm_df['sharpe'],
                marker_color=[C["positive"] if v > 0 else C["negative"] for v in wm_df['sharpe']],
                text=[f"{v:.3f}" for v in wm_df['sharpe']], textposition='outside',
            ))
            fig_ws.add_hline(y=0, line_color=C["grid"])
            fig_ws.update_layout(**PLOTLY_BASE, height=280, yaxis_title="Sharpe Ratio")
            st.plotly_chart(fig_ws, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Ann. Return per Window</div>', unsafe_allow_html=True)
            fig_wr = go.Figure(go.Bar(
                x=wm_df['Window'], y=wm_df['ann_ret'] * 100,
                marker_color=[C["positive"] if v > 0 else C["negative"] for v in wm_df['ann_ret']],
                text=[f"{v:.1%}" for v in wm_df['ann_ret']], textposition='outside',
            ))
            fig_wr.add_hline(y=0, line_color=C["grid"])
            fig_wr.update_layout(**PLOTLY_BASE, height=280, yaxis_title="Ann. Return (%)")
            st.plotly_chart(fig_wr, use_container_width=True)

        st.markdown('<div class="section-label">Window Detail Table</div>', unsafe_allow_html=True)
        tbl = wm_df[['Window', 'Start', 'End', 'ann_ret', 'vol', 'sharpe', 'sortino', 'max_dd', 'calmar']].copy()
        tbl['ann_ret'] = tbl['ann_ret'].map(lambda x: f"{x:.1%}")
        tbl['vol']     = tbl['vol'].map(lambda x: f"{x:.1%}")
        tbl['sharpe']  = tbl['sharpe'].map(lambda x: f"{x:.2f}")
        tbl['sortino'] = tbl['sortino'].map(lambda x: f"{x:.2f}")
        tbl['max_dd']  = tbl['max_dd'].map(lambda x: f"{x:.1%}")
        tbl['calmar']  = tbl['calmar'].map(lambda x: f"{x:.2f}")
        tbl = tbl.rename(columns={'ann_ret': 'Ann. Ret', 'vol': 'Vol', 'sharpe': 'Sharpe',
                                  'sortino': 'Sortino', 'max_dd': 'Max DD', 'calmar': 'Calmar'})
        st.dataframe(tbl.set_index('Window'), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">All Strategy Combinations</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def all_combo_metrics():
        rows = []
        for strat in ['mv', 'bl']:
            for fr in ['monthly', 'quarterly']:
                wf = load_wf_nav_stitched(strat, fr)
                if wf.empty:
                    continue
                m2 = metrics(wf['nav'])
                m2['Strategy'] = f"{strat.upper()}-{fr}"
                rows.append(m2)
        # Add EXP005
        wf = load_wf_nav_stitched('exp005', 'monthly')
        if not wf.empty:
            m2 = metrics(wf['nav'])
            m2['Strategy'] = "EXP005-monthly"
            rows.append(m2)
        return pd.DataFrame(rows)

    with st.spinner("Loading all combinations…"):
        combo_df = all_combo_metrics()

    if not combo_df.empty:
        metric_opts = {'Sharpe Ratio': 'sharpe', 'Ann. Return': 'ann_ret',
                       'Max Drawdown': 'max_dd', 'Calmar': 'calmar',
                       'Volatility': 'vol', 'Sortino': 'sortino'}
        sel  = st.selectbox("Compare by", list(metric_opts.keys()))
        key  = metric_opts[sel]
        scale = 100 if key in ['ann_ret', 'max_dd', 'vol'] else 1
        strat_colors = {'MV-monthly': C["strategy"], 'MV-quarterly': '#007D74',
                        'BL-monthly': C["purple"],   'BL-quarterly': '#4A3080',
                        'EXP005-monthly': '#FFD700'}
        fig_cmp = go.Figure(go.Bar(
            x=combo_df['Strategy'], y=combo_df[key] * scale,
            marker_color=[strat_colors.get(s, C["neutral"]) for s in combo_df['Strategy']],
            text=[f"{v*scale:.3f}{'%' if scale==100 else ''}" for v in combo_df[key]],
            textposition='outside',
        ))
        fig_cmp.update_layout(**PLOTLY_BASE, height=300, yaxis_title=sel,
                              title=f"{sel} — Strategy Comparison")
        st.plotly_chart(fig_cmp, use_container_width=True)

        cmp_tbl = combo_df[['Strategy', 'total_ret', 'ann_ret', 'vol', 'sharpe', 'sortino', 'max_dd', 'calmar']].copy()
        for col in ['total_ret', 'ann_ret', 'vol', 'max_dd']:
            cmp_tbl[col] = cmp_tbl[col].map(lambda x: f"{x:.1%}")
        for col in ['sharpe', 'sortino', 'calmar']:
            cmp_tbl[col] = cmp_tbl[col].map(lambda x: f"{x:.2f}")
        cmp_tbl = cmp_tbl.rename(columns={'total_ret': 'Total Ret', 'ann_ret': 'Ann. Ret', 'vol': 'Vol',
                                          'sharpe': 'Sharpe', 'sortino': 'Sortino', 'max_dd': 'Max DD', 'calmar': 'Calmar'})
        st.dataframe(cmp_tbl.set_index('Strategy'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">What is this system?</div>
          <p>
            QuantForge is a systematic, rules-based equity strategy that invests in S&P 500 stocks.
            It uses no human discretion — every buy and sell decision is driven entirely by data and a defined process.
            Think of it as a quantitative fund manager that runs the same disciplined playbook every month, regardless of news or sentiment.
          </p>
        </div>
        """, unsafe_allow_html=True)

        if is_exp005:
            st.markdown("""
            <div class="method-card">
              <div class="method-card-title">How does EXP005 differ from the base strategy?</div>
              <p>EXP005 extends the base 11-signal system with four additional factors and improved regime calibration:</p>
              <ul>
                <li><span class="teal">Revenue Growth</span> — quarter-over-quarter revenue acceleration captures forward earnings expectations that backward-looking accounting signals miss</li>
                <li><span class="teal">Low Volatility</span> — inverse of 252-day realized volatility provides defensive tilt during bear markets (2022: +31% alpha vs SPY)</li>
                <li><span class="teal">FCF Yield</span> — annualized free cash flow / market cap adds a cash-generation value signal orthogonal to P/E and P/B</li>
                <li><span class="teal">Volume Momentum</span> — 5-day vs 60-day average volume ratio captures unusual institutional activity</li>
              </ul>
              <br>
              <p>Additionally, the VIX regime threshold was recalibrated from 0.80 to 0.95 based on empirical analysis showing the original threshold classified only 21% of months as bull vs the actual ~55% proportion. Regime-conditional signal multipliers are applied to all 15 signals.</p>
            </div>
            """, unsafe_allow_html=True)

        signals_desc = "15 signals across seven categories" if is_exp005 else "11 signals across four categories"
        categories_html = """
            <li><span class="teal">Momentum</span> — stocks that have been rising tend to keep rising (12-month price trend, earnings acceleration)</li>
            <li><span class="teal">Value</span> — stocks trading cheaply relative to earnings and book value (P/E, P/B, EV/EBITDA vs sector peers)</li>
            <li><span class="teal">Quality</span> — financially healthy companies with stable earnings and improving margins (ROE stability, Piotroski F-Score, accruals)</li>
            <li><span class="teal">Mean Reversion</span> — stocks that have fallen too far, too fast and are likely to bounce back (5-day reversal, RSI extremes)</li>
        """
        if is_exp005:
            categories_html += """
            <li><span class="teal">Growth</span> — companies with accelerating revenue (QoQ revenue growth)</li>
            <li><span class="teal">Defensive</span> — lower-volatility stocks that protect capital in downturns (inverse 252d vol)</li>
            <li><span class="teal">Sentiment</span> — unusual trading volume signaling institutional interest (volume momentum)</li>
            """

        st.markdown(f"""
        <div class="method-card">
          <div class="method-card-title">How does it pick stocks?</div>
          <p>We score every S&P 500 stock on {signals_desc}:</p>
          <ul>
            {categories_html}
          </ul>
          <br>
          <p>Each signal is weighted by how predictive it has historically been (IC-IR weighting). The top-scoring stocks get bought; the rest get sold or ignored.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">How does it size positions?</div>
          <p>
            Position sizes are determined by a <span style="color:#E8EEF7; font-weight:500;">portfolio optimizer</span>, not gut feel.
            Two variants are tested:
          </p>
          <ul>
            <li><span class="hl">Mean-Variance (MV)</span> — finds the mathematically optimal mix of stocks that maximizes return for a given level of risk, using 13 years of return history and Ledoit-Wolf covariance shrinkage to avoid overfitting.</li>
            <li><span class="hl">Black-Litterman (BL)</span> — starts from market-cap weights (what the S&P 500 already looks like) and tilts toward our high-scoring stocks. More conservative, less aggressive rebalancing.</li>
          </ul>
          <br>
          <p>Both enforce a <span style="color:#E8EEF7; font-weight:500;">5% maximum weight per stock</span> — no single position can dominate the portfolio. The optimizer also penalizes excessive trading to keep transaction costs low.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">How does it manage risk?</div>
          <ul>
            <li><span class="hl">Trailing stops</span> — if a stock drops more than 2× its recent daily volatility from its recent high, it is automatically sold regardless of the rebalance schedule.</li>
            <li><span class="hl">Drawdown circuit breakers</span> — if the portfolio drops 10%, 15%, or 20% from peak, the system automatically reduces exposure and tightens position limits until it recovers.</li>
            <li><span class="hl">Sector neutrality</span> — the portfolio is kept balanced across sectors so it cannot accidentally become a pure tech or pure energy bet.</li>
            <li><span class="hl">Regime detection</span> — the system monitors VIX, yield curve, and credit spread data daily to classify the market as bull, bear, recovery, or crisis, and adjusts rebalance frequency accordingly. Regime states are shown as shaded overlays on the NAV chart.</li>
            <li><span class="hl">Tracking error cap</span> — the optimizer is constrained to stay within 6% tracking error of SPY, preventing extreme divergence from the benchmark.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">How was the backtest run?</div>
          <p>
            The backtest uses a technique called <span style="color:#E8EEF7; font-weight:500;">walk-forward testing</span> — the gold standard for validating systematic strategies. Here is how it works step by step:
          </p>
          <div class="pipeline-step">
            <div class="step-num">1</div>
            <div class="step-body">
              <div class="step-title">Train on expanding history</div>
              <div class="step-desc">The system learns signal weights and optimizer parameters using all data from 2009 up to the training cutoff. No future data is ever used in this step.</div>
            </div>
          </div>
          <div class="pipeline-step">
            <div class="step-num">2</div>
            <div class="step-body">
              <div class="step-title">Test on the next 12 months (out-of-sample)</div>
              <div class="step-desc">The trained model is deployed on the following year — data it has never seen. Returns from this period are the honest, real-world performance estimate.</div>
            </div>
          </div>
          <div class="pipeline-step">
            <div class="step-num">3</div>
            <div class="step-body">
              <div class="step-title">Slide the window forward and repeat</div>
              <div class="step-desc">The training window expands by one year and the process repeats. We ran 9 windows from 2013 to 2021, each independently trained and tested.</div>
            </div>
          </div>
          <div class="pipeline-step">
            <div class="step-num">4</div>
            <div class="step-body">
              <div class="step-title">Final holdout: 2022–2026</div>
              <div class="step-desc">The last 4+ years were completely untouched during development. This is the strictest test — a period the model was never shown. It covers a bear market (2022), recovery (2023), and a narrow mega-cap driven bull market (2024–2025).</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">Regime detection methodology</div>
          <p>Regime states are reconstructed point-in-time using only data available on each date — no lookahead bias. Two layers combine into one of four composite states:</p>
          <ul>
            <li><span class="teal">L1 — Market stress (daily)</span>: VIX vs its 252-day average + market breadth (% of S&P 500 constituents above 200d MA, point-in-time membership)</li>
            <li><span class="teal">L2 — Economic cycle (weekly)</span>: 10Y-2Y yield curve slope + HY credit spread direction</li>
          </ul>
          <br>
          <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:4px">
            <span style="font-family:'IBM Plex Mono'; font-size:11px; padding:4px 10px; border-radius:4px; background:rgba(0,196,180,0.15); color:#00C4B4">BULL — low stress + expansion</span>
            <span style="font-family:'IBM Plex Mono'; font-size:11px; padding:4px 10px; border-radius:4px; background:rgba(74,144,217,0.15); color:#4A90D9">RECOVERY — mixed signals</span>
            <span style="font-family:'IBM Plex Mono'; font-size:11px; padding:4px 10px; border-radius:4px; background:rgba(212,160,23,0.15); color:#D4A017">BEAR — elevated + contraction</span>
            <span style="font-family:'IBM Plex Mono'; font-size:11px; padding:4px 10px; border-radius:4px; background:rgba(224,82,82,0.15); color:#E05252">CRISIS — extreme stress</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">What are the honest limitations?</div>
          <div class="honesty-box">
            <p><strong>Transaction costs are estimated, not exact.</strong> We model a 10bps round-trip cost per trade. Real costs depend on market impact and timing.</p>
          </div>
          <div class="honesty-box">
            <p><strong>Fundamental data is point-in-time approximated, not exact.</strong> FMP quarterly financials are stamped by reporting period end date. In reality, companies file 30–60 days after quarter-end. We apply a 45-day announcement lag buffer, which approximates but does not perfectly replicate the exact filing date for every stock.</p>
          </div>
          <div class="honesty-box">
            <p><strong>The 2024–2025 lag is expected and explainable.</strong> The strategy is diversified across 20–80 stocks with a 5% cap per position. When the market is driven by 5–7 mega-cap tech stocks (NVIDIA, Apple, Microsoft), a diversified approach will underperform a market-cap-weighted index. This is a feature, not a bug — it reflects a deliberate risk management choice.</p>
          </div>
          <div class="honesty-box">
            <p><strong>This is a paper trading simulation.</strong> The strategy currently runs on Alpaca paper trading with $1M simulated capital. Live performance may differ from backtested results due to slippage, order fills, and data latency.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">Data sources</div>
          <ul>
            <li><span class="hl">Price & fundamental data</span> — Financial Modeling Prep (FMP) Premium, 30+ years of history, quarterly financials for all S&P 500 stocks</li>
            <li><span class="hl">Macro regime data</span> — FRED (Federal Reserve) for VIX, yield curve (10Y–2Y), and credit spreads. Full history from 2007 fetched for point-in-time regime reconstruction.</li>
            <li><span class="hl">Execution</span> — Alpaca paper trading API for order submission and fill confirmation</li>
            <li><span class="hl">Universe</span> — S&P 500 constituents with point-in-time historical membership to avoid survivorship bias</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)