"""
QuantForge — Systematic Equity Backtest Dashboard
Streamlit Cloud entry point: streamlit_app.py
Strategies: MV, BL, EXP005, EXP006, EXP007
"""

import os, glob, warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="QuantForge | Backtest", page_icon="▲", layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background-color: #080C14; }
[data-testid="stSidebar"] { background-color: #0C1220; border-right: 1px solid #1E2D45; }
.qf-header { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.2em; color: #4A90D9; text-transform: uppercase; margin-bottom: 2px; }
.qf-title { font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600; color: #E8EEF7; letter-spacing: -0.01em; }
.kpi-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 12px; margin: 18px 0; }
.kpi-card { background: #0C1220; border: 1px solid #1E2D45; border-radius: 6px; padding: 14px 16px; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #1B6EBE, #00C4B4); }
.kpi-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.12em; color: #4A6080; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600; color: #E8EEF7; }
.kpi-value.positive { color: #00C4B4; } .kpi-value.negative { color: #E05252; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.18em; color: #4A6080; text-transform: uppercase; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #1E2D45; }
.stTabs [data-baseweb="tab-list"] { background-color: #080C14; gap: 4px; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.08em; background: #0C1220; border: 1px solid #1E2D45; border-radius: 4px; color: #4A6080; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: #1B6EBE20; border-color: #1B6EBE; color: #4A90D9; }
.sidebar-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.2em; color: #4A6080; text-transform: uppercase; margin-bottom: 4px; }
.sidebar-stat { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #8BAEC8; margin-bottom: 10px; }
.regime-legend { display: flex; align-items: center; gap: 20px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.12em; color: #4A6080; padding: 6px 0; }
.regime-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
.method-card { background: #0C1220; border: 1px solid #1E2D45; border-radius: 8px; padding: 20px 24px; margin-bottom: 16px; }
.method-card-title { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.15em; color: #4A90D9; text-transform: uppercase; margin-bottom: 10px; }
.method-card p { font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; color: #8BAEC8; line-height: 1.7; margin: 0; }
.method-card ul { font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; color: #8BAEC8; line-height: 1.9; margin: 8px 0 0 0; padding-left: 18px; }
.method-card li span.hl { color: #E8EEF7; font-weight: 500; }
.method-card li span.teal { color: #00C4B4; font-weight: 500; }
.pipeline-step { display: flex; align-items: flex-start; gap: 16px; padding: 14px 0; border-bottom: 1px solid #1E2D45; }
.pipeline-step:last-child { border-bottom: none; }
.step-num { font-family: 'IBM Plex Mono', monospace; font-size: 18px; font-weight: 600; color: #1B6EBE; min-width: 32px; line-height: 1.4; }
.step-body { flex: 1; } .step-title { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #E8EEF7; font-weight: 600; margin-bottom: 4px; }
.step-desc { font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: #8BAEC8; line-height: 1.6; }
.honesty-box { background: #0D1A14; border: 1px solid #1A3828; border-left: 3px solid #00C4B4; border-radius: 6px; padding: 16px 20px; margin-bottom: 12px; }
.honesty-box p { font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: #8BAEC8; line-height: 1.7; margin: 0; }
.honesty-box strong { color: #E8EEF7; }
</style>""", unsafe_allow_html=True)

C = {"strategy": "#00C4B4", "benchmark": "#E05252", "positive": "#00C4B4", "negative": "#E05252", "neutral": "#4A6080", "accent": "#4A90D9", "grid": "#1E2D45", "bg": "#080C14", "card": "#0C1220", "text": "#E8EEF7", "muted": "#8BAEC8", "purple": "#7C65D4", "amber": "#D4A017"}
YAXIS = dict(gridcolor="#1E2D45", zeroline=False, showgrid=True)
PLOTLY_BASE = dict(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], font=dict(family="IBM Plex Mono", color=C["muted"], size=11), xaxis=dict(gridcolor=C["grid"], zeroline=False, showgrid=True), legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
REGIME_COLORS = {"bull": "rgba(0,196,180,0.13)", "recovery": "rgba(74,144,217,0.10)", "bear": "rgba(212,160,23,0.13)", "crisis": "rgba(224,82,82,0.14)"}
BASE = os.path.dirname(os.path.abspath(__file__))
def dpath(*parts): return os.path.join(BASE, "data", *parts)

STRATEGY_OPTIONS = {"mv": "Mean-Variance (MV)", "bl": "Black-Litterman (BL)", "exp005": "EXP005 \u2014 15 Signals", "exp006": "EXP006 \u2014 Extended", "exp007": "EXP007 \u2014 Sliding Params"}
SIGNAL_COLS_11 = ['momentum_12_1','earnings_momentum','pe_zscore','pb_zscore','ev_ebitda_zscore','roe_stability','gross_margin_trend','piotroski','earnings_accruals','short_term_reversal','rsi_extremes']
SIGNAL_COLS_15 = SIGNAL_COLS_11 + ['revenue_growth','low_volatility','fcf_yield','volume_momentum']
SIGNAL_LABELS = {'momentum_12_1':'12-1 Momentum','earnings_momentum':'Earnings Momentum','pe_zscore':'P/E Z-Score','pb_zscore':'P/B Z-Score','ev_ebitda_zscore':'EV/EBITDA Z-Score','roe_stability':'ROE Stability','gross_margin_trend':'Gross Margin Trend','piotroski':'Piotroski F-Score','earnings_accruals':'Earnings Accruals','short_term_reversal':'Short-Term Reversal','rsi_extremes':'RSI Extremes','revenue_growth':'Revenue Growth (QoQ)','low_volatility':'Low Volatility','fcf_yield':'FCF Yield','volume_momentum':'Volume Momentum'}
GROUPS = {'Momentum':['momentum_12_1','earnings_momentum'],'Value':['pe_zscore','pb_zscore','ev_ebitda_zscore','fcf_yield'],'Quality':['roe_stability','gross_margin_trend','piotroski','earnings_accruals'],'Mean Rev':['short_term_reversal','rsi_extremes'],'Growth':['revenue_growth'],'Defensive':['low_volatility'],'Sentiment':['volume_momentum']}
GROUP_COLORS = {'Momentum':C["strategy"],'Value':C["accent"],'Quality':C["purple"],'Mean Rev':C["amber"],'Growth':'#FF6B6B','Defensive':'#4ECDC4','Sentiment':'#FFE66D'}

# ── DATA LOADERS ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_signals(strat_key):
    m = {"exp007":"signals_history_exp007.parquet","exp006":"signals_history_exp006.parquet","exp005":"signals_history_exp005.parquet"}
    fname = m.get(strat_key, "signals_history.parquet")
    df = pd.read_parquet(dpath(f"backtest/precomputed/{fname}")); df['date']=pd.to_datetime(df['date']); return df

@st.cache_data(show_spinner=False)
def load_fwd_returns(strat_key):
    m = {"exp007":"forward_returns_exp007.parquet","exp006":"forward_returns_exp006.parquet","exp005":"forward_returns_exp005.parquet"}
    fname = m.get(strat_key, "forward_returns.parquet")
    df = pd.read_parquet(dpath(f"backtest/precomputed/{fname}")); df['date']=pd.to_datetime(df['date']); return df

@st.cache_data(show_spinner=False)
def load_signal_decay():
    df = pd.read_parquet(dpath("processed/signal_decay.parquet")); df['date']=pd.to_datetime(df['date']); return df

@st.cache_data(show_spinner=False)
def load_regime_history():
    p = dpath("backtest/regime_history.parquet")
    if not os.path.exists(p): return pd.DataFrame()
    df = pd.read_parquet(p); df["date"]=pd.to_datetime(df["date"]); return df.sort_values("date").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_wf_nav_stitched(strategy, freq):
    windows = [str(i) for i in range(1,30) if os.path.exists(dpath(f'backtest/wf_results/nav_window_{i}_{strategy}_{freq}.parquet'))]
    windows.append('holdout')
    rs = []
    for w in windows:
        f = dpath(f"backtest/wf_results/nav_window_{w}_{strategy}_{freq}.parquet")
        if not os.path.exists(f): continue
        df = pd.read_parquet(f); df.index=pd.to_datetime(df.index); df=df.sort_index()
        rs.append(df['nav'].pct_change().dropna())
    if not rs: return pd.DataFrame()
    ar = pd.concat(rs).sort_index(); ar=ar[~ar.index.duplicated(keep='first')]
    return pd.DataFrame({'nav': (1+ar).cumprod()*1_000_000})

@st.cache_data(show_spinner=False)
def load_all_wf_portfolios(strategy, freq):
    dfs = [pd.read_parquet(f) for f in sorted(glob.glob(dpath(f"backtest/wf_results/portfolios_window_*_{strategy}_{freq}.parquet")))]
    return pd.concat(dfs) if dfs else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_all_wf_trades(strategy, freq):
    dfs = [pd.read_parquet(f) for f in sorted(glob.glob(dpath(f"backtest/wf_results/trades_window_*_{strategy}_{freq}.parquet")))]
    return pd.concat(dfs) if dfs else pd.DataFrame()

def get_spy_nav(index, start_nav):
    try:
        spy_df = pd.read_parquet(dpath("backtest/spy_benchmark.parquet")); spy_df.index=pd.to_datetime(spy_df.index)
        if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns=[c[0].lower() for c in spy_df.columns]
        spy = spy_df['close'].sort_index(); spy=spy.reindex(index, method='ffill').dropna()
        return spy/spy.iloc[0]*start_nav if not spy.empty else pd.Series(dtype=float)
    except: return pd.Series(dtype=float)

def add_regime_overlay(fig, rdf, s, e):
    if rdf.empty: return
    dates, comps = rdf["date"].tolist(), rdf["composite"].tolist()
    for i,(dt,st_) in enumerate(zip(dates,comps)):
        x0,x1 = max(dt,s), min(dates[i+1] if i+1<len(dates) else e, e)
        if x0<x1: fig.add_vrect(x0=x0,x1=x1,fillcolor=REGIME_COLORS.get(st_,"rgba(255,255,255,0.03)"),line_width=0,layer="below")

def metrics(nav_series, spy_series=None, rf=0.02):
    nav=nav_series.dropna()
    if len(nav)<10: return {}
    r=nav.pct_change().dropna(); n_yr=len(nav)/252; total_ret=nav.iloc[-1]/nav.iloc[0]-1
    ann_ret=(1+total_ret)**(1/n_yr)-1 if n_yr>0 else 0; vol=r.std()*np.sqrt(252)
    sharpe=(ann_ret-rf)/vol if vol>0 else 0; down=r[r<0].std()*np.sqrt(252); sortino=(ann_ret-rf)/down if down>0 else 0
    dd=(nav/nav.cummax()-1); max_dd=dd.min(); calmar=ann_ret/abs(max_dd) if max_dd!=0 else 0
    out=dict(total_ret=total_ret,ann_ret=ann_ret,vol=vol,sharpe=sharpe,sortino=sortino,max_dd=max_dd,calmar=calmar)
    if spy_series is not None and not spy_series.empty:
        spy_r=spy_series.pct_change().dropna(); common=r.index.intersection(spy_r.index)
        if len(common)>10:
            active=r.loc[common]-spy_r.loc[common]; te=active.std()*np.sqrt(252)
            cov_=np.cov(r.loc[common],spy_r.loc[common]); beta=cov_[0,1]/cov_[1,1] if cov_[1,1]>0 else 1
            spy_ann=(spy_series.iloc[-1]/spy_series.iloc[0])**(1/n_yr)-1; alpha=ann_ret-beta*spy_ann
            out.update(dict(te=te,beta=beta,alpha=alpha))
    return out

def fmt(v, style="pct"):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "\u2014"
    if style=="pct": return f"{v:+.1%}" if v!=0 else "0.0%"
    if style=="pct_plain": return f"{v:.1%}"
    if style=="x2": return f"{v:.2f}"
    return str(v)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='margin-bottom:20px'><div class='qf-header'>Systematic Equity Engine</div><div class='qf-title'>QuantForge</div></div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Strategy</div>', unsafe_allow_html=True)
    strategy = st.selectbox("", list(STRATEGY_OPTIONS.keys()), format_func=lambda x: STRATEGY_OPTIONS[x], label_visibility="collapsed")
    is_exp = strategy in ["exp005","exp006","exp007"]
    if is_exp:
        freq="monthly"
        st.markdown('<div class="sidebar-label" style="margin-top:12px">Rebalance Frequency</div><div class="sidebar-stat">Monthly (fixed)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-label" style="margin-top:12px">Rebalance Frequency</div>', unsafe_allow_html=True)
        freq = st.selectbox("", ["monthly","quarterly"], format_func=lambda x:x.capitalize(), label_visibility="collapsed")
    n_sig = 15 if is_exp else 11
    wf_w = "17 OOS + Holdout" if strategy in ["exp006","exp007"] else "9 OOS + Holdout"
    tw = "Sliding 5yr + Expanding IC" if strategy=="exp007" else ("Expanding (1997-based)" if strategy=="exp006" else "Expanding (2009-based)")
    st.markdown("---")
    st.markdown(f'<div class="sidebar-label">Universe</div><div class="sidebar-stat">S&P 500 Constituents</div><div class="sidebar-label">Capital</div><div class="sidebar-stat">$1,000,000 Paper</div><div class="sidebar-label">Benchmark</div><div class="sidebar-stat">SPY (Total Return)</div><div class="sidebar-label">Walk-Forward Windows</div><div class="sidebar-stat">{wf_w}</div><div class="sidebar-label">Training Window</div><div class="sidebar-stat">{tw}</div><div class="sidebar-label">Max Position Size</div><div class="sidebar-stat">5% per Stock</div><div class="sidebar-label">Signals</div><div class="sidebar-stat">{n_sig} Multi-Factor</div>', unsafe_allow_html=True)
    if is_exp:
        st.markdown('<div style="margin-top:12px; padding:10px; background:#0D1A14; border:1px solid #1A3828; border-radius:6px"><div class="sidebar-label" style="color:#00C4B4">EXP Additions</div><div style="font-size:11px; color:#8BAEC8; line-height:1.6">+ Revenue Growth<br>+ Low Volatility<br>+ FCF Yield<br>+ Volume Momentum<br>+ VIX threshold 0.95<br>+ Regime multipliers</div></div>', unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Loading backtest data\u2026"):
    wf_nav = load_wf_nav_stitched(strategy, freq); regime_df = load_regime_history()
if wf_nav.empty: st.error(f"No NAV data for {strategy}/{freq}"); st.stop()
primary_nav = wf_nav['nav']; spy_nav = get_spy_nav(primary_nav.index, primary_nav.iloc[0])
strategy_label = STRATEGY_OPTIONS[strategy]
col_h1,col_h2 = st.columns([3,1])
with col_h1: st.markdown(f"<div class='qf-header' style='margin-top:8px'>Backtest Results \u00b7 {strategy_label} \u00b7 {freq.capitalize()}</div><div class='qf-title'>Performance Dashboard</div>", unsafe_allow_html=True)
with col_h2: st.markdown(f"<div style='text-align:right; margin-top:12px'><div class='sidebar-label'>Period</div><div style='font-family:IBM Plex Mono; color:#8BAEC8; font-size:13px'>{primary_nav.index[0].strftime('%b %Y')} \u2192 {primary_nav.index[-1].strftime('%b %Y')}</div></div>", unsafe_allow_html=True)
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["▲  PERFORMANCE","◈  RISK","◻  PORTFOLIO","◉  SIGNALS","⟳  WALK-FORWARD","≡  METHODOLOGY"])

# ── TAB 1: PERFORMANCE ────────────────────────────────────────────────────────
with tab1:
    m = metrics(primary_nav, spy_nav)
    kpi_data = [("Total Return",fmt(m.get('total_ret'),'pct'),m.get('total_ret',0)>0),("Ann. Return",fmt(m.get('ann_ret'),'pct'),m.get('ann_ret',0)>0),("Volatility",fmt(m.get('vol'),'pct_plain'),None),("Sharpe Ratio",fmt(m.get('sharpe'),'x2'),m.get('sharpe',0)>1),("Sortino Ratio",fmt(m.get('sortino'),'x2'),m.get('sortino',0)>1),("Max Drawdown",fmt(m.get('max_dd'),'pct'),False),("Calmar Ratio",fmt(m.get('calmar'),'x2'),m.get('calmar',0)>0.5)]
    kpi_html='<div class="kpi-grid">'
    for label,value,is_good in kpi_data:
        cls="positive" if is_good is True else ("negative" if is_good is False and label!="Volatility" else "")
        kpi_html+=f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value {cls}">{value}</div></div>'
    kpi_html+='</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown('<div class="section-label">Cumulative Growth of $1,000,000</div>', unsafe_allow_html=True)
    show_regime=False
    if not regime_df.empty:
        ct,cl=st.columns([1,5])
        with ct: show_regime=st.toggle("Regime overlay",value=True)
        if show_regime:
            with cl: st.markdown('<div class="regime-legend"><span><span class="regime-swatch" style="background:rgba(0,196,180,0.45)"></span>BULL</span><span><span class="regime-swatch" style="background:rgba(74,144,217,0.40)"></span>RECOVERY</span><span><span class="regime-swatch" style="background:rgba(212,160,23,0.45)"></span>BEAR</span><span><span class="regime-swatch" style="background:rgba(224,82,82,0.45)"></span>CRISIS</span></div>', unsafe_allow_html=True)
    fig=go.Figure()
    if show_regime and not regime_df.empty: add_regime_overlay(fig,regime_df,primary_nav.index[0],primary_nav.index[-1])
    fig.add_trace(go.Scatter(x=primary_nav.index,y=primary_nav,name=f"{strategy.upper()}" if is_exp else f"{strategy.upper()}-{freq}",line=dict(color=C["strategy"],width=2),fill='tozeroy',fillcolor='rgba(0,196,180,0.06)'))
    if not spy_nav.empty: fig.add_trace(go.Scatter(x=spy_nav.index,y=spy_nav,name="SPY Benchmark",line=dict(color=C["benchmark"],width=1.5,dash='dot')))
    fig.update_layout(**PLOTLY_BASE,height=400,yaxis=dict(**YAXIS,tickprefix="$",tickformat=","))
    st.plotly_chart(fig, use_container_width=True)
    col_l,col_r=st.columns(2)
    with col_l:
        st.markdown('<div class="section-label">Annual Returns</div>', unsafe_allow_html=True)
        ann_ret=primary_nav.resample('YE').last().pct_change().dropna(); ann_ret.index=ann_ret.index.year
        fig2=go.Figure(); fig2.add_trace(go.Bar(x=ann_ret.index,y=ann_ret.values*100,name="Strategy",marker_color=[C["positive"] if r>0 else C["negative"] for r in ann_ret.values],text=[f"{r:.1%}" for r in ann_ret.values],textposition='outside',textfont=dict(size=10)))
        if not spy_nav.empty:
            sa=spy_nav.resample('YE').last().pct_change().dropna(); sa.index=sa.index.year
            fig2.add_trace(go.Scatter(x=sa.index,y=sa.values*100,mode='lines+markers',name="SPY",line=dict(color=C["benchmark"],width=1.5),marker=dict(size=5)))
        fig2.update_layout(**PLOTLY_BASE,height=300,yaxis_title="Return (%)"); st.plotly_chart(fig2,use_container_width=True)
    with col_r:
        st.markdown('<div class="section-label">Strategy vs Benchmark</div>', unsafe_allow_html=True)
        if not spy_nav.empty:
            sm=metrics(spy_nav)
            rows=[("Total Return",fmt(m.get('total_ret'),'pct'),fmt(sm.get('total_ret'),'pct')),("Ann. Return",fmt(m.get('ann_ret'),'pct'),fmt(sm.get('ann_ret'),'pct')),("Volatility",fmt(m.get('vol'),'pct_plain'),fmt(sm.get('vol'),'pct_plain')),("Sharpe",fmt(m.get('sharpe'),'x2'),fmt(sm.get('sharpe'),'x2')),("Sortino",fmt(m.get('sortino'),'x2'),fmt(sm.get('sortino'),'x2')),("Max Drawdown",fmt(m.get('max_dd'),'pct'),fmt(sm.get('max_dd'),'pct')),("Calmar",fmt(m.get('calmar'),'x2'),fmt(sm.get('calmar'),'x2')),("Beta vs SPY",fmt(m.get('beta'),'x2'),"1.00"),("Alpha (Ann.)",fmt(m.get('alpha'),'pct'),"\u2014"),("Tracking Error",fmt(m.get('te'),'pct_plain'),"\u2014")]
            cmp_df=pd.DataFrame(rows,columns=["Metric",strategy_label,"SPY"]); st.dataframe(cmp_df.set_index("Metric"),use_container_width=True,height=340)

# ── TAB 2: RISK ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Drawdown Analysis</div>', unsafe_allow_html=True)
    dd_s=(primary_nav/primary_nav.cummax()-1)*100; fig_dd=go.Figure()
    if show_regime and not regime_df.empty: add_regime_overlay(fig_dd,regime_df,primary_nav.index[0],primary_nav.index[-1])
    fig_dd.add_trace(go.Scatter(x=dd_s.index,y=dd_s,fill='tozeroy',fillcolor='rgba(224,82,82,0.15)',line=dict(color=C["negative"],width=1.5),name="Drawdown %"))
    mi=dd_s.idxmin(); fig_dd.add_trace(go.Scatter(x=[mi],y=[dd_s.min()],mode='markers+text',marker=dict(color=C["negative"],size=10,symbol='x'),text=[f"  Max DD: {dd_s.min():.1f}%"],textfont=dict(color=C["muted"],size=10),textposition='middle right',name="Max DD"))
    fig_dd.update_layout(**PLOTLY_BASE,height=280,yaxis_title="Drawdown (%)"); st.plotly_chart(fig_dd,use_container_width=True)
    r=primary_nav.pct_change().dropna(); c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="section-label">Rolling 60-Day Sharpe</div>', unsafe_allow_html=True)
        rs_=(r.rolling(60).mean()/r.rolling(60).std()*np.sqrt(252)).dropna(); fig_rs=go.Figure(); fig_rs.add_hline(y=0,line_color=C["grid"]); fig_rs.add_hline(y=1,line_dash="dot",line_color=C["accent"],annotation_text="Sharpe=1",annotation_font_color=C["accent"])
        fig_rs.add_trace(go.Scatter(x=rs_.index,y=rs_,line=dict(color=C["strategy"],width=1.5),fill='tozeroy',fillcolor='rgba(0,196,180,0.08)',name="Rolling Sharpe")); fig_rs.update_layout(**PLOTLY_BASE,height=270); st.plotly_chart(fig_rs,use_container_width=True)
    with c2:
        st.markdown('<div class="section-label">Rolling 60-Day Volatility</div>', unsafe_allow_html=True)
        rv=r.rolling(60).std()*np.sqrt(252)*100; fig_rv=go.Figure(); fig_rv.add_trace(go.Scatter(x=rv.index,y=rv,line=dict(color=C["strategy"],width=1.5),name="Strategy Vol"))
        if not spy_nav.empty: sr=spy_nav.pct_change().dropna(); srv=sr.rolling(60).std()*np.sqrt(252)*100; fig_rv.add_trace(go.Scatter(x=srv.index,y=srv,line=dict(color=C["benchmark"],width=1.2,dash='dot'),name="SPY Vol"))
        fig_rv.update_layout(**PLOTLY_BASE,height=270,yaxis_title="Vol (%)"); st.plotly_chart(fig_rv,use_container_width=True)
    if not spy_nav.empty:
        st.markdown('<div class="section-label">Tracking Error vs SPY (Rolling 60-Day)</div>', unsafe_allow_html=True)
        sr=spy_nav.pct_change().dropna(); cm=r.index.intersection(sr.index)
        if len(cm)>60:
            ac=r.loc[cm]-sr.loc[cm]; te_=ac.rolling(60).std()*np.sqrt(252)*100; fig_te=go.Figure()
            fig_te.add_hrect(y0=4,y1=6,fillcolor='rgba(212,160,23,0.08)',line_width=0); fig_te.add_hline(y=4,line_dash="dot",line_color=C["amber"]); fig_te.add_hline(y=6,line_dash="dot",line_color=C["negative"])
            fig_te.add_trace(go.Scatter(x=te_.index,y=te_,fill='tozeroy',fillcolor='rgba(0,196,180,0.08)',line=dict(color=C["strategy"],width=1.5),name="TE")); fig_te.update_layout(**PLOTLY_BASE,height=260,yaxis_title="TE (%)"); st.plotly_chart(fig_te,use_container_width=True)
    st.markdown('<div class="section-label">Monthly Return Distribution</div>', unsafe_allow_html=True)
    mr=primary_nav.resample('ME').last().pct_change().dropna()*100; fh=go.Figure(); fh.add_trace(go.Histogram(x=mr,nbinsx=40,marker_color=C["strategy"],opacity=0.7,name="Strategy"))
    if not spy_nav.empty: sm_=spy_nav.resample('ME').last().pct_change().dropna()*100; fh.add_trace(go.Histogram(x=sm_,nbinsx=40,marker_color=C["benchmark"],opacity=0.5,name="SPY"))
    fh.update_layout(**PLOTLY_BASE,height=250,barmode='overlay',xaxis_title="Monthly Return (%)",yaxis_title="Count"); st.plotly_chart(fh,use_container_width=True)

# ── TAB 3: PORTFOLIO ──────────────────────────────────────────────────────────
with tab3:
    portfolios=load_all_wf_portfolios(strategy,freq); trades_df=load_all_wf_trades(strategy,freq)
    if not portfolios.empty:
        portfolios['date']=pd.to_datetime(portfolios['date']); active=portfolios[portfolios['target_weight']>0]
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Active Position Count per Rebalance</div>',unsafe_allow_html=True)
            pc=active.groupby('date')['ticker'].count(); fp=go.Figure(); fp.add_trace(go.Scatter(x=pc.index,y=pc,mode='lines+markers',marker=dict(size=4),line=dict(color=C["strategy"]),fill='tozeroy',fillcolor='rgba(0,196,180,0.06)',name="Positions"))
            fp.add_hline(y=pc.mean(),line_dash="dot",line_color=C["muted"],annotation_text=f"  Avg: {pc.mean():.0f}",annotation_font_color=C["muted"]); fp.update_layout(**PLOTLY_BASE,height=280,yaxis_title="# Positions"); st.plotly_chart(fp,use_container_width=True)
        with c2:
            st.markdown('<div class="section-label">Top-10 Holdings Concentration</div>',unsafe_allow_html=True)
            t10=active.groupby('date').apply(lambda g:g['target_weight'].nlargest(10).sum()*100); fc=go.Figure(); fc.add_trace(go.Scatter(x=t10.index,y=t10,fill='tozeroy',fillcolor='rgba(74,144,217,0.08)',line=dict(color=C["accent"],width=1.5),name="Top-10 Weight")); fc.update_layout(**PLOTLY_BASE,height=280,yaxis_title="Weight (%)"); st.plotly_chart(fc,use_container_width=True)
        if not trades_df.empty:
            st.markdown('<div class="section-label">Monthly Portfolio Turnover ($M)</div>',unsafe_allow_html=True)
            trades_df['date']=pd.to_datetime(trades_df['date']); trades_df['value']=trades_df['shares']*trades_df['price']; mt=trades_df.groupby(pd.Grouper(key='date',freq='ME'))['value'].sum()/1e6
            ft=go.Figure(); ft.add_trace(go.Bar(x=mt.index,y=mt,marker_color=C["accent"],opacity=0.8,name="Turnover ($M)")); ft.update_layout(**PLOTLY_BASE,height=250,yaxis_title="$ Millions"); st.plotly_chart(ft,use_container_width=True)
        st.markdown('<div class="section-label">Most Frequently Held Stocks</div>',unsafe_allow_html=True)
        th=active['ticker'].value_counts().head(25); fth=go.Figure(go.Bar(x=th.index,y=th.values,marker=dict(color=th.values,colorscale=[[0,'#1B6EBE'],[1,'#00C4B4']],showscale=False),text=th.values,textposition='outside',name="Appearances")); fth.update_layout(**PLOTLY_BASE,height=300,yaxis_title="Rebalance Appearances"); st.plotly_chart(fth,use_container_width=True)
    else: st.warning("No portfolio data found.")

# ── TAB 4: SIGNALS ────────────────────────────────────────────────────────────
with tab4:
    ACT_SIG = SIGNAL_COLS_15 if is_exp else SIGNAL_COLS_11
    @st.cache_data(show_spinner=False)
    def compute_ic(_strat):
        sig=load_signals(_strat); fwd=load_fwd_returns(_strat); merged=sig.merge(fwd[['ticker','date','fwd_1m']],on=['ticker','date'],how='inner')
        sc=SIGNAL_COLS_15 if _strat in ["exp005","exp006","exp007"] else SIGNAL_COLS_11; results=[]
        for col in sc:
            if col not in merged.columns: continue
            ic_ts=merged.groupby('date').apply(lambda g:g[[col,'fwd_1m']].dropna().corr(method='spearman').iloc[0,1]).dropna()
            group=next((g for g,cols in GROUPS.items() if col in cols),'Other')
            results.append({'signal':col,'Signal':SIGNAL_LABELS.get(col,col),'Group':group,'Mean IC':ic_ts.mean(),'IC Std':ic_ts.std(),'IC IR':ic_ts.mean()/ic_ts.std() if ic_ts.std()>0 else 0,'Hit Rate':(ic_ts>0).mean(),'ic_ts':ic_ts})
        return results
    with st.spinner("Computing signal ICs\u2026"):
        try:
            ic_results=compute_ic(strategy)
            c1,c2=st.columns([3,2])
            with c1:
                st.markdown('<div class="section-label">IC \u2014 Mean Rank Correlation with 1-Month Forward Returns</div>',unsafe_allow_html=True)
                ic_df=pd.DataFrame([{k:v for k,v in r.items() if k!='ic_ts'} for r in ic_results]).sort_values('Mean IC',ascending=True)
                fig_ic=go.Figure()
                for group,color in GROUP_COLORS.items():
                    sub=ic_df[ic_df['Group']==group]
                    if sub.empty: continue
                    fig_ic.add_trace(go.Bar(x=sub['Mean IC'],y=sub['Signal'],orientation='h',name=group,marker_color=color,text=[f"{v:.4f}" for v in sub['Mean IC']],textposition='outside',textfont=dict(size=10)))
                fig_ic.add_vline(x=0,line_color=C["grid"]); fig_ic.update_layout(**{**PLOTLY_BASE,'margin':dict(l=0,r=80,t=30,b=10)},height=450 if is_exp else 400,xaxis_title="Mean IC",barmode='relative'); st.plotly_chart(fig_ic,use_container_width=True)
            with c2:
                st.markdown('<div class="section-label">IC Summary Table</div>',unsafe_allow_html=True)
                tbl=ic_df[['Signal','Group','Mean IC','IC IR','Hit Rate']].sort_values('Mean IC',ascending=False).copy()
                tbl['Mean IC']=tbl['Mean IC'].map(lambda x:f"{x:.4f}"); tbl['IC IR']=tbl['IC IR'].map(lambda x:f"{x:.2f}"); tbl['Hit Rate']=tbl['Hit Rate'].map(lambda x:f"{x:.1%}")
                st.dataframe(tbl.set_index('Signal'),use_container_width=True,height=450 if is_exp else 400)
            st.markdown('<div class="section-label">Rolling IC Over Time \u2014 Top Signals</div>',unsafe_allow_html=True)
            top5=sorted(ic_results,key=lambda x:abs(x['Mean IC']),reverse=True)[:5]; pal=[C["strategy"],C["accent"],C["purple"],C["amber"],C["negative"]]
            fig_ric=go.Figure()
            for i,sd in enumerate(top5): ic_ts=sd['ic_ts'].rolling(12).mean(); fig_ric.add_trace(go.Scatter(x=ic_ts.index,y=ic_ts,name=sd['Signal'],line=dict(color=pal[i],width=1.5)))
            fig_ric.add_hline(y=0,line_color=C["grid"]); fig_ric.update_layout(**PLOTLY_BASE,height=280,yaxis_title="12M Rolling Mean IC"); st.plotly_chart(fig_ric,use_container_width=True)
        except Exception as e: st.error(f"IC error: {e}")

# ── TAB 5: WALK-FORWARD ──────────────────────────────────────────────────────
with tab5:
    st.markdown(f'<div class="section-label">Walk-Forward \u00b7 {wf_w} \u00b7 {strategy_label}</div>',unsafe_allow_html=True)
    @st.cache_data(show_spinner=False)
    def per_window_metrics(strat,fr):
        rows=[]; wl=[str(i) for i in range(1,30) if os.path.exists(dpath(f"backtest/wf_results/nav_window_{i}_{strat}_{fr}.parquet"))]+['holdout']
        for w in wl:
            f=dpath(f"backtest/wf_results/nav_window_{w}_{strat}_{fr}.parquet")
            if not os.path.exists(f): continue
            try:
                df=pd.read_parquet(f); df.index=pd.to_datetime(df.index); m2=metrics(df['nav'])
                rows.append({'Window':f"W{w}" if w!='holdout' else 'Holdout','Start':df.index[0].strftime('%Y-%m'),'End':df.index[-1].strftime('%Y-%m'),**m2})
            except: pass
        return pd.DataFrame(rows)
    wm_df=per_window_metrics(strategy,freq)
    if not wm_df.empty:
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Sharpe per Window</div>',unsafe_allow_html=True)
            fw=go.Figure(go.Bar(x=wm_df['Window'],y=wm_df['sharpe'],marker_color=[C["positive"] if v>0 else C["negative"] for v in wm_df['sharpe']],text=[f"{v:.3f}" for v in wm_df['sharpe']],textposition='outside')); fw.add_hline(y=0,line_color=C["grid"]); fw.update_layout(**PLOTLY_BASE,height=280,yaxis_title="Sharpe"); st.plotly_chart(fw,use_container_width=True)
        with c2:
            st.markdown('<div class="section-label">Ann. Return per Window</div>',unsafe_allow_html=True)
            fw2=go.Figure(go.Bar(x=wm_df['Window'],y=wm_df['ann_ret']*100,marker_color=[C["positive"] if v>0 else C["negative"] for v in wm_df['ann_ret']],text=[f"{v:.1%}" for v in wm_df['ann_ret']],textposition='outside')); fw2.add_hline(y=0,line_color=C["grid"]); fw2.update_layout(**PLOTLY_BASE,height=280,yaxis_title="Ann. Return (%)"); st.plotly_chart(fw2,use_container_width=True)
        st.markdown('<div class="section-label">Window Detail Table</div>',unsafe_allow_html=True)
        tbl=wm_df[['Window','Start','End','ann_ret','vol','sharpe','sortino','max_dd','calmar']].copy()
        tbl['ann_ret']=tbl['ann_ret'].map(lambda x:f"{x:.1%}"); tbl['vol']=tbl['vol'].map(lambda x:f"{x:.1%}"); tbl['sharpe']=tbl['sharpe'].map(lambda x:f"{x:.2f}"); tbl['sortino']=tbl['sortino'].map(lambda x:f"{x:.2f}"); tbl['max_dd']=tbl['max_dd'].map(lambda x:f"{x:.1%}"); tbl['calmar']=tbl['calmar'].map(lambda x:f"{x:.2f}")
        tbl=tbl.rename(columns={'ann_ret':'Ann. Ret','vol':'Vol','sharpe':'Sharpe','sortino':'Sortino','max_dd':'Max DD','calmar':'Calmar'}); st.dataframe(tbl.set_index('Window'),use_container_width=True)
    st.markdown("---"); st.markdown('<div class="section-label">All Strategy Combinations</div>',unsafe_allow_html=True)
    @st.cache_data(show_spinner=False)
    def all_combo_metrics():
        rows=[]
        for s in ['mv','bl']:
            for f in ['monthly','quarterly']:
                wf=load_wf_nav_stitched(s,f)
                if wf.empty: continue
                m2=metrics(wf['nav']); m2['Strategy']=f"{s.upper()}-{f}"; rows.append(m2)
        for exp in ['exp005','exp006','exp007']:
            wf=load_wf_nav_stitched(exp,'monthly')
            if wf.empty: continue
            m2=metrics(wf['nav']); m2['Strategy']=f"{exp.upper()}-monthly"; rows.append(m2)
        return pd.DataFrame(rows)
    with st.spinner("Loading all combinations\u2026"): combo_df=all_combo_metrics()
    if not combo_df.empty:
        mo={'Sharpe Ratio':'sharpe','Ann. Return':'ann_ret','Max Drawdown':'max_dd','Calmar':'calmar','Volatility':'vol','Sortino':'sortino'}
        sel=st.selectbox("Compare by",list(mo.keys())); key=mo[sel]; scale=100 if key in ['ann_ret','max_dd','vol'] else 1
        sc={'MV-monthly':C["strategy"],'MV-quarterly':'#007D74','BL-monthly':C["purple"],'BL-quarterly':'#4A3080','EXP005-monthly':'#FFD700','EXP006-monthly':'#FF6B6B','EXP007-monthly':'#4ECDC4'}
        fc=go.Figure(go.Bar(x=combo_df['Strategy'],y=combo_df[key]*scale,marker_color=[sc.get(s,C["neutral"]) for s in combo_df['Strategy']],text=[f"{v*scale:.3f}{'%' if scale==100 else ''}" for v in combo_df[key]],textposition='outside'))
        fc.update_layout(**PLOTLY_BASE,height=300,yaxis_title=sel,title=f"{sel} \u2014 Strategy Comparison"); st.plotly_chart(fc,use_container_width=True)
        ct=combo_df[['Strategy','total_ret','ann_ret','vol','sharpe','sortino','max_dd','calmar']].copy()
        for col in ['total_ret','ann_ret','vol','max_dd']: ct[col]=ct[col].map(lambda x:f"{x:.1%}")
        for col in ['sharpe','sortino','calmar']: ct[col]=ct[col].map(lambda x:f"{x:.2f}")
        ct=ct.rename(columns={'total_ret':'Total Ret','ann_ret':'Ann. Ret','vol':'Vol','sharpe':'Sharpe','sortino':'Sortino','max_dd':'Max DD','calmar':'Calmar'}); st.dataframe(ct.set_index('Strategy'),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — METHODOLOGY (complete rewrite with cumulative strategy cards)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        # ── CARD 1: What is this system? (all strategies) ──
        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">What is this system?</div>
          <p>
            QuantForge is a systematic, rules-based equity strategy that invests in S&P 500 stocks.
            It uses no human discretion — every buy and sell decision is driven entirely by data and
            a defined process. It scores stocks on multiple factors, optimizes position sizes
            mathematically, and manages risk through automated trailing stops and circuit breakers.
            Think of it as a quantitative fund manager running the same disciplined playbook every
            month, regardless of news or sentiment.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # ── STRATEGY-SPECIFIC CARD (cumulative, self-contained) ──
        if strategy == "exp005":
            st.markdown("""
            <div class="method-card">
              <div class="method-card-title">What makes EXP005 different?</div>
              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">Signal expansion: 11 → 15 factors</p>
              <p>The base strategy uses 11 signals across momentum, value, quality, and mean reversion.
                 EXP005 adds four new signals that capture dimensions the original set missed:</p>
              <ul>
                <li><span class="teal">Revenue Growth</span> — quarter-over-quarter revenue acceleration.
                    Earnings momentum (already in the base set) looks at EPS, but revenue growth captures
                    top-line expansion that often precedes earnings surprises.</li>
                <li><span class="teal">Low Volatility</span> — inverse of 252-day realized volatility.
                    The low-vol anomaly is one of the most robust findings in empirical finance: less volatile
                    stocks deliver higher risk-adjusted returns. This signal provides defensive tilt during
                    bear markets (2022: the strategy generated +31% alpha vs SPY).</li>
                <li><span class="teal">FCF Yield</span> — annualized free cash flow divided by market cap.
                    Unlike P/E or P/B, FCF yield measures actual cash generation. A company can report
                    high earnings through accounting choices, but free cash flow is harder to manipulate.</li>
                <li><span class="teal">Volume Momentum</span> — ratio of 5-day to 60-day average volume.
                    Unusual volume spikes often signal institutional accumulation or distribution before
                    price moves. This captures sentiment that pure price-based signals miss.</li>
              </ul>
              <br>
              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">VIX regime recalibration</p>
              <p>The base strategy classified market stress using a VIX threshold of 0.80× its 252-day
                 average. Empirical analysis revealed this was too strict — only 21% of months were
                 labeled "bull" vs the actual ~55% proportion. The strategy was stuck in defensive mode
                 most of the time, missing bull market alpha. EXP005 recalibrated the threshold to 0.95,
                 correctly identifying ~55% of months as bull. This alone improved returns in years like
                 2013 (1/12 → 7/12 bull months) and 2017 (3/12 → 11/12).</p>
              <br>
              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">Regime-conditional signal multipliers</p>
              <p>Each signal's weight is multiplied by a regime-specific factor. For example, momentum
                 signals get a 1.3× boost in bull markets but are dampened to 0.5× in crisis.
                 Quality and defensive signals get boosted in bear/crisis regimes. This ensures the
                 portfolio tilts toward the right factors for the current market environment.</p>
              <br>
              <p style="color:#D4A017; font-weight:500; margin-bottom:4px">Result vs base strategy (2013–2025):</p>
              <p>Sharpe improved from 1.02 to 1.24. CAGR from 18.6% to 22.2%. Max drawdown improved
                 from −29.8% to −26.1%. All 15 signals, VIX recalibration, and regime multipliers
                 contributed to the improvement.</p>
            </div>
            """, unsafe_allow_html=True)

        if strategy == "exp006":
            st.markdown("""
            <div class="method-card">
              <div class="method-card-title">What makes EXP006 different?</div>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">Inherited from EXP005</p>
              <p>EXP006 uses the same 15 signals (11 base + revenue growth, low volatility, FCF yield,
                 volume momentum), VIX threshold of 0.95, and regime-conditional signal multipliers
                 introduced in EXP005.</p>
              <br>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">New: Extended training history back to 1997</p>
              <p>The base strategy and EXP005 train from 2009, producing the first out-of-sample trades
                 in 2013. With only 4 years of training data in the first window, the IC-IR signal weights
                 were unstable — explaining the flat returns from 2013–2016.</p>
              <p style="margin-top:8px">EXP006 pushes the training start back to 1997 by backfilling
                 macro data (VIX, yield curve, credit spreads) from FRED and recomputing market breadth
                 from price data back to 1996. This gives the first window 8 years of training data
                 (96+ monthly dates), making IC-IR weights stable from day one.</p>
              <br>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">17 walk-forward windows (2005–2021)</p>
              <p>With training from 1997, the first out-of-sample trade moves to January 2005. This adds
                 8 years of tested performance including the dot-com aftermath (2005–2006), the pre-GFC
                 bull market (2007), the Global Financial Crisis (2008), and the post-crisis recovery
                 (2009–2012). These are all out-of-sample — the model was trained only on pre-2005 data
                 for the first window.</p>
              <br>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">GFC inclusion and max drawdown</p>
              <p>The −37.6% max drawdown comes from the 2008 window. The strategy lost 17.9% that year
                 while SPY lost 37.7% — cutting the crisis drawdown roughly in half. Strategies starting
                 after 2009 (MV, BL, EXP005) avoid this period entirely, which is why their max drawdowns
                 appear better. EXP006's drawdown is the honest cost of testing through a once-in-a-generation
                 financial crisis.</p>
              <br>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">Key finding: stability vs adaptability</p>
              <p>On the common period (2013–2025), EXP006 beats EXP005 by +0.8% CAGR (23.0% vs 22.2%),
                 confirming that deeper training history produces more stable and more accurate IC-IR
                 signal weights. The tradeoff: expanding windows from 1997 include data from the
                 value-dominated era, which can slightly dilute signal weights in modern growth-driven
                 markets.</p>
              <br>

              <p style="color:#D4A017; font-weight:500; margin-bottom:4px">Realistic live performance estimate:</p>
              <p>The holdout period (2022–2026) uses fixed median parameters with no per-year optimization.
                 Holdout CAGR is ~20% — this is the honest forward-looking estimate, not the full
                 walk-forward CAGR of 23.6% which benefits from per-window parameter tuning that
                 doesn't exist in live trading.</p>
            </div>
            """, unsafe_allow_html=True)

        if strategy == "exp007":
            st.markdown("""
            <div class="method-card">
              <div class="method-card-title">What makes EXP007 different?</div>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">Inherited from EXP005 + EXP006</p>
              <p>EXP007 uses the same 15 signals, VIX threshold 0.95, and regime multipliers from EXP005.
                 It also inherits EXP006's extended data depth — training from 1997, 17 walk-forward
                 windows (2005–2021), and backfilled macro data.</p>
              <br>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">New: Split approach — expanding IC, sliding params</p>
              <p>EXP006 proved that longer training history improves IC-IR signal weights. But does the
                 same apply to execution parameters (lambda, risk_aversion)? EXP007 tests this by splitting
                 the two concerns:</p>
              <ul>
                <li><span class="teal">IC-IR signal weights</span> — computed from the full expanding history
                    (1997 to train_end) with EWM halflife of 60 months. Recent signal performance gets
                    ~2× the weight of data from 5 years ago, but old data is never fully dropped. This
                    preserves the stability benefit proven in EXP006.</li>
                <li><span class="teal">Parameter calibration</span> — lambda (turnover penalty) and
                    risk_aversion are swept on a 5-year sliding window only. The hypothesis: optimal
                    execution parameters shift with market regimes, so the optimizer should learn from
                    recent conditions rather than decades-old data.</li>
              </ul>
              <br>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">New: MIN_SIGNALS filter</p>
              <p>Stocks now need 10 of 15 valid signals to enter the universe (vs 6 in EXP005/006).
                 Analysis showed that the 3 value signals (P/E, P/B, EV/EBITDA) are missing for ~30%
                 of stocks due to incomplete fundamental data. With MIN_SIGNALS=6, these stocks entered
                 the universe with nearly half their composite score driven by zero-filled missing values —
                 pure noise. At MIN_SIGNALS=10, only 2.5% of stock-dates are excluded, but those excluded
                 are the ones with the weakest signal coverage.</p>
              <br>

              <p style="color:#E8EEF7; font-weight:500; margin-bottom:10px">Result: sliding params hurt</p>
              <p>With only 60 training dates per sliding window (vs 96–288 in EXP006's expanding windows),
                 the optimizer produced noisier parameter estimates. The calibration Sharpe ratios were
                 consistently lower (0.35–0.46 vs EXP006's 0.52–0.60 for later windows).</p>
              <ul>
                <li>Full-period CAGR: 22.8% vs EXP006's 23.6%</li>
                <li>Holdout CAGR: 17.7% vs EXP006's 19.8%</li>
                <li>Common-period max drawdown: −26.0% vs EXP006's −27.6% (EXP007 wins here)</li>
              </ul>
              <br>
              <p>EXP007 adapted faster during regime transitions — COVID (2020: +29.3% vs +26.0%) and
                 the 2022 bear (+10.7% vs +7.3%). But this marginal drawdown improvement came at the cost
                 of ~2% annual return, which compounds to a massive gap over 21 years.</p>
              <br>

              <p style="color:#D4A017; font-weight:500; margin-bottom:4px">Conclusion:</p>
              <p>For this strategy, both IC-IR weights AND execution parameters benefit from longer
                 training history. The expanding window approach (EXP006) is superior. The sliding window
                 idea — intuitive as it sounds — sacrifices return stability for marginal drawdown
                 improvement that does not justify the cost. The EWM IC decay and MIN_SIGNALS=10 filter
                 are worth keeping; the sliding param window is not.</p>
            </div>
            """, unsafe_allow_html=True)

        # ── CARD 2: How does it pick stocks? (all strategies, dynamic) ──
        signals_desc = "15 signals across seven categories" if is_exp else "11 signals across four categories"
        categories_html = """
            <li><span class="teal">Momentum</span> — stocks that have been rising tend to keep rising.
                We measure 12-month price trend (excluding the most recent month to avoid microstructure
                noise) and earnings acceleration (quarter-over-quarter EPS change).</li>
            <li><span class="teal">Value</span> — stocks trading cheaply relative to fundamentals.
                P/E, P/B, and EV/EBITDA are z-scored against sector peers so a "cheap" tech stock
                is compared to other tech stocks, not to utilities.</li>
            <li><span class="teal">Quality</span> — financially healthy companies with stable, real
                earnings. ROE stability (low variance over 8 quarters), gross margin trend (improving
                margins), Piotroski F-Score (9-point financial health checklist), and earnings accruals
                (low accruals = earnings backed by real cash flow).</li>
            <li><span class="teal">Mean Reversion</span> — stocks that have fallen too far, too fast.
                5-day reversal captures short-term overselling; RSI extremes flag technically
                oversold conditions.</li>
        """
        if is_exp:
            categories_html += """
            <li><span class="teal">Growth</span> — quarter-over-quarter revenue acceleration,
                capturing top-line expansion that precedes earnings surprises.</li>
            <li><span class="teal">Defensive</span> — inverse of 252-day realized volatility,
                exploiting the low-vol anomaly for downside protection.</li>
            <li><span class="teal">Sentiment</span> — 5-day vs 60-day average volume ratio,
                flagging unusual institutional activity before price moves.</li>
            """

        st.markdown(f"""
        <div class="method-card">
          <div class="method-card-title">How does it pick stocks?</div>
          <p>Every S&P 500 stock is scored on {signals_desc}:</p>
          <ul>{categories_html}</ul>
          <br>
          <p>Signals are combined using <span style="color:#E8EEF7; font-weight:500">IC-IR weighting</span>:
             for each signal, we measure its Information Coefficient (Spearman rank correlation between
             the signal score and next-month stock returns) across all training dates, then weight
             each signal by IC / std(IC). Signals that consistently predict returns get high weight;
             noisy or negative-IC signals get zero weight. The top-scoring stocks are bought; the rest
             are sold or ignored.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── CARD 3: Position sizing (all strategies) ──
        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">How does it size positions?</div>
          <p>Position sizes are determined by a portfolio optimizer, not gut feel. Two variants:</p>
          <ul>
            <li><span class="hl">Mean-Variance (MV)</span> — finds the mathematically optimal mix of
                stocks maximizing return per unit of risk. Uses Ledoit-Wolf covariance shrinkage
                rather than raw sample covariance, which is underdetermined for 500 stocks on 252
                trading days.</li>
            <li><span class="hl">Black-Litterman (BL)</span> — starts from market-cap weights (what
                the S&P 500 already looks like) and tilts toward high-scoring stocks. More conservative
                with less aggressive rebalancing.</li>
          </ul>
          <br>
          <p>Both enforce a <span style="color:#E8EEF7; font-weight:500">5% maximum weight per stock</span>
             so no single position can dominate the portfolio. A turnover penalty (lambda) discourages
             excessive trading — the optimizer must believe a trade improves the portfolio enough to
             justify its transaction cost. Tracking error is constrained to 6% vs SPY, preventing
             extreme benchmark divergence.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── CARD 4: Risk management (all strategies) ──
        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">How does it manage risk?</div>
          <ul>
            <li><span class="hl">Trailing stops</span> — if a stock drops more than 2× its 25-day
                exponentially-weighted volatility from its recent high, it is automatically sold
                regardless of the rebalance schedule. Stop distance has a floor of 5% and cap of 20%.
                Stops are checked daily on the LOW price and executed as market sells at next open.
                A 1-day seasoning rule prevents false triggers on newly opened positions.</li>
            <li><span class="hl">Drawdown circuit breakers</span> — four tiers at 5%, 10%, 15%, and
                20% from peak. Each tier progressively reduces maximum position sizes, increases
                rebalance frequency, and restricts new positions. At 20%, trading pauses for 5 days
                and manual review is required. Recovery requires 40% retracement of the drawdown.</li>
            <li><span class="hl">Sector neutrality</span> — the portfolio is balanced across GICS
                sectors using pooled neutralization for small sectors (Materials+Industrials,
                Real Estate+Financials, Utilities+Energy) to prevent accidental sector concentration.</li>
            <li><span class="hl">Regime detection</span> — daily monitoring of VIX and market breadth
                for stress classification, weekly monitoring of yield curve and credit spreads for
                economic cycle. Four composite states (bull, recovery, bear, crisis) adjust both
                signal multipliers and rebalance frequency. Regime overlays are shown on the NAV chart.</li>
            <li><span class="hl">Tracking error cap</span> — the optimizer is constrained to stay within
                6% annualized tracking error of SPY, preventing extreme benchmark divergence even when
                signals are strongly directional.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        # ── CARD 5: Walk-forward methodology (all strategies, dynamic) ──
        if strategy == "exp007":
            train_desc = "Signal weights are learned from expanding history starting 1997; execution parameters (lambda, risk aversion) are calibrated on a 5-year sliding window of recent data only."
        elif strategy == "exp006":
            train_desc = "The system learns signal weights and optimizer parameters using all data from 1997 up to the training cutoff."
        elif is_exp:
            train_desc = "The system learns signal weights and optimizer parameters using all data from 2009 up to the training cutoff."
        else:
            train_desc = "The system learns signal weights and optimizer parameters using all data from 2009 up to the training cutoff."

        if strategy in ["exp006", "exp007"]:
            slide_desc = "The training window expands by one year and the process repeats. 17 windows from 2005 to 2021, each independently trained and tested."
        else:
            slide_desc = "The training window expands by one year and the process repeats. 9 windows from 2013 to 2021, each independently trained and tested."

        st.markdown(f"""
        <div class="method-card">
          <div class="method-card-title">How was the backtest run?</div>
          <p>The backtest uses <span style="color:#E8EEF7; font-weight:500">walk-forward testing</span> —
             the gold standard for validating systematic strategies. Unlike a single backtest where the
             model sees all data at once, walk-forward enforces strict temporal separation between
             training and testing.</p>
          <div class="pipeline-step">
            <div class="step-num">1</div>
            <div class="step-body">
              <div class="step-title">Train on historical data</div>
              <div class="step-desc">{train_desc} No future data is ever used in this step.</div>
            </div>
          </div>
          <div class="pipeline-step">
            <div class="step-num">2</div>
            <div class="step-body">
              <div class="step-title">Test on the next 12 months (out-of-sample)</div>
              <div class="step-desc">The trained model is deployed on the following year — data it has
                never seen. Every trade, every stop trigger, every rebalance in the test period uses
                only the parameters learned in step 1. Returns from this period are the honest,
                real-world performance estimate.</div>
            </div>
          </div>
          <div class="pipeline-step">
            <div class="step-num">3</div>
            <div class="step-body">
              <div class="step-title">Slide the window forward and repeat</div>
              <div class="step-desc">{slide_desc}</div>
            </div>
          </div>
          <div class="pipeline-step">
            <div class="step-num">4</div>
            <div class="step-body">
              <div class="step-title">Final holdout: 2022–2026</div>
              <div class="step-desc">The last 4+ years were completely untouched during development.
                Parameters are fixed at the median of all walk-forward windows — no per-year
                optimization. This is the strictest and most realistic test of what live performance
                would look like. It covers a bear market (2022), recovery (2023), and a narrow
                mega-cap-driven bull market (2024–2025).</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CARD 6: Regime detection (all strategies) ──
        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">Regime detection methodology</div>
          <p>Regime states are reconstructed point-in-time using only data available on each date —
             no lookahead bias. Two layers combine into one of four composite states:</p>
          <ul>
            <li><span class="teal">L1 — Market stress (daily)</span>: VIX vs its 252-day average
                combined with market breadth (percentage of S&P 500 constituents above their 200-day
                moving average, using point-in-time index membership).</li>
            <li><span class="teal">L2 — Economic cycle (weekly)</span>: 10Y–2Y Treasury yield curve
                slope combined with high-yield credit spread direction (HYG/LQD ratio).</li>
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

        # ── CARD 7: Honest limitations (all strategies, some dynamic) ──
        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">What are the honest limitations?</div>
          <div class="honesty-box">
            <p><strong>Transaction costs are estimated, not exact.</strong> We model a 10bps round-trip
               cost per trade. Real costs depend on market impact, bid-ask spread, and execution timing.
               For a $1M portfolio trading ~20 stocks monthly, this is a reasonable approximation but
               not a guarantee.</p>
          </div>
          <div class="honesty-box">
            <p><strong>Fundamental data is point-in-time approximated.</strong> FMP quarterly financials
               are stamped by reporting period end date. In reality, companies file 30–60 days after
               quarter-end. We apply a 45-day announcement lag buffer, which approximates but does not
               perfectly replicate the exact filing date for every stock.</p>
          </div>
        """, unsafe_allow_html=True)

        if strategy in ["exp006", "exp007"]:
            st.markdown("""
          <div class="honesty-box">
            <p><strong>Max drawdown includes the 2008 financial crisis.</strong> The −37% drawdown
               occurred during the GFC. Strategies starting after 2009 (MV, BL, EXP005) do not include
               this period, which is why their max drawdowns appear better. The strategy still
               outperformed SPY during the crisis (−18% vs −38%), cutting the drawdown roughly in half.
               This is the honest cost of testing through a once-in-a-generation event.</p>
          </div>
            """, unsafe_allow_html=True)

        st.markdown("""
          <div class="honesty-box">
            <p><strong>The 2024–2025 underperformance is expected and explainable.</strong> The strategy
               is diversified across 20–80 stocks with a 5% cap per position. When the market is driven
               by 5–7 mega-cap tech stocks (NVIDIA, Apple, Microsoft), a diversified approach will
               underperform a market-cap-weighted index. This is a feature, not a bug — it reflects a
               deliberate risk management choice that protects against concentration risk in all other
               market environments.</p>
          </div>
          <div class="honesty-box">
            <p><strong>This is a paper trading simulation.</strong> The strategy currently runs on Alpaca
               paper trading with $1M simulated capital. Live performance may differ from backtested
               results due to slippage, partial fills, and data latency.</p>
          </div>
          <div class="honesty-box">
            <p><strong>Walk-forward CAGR overstates realistic live expectations.</strong> Each walk-forward
               window calibrates its own optimal parameters (lambda, risk aversion) on its training data.
               In live trading, parameters are fixed at the median across all windows — there is no
               per-year optimization. The holdout period (2022–2026) uses these fixed median parameters
               and is the most honest estimate of live performance. Expect the holdout CAGR (~18–20%)
               rather than the full walk-forward CAGR (~23%) as a realistic forward-looking estimate.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CARD 8: Data sources (all strategies) ──
        st.markdown("""
        <div class="method-card">
          <div class="method-card-title">Data sources</div>
          <ul>
            <li><span class="hl">Price & fundamental data</span> — Financial Modeling Prep (FMP)
                Premium, 30+ years of history, quarterly financials for all S&P 500 stocks.</li>
            <li><span class="hl">Macro regime data</span> — FRED (Federal Reserve) for VIX, yield
                curve (10Y–2Y), and credit spreads. Full history from 1997 for EXP006/007; from 2007
                for base strategies.</li>
            <li><span class="hl">Execution</span> — Alpaca paper trading API for order submission and
                fill confirmation.</li>
            <li><span class="hl">Universe</span> — S&P 500 constituents with point-in-time historical
                membership via FMP's historical constituent endpoint, avoiding survivorship bias.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)