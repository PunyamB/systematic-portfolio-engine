# precompute_dashboard.py
# Precomputes ALL dashboard data into a single pickle file.
# Run locally after any backtest changes, then push the pickle to GitHub.
# The Streamlit app loads this one file instead of 258+ parquet files.
#
# Usage: python precompute_dashboard.py
# Output: data/backtest/dashboard_cache.pkl

import os
import glob
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(".")
def dpath(*parts): return str(BASE / "data" / Path(*parts))

STRATEGIES = {
    ("mv", "monthly"): "MV-monthly",
    ("mv", "quarterly"): "MV-quarterly",
    ("bl", "monthly"): "BL-monthly",
    ("bl", "quarterly"): "BL-quarterly",
    ("exp005", "monthly"): "EXP005-monthly",
    ("exp006", "monthly"): "EXP006-monthly",
    ("exp007", "monthly"): "EXP007-monthly",
}

SIGNAL_COLS_11 = ['momentum_12_1','earnings_momentum','pe_zscore','pb_zscore',
                  'ev_ebitda_zscore','roe_stability','gross_margin_trend','piotroski',
                  'earnings_accruals','short_term_reversal','rsi_extremes']
SIGNAL_COLS_15 = SIGNAL_COLS_11 + ['revenue_growth','low_volatility','fcf_yield','volume_momentum']

SIGNAL_LABELS = {
    'momentum_12_1':'12-1 Momentum','earnings_momentum':'Earnings Momentum',
    'pe_zscore':'P/E Z-Score','pb_zscore':'P/B Z-Score','ev_ebitda_zscore':'EV/EBITDA Z-Score',
    'roe_stability':'ROE Stability','gross_margin_trend':'Gross Margin Trend',
    'piotroski':'Piotroski F-Score','earnings_accruals':'Earnings Accruals',
    'short_term_reversal':'Short-Term Reversal','rsi_extremes':'RSI Extremes',
    'revenue_growth':'Revenue Growth (QoQ)','low_volatility':'Low Volatility',
    'fcf_yield':'FCF Yield','volume_momentum':'Volume Momentum',
}

GROUPS = {
    'Momentum':['momentum_12_1','earnings_momentum'],
    'Value':['pe_zscore','pb_zscore','ev_ebitda_zscore','fcf_yield'],
    'Quality':['roe_stability','gross_margin_trend','piotroski','earnings_accruals'],
    'Mean Rev':['short_term_reversal','rsi_extremes'],
    'Growth':['revenue_growth'],
    'Defensive':['low_volatility'],
    'Sentiment':['volume_momentum'],
}


# ── HELPERS ──

def stitch_nav(strategy, freq):
    """Stitches walk-forward NAV windows into a single return series."""
    windows = [str(i) for i in range(1, 30)
               if os.path.exists(dpath(f'backtest/wf_results/nav_window_{i}_{strategy}_{freq}.parquet'))]
    windows.append('holdout')
    rs = []
    for w in windows:
        f = dpath(f"backtest/wf_results/nav_window_{w}_{strategy}_{freq}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        rs.append(df['nav'].pct_change().dropna())
    if not rs:
        return pd.Series(dtype=float)
    ar = pd.concat(rs).sort_index()
    ar = ar[~ar.index.duplicated(keep='first')]
    nav = (1 + ar).cumprod() * 1_000_000
    return nav


def compute_metrics(nav_series, spy_series=None, rf=0.02):
    """Compute performance metrics from a NAV series."""
    nav = nav_series.dropna()
    if len(nav) < 10:
        return {}
    r = nav.pct_change().dropna()
    n_yr = len(nav) / 252
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / n_yr) - 1 if n_yr > 0 else 0
    vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / vol if vol > 0 else 0
    down = r[r < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf) / down if down > 0 else 0
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    out = dict(total_ret=total_ret, ann_ret=ann_ret, vol=vol, sharpe=sharpe,
               sortino=sortino, max_dd=max_dd, calmar=calmar)

    if spy_series is not None and not spy_series.empty:
        spy_r = spy_series.pct_change().dropna()
        common = r.index.intersection(spy_r.index)
        if len(common) > 10:
            active = r.loc[common] - spy_r.loc[common]
            te = active.std() * np.sqrt(252)
            cov_ = np.cov(r.loc[common], spy_r.loc[common])
            beta = cov_[0, 1] / cov_[1, 1] if cov_[1, 1] > 0 else 1
            spy_ann = (spy_series.iloc[-1] / spy_series.iloc[0]) ** (1 / n_yr) - 1
            alpha = ann_ret - beta * spy_ann
            out.update(dict(te=te, beta=beta, alpha=alpha))
    return out


def get_spy_nav(index, start_nav):
    """Load SPY benchmark aligned to strategy index."""
    try:
        spy_df = pd.read_parquet(dpath("backtest/spy_benchmark.parquet"))
        spy_df.index = pd.to_datetime(spy_df.index)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = [c[0].lower() for c in spy_df.columns]
        spy = spy_df['close'].sort_index()
        spy = spy.reindex(index, method='ffill').dropna()
        return spy / spy.iloc[0] * start_nav if not spy.empty else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def compute_ic_table(strat_key):
    """Compute IC summary table for a strategy."""
    sig_map = {
        "exp007": "signals_history_exp007.parquet",
        "exp006": "signals_history_exp006.parquet",
        "exp005": "signals_history_exp005.parquet",
    }
    fwd_map = {
        "exp007": "forward_returns_exp007.parquet",
        "exp006": "forward_returns_exp006.parquet",
        "exp005": "forward_returns_exp005.parquet",
    }
    sig_fname = sig_map.get(strat_key, "signals_history.parquet")
    fwd_fname = fwd_map.get(strat_key, "forward_returns.parquet")

    sig = pd.read_parquet(dpath(f"backtest/precomputed/{sig_fname}"))
    fwd = pd.read_parquet(dpath(f"backtest/precomputed/{fwd_fname}"))
    sig['date'] = pd.to_datetime(sig['date'])
    fwd['date'] = pd.to_datetime(fwd['date'])

    merged = sig.merge(fwd[['ticker', 'date', 'fwd_1m']], on=['ticker', 'date'], how='inner')
    is_exp = strat_key in ["exp005", "exp006", "exp007"]
    sc = SIGNAL_COLS_15 if is_exp else SIGNAL_COLS_11

    results = []
    for col in sc:
        if col not in merged.columns:
            continue
        ic_ts = merged.groupby('date').apply(
            lambda g: g[[col, 'fwd_1m']].dropna().corr(method='spearman').iloc[0, 1]
        ).dropna()
        group = next((g for g, cols in GROUPS.items() if col in cols), 'Other')
        results.append({
            'signal': col,
            'Signal': SIGNAL_LABELS.get(col, col),
            'Group': group,
            'Mean IC': float(ic_ts.mean()),
            'IC Std': float(ic_ts.std()),
            'IC IR': float(ic_ts.mean() / ic_ts.std()) if ic_ts.std() > 0 else 0,
            'Hit Rate': float((ic_ts > 0).mean()),
            # Store rolling IC as a dict for JSON-friendliness
            'ic_rolling_12m': ic_ts.rolling(12).mean().dropna().to_dict(),
        })
    return results


def load_portfolios_trades(strategy, freq):
    """Load and concatenate portfolio/trade files for a strategy."""
    port_files = sorted(glob.glob(dpath(f"backtest/wf_results/portfolios_window_*_{strategy}_{freq}.parquet")))
    trade_files = sorted(glob.glob(dpath(f"backtest/wf_results/trades_window_*_{strategy}_{freq}.parquet")))

    portfolios = pd.concat([pd.read_parquet(f) for f in port_files]) if port_files else pd.DataFrame()
    trades = pd.concat([pd.read_parquet(f) for f in trade_files]) if trade_files else pd.DataFrame()

    # Compute summary stats instead of storing raw data
    port_summary = {}
    if not portfolios.empty:
        portfolios['date'] = pd.to_datetime(portfolios['date'])
        active = portfolios[portfolios['target_weight'] > 0]
        # Position count per rebalance
        pc = active.groupby('date')['ticker'].count()
        port_summary['position_count'] = pc.to_dict()
        # Top-10 concentration
        t10 = active.groupby('date').apply(lambda g: g['target_weight'].nlargest(10).sum() * 100)
        port_summary['top10_concentration'] = t10.to_dict()
        # Most held tickers
        port_summary['most_held'] = active['ticker'].value_counts().head(25).to_dict()

    trade_summary = {}
    if not trades.empty:
        trades['date'] = pd.to_datetime(trades['date'])
        trades['value'] = trades['shares'] * trades['price']
        mt = trades.groupby(pd.Grouper(key='date', freq='ME'))['value'].sum() / 1e6
        trade_summary['monthly_turnover'] = mt.to_dict()

    return port_summary, trade_summary


def per_window_metrics(strategy, freq, spy_nav_full=None):
    """Compute metrics per walk-forward window."""
    windows = [str(i) for i in range(1, 30)
               if os.path.exists(dpath(f"backtest/wf_results/nav_window_{i}_{strategy}_{freq}.parquet"))]
    windows.append('holdout')

    rows = []
    for w in windows:
        f = dpath(f"backtest/wf_results/nav_window_{w}_{strategy}_{freq}.parquet")
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index)
            m = compute_metrics(df['nav'])
            rows.append({
                'Window': f"W{w}" if w != 'holdout' else 'Holdout',
                'Start': df.index[0].strftime('%Y-%m'),
                'End': df.index[-1].strftime('%Y-%m'),
                **m
            })
        except Exception:
            pass
    return rows


# ── MAIN ──

def main():
    print("=" * 60)
    print("  Precomputing dashboard cache")
    print("=" * 60)

    cache = {}

    # 1. Stitch NAV for all strategy combos
    print("\n[1/7] Stitching NAV series...")
    nav_cache = {}
    for (strat, freq), label in STRATEGIES.items():
        nav = stitch_nav(strat, freq)
        if not nav.empty:
            nav_cache[label] = nav.to_dict()
            print(f"  {label}: {len(nav)} days, {nav.index[0].date()} to {nav.index[-1].date()}")
        else:
            print(f"  {label}: EMPTY")
    cache['nav'] = nav_cache

    # 2. SPY benchmark
    print("\n[2/7] Loading SPY benchmark...")
    # Use the longest NAV series as reference index
    ref_key = max(nav_cache.keys(), key=lambda k: len(nav_cache[k]))
    ref_nav = pd.Series(nav_cache[ref_key])
    ref_nav.index = pd.to_datetime(ref_nav.index)
    spy_nav = get_spy_nav(ref_nav.index, 1_000_000)
    if not spy_nav.empty:
        cache['spy'] = spy_nav.to_dict()
        print(f"  SPY: {len(spy_nav)} days")
    else:
        cache['spy'] = {}
        print("  SPY: NOT FOUND")

    # 3. Metrics for all combos
    print("\n[3/7] Computing metrics...")
    metrics_cache = {}
    spy_series = pd.Series(cache['spy']) if cache['spy'] else None
    if spy_series is not None and not spy_series.empty:
        spy_series.index = pd.to_datetime(spy_series.index)

    for label, nav_dict in nav_cache.items():
        nav_s = pd.Series(nav_dict)
        nav_s.index = pd.to_datetime(nav_s.index)
        # Align SPY to this strategy's index
        if spy_series is not None and not spy_series.empty:
            spy_aligned = spy_series.reindex(nav_s.index, method='ffill').dropna()
            spy_aligned = spy_aligned / spy_aligned.iloc[0] * nav_s.iloc[0]
        else:
            spy_aligned = None
        m = compute_metrics(nav_s, spy_aligned)
        metrics_cache[label] = m
        print(f"  {label}: Sharpe={m.get('sharpe', 0):.3f} CAGR={m.get('ann_ret', 0):.1%}")
    cache['metrics'] = metrics_cache

    # 4. Per-window metrics
    print("\n[4/7] Computing per-window metrics...")
    window_cache = {}
    for (strat, freq), label in STRATEGIES.items():
        rows = per_window_metrics(strat, freq)
        if rows:
            window_cache[label] = rows
            print(f"  {label}: {len(rows)} windows")
    cache['windows'] = window_cache

    # 5. IC tables
    print("\n[5/7] Computing IC tables...")
    ic_cache = {}
    for strat_key in ['mv', 'bl', 'exp005', 'exp006', 'exp007']:
        try:
            ic_data = compute_ic_table(strat_key)
            ic_cache[strat_key] = ic_data
            print(f"  {strat_key}: {len(ic_data)} signals")
        except Exception as e:
            print(f"  {strat_key}: FAILED - {e}")
    cache['ic'] = ic_cache

    # 6. Portfolio/trade summaries
    print("\n[6/7] Computing portfolio/trade summaries...")
    port_cache = {}
    trade_cache = {}
    for (strat, freq), label in STRATEGIES.items():
        ps, ts = load_portfolios_trades(strat, freq)
        if ps:
            port_cache[label] = ps
        if ts:
            trade_cache[label] = ts
        print(f"  {label}: portfolio={'yes' if ps else 'no'}, trades={'yes' if ts else 'no'}")
    cache['portfolios'] = port_cache
    cache['trades'] = trade_cache

    # 7. Regime history + signal decay
    print("\n[7/7] Loading regime history + signal decay...")
    regime_path = dpath("backtest/regime_history.parquet")
    if os.path.exists(regime_path):
        rdf = pd.read_parquet(regime_path)
        rdf['date'] = pd.to_datetime(rdf['date'])
        cache['regime'] = rdf.to_dict('records')
        print(f"  Regime: {len(rdf)} records")
    else:
        cache['regime'] = []

    decay_path = dpath("processed/signal_decay.parquet")
    if os.path.exists(decay_path):
        ddf = pd.read_parquet(decay_path)
        ddf['date'] = pd.to_datetime(ddf['date'])
        cache['signal_decay'] = ddf.to_dict('records')
        print(f"  Signal decay: {len(ddf)} records")
    else:
        cache['signal_decay'] = []

    # ── SAVE ──
    out_path = dpath("backtest/dashboard_cache.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"  Cache saved: {out_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Keys: {list(cache.keys())}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
