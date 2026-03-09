# dashboard/app.py
# Streamlit read-only dashboard.
# Displays NAV, regime, pipeline health, portfolio positions,
# signal scores, risk metrics, and proposed trades.
# STRICTLY READ-ONLY — no parameter changes or system modifications.
# All settings changes via VS Code editing config/settings.yaml.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

from data.storage import load_signals, load_portfolio, load_constituents
from fund_accounting.nav import load_nav_history, load_cash
from utils.config_loader import get_config

cfg = get_config()

HEALTH_FILE   = Path("data/pipeline_health.json")
PROPOSED_DIR  = Path("data/proposed")
APPROVED_DIR  = Path("data/approved")

st.set_page_config(
    page_title="SirAlgot Portfolio Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def load_health() -> dict:
    if not HEALTH_FILE.exists():
        return {}
    with open(HEALTH_FILE) as f:
        return json.load(f)


def load_proposed_trades() -> pd.DataFrame:
    files = sorted(PROPOSED_DIR.glob("proposed_trades_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[0])


def load_approved_trades() -> pd.DataFrame:
    files = sorted(APPROVED_DIR.glob("proposed_trades_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[0])


def color_status(status: str) -> str:
    return {
        "success": "green",
        "failed":  "red",
        "skipped": "gray",
    }.get(status, "gray")


def fmt_pct(val) -> str:
    try:
        return f"{float(val):.2%}"
    except:
        return "—"


def fmt_usd(val) -> str:
    try:
        return f"${float(val):,.0f}"
    except:
        return "—"


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("SirAlgot Portfolio Engine")
st.caption("Read-only dashboard. All settings changes via config/settings.yaml in VS Code.")

health      = load_health()
nav_history = load_nav_history()
cash        = load_cash()

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)

if not nav_history.empty:
    latest_nav    = nav_history.iloc[-1]
    nav_val       = latest_nav["nav"]
    daily_ret     = latest_nav["daily_return"]
    equity_val    = latest_nav["equity_value"]
    run_date      = latest_nav["date"]

    col1.metric("NAV",          fmt_usd(nav_val))
    col2.metric("Daily Return", fmt_pct(daily_ret))
    col3.metric("Equity",       fmt_usd(equity_val))
    col4.metric("Cash",         fmt_usd(cash))
    col5.metric("As Of",        str(run_date))
else:
    col1.metric("NAV",    "—")
    col2.metric("Return", "—")
    col3.metric("Equity", "—")
    col4.metric("Cash",   fmt_usd(cash))
    col5.metric("As Of",  "—")

st.divider()

# ------------------------------------------------------------
# REGIME + PIPELINE HEALTH
# ------------------------------------------------------------

left, right = st.columns([1, 2])

with left:
    st.subheader("Regime")
    if health:
        regime_stage = health.get("stages", {}).get("regime", {})
        regime_detail = regime_stage.get("detail", "")

        # Pull regime from pipeline health run_date context
        run_date_str = health.get("run_date", "—")
        st.metric("Last Run", run_date_str)
        st.metric("Total Runtime", f"{health.get('total_sec', 0):.0f}s")

    # Load regime from signals if available
    signals = load_signals()
    if not signals.empty:
        regime_cfg = cfg.get("regime", {})
        st.write("")

    # Show from health file if available
    if health:
        stages = health.get("stages", {})
        for stage_name in ["regime"]:
            s = stages.get(stage_name, {})
            status = s.get("status", "—")
            detail = s.get("detail", "")
            color  = color_status(status)
            st.markdown(f":{color}[**{stage_name}** — {status}] {detail}")

with right:
    st.subheader("Pipeline Health")
    if health:
        stages = health.get("stages", {})
        rows   = []
        for name, info in stages.items():
            rows.append({
                "Stage":    name,
                "Status":   info.get("status", "—"),
                "Time (s)": info.get("duration_sec", 0),
                "Detail":   info.get("detail", ""),
            })
        if rows:
            df_health = pd.DataFrame(rows)
            st.dataframe(
                df_health,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn(width="small"),
                    "Time (s)": st.column_config.NumberColumn(format="%.1f", width="small"),
                }
            )
    else:
        st.info("No pipeline health data. Run pipeline first.")

st.divider()

# ------------------------------------------------------------
# PORTFOLIO POSITIONS
# ------------------------------------------------------------

st.subheader("Current Portfolio")

portfolio = load_portfolio()

if not portfolio.empty:
    constituents = load_constituents()
    portfolio    = portfolio.merge(
        constituents[["ticker", "sector"]],
        on="ticker", how="left"
    )

    col_a, col_b = st.columns([2, 1])

    with col_a:
        display_cols = [c for c in ["ticker", "shares", "market_value", "weight", "sector", "unrealized_pnl", "entry_price"] if c in portfolio.columns]
        st.dataframe(
            portfolio[display_cols].sort_values("weight", ascending=False) if "weight" in portfolio.columns
            else portfolio[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    with col_b:
        st.write("**Sector Allocation**")
        if "sector" in portfolio.columns and "market_value" in portfolio.columns:
            sector_alloc = (
                portfolio.groupby("sector")["market_value"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            sector_alloc.columns = ["Sector", "Market Value"]
            sector_alloc["Weight"] = sector_alloc["Market Value"] / sector_alloc["Market Value"].sum()
            sector_alloc["Weight"] = sector_alloc["Weight"].apply(fmt_pct)
            sector_alloc["Market Value"] = sector_alloc["Market Value"].apply(fmt_usd)
            st.dataframe(sector_alloc, use_container_width=True, hide_index=True)
else:
    st.info("No portfolio positions. Approve and execute proposed trades first.")

st.divider()

# ------------------------------------------------------------
# SIGNAL SCORES
# ------------------------------------------------------------

st.subheader("Signal Scores")

if not signals.empty:
    tab1, tab2 = st.tabs(["Top 20 Tickers", "All Signals"])

    with tab1:
        top20 = signals.nlargest(20, "composite_rank")[
            ["ticker", "composite_score", "composite_rank",
             "momentum_12_1", "piotroski", "roe_stability",
             "earnings_accruals", "rsi_extremes"]
        ].reset_index(drop=True)
        st.dataframe(top20, use_container_width=True, hide_index=True)

    with tab2:
        signal_cols = ["ticker", "composite_score", "composite_rank",
                       "momentum_12_1", "earnings_momentum", "pe_zscore",
                       "pb_zscore", "ev_ebitda_zscore", "roe_stability",
                       "gross_margin_trend", "piotroski", "earnings_accruals",
                       "short_term_reversal", "rsi_extremes"]
        available = [c for c in signal_cols if c in signals.columns]
        st.dataframe(
            signals[available].sort_values("composite_rank", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("No signal data. Run pipeline first.")

st.divider()

# ------------------------------------------------------------
# PROPOSED TRADES
# ------------------------------------------------------------

st.subheader("Proposed Trades")

proposed = load_proposed_trades()
approved = load_approved_trades()

if not proposed.empty:
    approved_file = None
    if not approved.empty:
        files = sorted(APPROVED_DIR.glob("proposed_trades_*.csv"), reverse=True)
        approved_file = files[0].name if files else None

    prop_files = sorted(PROPOSED_DIR.glob("proposed_trades_*.csv"), reverse=True)
    prop_file  = prop_files[0].name if prop_files else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Proposed File",  prop_file)
    c2.metric("Total Trades",   len(proposed))
    c3.metric("Approved",       "Yes" if approved_file else "No")

    col_order = [c for c in ["ticker", "direction", "current_weight", "target_weight",
                              "delta_weight", "trade_value_usd", "sector"] if c in proposed.columns]
    st.dataframe(proposed[col_order], use_container_width=True, hide_index=True)

    if approved_file:
        st.success(f"Approved: {approved_file} — ready for execution")
    else:
        st.warning(
            f"Not yet approved. Copy {prop_file} from data/proposed/ to data/approved/ to approve."
        )
else:
    st.info("No proposed trades. Run pipeline on a rebalance day first.")

st.divider()

# ------------------------------------------------------------
# NAV HISTORY
# ------------------------------------------------------------

st.subheader("NAV History")

if not nav_history.empty and len(nav_history) > 1:
    st.line_chart(
        nav_history.set_index("date")["nav"],
        use_container_width=True,
    )
else:
    st.info("Insufficient NAV history for chart. Needs at least 2 data points.")

st.divider()

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.caption(
    "SirAlgot Portfolio Engine — Paper trading on Alpaca | "
    "Universe: S&P 500 | Benchmark: SPY | Capital: $1,000,000"
)