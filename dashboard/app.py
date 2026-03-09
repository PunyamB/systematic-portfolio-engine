# dashboard/app.py
# Streamlit operational dashboard — STRICTLY READ-ONLY.
# 7 pages covering full portfolio operations.
# All data read from Parquet logs and storage files.
# No parameter changes or system modifications.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

from data.storage import (
    load_signals, load_portfolio, load_constituents,
    load_decision_log, load_portfolio_history,
    load_pipeline_history, load_execution_log,
)
from fund_accounting.nav import load_nav_history, load_cash
from utils.config_loader import get_config

cfg = get_config()

HEALTH_FILE  = Path("logs/pipeline_health.json")
PROPOSED_DIR = Path("data/proposed")

st.set_page_config(
    page_title="QuantForge | Operations",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SIDEBAR NAV ───────────────────────────────────────────────
st.sidebar.markdown("## ▲ QuantForge")
st.sidebar.markdown("**Operational Dashboard**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Portfolio",
    "History",
    "Signals",
    "Risk",
    "Execution Log",
    "Pipeline Health",
])


# ── HELPER ────────────────────────────────────────────────────

def empty_state(message: str):
    st.info(f"📭 {message}")


def fmt_pct(val, decimals=2):
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.{decimals}%}"


def fmt_usd(val):
    if val is None or pd.isna(val):
        return "—"
    return f"${val:,.2f}"


# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Portfolio Overview")

    nav_history = load_nav_history()
    decision_log = load_decision_log()

    if nav_history.empty:
        empty_state("No NAV history yet — run the pipeline to start tracking.")
    else:
        nav_history["date"] = pd.to_datetime(nav_history["date"])
        latest = nav_history.iloc[-1]

        # KPI row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("NAV", fmt_usd(latest["nav"]))
        col2.metric("Daily Return", fmt_pct(latest.get("daily_return", 0)))
        col3.metric("Cash", fmt_usd(latest.get("cash", 0)))
        col4.metric("Equity", fmt_usd(latest.get("equity_value", 0)))

        # Regime and CB from latest decision log
        if not decision_log.empty:
            last_decision = decision_log.iloc[-1]
            col5.metric("Regime", str(last_decision.get("regime", "—")).upper())
            cb = last_decision.get("cb_tier", 0)
            col6.metric("CB Tier", f"T{int(cb)}" if cb and cb > 0 else "None")
        else:
            col5.metric("Regime", "—")
            col6.metric("CB Tier", "—")

        # NAV chart
        st.subheader("NAV Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nav_history["date"], y=nav_history["nav"],
            mode="lines", name="NAV",
            line=dict(color="#2196F3", width=2)
        ))
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="", yaxis_title="NAV ($)",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Next rebalance
        from utils.rebalance_calendar import get_next_rebalance_date
        next_reb = get_next_rebalance_date(date.today())
        st.caption(f"Rebalance frequency: {cfg['rebalance']['frequency']} | Next rebalance: {next_reb}")


# ══════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO
# ══════════════════════════════════════════════════════════════

elif page == "Portfolio":
    st.title("Current Portfolio")

    portfolio = load_portfolio()
    nav_history = load_nav_history()

    if portfolio.empty:
        empty_state("No positions — portfolio is empty (pre-execution or all cash).")
    else:
        nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else cfg["portfolio"]["initial_capital"]
        portfolio["weight"] = portfolio["market_value"] / nav if nav > 0 else 0

        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Positions", len(portfolio))
        col2.metric("Max Weight", fmt_pct(portfolio["weight"].max()))
        col3.metric("Invested %", fmt_pct(portfolio["market_value"].sum() / nav if nav > 0 else 0))

        # Positions table
        st.subheader("Positions")
        display_cols = ["ticker", "shares", "market_value", "weight", "cost_basis",
                        "unrealized_pnl", "entry_date", "stop_price"]
        display_cols = [c for c in display_cols if c in portfolio.columns]
        st.dataframe(
            portfolio[display_cols].sort_values("weight", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "weight": st.column_config.NumberColumn(format="%.2%%"),
                "market_value": st.column_config.NumberColumn(format="$%.2f"),
                "cost_basis": st.column_config.NumberColumn(format="$%.2f"),
                "unrealized_pnl": st.column_config.NumberColumn(format="$%.2f"),
                "stop_price": st.column_config.NumberColumn(format="$%.2f"),
            }
        )

        # Sector breakdown
        constituents = load_constituents()
        if not constituents.empty:
            merged = portfolio.merge(
                constituents[["ticker", "sector"]], on="ticker", how="left"
            )
            sector_wts = merged.groupby("sector")["weight"].sum().sort_values(ascending=True)

            st.subheader("Sector Allocation")
            fig = go.Figure(go.Bar(
                x=sector_wts.values, y=sector_wts.index,
                orientation="h",
                marker_color="#2196F3"
            ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Weight", template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════

elif page == "History":
    st.title("Decision History")

    decision_log = load_decision_log()

    if decision_log.empty:
        empty_state("No decision history yet — run the pipeline to start tracking.")
    else:
        decision_log["date"] = pd.to_datetime(decision_log["date"])

        # Date picker
        available_dates = sorted(decision_log["date"].dt.date.unique(), reverse=True)
        selected_date = st.selectbox("Select date", available_dates)

        day_row = decision_log[decision_log["date"].dt.date == selected_date].iloc[-1]

        # Day summary
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("NAV", fmt_usd(day_row.get("nav")))
        col2.metric("Return", fmt_pct(day_row.get("daily_return")))
        col3.metric("Regime", str(day_row.get("regime", "")).upper())
        col4.metric("CB Tier", f"T{int(day_row.get('cb_tier', 0))}" if day_row.get("cb_tier", 0) else "None")
        col5.metric("Status", str(day_row.get("status", "")).upper())

        # Full decision record
        st.subheader("Full Decision Record")
        st.json({k: (str(v) if pd.notna(v) else None) for k, v in day_row.to_dict().items()})

        # Portfolio on that date
        st.subheader("Portfolio Snapshot")
        portfolio_history = load_portfolio_history()
        if not portfolio_history.empty:
            portfolio_history["date"] = pd.to_datetime(portfolio_history["date"])
            day_portfolio = portfolio_history[portfolio_history["date"].dt.date == selected_date]
            if not day_portfolio.empty and not (len(day_portfolio) == 1 and day_portfolio.iloc[0].get("ticker") == "CASH_ONLY"):
                display = day_portfolio[day_portfolio["ticker"] != "CASH_ONLY"]
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.caption("No positions held on this date.")
        else:
            st.caption("No portfolio history available.")

        # Drawdown chart
        st.subheader("Drawdown Over Time")
        nav_history = load_nav_history()
        if not nav_history.empty:
            nav_history["date"] = pd.to_datetime(nav_history["date"])
            nav_history = nav_history.sort_values("date")
            nav_history["peak"] = nav_history["nav"].cummax()
            nav_history["drawdown"] = (nav_history["nav"] - nav_history["peak"]) / nav_history["peak"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=nav_history["date"], y=nav_history["drawdown"],
                fill="tozeroy", name="Drawdown",
                line=dict(color="#F44336", width=1),
                fillcolor="rgba(244,67,54,0.3)"
            ))
            fig.update_layout(
                height=250, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_tickformat=".1%", template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: SIGNALS
# ══════════════════════════════════════════════════════════════

elif page == "Signals":
    st.title("Signal Scores")

    signals = load_signals()

    if signals.empty:
        empty_state("No signal data — run the pipeline to compute signals.")
    else:
        # Top composite scores
        st.subheader("Top 20 by Composite Score")
        top = signals.nlargest(20, "composite_score")[["ticker", "composite_score", "composite_rank"]]
        st.dataframe(top, use_container_width=True, hide_index=True)

        # Signal heatmap for top 20
        signal_cols = [c for c in signals.columns if c not in [
            "ticker", "date", "composite_score", "composite_rank"
        ]]
        if signal_cols:
            st.subheader("Individual Signal Scores (Top 20)")
            top_tickers = signals.nlargest(20, "composite_score")["ticker"].tolist()
            heatmap_data = signals[signals["ticker"].isin(top_tickers)].set_index("ticker")[signal_cols]

            fig = px.imshow(
                heatmap_data, aspect="auto",
                color_continuous_scale="RdYlGn",
                labels=dict(color="Z-Score")
            )
            fig.update_layout(height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: RISK
# ══════════════════════════════════════════════════════════════

elif page == "Risk":
    st.title("Risk Monitor")

    decision_log = load_decision_log()

    if decision_log.empty:
        empty_state("No risk data — run the pipeline to start monitoring.")
    else:
        latest = decision_log.iloc[-1]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Beta", f"{latest.get('beta', 1.0):.3f}")
        col2.metric("Tracking Error", fmt_pct(latest.get("tracking_error", 0)))
        col3.metric("Drawdown", fmt_pct(latest.get("drawdown", 0)))
        col4.metric("CB Tier", f"T{int(latest.get('cb_tier', 0))}" if latest.get("cb_tier") else "None")
        col5.metric("VIX", f"{latest.get('vix', '—')}")

        # Risk metrics over time
        if len(decision_log) > 1:
            decision_log["date"] = pd.to_datetime(decision_log["date"])

            st.subheader("Tracking Error Over Time")
            if "tracking_error" in decision_log.columns:
                te_data = decision_log[["date", "tracking_error"]].dropna()
                if not te_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=te_data["date"], y=te_data["tracking_error"],
                        mode="lines", name="TE",
                        line=dict(color="#FF9800", width=2)
                    ))
                    fig.add_hline(
                        y=cfg["optimizer"]["tracking_error_cap"],
                        line_dash="dash", line_color="red",
                        annotation_text="TE Cap"
                    )
                    fig.update_layout(
                        height=250, margin=dict(l=0, r=0, t=30, b=0),
                        yaxis_tickformat=".1%", template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Beta Over Time")
            if "beta" in decision_log.columns:
                beta_data = decision_log[["date", "beta"]].dropna()
                if not beta_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=beta_data["date"], y=beta_data["beta"],
                        mode="lines", name="Beta",
                        line=dict(color="#4CAF50", width=2)
                    ))
                    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        height=250, margin=dict(l=0, r=0, t=30, b=0),
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Stop exits
        stop_exits = latest.get("stop_exits", "")
        if stop_exits:
            st.subheader("Recent Stop Exits")
            st.warning(f"Trailing stops triggered: {stop_exits}")


# ══════════════════════════════════════════════════════════════
# PAGE: EXECUTION LOG
# ══════════════════════════════════════════════════════════════

elif page == "Execution Log":
    st.title("Execution Log")

    exec_log = load_execution_log()

    if exec_log.empty:
        empty_state("No execution history — approve and execute trades to start logging.")
    else:
        st.subheader(f"Total Records: {len(exec_log)}")

        # Summary stats
        if "filled" in exec_log.columns:
            filled = exec_log[exec_log["filled"] == True]
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Fills", len(filled))
            col2.metric("Failed", len(exec_log) - len(filled))
            if "slippage_bps" in filled.columns:
                avg_slip = filled["slippage_bps"].mean()
                col3.metric("Avg Slippage", f"{avg_slip:.1f} bps")

        # Full log table
        display_cols = [c for c in exec_log.columns if c not in ["order_id"]]
        st.dataframe(
            exec_log[display_cols].sort_values("execution_date", ascending=False)
            if "execution_date" in exec_log.columns else exec_log,
            use_container_width=True, hide_index=True
        )


# ══════════════════════════════════════════════════════════════
# PAGE: PIPELINE HEALTH
# ══════════════════════════════════════════════════════════════

elif page == "Pipeline Health":
    st.title("Pipeline Health")

    # Current run status from JSON
    if HEALTH_FILE.exists():
        with open(HEALTH_FILE) as f:
            health = json.load(f)

        st.subheader(f"Last Run: {health.get('run_date', '—')}")

        col1, col2 = st.columns(2)
        col1.metric("Total Time", f"{health.get('total_sec', 0):.1f}s")
        failed = [s for s, v in health.get("stages", {}).items() if v["status"] == "failed"]
        col2.metric("Status", "FAILED" if failed else "SUCCESS")

        # Stage breakdown
        st.subheader("Stage Breakdown")
        stages = health.get("stages", {})
        stage_data = []
        for name, info in stages.items():
            stage_data.append({
                "Stage": name,
                "Status": info["status"].upper(),
                "Duration (s)": info["duration_sec"],
                "Detail": info.get("detail", ""),
            })

        stage_df = pd.DataFrame(stage_data)
        st.dataframe(stage_df, use_container_width=True, hide_index=True)

        # Timing bar chart
        fig = go.Figure(go.Bar(
            x=[s["Duration (s)"] for s in stage_data],
            y=[s["Stage"] for s in stage_data],
            orientation="h",
            marker_color=[
                "#4CAF50" if s["Status"] == "SUCCESS"
                else ("#FF9800" if s["Status"] == "SKIPPED" else "#F44336")
                for s in stage_data
            ]
        ))
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Seconds", template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty_state("No pipeline health data — run the pipeline first.")

    # History
    pipeline_history = load_pipeline_history()
    if not pipeline_history.empty:
        st.subheader("Run History")
        display_cols = ["run_date", "status", "total_sec", "run_at"]
        display_cols = [c for c in display_cols if c in pipeline_history.columns]
        st.dataframe(
            pipeline_history[display_cols].sort_values("run_date", ascending=False),
            use_container_width=True, hide_index=True
        )
