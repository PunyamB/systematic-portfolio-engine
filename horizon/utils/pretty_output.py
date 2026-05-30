"""utils/pretty_output.py - Streaming pretty output for Horizon."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date

W = 72  # total box width


def _line(char="="):
    return "+" + char * (W - 2) + "+"


def _row(text=""):
    # Strip control, truncate to fit
    inner = W - 4
    if len(text) > inner:
        text = text[:inner]
    return "| " + text + " " * (inner - len(text)) + " |"


def _section_header(label):
    label = f" {label} "
    fill = "=" * (W - 4 - len(label))
    if len(label) + 4 > W:
        label = label[:W - 8]
        fill = "=="
    return "+==" + label + fill + "+"


def _hr():
    return "-" * (W - 6)


def _icon(status):
    return {
        "ok":   "OK  ",
        "skip": "SKIP",
        "fail": "FAIL",
        "warn": "WARN",
    }.get(status, "?   ")


@dataclass
class PipelineReport:
    run_date: date
    started_at: datetime
    is_weekend: bool = False
    is_holiday: bool = False
    holiday_name: str = ""
    is_cold_start: bool = False

    cb_tier: int = 0
    regime_composite: str = ""

    def print_header(self):
        day = self.run_date.strftime("%A, %Y-%m-%d")
        time_str = self.started_at.strftime("%H:%M:%S")
        ctx = ""
        if self.is_holiday:
            ctx = f" * {self.holiday_name}"
        elif self.is_weekend:
            ctx = " * weekend"
        if self.is_cold_start:
            ctx += " * cold start"
        print()
        print(_line("="))
        print(_row(f"  HORIZON  *  {day}  *  {time_str}{ctx}"))
        print(_line("="))

    def log_stage(self, name, status, detail="", seconds=0.0):
        t = f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds//60)}m{int(seconds%60):02d}s"
        d = detail[:30] if detail else ""
        line = f"  [{_icon(status)}] {name:<18} {d:<32} {t:>8}"
        print(_row(line))

    def print_broker(self, status, portfolio, cash, equity):
        print(_section_header("BROKER"))
        print(_row(f"  Status: {status:<10}  Portfolio: ${portfolio:>14,.2f}"))
        print(_row(f"  Cash:   ${cash:>14,.2f}  Equity:    ${equity:>14,.2f}"))
        print(_row(""))

    def print_reconciliation_drift(self, only_alpaca, only_ours, mismatches):
        print(_section_header("RECONCILIATION DRIFT"))
        if only_alpaca:
            print(_row(f"  Only in Alpaca       {', '.join(only_alpaca)}"))
        if only_ours:
            print(_row(f"  Only in our records  {', '.join(only_ours)}"))
        if mismatches:
            for m in mismatches:
                print(_row(f"  Share mismatch       {m}"))
        print(_row("  -> Manual review. Continued with Alpaca state."))
        print(_row(""))

    def print_corporate_actions(self, splits, dividends, cash_income):
        if not splits and not dividends:
            return
        print(_section_header("CORPORATE ACTIONS"))
        for tkr, ratio, old, new in splits:
            print(_row(f"  Split     {tkr}  {ratio}  -> shares {old} -> {new}"))
        for tkr, ps, income in dividends:
            print(_row(f"  Dividend  {tkr}  ${ps:.2f}/sh -> ${income:.2f}"))
        if cash_income > 0:
            print(_row(f"  Cash income  +${cash_income:,.2f}"))
        print(_row(""))

    def print_data_refresh(self, constituents, constituents_cached, constituents_age_days,
                           prices_new_rows, prices_backfill, prices_range,
                           financials_status, keymetrics_status, macro_status):
        print(_section_header("DATA REFRESH"))
        if constituents_cached:
            print(_row(f"  Constituents  {constituents} tickers (cached, {constituents_age_days}d old)"))
        else:
            print(_row(f"  Constituents  {constituents} S&P 500 tickers"))
        if prices_backfill:
            print(_row(f"  Prices        full backfill {prices_range}"))
            print(_row(f"                {prices_new_rows:,} rows"))
        else:
            print(_row(f"  Prices        +{prices_new_rows:,} rows ({prices_range})"))
        print(_row(f"  Financials    {financials_status}"))
        print(_row(f"  Key Metrics   {keymetrics_status}"))
        print(_row(f"  Macro (FRED)  {macro_status}"))
        print(_row(""))

    def print_nav(self, nav, daily_return, equity, cash, history_count, peak, top_pnl=None):
        print(_section_header("FUND ACCOUNTING"))
        print(_row(f"  NAV       ${nav:>14,.2f}   Daily Return  {daily_return:+.4%}"))
        print(_row(f"  Equity    ${equity:>14,.2f}   History       {history_count} records"))
        print(_row(f"  Cash      ${cash:>14,.2f}   Peak NAV      ${peak:,.2f}"))
        if top_pnl:
            print(_row(""))
            print(_row("  Top P&L Today"))
            for tkr, pnl, pct in top_pnl:
                print(_row(f"    {tkr:<6}  {pnl:+>10,.2f}  ({pct:+.2%})"))
        print(_row(""))

    def print_regime(self, composite, l1_stress, vix, vix_ratio, breadth, breadth_count,
                     l2_cycle, yield_spread, curve_inverted, credit_trend,
                     changed=False, previous=""):
        self.regime_composite = composite
        print(_section_header("REGIME"))
        change = f"  (!) changed from {previous.upper()}" if changed and previous else ""
        print(_row(f"  Composite      {composite.upper()}{change}"))
        print(_row(f"  L1 Stress      {l1_stress:<12} VIX {vix:.1f}  ratio {vix_ratio:.2f}"))
        print(_row(f"  Breadth        {breadth:.2%}    ({breadth_count[0]}/{breadth_count[1]} above 200d MA)"))
        curve = "INVERTED" if curve_inverted else "normal"
        print(_row(f"  L2 Cycle       {l2_cycle:<12} spread {yield_spread:+.2f} * curve {curve}"))
        print(_row(f"  Credit Spreads {credit_trend}"))
        if changed:
            print(_row("  -> Regime change triggers immediate rebalance"))
        print(_row(""))

    def print_risk(self, cb_tier, drawdown, peak_nav, beta, cb_actions, stop_exits,
                   drift_positions, drift_portfolio, drift_portfolio_total,
                   drift_sectors, liquidity_flags, cb_tier_changed=False, cb_tier_previous=0):
        self.cb_tier = cb_tier
        print(_section_header("RISK"))
        tier_ch = f"  (!) from T{cb_tier_previous}" if cb_tier_changed else ""
        print(_row(f"  CB Tier        T{cb_tier}{tier_ch}    Drawdown   {drawdown:.2%}"))
        print(_row(f"  Peak NAV       ${peak_nav:,.2f}   Beta  {beta:.3f}"))
        if cb_actions:
            print(_row(f"  T{cb_tier} Actions:"))
            for action in cb_actions:
                print(_row(f"    * {action}"))

        if stop_exits:
            print(_row(""))
            print(_row(f"  Stop Exits ({len(stop_exits)})"))
            print(_row("  Ticker   Close      Stop       Distance"))
            for tkr, close, stop, dist in stop_exits:
                print(_row(f"  {tkr:<8} ${close:<9,.2f} ${stop:<9,.2f} {dist:+.2%}"))
        else:
            print(_row(f"  Stop Exits     none triggered"))

        any_drift = drift_positions or drift_portfolio or drift_sectors
        if any_drift:
            if drift_positions:
                ds = [f"{t}({d:+.1%})" if d else t for t, d in drift_positions]
                print(_row(f"  Position drift {', '.join(ds)[:50]}"))
            if drift_portfolio:
                print(_row(f"  Portfolio drift {drift_portfolio_total:.1%} (threshold 5%)"))
            if drift_sectors:
                ss = [f"{s}({d:+.1%})" if d else s for s, d in drift_sectors]
                print(_row(f"  Sector drift   {', '.join(ss)[:50]}"))
        else:
            print(_row(f"  Drift          none"))

        if liquidity_flags:
            for tkr, ratio in liquidity_flags:
                print(_row(f"  Liquidity flag {tkr} ratio {ratio:.1%}"))
        else:
            print(_row(f"  Liquidity      0 flags"))
        print(_row(""))

    def print_stop_execution(self, market_open, submitted=0, filled=0, value=0.0, cooldown_tickers=None):
        print(_section_header("STOP EXECUTION"))
        if market_open:
            print(_row(f"  Market open      Yes - auto-executed"))
            print(_row(f"  Orders submitted {submitted} sells"))
            print(_row(f"  Filled           {filled}/{submitted}  (${value:,.2f})"))
            if cooldown_tickers:
                print(_row(f"  Rebuy cooldown   {', '.join(cooldown_tickers)[:40]}"))
        else:
            print(_row(f"  Market closed    Manual execution required"))
            print(_row(f"  -> python execute_stops.py at next market open"))
        print(_row(""))

    def print_signals(self, signals_table, weighting, top5, decay_alerts, decay_mean_corr):
        regime_label = f" ({self.regime_composite})" if self.regime_composite else ""
        print(_section_header(f"SIGNALS ({len(signals_table)}) * {weighting}{regime_label}"))
        print(_row(f"  {'Signal':<22} {'N':>5}  {'Mean':>10}  {'Std':>10}"))
        for name, n, mean, std in signals_table:
            print(_row(f"  {name:<22} {n:>5}  {mean:>+10.3f}  {std:>10.3f}"))
        if top5:
            print(_row(""))
            print(_row(f"  Top 5: {' * '.join(top5)[:55]}"))

        if decay_alerts:
            print(_row(""))
            print(_row(f"  (!) Decay Alerts ({len(decay_alerts)} below 0.65)"))
            for sig, corr, hl in decay_alerts:
                print(_row(f"    {sig:<22} corr {corr:.2f}  HL {hl:.0f}d"))
        else:
            print(_row(f"  Decay          no triggers  mean corr {decay_mean_corr:.2f}"))
        print(_row(""))

    def print_rebalance(self, due, reason, next_date="", prev_date="", prev_regime="", prev_cb=0):
        print(_section_header("REBALANCE"))
        print(_row(f"  Due            {due}  Reason: {reason}"))
        if prev_date:
            print(_row(f"  Previous       {prev_date} ({prev_regime}, T{prev_cb})"))
        if next_date:
            print(_row(f"  Next           {next_date}"))
        print(_row(""))

    def print_optimizer(self, params_lambda, params_ra, cb_tier, cb_invested, cb_maxwt,
                        cb_floor, no_new_pos, universe_size, universe_filtered_to_held,
                        solver, solver_status, n_positions, sum_weights, max_weight,
                        turnover_pct, turnover_binding, sectors, trades, trades_total_value):
        print(_section_header("OPTIMIZER (P3 FULL REBALANCE)"))
        print(_row(f"  Params         lambda {params_lambda} * ra {params_ra}"))
        nonew = " * NO NEW POS" if no_new_pos else ""
        print(_row(f"  CB tier        T{cb_tier} inv {cb_invested:.0%} max {cb_maxwt:.1%} floor {cb_floor:.2f}{nonew}"))
        if universe_filtered_to_held:
            print(_row(f"  Universe       {universe_size} held positions (filtered)"))
        else:
            print(_row(f"  Universe       {universe_size} tickers"))
        print(_row(f"  Solver         {solver} - {solver_status}"))

        if solver_status == "infeasible":
            print(_row(f"  Action         Held current portfolio"))
            print(_row(""))
            return

        print(_row(f"  Result         {n_positions} positions * sum {sum_weights:.2%} * max {max_weight:.2%}"))
        binding = "YES" if turnover_binding else "no"
        print(_row(f"  Turnover       {turnover_pct:.1%} (cap 30.0%, binding: {binding})"))

        if sectors:
            sector_strs = " * ".join(f"{s} {w:.0%}" for s, w in sectors[:5])
            print(_row(f"  Sectors        {sector_strs[:55]}"))

        if trades:
            print(_row(""))
            print(_row(f"  Trades ({len(trades)}) * turnover ${trades_total_value:,.0f}"))
            print(_row(f"  {'Ticker':<8} {'Dir':<5} {'Cur':>7} {'Tgt':>7} {'Delta':>8} {'Value':>11}"))
            for tkr, direction, cur, tgt, delta, val in trades:
                print(_row(f"  {tkr:<8} {direction:<5} {cur:>6.2%} {tgt:>6.2%} {delta:>+7.2%} {val:>+11,.0f}"))
        print(_row(""))

    def print_replacement(self, excess_cash, universe, cb_tier, buys, trims, value, trades):
        print(_section_header("CASH DEPLOYMENT"))
        print(_row(f"  Trigger        Excess cash ${excess_cash:,.0f} > $100k"))
        print(_row(f"  Universe       {universe} eligible tickers"))
        print(_row(f"  CB Tier        T{cb_tier}"))
        print(_row(f"  Trades         {buys} buys * {trims} trims * ${value:,.0f}"))
        if trades:
            print(_row(""))
            print(_row(f"  {'Ticker':<8} {'Dir':<5} {'Cur':>7} {'Tgt':>7} {'Value':>13}"))
            for tkr, direction, cur, tgt, delta, val in trades:
                print(_row(f"  {tkr:<8} {direction:<5} {cur:>6.2%} {tgt:>6.2%} {val:>+13,.0f}"))
            print(_row(f"  {'':>30}{'-'*10}"))
            print(_row(f"  {'TOTAL':>32}{value:>+13,.0f}"))
        print(_row(""))

    def print_actions(self, actions):
        print(_section_header("ACTION REQUIRED"))
        if not actions:
            print(_row(f"  None - no trades proposed"))
        else:
            for action in actions:
                print(_row(f"  -> {action}"))
        print(_row(""))

    def print_halt(self, reason):
        print(_section_header("HALT"))
        print(_row(f"  Reason  {reason[:55]}"))
        print(_row(""))

    def print_fatal(self, error, tb=""):
        print(_section_header("FATAL"))
        print(_row(f"  Error  {error[:55]}"))
        if tb:
            for line in tb.splitlines()[-4:]:
                print(_row(f"  {line[:W-6]}"))
        print(_row(f"  -> Check logs/pipeline_health.json"))
        print(_row(""))

    def print_footer(self, status="success"):
        completed = datetime.now()
        total_sec = (completed - self.started_at).total_seconds()
        mins, secs = divmod(int(total_sec), 60)
        time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        label = {
            "success": "OK  Pipeline complete",
            "halted":  "!   Pipeline halted",
            "fatal":   "X   Pipeline FATAL",
        }.get(status, "Pipeline done")
        print(_line("="))
        print(_row(f"  {label} * {time_str} * health saved"))
        print(_line("="))
        print()


US_HOLIDAYS_2026 = {
    date(2026, 1, 1):  "New Year's Day",
    date(2026, 1, 19): "MLK Day",
    date(2026, 2, 16): "Presidents Day",
    date(2026, 4, 3):  "Good Friday",
    date(2026, 5, 25): "Memorial Day",
    date(2026, 6, 19): "Juneteenth",
    date(2026, 7, 3):  "Independence Day",
    date(2026, 9, 7):  "Labor Day",
    date(2026, 11, 26):"Thanksgiving",
    date(2026, 12, 25):"Christmas",
}


def is_market_holiday(d):
    return (d in US_HOLIDAYS_2026, US_HOLIDAYS_2026.get(d, ""))
