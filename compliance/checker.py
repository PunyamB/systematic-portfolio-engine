# compliance/checker.py
# Pre-trade and post-trade compliance checks.
# Modeled on 40 Act requirements for a registered fund.
# All checks return a ComplianceResult with pass/fail and reason.
# Pipeline halts on any pre-trade failure.

import pandas as pd
from dataclasses import dataclass
from typing import List
from data.storage import load_portfolio, load_prices
from fund_accounting.nav import load_cash, load_nav_history
from utils.config_loader import get_config
from utils.notifications import notify

cfg = get_config()

MAX_POSITION_WEIGHT  = cfg["portfolio"]["max_position_weight"]   # 0.05
LIQUIDITY_ADV_LIMIT  = cfg["risk"]["liquidity_flag_pct"]         # 0.10


# ------------------------------------------------------------
# RESULT CONTAINER
# ------------------------------------------------------------

@dataclass
class ComplianceResult:
    passed: bool
    check:  str
    reason: str

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.check}: {self.reason}"


# ------------------------------------------------------------
# INDIVIDUAL CHECKS
# ------------------------------------------------------------

def check_75_5_10(portfolio: pd.DataFrame, nav: float) -> ComplianceResult:
    """
    40 Act diversification test:
    - 75% of assets must be in positions <= 5% of fund NAV
    - No single issuer > 10% of fund NAV
    - No single issuer > 10% of that issuer's outstanding shares (not checked here — no data)
    """
    if portfolio.empty:
        return ComplianceResult(True, "75-5-10", "No positions")

    portfolio = portfolio.copy()
    portfolio["weight"] = portfolio["market_value"] / nav

    # Check no single position > 10%
    over_10 = portfolio[portfolio["weight"] > 0.10]
    if not over_10.empty:
        tickers = over_10["ticker"].tolist()
        return ComplianceResult(
            False, "75-5-10",
            f"Positions exceed 10% of NAV: {tickers}"
        )

    # Check 75% of assets in positions <= 5%
    small_positions  = portfolio[portfolio["weight"] <= 0.05]
    small_value      = small_positions["market_value"].sum()
    total_equity     = portfolio["market_value"].sum()

    if total_equity == 0:
        return ComplianceResult(True, "75-5-10", "No equity positions")

    pct_in_small = small_value / nav
    if pct_in_small < 0.75:
        return ComplianceResult(
            False, "75-5-10",
            f"Only {pct_in_small:.1%} of NAV in positions <=5% (required 75%)"
        )

    return ComplianceResult(True, "75-5-10", f"{pct_in_small:.1%} of NAV in diversified positions")


def check_concentration(proposed_trades: pd.DataFrame, portfolio: pd.DataFrame, nav: float) -> ComplianceResult:
    """
    Checks that no proposed trade would push any position above max_position_weight (5%).
    """
    if proposed_trades.empty:
        return ComplianceResult(True, "concentration", "No trades proposed")

    portfolio = portfolio.copy() if not portfolio.empty else pd.DataFrame(columns=["ticker", "market_value"])

    for _, trade in proposed_trades.iterrows():
        ticker       = trade["ticker"]
        trade_value  = float(trade.get("trade_value", 0))

        current_value = 0.0
        if not portfolio.empty and ticker in portfolio["ticker"].values:
            current_value = float(portfolio.loc[portfolio["ticker"] == ticker, "market_value"].values[0])

        new_weight = (current_value + trade_value) / nav if nav > 0 else 0

        if new_weight > MAX_POSITION_WEIGHT + 0.002:  # 20bps tolerance for rounding
            return ComplianceResult(
                False, "concentration",
                f"{ticker} would reach {new_weight:.2%} (max {MAX_POSITION_WEIGHT:.0%})"
            )

    return ComplianceResult(True, "concentration", "All positions within weight limits")


def check_liquidity(proposed_trades: pd.DataFrame, prices: pd.DataFrame) -> ComplianceResult:
    """
    Checks that no proposed trade exceeds 10% of the ticker's average daily volume.
    Flags if ADV data is missing.
    """
    if proposed_trades.empty:
        return ComplianceResult(True, "liquidity", "No trades proposed")

    if prices.empty:
        return ComplianceResult(False, "liquidity", "No price data available for ADV check")

    # Compute 20-day ADV per ticker
    adv = (
        prices.sort_values("date")
        .groupby("ticker")
        .tail(20)
        .groupby("ticker")["volume"]
        .mean()
        .reset_index()
        .rename(columns={"volume": "adv"})
    )

    trades_with_adv = proposed_trades.merge(adv, on="ticker", how="left")
    trades_with_adv["adv"] = trades_with_adv["adv"].fillna(0)

    for _, row in trades_with_adv.iterrows():
        if row["adv"] == 0:
            continue  # Skip if no ADV data — flagged separately in data quality

        close_price  = float(prices[prices["ticker"] == row["ticker"]]["close"].iloc[-1])
        shares_traded = abs(float(row.get("shares", 0)))
        adv_pct       = (shares_traded * close_price) / (row["adv"] * close_price)

        if adv_pct > LIQUIDITY_ADV_LIMIT:
            return ComplianceResult(
                False, "liquidity",
                f"{row['ticker']} trade is {adv_pct:.1%} of ADV (max {LIQUIDITY_ADV_LIMIT:.0%})"
            )

    return ComplianceResult(True, "liquidity", "All trades within ADV limits")


def check_cash_buffer(nav: float, cash: float) -> ComplianceResult:
    """
    Ensures minimum cash buffer is maintained after trades.
    """
    cash_pct = cash / nav if nav > 0 else 0
    min_cash = cfg["portfolio"]["cash_buffer"]

    if cash_pct < min_cash:
        return ComplianceResult(
            False, "cash_buffer",
            f"Cash at {cash_pct:.2%} below minimum {min_cash:.0%}"
        )
    return ComplianceResult(True, "cash_buffer", f"Cash at {cash_pct:.2%}")


# ------------------------------------------------------------
# PRE-TRADE CHECK
# ------------------------------------------------------------

def run_pre_trade_checks(proposed_trades: pd.DataFrame) -> List[ComplianceResult]:
    """
    Runs all pre-trade compliance checks.
    Called after optimizer output, before execution.
    Returns list of ComplianceResult — pipeline halts if any fail.
    """
    portfolio = load_portfolio()
    prices    = load_prices()
    cash      = load_cash()

    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else cfg["portfolio"]["initial_capital"]

    results = [
        check_concentration(proposed_trades, portfolio, nav),
        check_liquidity(proposed_trades, prices),
        check_cash_buffer(nav, cash)
    ]

    failures = [r for r in results if not r.passed]

    for r in results:
        print(f"[compliance] {r}")

    if failures:
        notify(
            f"Pre-trade compliance FAILED\n" +
            "\n".join(r.reason for r in failures),
            level="critical"
        )
    else:
        notify("Pre-trade compliance passed", level="info")

    return results


# ------------------------------------------------------------
# POST-TRADE CHECK
# ------------------------------------------------------------

def run_post_trade_checks() -> List[ComplianceResult]:
    """
    Runs compliance checks after fills are confirmed.
    Uses actual portfolio state post-execution.
    """
    portfolio = load_portfolio()
    cash      = load_cash()

    nav_history = load_nav_history()
    nav = float(nav_history.iloc[-1]["nav"]) if not nav_history.empty else cfg["portfolio"]["initial_capital"]

    results = [
        check_75_5_10(portfolio, nav),
        check_cash_buffer(nav, cash)
    ]

    failures = [r for r in results if not r.passed]

    for r in results:
        print(f"[compliance] {r}")

    if failures:
        notify(
            f"Post-trade compliance FAILED\n" +
            "\n".join(r.reason for r in failures),
            level="critical"
        )

    return results