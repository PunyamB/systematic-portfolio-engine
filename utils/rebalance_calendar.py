# utils/rebalance_calendar.py
# Regime + circuit-breaker conditional rebalance scheduling.
#
# Trading-day intervals per regime (configured in settings.yaml):
#   - bull / recovery: 21 trading days  (~monthly)
#   - bear:             7 trading days  (~biweekly)
#   - crisis:           4 trading days  (~weekly)
#
# CB tier intervals tighten the schedule when drawdown breaches thresholds:
#   - T1: 21 trading days  (no effective override)
#   - T2:  5 trading days  (~weekly)
#   - T3:  1 trading day   (daily)
#   - T4:  1 trading day   (daily; trade pause handled separately)
#
# Effective interval = min(regime_interval, cb_interval).
# Regime change OR CB tier change forces immediate rebalance and resets clock.
# State persisted in data/processed/rebalance_state.json:
#   { "last_rebalance_date": "YYYY-MM-DD",
#     "last_regime":         "<regime>",
#     "last_cb_tier":        <int> }
#
# Calendar functions are READ-ONLY for state. State is updated only by
# execute.py via mark_rebalance_complete() after a successful execution.

import json
from datetime import date
from pathlib import Path

import pandas as pd

from utils.config_loader import get_config

STATE_FILE = Path("data/processed/rebalance_state.json")


# ------------------------------------------------------------
# STATE I/O
# ------------------------------------------------------------

def _load_state() -> dict | None:
    """Returns state dict or None if file missing/corrupt."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        if "last_rebalance_date" not in data or "last_regime" not in data:
            return None
        # Backfill last_cb_tier=0 for old state files (pre-A2)
        if "last_cb_tier" not in data:
            data["last_cb_tier"] = 0
        return data
    except Exception:
        return None


def _save_state(last_rebalance_date: date, last_regime: str,
                last_cb_tier: int = 0) -> None:
    """Writes state file. Called by mark_rebalance_complete()."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_rebalance_date": str(last_rebalance_date),
        "last_regime":         str(last_regime).lower(),
        "last_cb_tier":        int(last_cb_tier),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def _trading_days_between(start_date: date, end_date: date) -> int:
    """
    Returns count of trading days strictly after start_date through end_date,
    using actual trading dates from prices.parquet.
    """
    from data.storage import load_prices
    prices = load_prices()
    if prices.empty:
        return 0
    trading_dates = sorted(pd.to_datetime(prices["date"]).unique())
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
    return len([d for d in trading_dates if start_ts < d <= end_ts])


def _interval_for_regime(regime: str) -> int:
    """Returns trading-day interval for the given regime."""
    cfg = get_config()
    intervals = cfg["rebalance"]["regime_trading_days"]
    return int(intervals.get(str(regime).lower(), intervals.get("recovery", 21)))


def _interval_for_cb_tier(cb_tier: int) -> int:
    """
    Returns trading-day interval for the given CB tier.
    Tier 0 (no breach) returns a large value so min() defers to regime.
    """
    if cb_tier <= 0:
        return 999  # effectively no override
    cfg = get_config()
    intervals = cfg.get("circuit_breaker", {}).get("cb_trading_days", {})
    # YAML keys may be int or str depending on parser; check both
    val = intervals.get(cb_tier, intervals.get(str(cb_tier), 21))
    return int(val)


def _effective_interval(regime: str, cb_tier: int) -> int:
    """
    Returns the tighter of regime interval and CB tier interval.
    Whichever fires more often wins -- safer regime always dominates.
    """
    return min(_interval_for_regime(regime), _interval_for_cb_tier(cb_tier))


# ------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------

def is_rebalance_day(today: date = None, regime: str = "recovery",
                     cb_tier: int = 0,
                     force: bool = False) -> tuple[bool, str]:
    """
    Returns (is_due, reason). Reason is one of:
      - "scheduled"     : effective interval reached for current regime/CB
      - "regime_change" : regime differs from last rebalance regime
      - "cb_change"     : CB tier differs from last rebalance CB tier
      - "first_run"     : no state file yet, force initial rebalance
      - "forced"        : caller passed force=True
      - "weekend"       : today is a weekend, skip
      - "not_due"       : neither condition met
    """
    if today is None:
        today = date.today()

    if force:
        return True, "forced"

    # No trading on weekends
    if today.weekday() >= 5:
        return False, "weekend"

    state = _load_state()

    # First run -- no prior state
    if state is None:
        return True, "first_run"

    last_date    = date.fromisoformat(state["last_rebalance_date"])
    last_regime  = state["last_regime"]
    last_cb_tier = int(state.get("last_cb_tier", 0))

    # Regime change forces immediate rebalance
    if str(regime).lower() != last_regime:
        return True, "regime_change"

    # CB tier change forces immediate rebalance (escalation OR de-escalation)
    if int(cb_tier) != last_cb_tier:
        return True, "cb_change"

    # Trading-day interval check (effective = min(regime, cb))
    interval     = _effective_interval(regime, cb_tier)
    days_elapsed = _trading_days_between(last_date, today)
    if days_elapsed >= interval:
        return True, "scheduled"

    return False, "not_due"


def get_next_rebalance_date(today: date = None,
                            regime: str = "recovery",
                            cb_tier: int = 0) -> date | None:
    """
    Estimates next scheduled rebalance date based on the effective
    interval (min of regime and CB tier). May shift if regime or CB
    tier changes before then.
    Returns None if state missing or no future trading dates available.
    """
    from data.storage import load_prices

    if today is None:
        today = date.today()

    state = _load_state()
    if state is None:
        return today  # first run -- next rebalance is today

    last_date    = date.fromisoformat(state["last_rebalance_date"])
    interval     = _effective_interval(regime, cb_tier)
    days_elapsed = _trading_days_between(last_date, today)
    days_remaining = max(0, interval - days_elapsed)

    prices = load_prices()
    if prices.empty:
        return None
    trading_dates = sorted(pd.to_datetime(prices["date"]).unique())
    future_dates  = [d for d in trading_dates if d > pd.Timestamp(today)]
    if days_remaining == 0:
        return today
    if len(future_dates) < days_remaining:
        return None
    return future_dates[days_remaining - 1].date()


def mark_rebalance_complete(today: date, regime: str,
                            cb_tier: int = 0) -> None:
    """
    Called by execute.py after a successful rebalance execution.
    Updates state file with the rebalance date, regime, and CB tier,
    which resets the clock for the next interval.
    """
    _save_state(today, regime, cb_tier)
    print(f"[rebalance_calendar] State updated: {today} | "
          f"regime={regime} | cb_tier=T{cb_tier}")
