# utils/rebalance_calendar.py
# Horizon regime + V4 6-tier CB rebalance scheduling.
#
# Regime trading-day intervals (settings.yaml):
#   bull / recovery: 21 trading days
#   bear:             7 trading days
#   crisis:           4 trading days
#
# V4 CB tier intervals (settings.yaml):
#   T0: 999 (no override; defer to regime)
#   T1:  21
#   T2:   5
#   T3:   3
#   T4:   1
#   T5:   1
#
# Effective interval = min(regime_interval, cb_interval).
# Regime change OR CB tier change forces immediate rebalance + clock reset.
# State persisted in data/processed/rebalance_state.json.
#
# Calendar functions are READ-ONLY for state. State is updated only by
# execute.py via mark_rebalance_complete() after successful execution.

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
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        if "last_rebalance_date" not in data or "last_regime" not in data:
            return None
        if "last_cb_tier" not in data:
            data["last_cb_tier"] = 0
        return data
    except Exception:
        return None


def _save_state(last_rebalance_date: date, last_regime: str,
                last_cb_tier: int = 0) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_rebalance_date": str(last_rebalance_date),
        "last_regime":         str(last_regime).lower(),
        "last_cb_tier":        int(last_cb_tier),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def _trading_days_between(start_date: date, end_date: date) -> int:
    from data.storage import load_prices
    prices = load_prices()
    if prices.empty:
        return 0
    trading_dates = sorted(pd.to_datetime(prices["date"]).unique())
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
    return len([d for d in trading_dates if start_ts < d <= end_ts])


def _interval_for_regime(regime: str) -> int:
    cfg = get_config()
    intervals = cfg["rebalance"]["regime_trading_days"]
    return int(intervals.get(str(regime).lower(), intervals.get("recovery", 21)))


def _interval_for_cb_tier(cb_tier: int) -> int:
    """
    Returns trading-day interval for the given V4 CB tier (0-5).
    Tier 0 returns 999 (no override; defer to regime).
    """
    cfg = get_config()
    intervals = cfg.get("circuit_breaker", {}).get("cb_trading_days", {})
    val = intervals.get(cb_tier, intervals.get(str(cb_tier), 999))
    return int(val)


def _effective_interval(regime: str, cb_tier: int) -> int:
    """Whichever fires more often wins -- safer regime always dominates."""
    return min(_interval_for_regime(regime), _interval_for_cb_tier(cb_tier))


# ------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------

def is_rebalance_day(today: date = None, regime: str = "recovery",
                     cb_tier: int = 0,
                     force: bool = False) -> tuple[bool, str]:
    """
    Returns (is_due, reason). Reason ∈
      {scheduled, regime_change, cb_change, first_run, forced, weekend, not_due}
    """
    if today is None:
        today = date.today()

    if force:
        return True, "forced"

    if today.weekday() >= 5:
        return False, "weekend"

    state = _load_state()

    if state is None:
        return True, "first_run"

    last_date    = date.fromisoformat(state["last_rebalance_date"])
    last_regime  = state["last_regime"]
    last_cb_tier = int(state.get("last_cb_tier", 0))

    if str(regime).lower() != last_regime:
        return True, "regime_change"

    if int(cb_tier) != last_cb_tier:
        return True, "cb_change"

    interval     = _effective_interval(regime, cb_tier)
    days_elapsed = _trading_days_between(last_date, today)
    if days_elapsed >= interval:
        return True, "scheduled"

    return False, "not_due"


def get_next_rebalance_date(today: date = None,
                            regime: str = "recovery",
                            cb_tier: int = 0) -> date | None:
    from data.storage import load_prices

    if today is None:
        today = date.today()

    state = _load_state()
    if state is None:
        return today

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
    Called by execute.py after successful rebalance.
    Resets the clock for the next interval.
    """
    _save_state(today, regime, cb_tier)
    print(f"[rebalance_calendar] State updated: {today} | "
          f"regime={regime} | cb_tier=T{cb_tier}")
