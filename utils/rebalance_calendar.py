# utils/rebalance_calendar.py
# Determines whether today is a rebalance day based on frequency set in settings.yaml.
# All pipeline branching (full rebalance vs daily-only) runs through this.

import pandas as pd
from datetime import date
from utils.config_loader import get_config


def is_rebalance_day(today: date = None) -> bool:
    """
    Returns True if today is a scheduled rebalance day.
    Frequency options: daily | weekly | monthly | quarterly
    """
    cfg = get_config()
    frequency = cfg["rebalance"]["frequency"]

    if today is None:
        today = date.today()

    # Skip weekends
    if today.weekday() >= 5:
        return False

    if frequency == "daily":
        return True

    elif frequency == "weekly":
        # Every Monday
        return today.weekday() == 0

    elif frequency == "monthly":
        # First trading day of the month
        return _is_first_trading_day_of_month(today)

    elif frequency == "quarterly":
        # First trading day of Jan, Apr, Jul, Oct
        if today.month in [1, 4, 7, 10]:
            return _is_first_trading_day_of_month(today)
        return False

    else:
        raise ValueError(f"Unknown rebalance frequency: {frequency}")


def _is_first_trading_day_of_month(today: date) -> bool:
    """
    Returns True if today is the first weekday of the current month.
    """
    first = date(today.year, today.month, 1)
    # Walk forward from the 1st until we hit a weekday
    while first.weekday() >= 5:
        first = date(first.year, first.month, first.day + 1)
    return today == first


def get_next_rebalance_date(today: date = None) -> date:
    """
    Returns the next scheduled rebalance date after today.
    Useful for Slack alerts and dashboard display.
    """
    if today is None:
        today = date.today()

    check = date(today.year, today.month, today.day)
    for _ in range(95):  # max ~1 quarter ahead
        check = date(check.year, check.month, check.day + 1) if check.day < 28 else \
                (check.replace(day=check.day + 1) if check.day < _days_in_month(check) \
                else date(check.year + (check.month == 12), (check.month % 12) + 1, 1))
        if is_rebalance_day(check):
            return check

    return None


def _days_in_month(d: date) -> int:
    import calendar
    return calendar.monthrange(d.year, d.month)[1]