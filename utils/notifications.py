# utils/notifications.py
# Sends Slack alerts via webhook. Used across the entire pipeline.
# Call notify() from anywhere — it never crashes the pipeline on failure.

import requests
import json
import traceback
from datetime import datetime
from utils.config_loader import get_env


def notify(message: str, level: str = "info") -> bool:
    """
    Sends a Slack message via webhook.
    level options: "info" | "warning" | "critical"
    Returns True if sent successfully, False if failed.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "text": f"[{level.upper()}] {timestamp}\n{message}"
    }

    try:
        webhook_url = get_env("SLACK_WEBHOOK_URL")
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        return response.status_code == 200

    except Exception:
        print(f"[notifications] Failed to send Slack alert:\n{traceback.format_exc()}")
        return False


def notify_pipeline_start(run_date: str) -> None:
    notify(f"Pipeline started for {run_date}", level="info")


def notify_pipeline_complete(run_date: str, nav: float, daily_return: float, regime: str) -> None:
    msg = (
        f"Pipeline complete for {run_date}\n"
        f"NAV: ${nav:,.2f}\n"
        f"Daily Return: {daily_return:.2%}\n"
        f"Regime: {regime}"
    )
    notify(msg, level="info")


def notify_trade_approval_needed(n_trades: int, file_path: str) -> None:
    msg = (
        f"Trade approval required -- {n_trades} trades pending\n"
        f"Review file: {file_path}\n"
        f"Copy to data/approved/ with today's date to execute"
    )
    notify(msg, level="warning")


def notify_circuit_breaker(tier: int, drawdown: float) -> None:
    msg = (
        f"Circuit Breaker Triggered -- Tier {tier}\n"
        f"Drawdown: {drawdown:.2%}\n"
        f"Risk controls activated"
    )
    notify(msg, level="critical")


def notify_data_quality_failure(pct_missing: float) -> None:
    msg = (
        f"Pipeline halted -- Data quality gate failed\n"
        f"Missing data: {pct_missing:.2%} (threshold: 5%)"
    )
    notify(msg, level="critical")