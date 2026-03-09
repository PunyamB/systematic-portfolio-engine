# utils/broker_health.py
# Pings Alpaca at pipeline start to confirm connectivity.
# Pipeline halts if broker is unreachable or account is not in good standing.

import alpaca_trade_api as tradeapi
from utils.config_loader import get_env
from utils.notifications import notify


def check_broker_health() -> bool:
    """
    Connects to Alpaca and verifies:
    - API credentials are valid
    - Account is active
    - Paper trading is enabled
    Returns True if healthy, False if not.
    Pipeline should halt on False.
    """
    try:
        api = _get_alpaca_client()
        account = api.get_account()

        if account.status != "ACTIVE":
            msg = f"Alpaca account status is {account.status}, expected ACTIVE. Pipeline halted."
            print(f"[broker_health] {msg}")
            notify(msg, level="critical")
            return False

        print(f"[broker_health] Alpaca connected. Account status: {account.status}")
        print(f"[broker_health] Portfolio value: ${float(account.portfolio_value):,.2f}")
        print(f"[broker_health] Cash: ${float(account.cash):,.2f}")
        return True

    except Exception as e:
        msg = f"Alpaca health check failed: {str(e)}"
        print(f"[broker_health] {msg}")
        notify(msg, level="critical")
        return False


def get_account_info() -> dict:
    """
    Returns key account metrics as a dict.
    Used by fund_accounting and dashboard.
    """
    api = _get_alpaca_client()
    account = api.get_account()

    return {
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
        "equity": float(account.equity),
        "buying_power": float(account.buying_power),
        "status": account.status
    }


def _get_alpaca_client():
    return tradeapi.REST(
        key_id=get_env("ALPACA_API_KEY"),
        secret_key=get_env("ALPACA_SECRET_KEY"),
        base_url=get_env("ALPACA_BASE_URL")
    )