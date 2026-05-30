"""setup_horizon.py - One-time setup for Horizon paper trading system.

Builds D:\\Projects\\SystematicPortfolioEngine\\horizon\\ by:
  1. Creating the folder structure
  2. Copying unchanged operational modules from Meridian
  3. Copying signal modules (15) from Meridian
  4. Reminding user to drop in the 6 modified Horizon files manually
  5. Reminding user to create .env and run seed_ic_history.py

Does NOT touch Meridian. Read-only on Meridian source paths.

Usage:
    python setup_horizon.py
"""
import shutil
import sys
from pathlib import Path

MERIDIAN_ROOT = Path(r"D:\Projects\SystematicPortfolioEngine")
HORIZON_ROOT  = Path(r"D:\Projects\SystematicPortfolioEngine\horizon")

# Folder structure to create
FOLDERS = [
    "compliance",
    "config",
    "corporate_actions",
    "dashboard",
    "data",
    "data/approved",
    "data/processed",
    "data/snapshots",
    "data/raw",
    "execution",
    "fund_accounting",
    "logs",
    "optimizer",
    "pipeline",
    "regime",
    "reporting",
    "risk",
    "signals",
    "utils",
]

# Files to COPY AS-IS from Meridian (relative to MERIDIAN_ROOT)
COPY_AS_IS = [
    # Top-level scripts
    "approve.py",
    "execute.py",
    "execute_replacement.py",
    "execute_stops.py",
    "precompute_dashboard.py",
    "streamlit_app.py",
    "main.py",
    "requirements.txt",

    # Modules (whole directories)
    "compliance/__init__.py",
    "compliance/checker.py",
    "corporate_actions/__init__.py",
    "corporate_actions/processor.py",
    "dashboard/__init__.py",
    "dashboard/app.py",
    "data/__init__.py",
    "data/pipeline_data.py",
    "data/storage.py",
    "execution/__init__.py",
    "execution/order_manager.py",
    "execution/execute_stops.py",
    "fund_accounting/__init__.py",
    "fund_accounting/nav.py",
    "regime/__init__.py",
    "regime/detector.py",
    "utils/__init__.py",
    "utils/broker_health.py",
    "utils/config_loader.py",
    "utils/notifications.py",

    # All 15 signals
    "signals/__init__.py",
    "signals/combiner.py",
    "signals/decay_tracker.py",
    "signals/earnings_accruals.py",
    "signals/earnings_momentum.py",
    "signals/ev_ebitda_zscore.py",
    "signals/fcf_yield.py",
    "signals/gross_margin_trend.py",
    "signals/low_volatility.py",
    "signals/momentum_12_1.py",
    "signals/pb_zscore.py",
    "signals/pe_zscore.py",
    "signals/piotroski.py",
    "signals/revenue_growth.py",
    "signals/roe_stability.py",
    "signals/rsi_extremes.py",
    "signals/short_term_reversal.py",
    "signals/volume_momentum.py",

    # Optimizer __init__
    "optimizer/__init__.py",
    "pipeline/__init__.py",
    "risk/__init__.py",
]

# Files that MUST be manually placed (the 6 Horizon-specific files)
MANUAL_PLACE = [
    "config/settings.yaml",
    "risk/monitor.py",
    "optimizer/portfolio_optimizer.py",
    "utils/rebalance_calendar.py",
    "pipeline/runner.py",
    "seed_ic_history.py",
]


def create_folders():
    print(f"\n[setup] Creating folder structure under {HORIZON_ROOT}")
    HORIZON_ROOT.mkdir(parents=True, exist_ok=True)
    for f in FOLDERS:
        p = HORIZON_ROOT / f
        p.mkdir(parents=True, exist_ok=True)
        # Create __init__.py for python packages
        if f in {"compliance", "corporate_actions", "dashboard", "data", "execution",
                 "fund_accounting", "optimizer", "pipeline", "regime", "reporting",
                 "risk", "signals", "utils"}:
            init = p / "__init__.py"
            if not init.exists():
                init.touch()
    print(f"[setup]   {len(FOLDERS)} folders ready")


def copy_files():
    print(f"\n[setup] Copying unchanged files from Meridian")
    copied = 0
    missing = []
    for rel in COPY_AS_IS:
        src = MERIDIAN_ROOT / rel
        dst = HORIZON_ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"[setup]   {copied}/{len(COPY_AS_IS)} files copied")
    if missing:
        print(f"[setup]   WARNING: {len(missing)} files not found in Meridian:")
        for m in missing:
            print(f"             - {m}")


def check_manual_files():
    print(f"\n[setup] Checking for manual Horizon files")
    missing_manual = []
    for rel in MANUAL_PLACE:
        if not (HORIZON_ROOT / rel).exists():
            missing_manual.append(rel)
    if missing_manual:
        print(f"[setup]   {len(missing_manual)} Horizon files NOT YET placed:")
        for m in missing_manual:
            print(f"             - {HORIZON_ROOT / m}")
    else:
        print(f"[setup]   All {len(MANUAL_PLACE)} Horizon files in place")
    return len(missing_manual) == 0


def write_env_template():
    env_template = HORIZON_ROOT / ".env.template"
    content = """# Horizon paper trading environment
# Copy to .env and fill in real credentials.
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
SLACK_WEBHOOK_URL=
FMP_API_KEY=
FRED_API_KEY=
"""
    env_template.write_text(content)
    print(f"\n[setup] Created .env.template at {env_template}")
    print(f"[setup]   Copy to .env and fill in keys before first run")


def print_next_steps():
    print(f"\n{'='*60}")
    print(f"  HORIZON SETUP COMPLETE")
    print(f"{'='*60}")
    print(f"""
Next steps:

  1. Place the 6 Horizon files (download from Claude output) into:
     - {HORIZON_ROOT}\\config\\settings.yaml
     - {HORIZON_ROOT}\\risk\\monitor.py
     - {HORIZON_ROOT}\\optimizer\\portfolio_optimizer.py
     - {HORIZON_ROOT}\\utils\\rebalance_calendar.py
     - {HORIZON_ROOT}\\pipeline\\runner.py
     - {HORIZON_ROOT}\\seed_ic_history.py

  2. Copy .env.template to .env and fill in:
     - Alpaca paper account API keys (NEW account, not Meridian's)
     - Slack webhook URL
     - FMP and FRED API keys

  3. Seed the IC history:
     cd {HORIZON_ROOT}
     python seed_ic_history.py

  4. Verify everything loads:
     cd {HORIZON_ROOT}
     python -c "from utils.config_loader import get_config; print(get_config()['circuit_breaker'])"

  5. First dry-run (will halt cleanly if anything missing):
     cd {HORIZON_ROOT}
     python main.py

  6. Tag git state before going live:
     git tag horizon_v1_pre_live
""")


def main():
    print(f"\n{'='*60}\n  HORIZON SETUP\n{'='*60}")
    print(f"  Meridian source: {MERIDIAN_ROOT}")
    print(f"  Horizon target:  {HORIZON_ROOT}")

    if HORIZON_ROOT.exists() and any(HORIZON_ROOT.iterdir()):
        confirm = input(f"\n{HORIZON_ROOT} already exists and is not empty.\n"
                        f"Continue (will overwrite)? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    create_folders()
    copy_files()
    write_env_template()
    check_manual_files()
    print_next_steps()


if __name__ == "__main__":
    main()
