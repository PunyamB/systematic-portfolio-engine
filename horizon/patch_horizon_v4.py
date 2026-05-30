"""patch_horizon_v4.py - Surgically patches copied Meridian files for V4.

Modifies (in horizon/ only, never touches Meridian):
  1. risk/monitor.py        — add T5 threshold + 6-tier ladder + actions
  2. optimizer/portfolio_optimizer.py — extend CB_INVESTED_TARGET to 6 keys, NO_NEW_POSITIONS_TIER 3->4

settings.yaml and rebalance_calendar.py already V4-ready.

Idempotent: detects existing V4 patches and skips.
Creates .bak_preV4 backups before any modification.
"""
import sys
from pathlib import Path

HORIZON = Path(__file__).resolve().parent

MONITOR_PATH   = HORIZON / "risk" / "monitor.py"
OPTIMIZER_PATH = HORIZON / "optimizer" / "portfolio_optimizer.py"


def backup(p: Path):
    bak = p.with_suffix(p.suffix + ".bak_preV4")
    if not bak.exists():
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[patch] Backup -> {bak.name}")


# ============================================================
# 1. risk/monitor.py
# ============================================================

MONITOR_OLD_CONSTS = """CB_T1 = cfg["circuit_breaker"]["t1_pct"]  # 0.05
CB_T2 = cfg["circuit_breaker"]["t2_pct"]  # 0.10
CB_T3 = cfg["circuit_breaker"]["t3_pct"]  # 0.15
CB_T4 = cfg["circuit_breaker"]["t4_pct"]  # 0.20"""

MONITOR_NEW_CONSTS = """# Horizon V4 6-tier thresholds (from settings.yaml)
CB_T1 = cfg["circuit_breaker"]["t1_pct"]  # 0.020
CB_T2 = cfg["circuit_breaker"]["t2_pct"]  # 0.030
CB_T3 = cfg["circuit_breaker"]["t3_pct"]  # 0.0685
CB_T4 = cfg["circuit_breaker"]["t4_pct"]  # 0.145
CB_T5 = cfg["circuit_breaker"]["t5_pct"]  # 0.215"""

MONITOR_OLD_LADDER = """    if drawdown >= CB_T4:
        tier = 4
        actions = [
            "Reduce to 40% invested",
            "Trading paused 5 days",
            "Manual review required"
        ]
    elif drawdown >= CB_T3:
        tier = 3
        actions = [
            "Gross exposure capped at 70%",
            "Max position size 2.5%",
            "Rebalance frequency daily",
            "No new positions for 3 days"
        ]
    elif drawdown >= CB_T2:
        tier = 2
        actions = [
            "Max position size 3.5%",
            "Rebalance frequency weekly",
            "No new low-liquidity positions"
        ]
    elif drawdown >= CB_T1:
        tier = 1
        actions = ["Info alert only"]
    else:
        tier = 0
        actions = []"""

MONITOR_NEW_LADDER = """    if drawdown >= CB_T5:
        tier = 5
        actions = [
            "Catastrophic regime",
            "Invested target 40%, max weight 2.5%, sum floor 0.40",
            "Daily rebalance, NO NEW POSITIONS"
        ]
    elif drawdown >= CB_T4:
        tier = 4
        actions = [
            "Crisis entry",
            "Invested target 60%, max weight 2.5%, sum floor 0.40",
            "Daily rebalance, NO NEW POSITIONS"
        ]
    elif drawdown >= CB_T3:
        tier = 3
        actions = [
            "Deep correction",
            "Invested target 75%, max weight 3.0%, sum floor 0.55",
            "Rebalance every 3 trading days"
        ]
    elif drawdown >= CB_T2:
        tier = 2
        actions = [
            "Notable correction",
            "Invested target 95%, max weight 3.5%, sum floor 0.70",
            "Rebalance every 5 trading days"
        ]
    elif drawdown >= CB_T1:
        tier = 1
        actions = [
            "Mild stress (info)",
            "Rebalance interval forced to 21 days"
        ]
    else:
        tier = 0
        actions = []"""

MONITOR_OLD_NOTIFY = '            level="critical" if tier >= 3 else "warning"'
MONITOR_NEW_NOTIFY = '            level="critical" if tier >= 4 else ("warning" if tier >= 2 else "info")'


def patch_monitor():
    if not MONITOR_PATH.exists():
        print(f"[patch] {MONITOR_PATH} missing, skipping")
        return
    txt = MONITOR_PATH.read_text(encoding="utf-8")

    if "CB_T5 = cfg" in txt:
        print(f"[patch] monitor.py already V4, skipping")
        return

    backup(MONITOR_PATH)

    if MONITOR_OLD_CONSTS not in txt:
        print(f"[patch] FAIL: monitor.py CB_T constants block not matched")
        sys.exit(1)
    txt = txt.replace(MONITOR_OLD_CONSTS, MONITOR_NEW_CONSTS)

    if MONITOR_OLD_LADDER not in txt:
        print(f"[patch] FAIL: monitor.py tier ladder block not matched")
        sys.exit(1)
    txt = txt.replace(MONITOR_OLD_LADDER, MONITOR_NEW_LADDER)

    if MONITOR_OLD_NOTIFY in txt:
        txt = txt.replace(MONITOR_OLD_NOTIFY, MONITOR_NEW_NOTIFY)

    MONITOR_PATH.write_text(txt, encoding="utf-8")
    print(f"[patch] monitor.py -> V4 6-tier ladder applied")


# ============================================================
# 2. optimizer/portfolio_optimizer.py
# ============================================================

OPT_OLD_DICT = """CB_INVESTED_TARGET = {
    0: 0.98,   # T0: normal, 2% cash buffer
    1: 0.98,   # T1: info only, no exposure change
    2: 0.95,   # T2: slightly more defensive
    3: 0.70,   # T3: 30% cash
    4: 0.40,   # T4: 60% cash
}"""

OPT_NEW_DICT = """CB_INVESTED_TARGET = {
    0: 0.98,   # T0: normal (DD<2.0%), 2% cash buffer
    1: 0.98,   # T1: mild stress (DD 2.0-3.0%), no exposure change
    2: 0.95,   # T2: notable (DD 3.0-6.85%), 5% cash
    3: 0.75,   # T3: deep correction (DD 6.85-14.5%), 25% cash
    4: 0.60,   # T4: crisis entry (DD 14.5-21.5%), 40% cash, NO NEW POS
    5: 0.40,   # T5: catastrophic (DD >=21.5%), 60% cash, NO NEW POS
}"""

OPT_OLD_NONEW = "NO_NEW_POSITIONS_TIER = 3"
OPT_NEW_NONEW = "NO_NEW_POSITIONS_TIER = 4"


def patch_optimizer():
    if not OPTIMIZER_PATH.exists():
        print(f"[patch] {OPTIMIZER_PATH} missing, skipping")
        return
    txt = OPTIMIZER_PATH.read_text(encoding="utf-8")

    if "5: 0.40" in txt and "NO_NEW_POSITIONS_TIER = 4" in txt:
        print(f"[patch] portfolio_optimizer.py already V4, skipping")
        return

    backup(OPTIMIZER_PATH)

    if OPT_OLD_DICT not in txt:
        print(f"[patch] FAIL: optimizer CB_INVESTED_TARGET block not matched")
        sys.exit(1)
    txt = txt.replace(OPT_OLD_DICT, OPT_NEW_DICT)

    if OPT_OLD_NONEW not in txt:
        print(f"[patch] FAIL: optimizer NO_NEW_POSITIONS_TIER constant not matched")
        sys.exit(1)
    txt = txt.replace(OPT_OLD_NONEW, OPT_NEW_NONEW)

    OPTIMIZER_PATH.write_text(txt, encoding="utf-8")
    print(f"[patch] portfolio_optimizer.py -> V4 CB_INVESTED_TARGET + NO_NEW_POSITIONS_TIER=4")


def main():
    print(f"[patch] HORIZON V4 PATCH")
    print(f"[patch] target: {HORIZON}\n")
    patch_monitor()
    patch_optimizer()
    print(f"\n[patch] DONE. Verify:")
    print(f"   python -c \"from risk.monitor import CB_T5; print('T5 threshold:', CB_T5)\"")
    print(f"   python -c \"from optimizer.portfolio_optimizer import CB_INVESTED_TARGET, NO_NEW_POSITIONS_TIER; print(CB_INVESTED_TARGET); print('NO_NEW:', NO_NEW_POSITIONS_TIER)\"")


if __name__ == "__main__":
    main()
