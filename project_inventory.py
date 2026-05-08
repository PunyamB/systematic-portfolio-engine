"""
Project Inventory Script
========================
Run this in your SirAlgotsAlot venv from any directory.
It scans SPE data files, EVTTailRisk project state, and settings.yaml,
then writes everything to a single text file you can paste back to Claude.

Usage:
    cd D:\Projects\SystematicPortfolioEngine
    SirAlgotsAlot\Scripts\Activate.ps1
    python project_inventory.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────
SPE_ROOT = Path(r"D:\Projects\SystematicPortfolioEngine")
EVT_ROOT = Path(r"D:\Projects\EVTTailRisk")
SRL_ROOT = Path(r"D:\Projects\StrategyResearchLab")
OUTPUT_FILE = Path(r"D:\Projects\project_inventory_output.txt")

# Parquets to inspect for date range + shape
PARQUETS_TO_CHECK = [
    SPE_ROOT / "data" / "raw" / "prices.parquet",
    SPE_ROOT / "data" / "raw" / "financials.parquet",
    SPE_ROOT / "data" / "raw" / "key_metrics.parquet",
    SPE_ROOT / "data" / "processed" / "nav_history.parquet",
    SPE_ROOT / "data" / "processed" / "ic_history.parquet",
    SPE_ROOT / "data" / "processed" / "signals.parquet",
    SPE_ROOT / "data" / "processed" / "portfolio.parquet",
]

# Files to print contents of
FILES_TO_PRINT = [
    SPE_ROOT / "config" / "settings.yaml",
]


def write(f, text=""):
    """Write a line to the output file."""
    f.write(text + "\n")


def dir_tree(root, f, max_depth=2, prefix=""):
    """Print directory tree up to max_depth, skipping hidden/venv/node_modules."""
    skip = {".git", "__pycache__", "node_modules", ".venv", "SirAlgotsAlot",
            "ReLabs", ".mypy_cache", ".pytest_cache", "SirAlgotsAlot"}
    
    if not root.exists():
        write(f, f"  [DOES NOT EXIST: {root}]")
        return

    items = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for item in items:
        if item.name in skip or item.name.startswith("."):
            continue
        if item.is_dir():
            write(f, f"{prefix}📁 {item.name}/")
            if max_depth > 1:
                dir_tree(item, f, max_depth - 1, prefix + "    ")
        else:
            size_kb = item.stat().st_size / 1024
            if size_kb > 1024:
                size_str = f"{size_kb/1024:.1f} MB"
            else:
                size_str = f"{size_kb:.1f} KB"
            write(f, f"{prefix}📄 {item.name}  ({size_str})")


def inspect_parquet(filepath, f):
    """Read a parquet file and print shape, columns, date range."""
    try:
        import pandas as pd
        df = pd.read_parquet(filepath)
        write(f, f"\n  File: {filepath.name}")
        write(f, f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        write(f, f"  Columns: {list(df.columns)}")
        
        # Try to find date columns and print range
        date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["date", "time", "period"])]
        if df.index.name and any(kw in str(df.index.name).lower() for kw in ["date", "time", "period"]):
            date_cols.insert(0, f"INDEX:{df.index.name}")
        
        # Also check if the index itself is datetime
        if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
            write(f, f"  Index type: {df.index.dtype}")
            write(f, f"  Date range (index): {df.index.min()} → {df.index.max()}")
        
        for col in date_cols:
            if col.startswith("INDEX:"):
                continue  # already handled above
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    write(f, f"  Date range ({col}): {dates.min()} → {dates.max()}")
            except Exception:
                pass
        
        # Print unique tickers if there's a ticker/symbol column
        ticker_cols = [c for c in df.columns if c.lower() in ["ticker", "symbol", "stock"]]
        for col in ticker_cols:
            n_unique = df[col].nunique()
            write(f, f"  Unique {col}s: {n_unique}")
            if n_unique <= 10:
                write(f, f"  Values: {sorted(df[col].unique())}")
        
        # Memory usage
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        write(f, f"  Memory: {mem_mb:.1f} MB")
        
        # First 3 rows
        write(f, f"  Head (3 rows):")
        head_str = df.head(3).to_string(max_colwidth=40)
        for line in head_str.split("\n"):
            write(f, f"    {line}")
            
    except ImportError:
        write(f, f"  [pandas not available — cannot read parquet]")
    except Exception as e:
        write(f, f"  [ERROR reading {filepath.name}: {e}]")


def print_file_contents(filepath, f, max_lines=200):
    """Print file contents, truncated if too long."""
    try:
        text = filepath.read_text(encoding="utf-8")
        lines = text.split("\n")
        if len(lines) > max_lines:
            write(f, f"  [Showing first {max_lines} of {len(lines)} lines]")
            lines = lines[:max_lines]
        for line in lines:
            write(f, f"  {line}")
    except Exception as e:
        write(f, f"  [ERROR reading: {e}]")


def main():
    print(f"Running project inventory...")
    print(f"Output will be written to: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        write(f, "=" * 70)
        write(f, f"PROJECT INVENTORY — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write(f, "=" * 70)
        
        # ── 1. SPE Directory Tree ──────────────────────────────────
        write(f, "\n" + "─" * 70)
        write(f, "1. SPE DIRECTORY TREE (2 levels)")
        write(f, "─" * 70)
        dir_tree(SPE_ROOT, f, max_depth=2)
        
        # ── 2. EVTTailRisk Directory Tree ──────────────────────────
        write(f, "\n" + "─" * 70)
        write(f, "2. EVTTailRisk DIRECTORY TREE (3 levels)")
        write(f, "─" * 70)
        if EVT_ROOT.exists():
            dir_tree(EVT_ROOT, f, max_depth=3)
        else:
            write(f, f"  [DOES NOT EXIST: {EVT_ROOT}]")
        
        # ── 3. StrategyResearchLab Directory Tree ──────────────────
        write(f, "\n" + "─" * 70)
        write(f, "3. StrategyResearchLab DIRECTORY TREE (2 levels)")
        write(f, "─" * 70)
        if SRL_ROOT.exists():
            dir_tree(SRL_ROOT, f, max_depth=2)
        else:
            write(f, f"  [DOES NOT EXIST: {SRL_ROOT}]")
        
        # ── 4. Parquet File Inspection ─────────────────────────────
        write(f, "\n" + "─" * 70)
        write(f, "4. PARQUET FILE INSPECTION (date ranges, shapes, columns)")
        write(f, "─" * 70)
        for pq in PARQUETS_TO_CHECK:
            if pq.exists():
                inspect_parquet(pq, f)
            else:
                write(f, f"\n  [NOT FOUND: {pq}]")
        
        # ── 5. settings.yaml ───────────────────────────────────────
        write(f, "\n" + "─" * 70)
        write(f, "5. SETTINGS.YAML CONTENTS")
        write(f, "─" * 70)
        for fp in FILES_TO_PRINT:
            write(f, f"\n  File: {fp}")
            if fp.exists():
                print_file_contents(fp, f)
            else:
                write(f, f"  [NOT FOUND]")
        
        # ── 6. Python / Package Versions ───────────────────────────
        write(f, "\n" + "─" * 70)
        write(f, "6. ENVIRONMENT INFO")
        write(f, "─" * 70)
        write(f, f"  Python: {sys.version}")
        write(f, f"  Platform: {sys.platform}")
        write(f, f"  CWD: {os.getcwd()}")
        
        for pkg in ["pandas", "numpy", "cvxpy", "scipy", "statsmodels", "arch", "sklearn"]:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
                write(f, f"  {pkg}: {ver}")
            except ImportError:
                write(f, f"  {pkg}: NOT INSTALLED")
        
        write(f, "\n" + "=" * 70)
        write(f, "END OF INVENTORY")
        write(f, "=" * 70)
    
    print(f"\nDone! Output written to: {OUTPUT_FILE}")
    print(f"Copy the contents of that file and paste it back to Claude.")


if __name__ == "__main__":
    main()
