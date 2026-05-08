from pathlib import Path

for p in Path("data/backtest").rglob("*.parquet"):
    print(p)