import pandas as pd
from pathlib import Path

snap_dir = Path("data/snapshots")
regime_files = sorted(snap_dir.glob("*_regime.parquet"))
print(f"Regime snapshots found: {len(regime_files)}")
for f in regime_files:
    df = pd.read_parquet(f)
    print(f.name, df.to_dict(orient="records"))