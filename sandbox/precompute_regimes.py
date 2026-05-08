import pandas as pd
import numpy as np
from pathlib import Path

# Provide a lightweight logic duplicate of your regime/detector.py
# that doesn't rely on yaml or duckdb so we skip installation headaches.

def precompute_regimes_light():
    print("🚀 Precomputing Regimes (Lightweight Engine)...")
    
    RAW = Path("data/raw")
    RESULTS = Path("sandbox/results")
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    # 1. Load raw parquet data using pandas directly
    print("-> Loading raw datasets...")
    prices = pd.read_parquet(RAW / "prices.parquet")[["date", "ticker", "close"]]
    prices["date"] = pd.to_datetime(prices["date"])
    
    vix = pd.read_parquet(RAW / "regime_vix.parquet")
    vix["date"] = pd.to_datetime(vix["date"])
    
    yc = pd.read_parquet(RAW / "regime_yield_curve.parquet")
    yc["date"] = pd.to_datetime(yc["date"])
    
    cs = pd.read_parquet(RAW / "regime_credit_spreads.parquet")
    cs["date"] = pd.to_datetime(cs["date"])
    
    trading_days = sorted(prices["date"].unique())
    
    # Hardcoded from your settings.yaml
    VIX_LOOKBACK = 252
    VIX_ELEVATED = 0.80
    VIX_CRISIS = 1.20
    BREADTH_HIGH = 0.60
    BREADTH_LOW = 0.40
    
    print("\n-> Computing Market Breadth (200d MA)...")
    # Breadth O(1) vectorized mapping
    pivot = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    ma_200 = pivot.rolling(200, min_periods=100).mean()
    above_ma = (pivot > ma_200).sum(axis=1)
    valid_count = ma_200.notna().sum(axis=1)
    breadth_s = (above_ma / valid_count).fillna(0.5)
    
    print("-> Computing VIX Ratio...")
    vix = vix.set_index("date").sort_index()
    vix_roll_avg = vix["vix"].rolling(VIX_LOOKBACK, min_periods=50).mean()
    vix_ratio_s = (vix["vix"] / vix_roll_avg).fillna(1.0)
    
    print("-> Computing Economic Cycle Triggers...")
    yc = yc.set_index("date").sort_index()
    cs = cs.set_index("date").sort_index()
    
    # Convert all indices strictly to pd.Timestamp to avoid mixups
    yc.index = pd.to_datetime(yc.index)
    cs.index = pd.to_datetime(cs.index)
    vix_ratio_s.index = pd.to_datetime(vix_ratio_s.index)
    breadth_s.index = pd.to_datetime(breadth_s.index)
    
    trading_days = pd.to_datetime(trading_days)
    
    # Forward fill missing economic numbers to match daily trading dates
    yc_daily = yc.reindex(trading_days, method="ffill").fillna(0)
    cs_daily = cs.reindex(trading_days, method="ffill").fillna(method="bfill")
    vix_ratio_daily = vix_ratio_s.reindex(trading_days, method="ffill").fillna(1.0)
    breadth_daily = breadth_s.reindex(trading_days, method="ffill").fillna(0.5)
    
    history = []
    
    print("-> Stitching composites...")
    for day in trading_days:
        b = float(breadth_daily.loc[day])
        vr = float(vix_ratio_daily.loc[day])
        
        # Stress State mapping
        if vr < VIX_ELEVATED and b > BREADTH_HIGH:
            stress = "low_stress"
        elif vr > VIX_CRISIS and b < BREADTH_LOW:
            stress = "crisis"
        else:
            stress = "elevated"
            
        # Cycle State mapping
        spread = yc_daily.loc[day, "spread_10y2y"] if "spread_10y2y" in yc_daily.columns and not pd.isna(yc_daily.loc[day, "spread_10y2y"]) else 1.0
        curve_inverted = spread < 0
        
        # Credit spread trend (5 day lookback)
        idx = cs_daily.index.get_loc(day)
        if isinstance(idx, slice):
            idx = idx.start
        elif isinstance(idx, np.ndarray):
            idx = idx[0]
            
        if idx >= 5 and "hy_spread" in cs_daily.columns:
            trend = cs_daily["hy_spread"].iloc[idx] - cs_daily["hy_spread"].iloc[idx-5]
            widening = trend > 0
        else:
            widening = False
            
        cycle = "contraction" if (curve_inverted and widening) else "expansion"
        
        # Composite mapping
        if stress == "crisis":
            composite = "crisis"
        elif stress == "elevated" and cycle == "contraction":
            composite = "bear"
        elif stress == "low_stress" and cycle == "expansion":
            composite = "bull"
        else:
            composite = "recovery"
            
        history.append({
            "date": day,
            "composite": composite,
            "stress_state": stress,
            "cycle_state": cycle,
            "vix_ratio": vr,
            "breadth": b,
            "yield_spread": spread
        })
        
    df = pd.DataFrame(history)
    out_path = RESULTS / "regime_history.parquet"
    df.to_parquet(out_path, index=False)
    
    print(df["composite"].value_counts())
    print(f"\n✅ Successfully saved lightweight regime history! {len(df)} days mapped to {out_path}")

if __name__ == "__main__":
    precompute_regimes_light()
