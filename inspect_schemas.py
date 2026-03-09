import pandas as pd

files = {
    "nav_monthly": "data/backtest/results/nav_monthly.parquet",
    "nav_w1_mv_monthly": "data/backtest/wf_results/nav_window_1_mv_monthly.parquet",
    "portfolios_w1_mv_monthly": "data/backtest/wf_results/portfolios_window_1_mv_monthly.parquet",
    "trades_w1_mv_monthly": "data/backtest/wf_results/trades_window_1_mv_monthly.parquet",
    "signals_history": "data/backtest/precomputed/signals_history.parquet",
    "signal_decay": "data/processed/signal_decay.parquet",
    "forward_returns": "data/backtest/precomputed/forward_returns.parquet",
}

for name, path in files.items():
    try:
        df = pd.read_parquet(path)
        print(f"\n=== {name} ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head(2).to_string())
    except Exception as e:
        print(f"\n=== {name} === ERROR: {e}")
