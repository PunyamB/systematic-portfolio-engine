import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def stitch_and_display_results(results_dir="sandbox/results"):
    """
    Stitches the individual walk-forward out-of-sample test windows
    and cleanly calculates/displays performance metrics exactly like a real backtest.
    """
    import os
    results_path = Path(results_dir)
    dfs = []
    
    # We want to pull Windows 1-9 and the holdout
    windows = list(range(1, 10)) + ["holdout"]
    for w in windows:
        p = results_path / f"sim_nav_window_{w}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            dfs.append(df)
            
    if not dfs:
        print(f"No results found in {results_path}. Please run execute_backtest() first.")
        return
        
    # Concatenate all true out-of-sample days
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Calculate daily returns
    daily_returns = combined['nav'].pct_change().dropna()
    
    # Stitch the equity curve geometrically (since each window simulated starting with 1M)
    cum_returns = (1 + daily_returns).cumprod()
    
    # Performance Metrics
    total_return = cum_returns.iloc[-1] - 1
    annualized_return = daily_returns.mean() * 252
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    # Print Metrics
    print("=" * 50)
    print("🔋 WALK-FORWARD OOS PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Total Return:         {total_return:.2%}")
    print(f"Annualized Return:    {annualized_return:.2%}")
    print(f"Annualized Vol:       {annualized_vol:.2%}")
    print(f"Sharpe Ratio:         {sharpe:.2f}")
    print(f"Max Drawdown:         {max_drawdown:.2%}")
    print("=" * 50)
    
    # Plotly interactive chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_returns.index, 
        y=cum_returns.values,
        mode='lines',
        name='Strategy Equity Curve',
        line=dict(color='#00ff9d', width=2)
    ))
    
    fig.update_layout(
        title="Out-of-Sample Walk-Forward Equity Curve (2013-2025)",
        yaxis_title="Cumulative Growth (1.0 = Base)",
        xaxis_title="Date",
        template="plotly_dark",
        height=600
    )
    
    fig.show()
