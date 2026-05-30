import pandas as pd, numpy as np, os
from evt_tail_risk import config

wf_dir = config.WF_RESULTS_DIR
nav_files = [f for f in os.listdir(wf_dir) if f.startswith('nav_window_') and 'exp006_monthly' in f]

# Load each window, compute daily returns WITHIN window (skip first day)
all_returns = []
for f in sorted(nav_files, key=lambda x: (0 if 'holdout' not in x else 1, x)):
    df = pd.read_parquet(os.path.join(wf_dir, f))
    if 'date' not in df.columns:
        df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date','nav']].sort_values('date').reset_index(drop=True)
    df['daily_return'] = df['nav'] / df['nav'].shift(1)
    df = df.dropna(subset=['daily_return'])
    all_returns.append(df[['date','daily_return']])
    print(f + ': ' + str(len(df)) + ' returns')

combined = pd.concat(all_returns, ignore_index=True).sort_values('date').reset_index(drop=True)
combined = combined.drop_duplicates(subset=['date'], keep='first')

# Chain into one continuous NAV from 1M
combined['nav'] = 1_000_000 * combined['daily_return'].cumprod()
combined['loss'] = -np.log(combined['daily_return'])

print()
print('Total: ' + str(len(combined)) + ' rows')
print('Date range: ' + str(combined['date'].min().date()) + ' to ' + str(combined['date'].max().date()))
print('Starting NAV: 1,000,000')
print('Ending NAV: ' + str(round(combined['nav'].iloc[-1], 2)))
print('Max loss: ' + str(round(combined['loss'].max(), 6)))
print('Min loss: ' + str(round(combined['loss'].min(), 6)))

top5 = combined.nlargest(5, 'loss')
print()
print('5 largest losses:')
print(top5[['date','loss','nav']].to_string(index=False))

# Save
combined[['date','loss']].to_parquet(os.path.join(config.DATA_DIR, 'meridian_losses.parquet'), index=False)
combined[['date','nav']].to_parquet(os.path.join(config.DATA_DIR, 'meridian_nav.parquet'), index=False)
print()
print('Saved meridian_losses.parquet and meridian_nav.parquet')
