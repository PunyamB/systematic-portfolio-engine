import pandas as pd
from evt_tail_risk.m1_data_loader import load_spy_losses
from evt_tail_risk.m8_visualizer import plot_loss_histogram, plot_qq_normal, plot_rolling_vol, plot_drawdown

spy = load_spy_losses(save=False)
mer = pd.read_parquet('evt_tail_risk/data/meridian_losses.parquet')
nav = pd.read_parquet('evt_tail_risk/data/meridian_nav.parquet')

print('Generating SPY plots...')
plot_loss_histogram(spy['loss'].values, name='SPY')
plot_qq_normal(spy['loss'].values, name='SPY')
plot_rolling_vol(spy['loss'].values, spy['date'].values, name='SPY')
plot_drawdown(spy['loss'].values, spy['date'].values, name='SPY')

print('Generating Meridian plots...')
plot_loss_histogram(mer['loss'].values, name='Meridian')
plot_qq_normal(mer['loss'].values, name='Meridian')
plot_rolling_vol(mer['loss'].values, mer['date'].values, name='Meridian')
plot_drawdown(mer['loss'].values, mer['date'].values, name='Meridian')

print('Done. Check evt_tail_risk/outputs/')
