import pandas as pd
from evt_tail_risk import config
from scipy import stats as sp_stats

spy = pd.read_parquet(config.DATA_DIR + '/spy_losses.parquet')
mer = pd.read_parquet(config.DATA_DIR + '/meridian_losses.parquet')

print('SPY: ' + str(len(spy)) + ' obs, ' + str(spy['date'].min().date()) + ' to ' + str(spy['date'].max().date()))
print('Meridian: ' + str(len(mer)) + ' obs, ' + str(mer['date'].min().date()) + ' to ' + str(mer['date'].max().date()))
print()

for name, df in [('SPY', spy), ('Meridian', mer)]:
    s = df['loss']
    print(name + ':')
    print('  count:    ' + str(len(s)))
    print('  mean:     ' + str(round(s.mean(), 6)))
    print('  std:      ' + str(round(s.std(), 6)))
    print('  min:      ' + str(round(s.min(), 6)))
    print('  max:      ' + str(round(s.max(), 6)))
    print('  skewness: ' + str(round(sp_stats.skew(s), 4)))
    print('  kurtosis: ' + str(round(sp_stats.kurtosis(s), 4)))
    print('  q95:      ' + str(round(s.quantile(0.95), 6)))
    print('  q99:      ' + str(round(s.quantile(0.99), 6)))
    print()
