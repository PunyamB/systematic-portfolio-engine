import requests, pandas as pd
KEY = 'chiezlHmDSi0a5A8OUPwMxhOBMuEIkSq'
frames = []
for start, end in [('1993-01-01','2006-06-15'), ('2006-06-16','2026-05-01')]:
    url = 'https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=SPY&from=' + start + '&to=' + end + '&apikey=' + KEY
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    d = r.json()
    if isinstance(d, list) and len(d) > 0:
        frames.append(pd.DataFrame(d))
        print(start + ' to ' + end + ': ' + str(len(d)) + ' rows')
combined = pd.concat(frames, ignore_index=True)
combined['date'] = pd.to_datetime(combined['date'])
combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
keep = [c for c in ['date','open','high','low','close','volume','adjClose'] if c in combined.columns]
combined = combined[keep]
combined.to_parquet(r'D:\Projects\SystematicPortfolioEngine\evt_tail_risk\data\spy_prices.parquet', index=False)
print('Total: ' + str(len(combined)) + ' rows, ' + str(combined['date'].min().date()) + ' to ' + str(combined['date'].max().date()))
