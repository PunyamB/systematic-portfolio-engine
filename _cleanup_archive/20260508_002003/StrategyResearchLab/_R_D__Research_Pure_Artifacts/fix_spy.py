import requests, os, pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path('.env'))
FMP_KEY = os.getenv('FRED_API_KEY')
FMP_KEY = os.getenv('FMP_API_KEY')

# Fetch 2004-2006
url = 'https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=SPY&from=2004-01-01&to=2006-12-31&apikey=' + FMP_KEY
r = requests.get(url)
old = pd.DataFrame(r.json())
old['date'] = pd.to_datetime(old['date'])
old = old.set_index('date').sort_index()
old = old[['open', 'high', 'low', 'close', 'volume']]

# Load existing
existing = pd.read_parquet('data/raw/spy_prices.parquet')

# Merge
combined = pd.concat([old, existing])
combined = combined[~combined.index.duplicated(keep='last')].sort_index()
combined.to_parquet('data/raw/spy_prices.parquet')
print('Combined SPY:', len(combined), 'rows')
print('Start:', combined.index.min().date())
print('End:', combined.index.max().date())