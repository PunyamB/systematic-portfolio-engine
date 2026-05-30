import requests, os, pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path('.env'))
FMP_KEY = os.getenv('FMP_API_KEY')

# Try fetching with to parameter to get older data
url = 'https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=SPY&from=2004-01-01&to=2006-12-31&apikey=' + FMP_KEY
r = requests.get(url)
data = r.json()
print('Type:', type(data))
if isinstance(data, list):
    print('Rows:', len(data))
    if data:
        df = pd.DataFrame(data)
        print('Cols:', df.columns.tolist())
        print('First:', df['date'].min())
        print('Last:', df['date'].max())
else:
    print(data)