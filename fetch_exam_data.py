import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["JPM", "XOM", "MSFT"]
BENCHMARK = "SPY"
data = yf.download(TICKERS + [BENCHMARK], start="2024-12-31", end="2026-01-01", auto_adjust=True, progress=False)["Close"].dropna().sort_index()
d = data[(data.index >= "2025-01-01") & (data.index <= "2025-12-31")]
print(f"Trading days: {len(d)}")
print("\n=== BLOCK 1: DAILY CLOSING PRICES ===")
print("Date,JPM,XOM,MSFT")
for dt, r in d[TICKERS].iterrows():
    print(f"{dt.strftime('%m/%d/%Y')},{r['JPM']:.4f},{r['XOM']:.4f},{r['MSFT']:.4f}")
rp = d[TICKERS].pct_change().dropna() * 100
print("\n=== BLOCK 2: SUMMARY STATS ===")
print(rp.describe().round(4))
print("\n=== BLOCK 3: ANNUALIZED ===")
print(f"Ann Ret: {dict((rp.mean()*252/100).round(4))}")
print(f"Ann Vol: {dict((rp.std()*np.sqrt(252)/100).round(4))}")
print("\n=== BLOCK 4: DAILY COV MATRIX ===")
print((rp/100).cov().round(8).to_string())
print("\n=== BLOCK 5: BETAS vs SPY ===")
spy = d[BENCHMARK].pct_change().dropna()
for t in TICKERS:
    a = pd.concat([d[t].pct_change().dropna(), spy], axis=1).dropna()
    print(f"  {t}: {(a.cov().iloc[0,1]/a.iloc[:,1].var()):.4f}")
print("\n=== BLOCK 6: LATEST PRICES ===")
for t in TICKERS + [BENCHMARK]:
    print(f"  {t}: ${d.iloc[-1][t]:.2f}")
print("\n=== BLOCK 7: MARKET CAPS ===")
for t in TICKERS:
    mc = yf.Ticker(t).info.get('marketCap')
    print(f"  {t}: ${mc/1e9:.2f}B" if mc else f"  {t}: unavailable")
