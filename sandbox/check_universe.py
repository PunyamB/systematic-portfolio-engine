import pickle
from pathlib import Path

pkl = Path("data/backtest/universe_history.pkl")
print("Exists:", pkl.exists())
if pkl.exists():
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    dates = sorted(d.keys())
    print("Date range:", dates[0], "→", dates[-1])
    print("Sample date members:", dates[100], "→", len(d[dates[100]]), "tickers")
    print("Sample tickers:", list(d[dates[100]])[:5])