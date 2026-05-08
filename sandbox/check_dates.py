import pandas as pd

prices = pd.read_parquet("data/backtest/prices.parquet")
print("Price date range:", prices["date"].min(), "→", prices["date"].max())
print("Tickers:", prices["ticker"].nunique())

constituents = pd.read_parquet("data/backtest/constituent_history.parquet")
print("Constituent columns:", constituents.columns.tolist())
print("Constituent rows:", len(constituents))
print(constituents.head(3))