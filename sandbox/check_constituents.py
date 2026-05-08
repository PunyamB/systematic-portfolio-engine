import pandas as pd

df = pd.read_parquet("data/backtest/constituent_history.parquet")
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.head(10).to_string())
print("---")
print(df.tail(10).to_string())