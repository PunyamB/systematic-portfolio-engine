path = "experiments/exp008_sweep/run_exp008.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '    pd.concat(all_trades, ignore_index=True).to_parquet(RESULTS_DIR / "all_trades.parquet")'
new = """    try:
        trades_combined = pd.concat(all_trades, ignore_index=True)
        trades_combined["window"] = trades_combined["window"].astype(str)
        trades_combined.to_parquet(RESULTS_DIR / "all_trades.parquet")
    except Exception as e:
        print(f"[exp008] WARNING: Could not save all_trades: {e}")"""

content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed: wrapped all_trades save in try/except with str cast")