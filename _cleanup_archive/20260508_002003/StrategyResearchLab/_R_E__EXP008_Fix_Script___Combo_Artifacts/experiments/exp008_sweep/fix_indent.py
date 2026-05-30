path = "experiments/exp008_sweep/run_exp008.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

bad = """    if all_trades:
        try:
        trades_combined = pd.concat(all_trades, ignore_index=True)
        trades_combined["window"] = trades_combined["window"].astype(str)
        trades_combined.to_parquet(RESULTS_DIR / "all_trades.parquet")
    except Exception as e:
        print(f"[exp008] WARNING: Could not save all_trades: {e}")"""

good = """    if all_trades:
        try:
            trades_combined = pd.concat(all_trades, ignore_index=True)
            trades_combined["window"] = trades_combined["window"].astype(str)
            trades_combined.to_parquet(RESULTS_DIR / "all_trades.parquet")
        except Exception as e:
            print(f"[exp008] WARNING: Could not save all_trades: {e}")"""

content = content.replace(bad, good)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

import py_compile
try:
    py_compile.compile(path, doraise=True)
    print("SYNTAX OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")