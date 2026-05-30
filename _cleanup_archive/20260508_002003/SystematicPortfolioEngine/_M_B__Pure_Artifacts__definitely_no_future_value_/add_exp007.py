path = "streamlit_app.py"
with open(path, "r") as f:
    content = f.read()

# 1. Add exp007 to STRATEGY_OPTIONS
content = content.replace(
    '"exp006": "EXP006 — Extended",',
    '"exp006": "EXP006 — Extended",\n    "exp007": "EXP007 — Sliding Params",'
)

# 2. Update is_exp005 to cover exp007
content = content.replace(
    'is_exp005 = (strategy in ["exp005", "exp006"])',
    'is_exp005 = (strategy in ["exp005", "exp006", "exp007"])'
)

# 3. Update signals loader for exp007
content = content.replace(
    'fname = "signals_history_exp006.parquet" if strategy == "exp006" else ("signals_history_exp005.parquet" if strategy == "exp005" else "signals_history.parquet")',
    'fname = "signals_history_exp007.parquet" if strategy == "exp007" else ("signals_history_exp006.parquet" if strategy == "exp006" else ("signals_history_exp005.parquet" if strategy == "exp005" else "signals_history.parquet"))'
)
content = content.replace(
    'fname = "forward_returns_exp007.parquet" if strategy == "exp007" else ("forward_returns_exp006.parquet" if strategy == "exp006" else ("forward_returns_exp005.parquet" if strategy == "exp005" else "forward_returns.parquet"))',
    'fname = "forward_returns_exp007.parquet" if strategy == "exp007" else ("forward_returns_exp006.parquet" if strategy == "exp006" else ("forward_returns_exp005.parquet" if strategy == "exp005" else "forward_returns.parquet"))'
)
# In case the above didn't match (exp007 not in fwd yet), do it from exp006 base
content = content.replace(
    'fname = "forward_returns_exp006.parquet" if strategy == "exp006" else ("forward_returns_exp005.parquet" if strategy == "exp005" else "forward_returns.parquet")',
    'fname = "forward_returns_exp007.parquet" if strategy == "exp007" else ("forward_returns_exp006.parquet" if strategy == "exp006" else ("forward_returns_exp005.parquet" if strategy == "exp005" else "forward_returns.parquet"))'
)

# 4. Add EXP007 to all-combo comparison
content = content.replace(
    """        # Add EXP006
        wf = load_wf_nav_stitched('exp006', 'monthly')
        if not wf.empty:
            m2 = metrics(wf['nav'])
            m2['Strategy'] = "EXP006-monthly"
            rows.append(m2)""",
    """        # Add EXP006
        wf = load_wf_nav_stitched('exp006', 'monthly')
        if not wf.empty:
            m2 = metrics(wf['nav'])
            m2['Strategy'] = "EXP006-monthly"
            rows.append(m2)
        # Add EXP007
        wf = load_wf_nav_stitched('exp007', 'monthly')
        if not wf.empty:
            m2 = metrics(wf['nav'])
            m2['Strategy'] = "EXP007-monthly"
            rows.append(m2)"""
)

# 5. Add EXP007 color
content = content.replace(
    "'EXP006-monthly': '#FF6B6B'}",
    "'EXP006-monthly': '#FF6B6B', 'EXP007-monthly': '#4ECDC4'}"
)

# 6. Update sidebar training window for exp007
content = content.replace(
    '"Expanding (1997-based)" if strategy == "exp006" else "Expanding (2009-based)"',
    '"Sliding 5yr (params) + Expanding (IC)" if strategy == "exp007" else ("Expanding (1997-based)" if strategy == "exp006" else "Expanding (2009-based)")'
)

# 7. Update sidebar walk-forward windows for exp007
content = content.replace(
    '"17 OOS + Holdout" if strategy == "exp006" else "9 OOS + Holdout"',
    '"17 OOS + Holdout" if strategy in ["exp006", "exp007"] else "9 OOS + Holdout"'
)

with open(path, "w") as f:
    f.write(content)

# Verify
import py_compile
try:
    py_compile.compile(path, doraise=True)
    print("SYNTAX OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")

checks = [
    ("exp007 in options", '"exp007"' in content),
    ("exp007 in is_exp005", '"exp007"' in content),
    ("exp007 combo", "EXP007-monthly" in content),
    ("exp007 color", "'EXP007-monthly'" in content),
    ("exp007 signals", "signals_history_exp007" in content),
]
for label, ok in checks:
    print(f"  {label}: {'OK' if ok else 'MISSING'}")