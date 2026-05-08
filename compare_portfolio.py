import os
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()
api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    base_url="https://paper-api.alpaca.markets"
)

# Alpaca positions
positions = api.list_positions()
alpaca = []
for p in positions:
    alpaca.append({
        "ticker": p.symbol,
        "shares": int(float(p.qty)),
        "market_value": float(p.market_value),
        "current_price": float(p.current_price),
        "avg_entry": float(p.avg_entry_price),
        "unrealized_pnl": float(p.unrealized_pl),
    })
alpaca_df = pd.DataFrame(alpaca).sort_values("ticker").reset_index(drop=True)

# Alpaca account
acct = api.get_account()
alpaca_cash = float(acct.cash)
alpaca_value = float(acct.portfolio_value)
alpaca_equity = alpaca_df["market_value"].sum()

# Internal portfolio
internal = pd.read_parquet("data/processed/portfolio.parquet")
internal = internal[["ticker", "shares", "market_value", "cost_basis", "unrealized_pnl"]].copy()
internal["shares"] = internal["shares"].astype(int)
internal = internal.sort_values("ticker").reset_index(drop=True)
internal_equity = internal["market_value"].sum()

# Internal NAV
nav_h = pd.read_parquet("data/processed/nav_history.parquet")
internal_nav = float(nav_h.iloc[-1]["nav"])
internal_cash = internal_nav - internal_equity

# Dashboard export
dash = pd.read_csv("data_export.csv")

print("=" * 70)
print("  ALPACA vs INTERNAL vs DASHBOARD")
print("=" * 70)

print(f"\n  ACCOUNT SUMMARY")
print(f"  {'':30} {'Alpaca':>14} {'Internal':>14}")
print(f"  {'-'*30} {'-'*14} {'-'*14}")
print(f"  {'Portfolio Value':30} {alpaca_value:>14,.2f} {internal_nav:>14,.2f}")
print(f"  {'Equity (positions)':30} {alpaca_equity:>14,.2f} {internal_equity:>14,.2f}")
print(f"  {'Cash':30} {alpaca_cash:>14,.2f} {internal_cash:>14,.2f}")
print(f"  {'Position Count':30} {len(alpaca_df):>14} {len(internal):>14}")

print(f"\n  POSITION COMPARISON")
print(f"  {'Ticker':<8} {'A.Shares':>8} {'I.Shares':>8} {'Match':>6} "
      f"{'A.MktVal':>12} {'I.MktVal':>12} {'Diff':>10}")
print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*12} {'-'*12} {'-'*10}")

merged = alpaca_df.merge(internal, on="ticker", how="outer", suffixes=("_a", "_i")).fillna(0)
merged = merged.sort_values("ticker")

mismatches = 0
for _, r in merged.iterrows():
    a_sh = int(r.get("shares_a", 0))
    i_sh = int(r.get("shares_i", 0))
    a_mv = float(r.get("market_value_a", 0))
    i_mv = float(r.get("market_value_i", 0))
    match = "OK" if a_sh == i_sh else "DIFF"
    diff = a_mv - i_mv
    if match == "DIFF":
        mismatches += 1
    print(f"  {r['ticker']:<8} {a_sh:>8} {i_sh:>8} {match:>6} "
          f"{a_mv:>12,.2f} {i_mv:>12,.2f} {diff:>10,.2f}")

# Dashboard weights check
print(f"\n  DASHBOARD WEIGHTS (from export)")
print(f"  {'Ticker':<8} {'Dash Wt':>8} {'Alpaca Wt':>10} {'Match':>6}")
print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

dash_mismatches = 0
for _, r in dash.iterrows():
    t = r["ticker"]
    dw = float(r["weight"])
    a_row = alpaca_df[alpaca_df["ticker"] == t]
    aw = float(a_row["market_value"].values[0]) / alpaca_value if not a_row.empty else 0
    match = "OK" if abs(dw - aw) < 0.005 else "DIFF"
    if match == "DIFF":
        dash_mismatches += 1
    print(f"  {t:<8} {dw:>7.2%} {aw:>9.2%} {match:>6}")

print(f"\n  SUMMARY")
print(f"  Position mismatches (shares): {mismatches}")
print(f"  Dashboard weight mismatches:  {dash_mismatches}")
print(f"  Cash discrepancy:             ${alpaca_cash - internal_cash:,.2f}")
print(f"  NAV discrepancy:              ${alpaca_value - internal_nav:,.2f}")
