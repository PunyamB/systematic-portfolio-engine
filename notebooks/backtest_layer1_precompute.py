# notebooks/backtest_layer1_precompute.py
# LAYER 1 — Precomputation (run once, ~45-90 mins)
#
# For every rebalance date in the backtest window, computes and saves:
#   - Universe membership (point-in-time S&P 500 constituents)
#   - All 11 signal scores per ticker
#   - Covariance matrix (Ledoit-Wolf, 252-day rolling)
#   - Forward returns per ticker (1-month and 1-quarter ahead closes)
#     needed for IC computation in Layer 2
#
# Outputs saved to data/backtest/precomputed/
# Resumable — skips already-computed rebalance dates on re-run.
#
# Run: python notebooks/backtest_layer1_precompute.py

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import get_config

cfg = get_config()

# ============================================================
# CONFIG
# ============================================================

BACKTEST_START = pd.Timestamp("2009-01-01")
BACKTEST_END   = pd.Timestamp("2026-01-01")
FREQUENCIES    = ["monthly", "quarterly"]
MIN_SIGNALS    = 6
COV_LOOKBACK   = 252

BACKTEST_DIR   = Path("data/backtest")
PRECOMP_DIR    = Path("data/backtest/precomputed")
PRECOMP_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_MAP = {
    "Consumer Cyclical":  "Consumer Discretionary",
    "Basic Materials":    "Materials",
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
}

print("[layer1] ================================================")
print(f"[layer1] Precomputation: {BACKTEST_START.date()} to {BACKTEST_END.date()}")
print(f"[layer1] Output: {PRECOMP_DIR}")
print("[layer1] ================================================\n")


# ============================================================
# DATA LOADING
# ============================================================

print("[layer1] Loading backtest data...")
prices_all      = pd.read_parquet(BACKTEST_DIR / "prices.parquet")
financials_all  = pd.read_parquet(BACKTEST_DIR / "financials.parquet")
key_metrics_all = pd.read_parquet(BACKTEST_DIR / "key_metrics.parquet")
const_history   = pd.read_parquet(BACKTEST_DIR / "constituent_history.parquet")
constituents_cur = pd.read_parquet("data/raw/constituents.parquet")

for df in [prices_all, financials_all, key_metrics_all, const_history]:
    df["date"] = pd.to_datetime(df["date"])

constituents_cur["sector"] = constituents_cur["sector"].replace(SECTOR_MAP)
prices_all = prices_all.sort_values(["ticker", "date"])

print(f"[layer1] Prices:      {len(prices_all):>10,} rows")
print(f"[layer1] Financials:  {len(financials_all):>10,} rows")
print(f"[layer1] Key metrics: {len(key_metrics_all):>10,} rows\n")


# ============================================================
# TRADING CALENDAR + REBALANCE DATES
# ============================================================

# Use AAPL for trading calendar — large-cap, continuous history since 2009
# SPY is not in backtest prices (ETF, not S&P 500 constituent)
# SPY benchmark comparison handled separately in Layer 3
aapl = prices_all[prices_all["ticker"] == "AAPL"].sort_values("date")
trading_days = aapl[
    (aapl["date"] >= BACKTEST_START) & (aapl["date"] < BACKTEST_END)
]["date"].tolist()

def get_rebalance_dates(trading_days: list, frequency: str) -> list:
    seen, dates = set(), []
    for day in trading_days:
        period = (day.year, day.month) if frequency == "monthly" else (day.year, (day.month-1)//3+1)
        if period not in seen:
            seen.add(period)
            dates.append(day)
    return dates

all_rebalance_dates = sorted(set(
    d for freq in FREQUENCIES for d in get_rebalance_dates(trading_days, freq)
))
print(f"[layer1] Total unique rebalance dates: {len(all_rebalance_dates)}")

# Load already-computed dates
done_file  = PRECOMP_DIR / "done_dates.txt"
done_dates = set()
if done_file.exists():
    with open(done_file) as f:
        done_dates = set(pd.to_datetime(line.strip()) for line in f if line.strip())

remaining = [d for d in all_rebalance_dates if d not in done_dates]
print(f"[layer1] Already done: {len(done_dates)} | Remaining: {len(remaining)}\n")


# ============================================================
# UNIVERSE RECONSTRUCTION
# ============================================================

def build_universe_on_date(as_of: pd.Timestamp) -> list:
    members = set(constituents_cur["ticker"].tolist())
    future  = const_history[const_history["date"] > as_of]
    for _, row in future.iterrows():
        added   = row.get("symbol")
        removed = row.get("removedTicker")
        if pd.notna(added) and added in members:
            members.discard(added)
        if pd.notna(removed) and removed not in members:
            members.add(removed)
    return sorted(list(members))


# ============================================================
# SIGNAL COMPUTATION FUNCTIONS
# ============================================================

def _zscore_winsorize(series: pd.Series) -> pd.Series:
    mean, std = series.mean(), series.std()
    if std == 0:
        return series * 0
    return ((series - mean) / std).clip(-3, 3)


def compute_signals(as_of: pd.Timestamp, universe: list) -> pd.DataFrame:
    prices_asof = prices_all[prices_all["date"] <= as_of]
    fin_asof    = financials_all[financials_all["date"] <= as_of]
    km_asof     = key_metrics_all[key_metrics_all["date"] <= as_of]
    cons        = constituents_cur.copy()

    raw = {}

    # --- Momentum 12-1 ---
    for ticker, px in prices_asof.groupby("ticker"):
        if ticker not in universe: continue
        px = px.sort_values("date")
        if len(px) < 252: continue
        p0, p12 = float(px["close"].iloc[-21]), float(px["close"].iloc[-252])
        if p12 == 0: continue
        raw.setdefault("momentum_12_1", {})[ticker] = (p0 / p12) - 1.0

    # --- Short-term reversal ---
    for ticker, px in prices_asof.groupby("ticker"):
        if ticker not in universe: continue
        px = px.sort_values("date")
        if len(px) < 6: continue
        p0, p5 = float(px["close"].iloc[-1]), float(px["close"].iloc[-6])
        if p5 == 0: continue
        raw.setdefault("short_term_reversal", {})[ticker] = -((p0 / p5) - 1.0)

    # --- RSI extremes ---
    period = 14
    for ticker, px in prices_asof.groupby("ticker"):
        if ticker not in universe: continue
        px = px.sort_values("date")
        if len(px) < period + 1: continue
        d  = px["close"].diff().dropna()
        ag = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        al = (-d).clip(lower=0).ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        rsi = 100.0 if al == 0 else 100.0 - (100.0 / (1.0 + ag / al))
        raw.setdefault("rsi_extremes", {})[ticker] = (50.0 - rsi) / 50.0

    # --- Earnings momentum ---
    if "eps" in fin_asof.columns:
        for ticker, fin in fin_asof.groupby("ticker"):
            if ticker not in universe: continue
            fin = fin.sort_values("date")
            if len(fin) < 3: continue
            eps = fin["eps"].values
            q0, q1, q2 = float(eps[-1]), float(eps[-2]), float(eps[-3])
            if q1 == 0 or q2 == 0: continue
            acc = ((q0 - q1) / abs(q1)) - ((q1 - q2) / abs(q2))
            raw.setdefault("earnings_momentum", {})[ticker] = acc

    # --- Earnings accruals ---
    req = ["netIncome", "operatingCashFlow", "totalAssets"]
    if all(c in fin_asof.columns for c in req):
        for ticker, fin in fin_asof.groupby("ticker"):
            if ticker not in universe: continue
            fin = fin.sort_values("date")
            if len(fin) < 4: continue
            row = fin.iloc[-1]
            if row["totalAssets"] <= 0: continue
            ratio = (row["netIncome"] - row["operatingCashFlow"]) / row["totalAssets"]
            raw.setdefault("earnings_accruals", {})[ticker] = -ratio

    # --- ROE stability ---
    if "roe" in fin_asof.columns:
        for ticker, fin in fin_asof.groupby("ticker"):
            if ticker not in universe: continue
            fin = fin.sort_values("date")
            if len(fin) < 8: continue
            std = fin["roe"].tail(8).dropna().std()
            if pd.isna(std): continue
            raw.setdefault("roe_stability", {})[ticker] = 1.0 / (std + 1e-6)

    # --- Gross margin trend ---
    if "grossMargin" in fin_asof.columns:
        for ticker, fin in fin_asof.groupby("ticker"):
            if ticker not in universe: continue
            fin = fin.sort_values("date")
            if len(fin) < 4: continue
            margins = fin["grossMargin"].tail(4).dropna().values
            if len(margins) < 4: continue
            slope = np.polyfit(np.arange(len(margins), dtype=float), margins, 1)[0]
            raw.setdefault("gross_margin_trend", {})[ticker] = slope

    # --- Piotroski ---
    for ticker, fin in fin_asof.groupby("ticker"):
        if ticker not in universe: continue
        fin = fin.sort_values("date")
        if len(fin) < 8: continue
        c, p  = fin.iloc[-1].to_dict(), fin.iloc[-2].to_dict()
        score = 0
        if c.get("totalAssets", 0) > 0:
            if c.get("netIncome", 0) / c["totalAssets"] > 0: score += 1
        if c.get("operatingCashFlow", 0) > 0: score += 1
        if c.get("totalAssets", 0) > 0 and p.get("totalAssets", 0) > 0:
            if c.get("netIncome", 0)/c["totalAssets"] > p.get("netIncome", 0)/p["totalAssets"]: score += 1
        if c.get("operatingCashFlow", 0) > c.get("netIncome", 0): score += 1
        if c.get("totalAssets", 0) > 0 and p.get("totalAssets", 0) > 0:
            if c.get("longTermDebt", 0)/c["totalAssets"] < p.get("longTermDebt", 0)/p["totalAssets"]: score += 1
        if c.get("totalCurrentLiabilities", 0) > 0 and p.get("totalCurrentLiabilities", 0) > 0:
            if c.get("totalCurrentAssets", 0)/c["totalCurrentLiabilities"] > p.get("totalCurrentAssets", 0)/p["totalCurrentLiabilities"]: score += 1
        sp, sc = p.get("weightedAverageShsOut", 0), c.get("weightedAverageShsOut", 0)
        if sp > 0 and sc <= sp * 1.02: score += 1
        if c.get("grossMargin") is not None and p.get("grossMargin") is not None:
            if c["grossMargin"] > p["grossMargin"]: score += 1
        if c.get("totalAssets", 0) > 0 and p.get("totalAssets", 0) > 0:
            if c.get("revenue", 0)/c["totalAssets"] > p.get("revenue", 0)/p["totalAssets"]: score += 1
        raw.setdefault("piotroski", {})[ticker] = float(score)

    # --- Sector z-scores: PE, EV/EBITDA ---
    for col, sig_name in [("peRatio", "pe_zscore"), ("evToEbitda", "ev_ebitda_zscore")]:
        if col not in km_asof.columns: continue
        counts  = km_asof.groupby("ticker")[col].count()
        eligible = counts[counts >= 4].index
        latest  = (km_asof[km_asof["ticker"].isin(eligible)].sort_values("date")
                   .groupby("ticker").last().reset_index()[["ticker", col]].dropna())
        if col == "peRatio":
            latest = latest[latest[col] > 0]
        else:
            latest = latest[(latest[col] > 0) & (latest[col] < 200)]
        merged = latest.merge(cons[["ticker", "sector"]], on="ticker", how="left").dropna(subset=["sector"])
        for sector, grp in merged.groupby("sector"):
            if len(grp) < 2: continue
            mean, std = grp[col].mean(), grp[col].std()
            if std == 0: continue
            for _, row in grp.iterrows():
                if row["ticker"] not in universe: continue
                raw.setdefault(sig_name, {})[row["ticker"]] = -((row[col] - mean) / std)

    # --- PB z-score ---
    req2 = ["totalStockholdersEquity", "weightedAverageShsOut"]
    if all(c in fin_asof.columns for c in req2):
        counts   = fin_asof.groupby("ticker")["date"].count()
        eligible = counts[counts >= 4].index
        lp = (prices_asof.sort_values("date").groupby("ticker").last()
              .reset_index()[["ticker", "close"]])
        lf = (fin_asof[fin_asof["ticker"].isin(eligible)].sort_values("date")
              .groupby("ticker").last().reset_index()[["ticker"] + req2])
        merged = lp.merge(lf, on="ticker", how="inner")
        merged = merged[(merged["totalStockholdersEquity"] > 0) & (merged["weightedAverageShsOut"] > 0)]
        if not merged.empty:
            merged["pb"] = (merged["close"] * merged["weightedAverageShsOut"]) / merged["totalStockholdersEquity"]
            merged = merged[merged["pb"] < 50]
            merged = merged.merge(cons[["ticker", "sector"]], on="ticker", how="left").dropna(subset=["sector"])
            for sector, grp in merged.groupby("sector"):
                if len(grp) < 2: continue
                mean, std = grp["pb"].mean(), grp["pb"].std()
                if std == 0: continue
                for _, row in grp.iterrows():
                    if row["ticker"] not in universe: continue
                    raw.setdefault("pb_zscore", {})[row["ticker"]] = -((row["pb"] - mean) / std)

    # --- Build combined DataFrame ---
    SIGNAL_COLS = [
        "momentum_12_1", "earnings_momentum", "pe_zscore", "pb_zscore",
        "ev_ebitda_zscore", "roe_stability", "gross_margin_trend",
        "piotroski", "earnings_accruals", "short_term_reversal", "rsi_extremes",
    ]

    all_tickers = set(t for d in raw.values() for t in d.keys())
    if not all_tickers:
        return pd.DataFrame()

    combined = pd.DataFrame({"ticker": sorted(list(all_tickers))})

    for sig in SIGNAL_COLS:
        if sig in raw:
            df_sig = pd.DataFrame(list(raw[sig].items()), columns=["ticker", sig])
            vals   = df_sig[sig]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                df_sig[sig] = ((vals - mean) / std).clip(-3, 3)
            combined = combined.merge(df_sig, on="ticker", how="left")

    sig_cols_present          = [c for c in SIGNAL_COLS if c in combined.columns]
    combined["n_signals"]     = combined[sig_cols_present].notna().sum(axis=1)
    combined                  = combined[combined["n_signals"] >= MIN_SIGNALS].copy()
    if combined.empty:
        return pd.DataFrame()

    combined["composite_score"] = combined[sig_cols_present].mean(axis=1)
    combined["composite_rank"]  = combined["composite_score"].rank(ascending=False)
    combined["date"]            = as_of

    return combined[["ticker", "date", "composite_score", "composite_rank"] + sig_cols_present]


# ============================================================
# COVARIANCE COMPUTATION
# ============================================================

def compute_covariance(as_of: pd.Timestamp, tickers: list) -> pd.DataFrame:
    """
    Computes Ledoit-Wolf covariance matrix for given tickers as of date.
    Returns DataFrame (tickers x tickers). Empty if insufficient data.
    """
    px = (prices_all[
            (prices_all["ticker"].isin(tickers)) &
            (prices_all["date"] <= as_of)
          ]
          .sort_values("date")
          .groupby("ticker")
          .tail(COV_LOOKBACK + 1)
          .pivot_table(index="date", columns="ticker", values="close")
          .pct_change()
          .dropna(how="all"))

    valid   = px.columns[px.notna().sum() >= COV_LOOKBACK // 2].tolist()
    px_clean = px[valid].dropna()

    if len(px_clean) < 30 or len(valid) < 5:
        return pd.DataFrame()

    try:
        sigma = LedoitWolf().fit(px_clean.values).covariance_ * 252
        return pd.DataFrame(sigma, index=valid, columns=valid)
    except Exception:
        return pd.DataFrame()


# ============================================================
# FORWARD RETURNS COMPUTATION
# ============================================================

def compute_forward_returns(as_of: pd.Timestamp, universe: list,
                              trading_days: list) -> pd.DataFrame:
    """
    Computes 1-month and 1-quarter ahead forward returns from as_of close.
    Uses actual future closes from prices_all.
    Returns DataFrame: ticker, fwd_1m, fwd_1q
    """
    idx = trading_days.index(as_of) if as_of in trading_days else None
    if idx is None:
        return pd.DataFrame()

    # 1-month ahead: ~21 trading days
    idx_1m = min(idx + 21, len(trading_days) - 1)
    # 1-quarter ahead: ~63 trading days
    idx_1q = min(idx + 63, len(trading_days) - 1)

    date_1m = trading_days[idx_1m]
    date_1q = trading_days[idx_1q]

    def get_close(date):
        return (prices_all[prices_all["date"] == date]
                .set_index("ticker")["close"])

    close_now = get_close(as_of)
    close_1m  = get_close(date_1m)
    close_1q  = get_close(date_1q)

    rows = []
    for ticker in universe:
        try:
            p0   = float(close_now[ticker])
            p_1m = float(close_1m[ticker])
            p_1q = float(close_1q[ticker])
            if p0 > 0:
                rows.append({
                    "ticker": ticker,
                    "fwd_1m": (p_1m / p0) - 1.0,
                    "fwd_1q": (p_1q / p0) - 1.0,
                })
        except (KeyError, TypeError):
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ============================================================
# MAIN PRECOMPUTATION LOOP
# ============================================================

trading_days_list = [d for d in trading_days]  # list for index lookup

all_signals_list  = []
all_cov_dict      = {}
all_fwd_list      = []
all_universe_dict = {}

for i, rebal_date in enumerate(remaining):
    print(f"[layer1] {rebal_date.date()} ({i+1}/{len(remaining)}) | ", end="", flush=True)

    universe = build_universe_on_date(rebal_date)
    all_universe_dict[rebal_date] = universe

    signals = compute_signals(rebal_date, universe)
    if not signals.empty:
        all_signals_list.append(signals)
        print(f"signals={len(signals)} | ", end="", flush=True)
    else:
        print(f"signals=0 | ", end="", flush=True)

    top_tickers = signals["ticker"].tolist() if not signals.empty else universe[:100]
    cov = compute_covariance(rebal_date, top_tickers)
    if not cov.empty:
        all_cov_dict[rebal_date] = cov
        print(f"cov={cov.shape[0]}x{cov.shape[1]} | ", end="", flush=True)

    fwd = compute_forward_returns(rebal_date, universe, trading_days_list)
    if not fwd.empty:
        fwd["date"] = rebal_date
        all_fwd_list.append(fwd)
        print(f"fwd={len(fwd)}", flush=True)
    else:
        print("fwd=0", flush=True)

    done_dates.add(rebal_date)

    # Checkpoint every 10 rebalance dates
    if (i + 1) % 10 == 0:
        # Save signals so far
        if all_signals_list:
            existing_sig_file = PRECOMP_DIR / "signals_history.parquet"
            new_sig = pd.concat(all_signals_list, ignore_index=True)
            if existing_sig_file.exists():
                existing = pd.read_parquet(existing_sig_file)
                new_sig  = pd.concat([existing, new_sig], ignore_index=True).drop_duplicates(subset=["ticker", "date"])
            new_sig.to_parquet(existing_sig_file, index=False)
            all_signals_list = []

        # Save forward returns so far
        if all_fwd_list:
            existing_fwd_file = PRECOMP_DIR / "forward_returns.parquet"
            new_fwd = pd.concat(all_fwd_list, ignore_index=True)
            if existing_fwd_file.exists():
                existing = pd.read_parquet(existing_fwd_file)
                new_fwd  = pd.concat([existing, new_fwd], ignore_index=True).drop_duplicates(subset=["ticker", "date"])
            new_fwd.to_parquet(existing_fwd_file, index=False)
            all_fwd_list = []

        # Save done dates
        with open(done_file, "w") as f:
            f.write("\n".join(str(d.date()) for d in done_dates))

        print(f"[layer1] Checkpoint saved at {rebal_date.date()}")

# Final save
if all_signals_list:
    existing_sig_file = PRECOMP_DIR / "signals_history.parquet"
    new_sig = pd.concat(all_signals_list, ignore_index=True)
    if existing_sig_file.exists():
        existing = pd.read_parquet(existing_sig_file)
        new_sig  = pd.concat([existing, new_sig], ignore_index=True).drop_duplicates(subset=["ticker", "date"])
    new_sig.to_parquet(existing_sig_file, index=False)

if all_fwd_list:
    existing_fwd_file = PRECOMP_DIR / "forward_returns.parquet"
    new_fwd = pd.concat(all_fwd_list, ignore_index=True)
    if existing_fwd_file.exists():
        existing = pd.read_parquet(existing_fwd_file)
        new_fwd  = pd.concat([existing, new_fwd], ignore_index=True).drop_duplicates(subset=["ticker", "date"])
    new_fwd.to_parquet(existing_fwd_file, index=False)

# Save covariance matrices as a single pickle (dict of DataFrames)
import pickle
cov_file = PRECOMP_DIR / "covariance_matrices.pkl"
if cov_file.exists():
    with open(cov_file, "rb") as f:
        existing_cov = pickle.load(f)
    existing_cov.update(all_cov_dict)
    all_cov_dict = existing_cov
with open(cov_file, "wb") as f:
    pickle.dump(all_cov_dict, f)

# Save universe dict
universe_file = PRECOMP_DIR / "universe_history.pkl"
if universe_file.exists():
    with open(universe_file, "rb") as f:
        existing_uni = pickle.load(f)
    existing_uni.update(all_universe_dict)
    all_universe_dict = existing_uni
with open(universe_file, "wb") as f:
    pickle.dump(all_universe_dict, f)

with open(done_file, "w") as f:
    f.write("\n".join(str(d.date()) for d in done_dates))

print("\n[layer1] ================================================")
print("[layer1] Precomputation complete.")
sig_count = pd.read_parquet(PRECOMP_DIR / "signals_history.parquet") if (PRECOMP_DIR / "signals_history.parquet").exists() else pd.DataFrame()
fwd_count = pd.read_parquet(PRECOMP_DIR / "forward_returns.parquet") if (PRECOMP_DIR / "forward_returns.parquet").exists() else pd.DataFrame()
print(f"[layer1] Signals history: {len(sig_count)} rows | {sig_count['date'].nunique() if not sig_count.empty else 0} dates")
print(f"[layer1] Forward returns: {len(fwd_count)} rows | {fwd_count['date'].nunique() if not fwd_count.empty else 0} dates")
print(f"[layer1] Covariance matrices: {len(all_cov_dict)} dates")
print(f"[layer1] Universe snapshots: {len(all_universe_dict)} dates")
print("[layer1] Ready for Layer 2.")
print("[layer1] ================================================")