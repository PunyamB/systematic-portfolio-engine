import numpy as np
from scipy.stats import norm

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    return delta, gamma

r = 0.043  # risk-free rate (~current 1-yr Treasury)

# SPY options - using May 1 expiry (2 days out, decent liquidity)
S_spy = 710.58
K_spy = 711.0
T_spy = 2 / 252  # 2 trading days to May 1

# SPY Call IV=0.1824, Put IV=0.1558
spy_call_delta, spy_call_gamma = bs_greeks(S_spy, K_spy, T_spy, r, 0.1824, "call")
spy_put_delta, spy_put_gamma = bs_greeks(S_spy, K_spy, T_spy, r, 0.1558, "put")

print("=" * 60)
print("SPY OPTIONS | Strike: 711 | Expiry: 2026-05-01")
print("=" * 60)
print(f"CALL -> Delta: {spy_call_delta:.5f}, Gamma: {spy_call_gamma:.5f}")
print(f"PUT  -> Delta: {spy_put_delta:.5f}, Gamma: {spy_put_gamma:.5f}")

# QQQ options - using May 1 expiry
S_qqq = 660.24
K_qqq = 660.0
T_qqq = 2 / 252

# QQQ Call IV=0.2734, Put IV=0.2320
qqq_call_delta, qqq_call_gamma = bs_greeks(S_qqq, K_qqq, T_qqq, r, 0.2734, "call")
qqq_put_delta, qqq_put_gamma = bs_greeks(S_qqq, K_qqq, T_qqq, r, 0.2320, "put")

print(f"\n{'=' * 60}")
print("QQQ OPTIONS | Strike: 660 | Expiry: 2026-05-01")
print("=" * 60)
print(f"CALL -> Delta: {qqq_call_delta:.5f}, Gamma: {qqq_call_gamma:.5f}")
print(f"PUT  -> Delta: {qqq_put_delta:.5f}, Gamma: {qqq_put_gamma:.5f}")

# Per-contract values (multiply by 100 shares per contract)
print(f"\n{'=' * 60}")
print("PER CONTRACT (x100)")
print("=" * 60)
print(f"SPY Call -> Delta: {spy_call_delta*100:.2f}, Gamma: {spy_call_gamma*100:.5f}")
print(f"SPY Put  -> Delta: {spy_put_delta*100:.2f}, Gamma: {spy_put_gamma*100:.5f}")
print(f"QQQ Call -> Delta: {qqq_call_delta*100:.2f}, Gamma: {qqq_call_gamma*100:.5f}")
print(f"QQQ Put  -> Delta: {qqq_put_delta*100:.2f}, Gamma: {qqq_put_gamma*100:.5f}")