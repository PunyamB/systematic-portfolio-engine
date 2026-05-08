# Quick check: is it stuck on synthetic or real data?
import time
print("Testing synthetic speed...")
from regime_switching.m2_fixed_ms import synthetic_recovery_test
t0 = time.time()
result = synthetic_recovery_test(K=2, n_reps=3, n_obs=2000)
print(f"2-state 3 reps took {time.time()-t0:.1f}s")
print(f"Result: {result}")
