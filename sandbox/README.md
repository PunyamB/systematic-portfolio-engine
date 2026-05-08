# Sandbox Environment

This directory (`sandbox/`) is dedicated to backtesting and mathematical experiments. 

**Rules:**
1. Do not modify production code from this directory.
2. Scripts here should only *read* from `/data/` and its subdirectories, or utilize existing components and classes functionally.
3. Outputs (like strategy performance metrics or simulated proposed trades) should be dumped into `sandbox/results/` to avoid corrupting live trading pipelines.
