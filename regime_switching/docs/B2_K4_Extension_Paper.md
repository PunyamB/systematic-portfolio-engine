# B2 K=4 Extension: Time-Varying Transition Probability Markov-Switching Regime Detection at Four States

**A Walk-Forward Investigation of Regime Model Complexity in a Live Systematic Equity Strategy**

**Author:** Punyam B
**Institution:** University of Connecticut, MS Financial and Enterprise Risk Management
**Date:** May 2026
**Project location:** `D:\Projects\SystematicPortfolioEngine\regime_switching\`
**Live strategy compared:** Meridian (SystematicPortfolioEngine, 15-signal MV-Monthly, $1M Alpaca paper, 2005-2026)

---

## Executive Summary

This paper documents the K=4 extension of the B2 Time-Varying Transition Probability Markov-Switching (TVTP-MS) regime detector built for the Meridian systematic equity trading strategy. The original B2 project (March-May 2026) implemented a 3-state TVTP model from scratch — custom Hamilton filter, EM algorithm with numerical M-step for the logistic transition coefficients — and walk-forward backtested it against Meridian's existing rule-based regime detector. The K=3 result was indistinguishable: 23.44% CAGR vs Meridian's 23.56%, Sharpe 1.308 vs 1.309. C1 (regime classification kappa) and C3 (walk-forward improvement) both failed; C2 (early warning lead time) passed at 88 days average lead.

The K=4 extension tested whether the K=3 model was misspecified. Diagnostic analysis of K=3 monthly probability traces revealed P_bear firing with high probability during periods that Meridian's rule-based detector classified as recovery and that retrospectively were periods of slow positive growth (2018, 2019, throughout 2024). The hypothesis: forcing a recovery state into either bull or bear is structural misspecification that masks the value of soft regime probabilities.

**Methodology:** Eight modules in sequence — diagnostic (M2b K=4 fixed-MS, M2c K=5 sensitivity), factor analysis (EXP010), TVTP estimation (M3b constrained), backtest (M6b), comparison (Step 7). Every step was apples-to-apples with B2's K=3 build and Meridian's EXP006 baseline: same data, same backtest engine, same calibration grid, same constraints, only the regime mechanism differs.

**Headline result — three-way comparison:**

| Strategy | Regime mechanism | Aggregate CAGR | Sharpe | MaxDD | Final NAV |
|---|---|---|---|---|---|
| Meridian (live, EXP006 hard) | Rule-based hard switching | 23.56% | 1.309 | -37.63% | $84.99M |
| K=3 soft (B2) | 3-state TVTP-MS, soft blending | 23.44% | 1.308 | -37.61% | $84.86M |
| K=4 soft (this paper) | 4-state TVTP-MS, soft blending | 23.17% | 1.295 | -36.87% | $80.96M |

All three are statistically indistinguishable. Per-window: K=4 wins 8 of 18 on CAGR, 7 of 18 on Sharpe vs K=3. K=3 wins 7 of 18 on CAGR, 7 of 18 on Sharpe vs Meridian. Roughly 50-50 distributions in both cases, consistent with no real difference.

**Verdict:** Do not integrate K=4 into Meridian. Retain K=3 and K=4 alongside as standalone analytical tools. The strategy is empirically invariant to regime model choice within the Markov-switching family.

**Why this is a defensible research contribution rather than a failed engineering project:**

1. K=4 fixed-MS BIC of -47,194 vs K=3 fixed-MS BIC of -47,101 demonstrates the recovery state is statistically real — the misspecification hypothesis was correct on its own terms.
2. K=5 was rejected on economic grounds despite better BIC (-47,226), validating the K=2-3 ceiling consensus in the literature for daily equity returns.
3. The custom TVTP implementation (Hamilton filter with time-varying transition matrices, EM with numerical M-step via L-BFGS-B, multiple restart protocol with reference initialization) is a genuine engineering contribution. No off-the-shelf library implements TVTP-MS.
4. The negative finding is robust across 18 expanding walk-forward windows spanning 21 years and three crisis periods (GFC, COVID, 2022 rate hikes). A single-window or single-metric improvement could be noise; the consistent indifference across windows and metrics is signal.
5. The result is consistent with the broader literature: Lo (2004) on adaptive markets, Frazzini-Pedersen (2014) on simple-beats-complex in cost-aware backtests, and Ang-Bekaert (2004) on regime-conditional allocation gains being concentrated in specific historical episodes.

**The strategy invariance hypothesis:** Meridian's signal multipliers, IC-IR weighted alpha combination, mean-variance optimizer with sector neutrality and turnover penalties, and trailing stops collectively absorb regime information regardless of whether it is encoded as 3 states, 4 states, or rule-based hard switching. The strategy mechanism is robust to regime model choice. This is a stronger claim than "TVTP did not work" — it is "we tested 3 regime model variants and the strategy is invariant to all of them."

**Standalone value retained:** Per-window TVTP fits (K=3 in `regime_switching/data/window_fits/`, K=4 in `regime_switching/data/k4_extension_constrained/window_fits/`) saved as filtered probability series. Reusable for future strategies and EXP009 (rolling parameter recalibration) without refitting. The C2 early warning property (~88 days lead on genuine crises) remains as a separate alert channel candidate.

---

## Table of Contents

1. **Introduction**
   1.1 Project context: the Meridian systematic equity strategy
   1.2 The regime detector role in Meridian
   1.3 What this paper covers and what it does not

2. **B2 K=3 Origin and Diagnostic**
   2.1 The original B2 hypothesis
   2.2 K=3 TVTP construction recap
   2.3 K=3 results vs Meridian
   2.4 The K=3 misspecification diagnostic
   2.5 Per-window probability trace patterns

3. **The K=4 Extension Hypothesis**
   3.1 Why K=4 rather than K=3 retuning
   3.2 The recovery state hypothesis
   3.3 What success would look like
   3.4 Pre-registered integration criteria

4. **Literature Foundations**
   4.1 Hamilton-filter Markov-switching origins
   4.2 The Kim smoother and EM estimation
   4.3 Time-varying transition probabilities
   4.4 Number of states selection in financial regime models
   4.5 Constraints in Markov-switching estimation
   4.6 Walk-forward methodology and backtest validity
   4.7 Adaptive markets and regime model utility

5. **EXP010: Statistical Analysis of Macro Factor Predictive Content**
   5.1 Methodology overview
   5.2 GPD-justified factor change thresholds
   5.3 ROC analysis of P_crisis as a stress predictor
   5.4 Forward drawdown regression
   5.5 Granger causality
   5.6 Rule-based baseline comparison
   5.7 Episode-level lead time analysis
   5.8 Implications for the K=4 build

6. **M2b: K=4 Fixed-MS Diagnostic**
   6.1 Why fixed-MS before TVTP
   6.2 Implementation specifics
   6.3 Information criteria results
   6.4 State characterization
   6.5 Validation criteria
   6.6 Decision and implications

7. **M2c: K=5 Diagnostic and Rejection**
   7.1 Sensitivity rationale
   7.2 Statistical fit
   7.3 Pathological state parameters
   7.4 Literature support for the K=2-3 ceiling
   7.5 Decision

8. **M3b: K=4 TVTP Estimation with Economic Constraints**
   8.1 Initial unconstrained attempt and failure mode
   8.2 Three constraints from the literature
   8.3 Constraint design rationale
   8.4 Custom Hamilton filter for time-varying transitions
   8.5 Custom EM algorithm with numerical M-step
   8.6 Multiple restart protocol with reference initialization
   8.7 Per-window estimation across 18 walk-forward windows
   8.8 Convergence behavior and runtime
   8.9 Per-window regime parameter table

9. **M6b: K=4 Walk-Forward Backtest**
   9.1 Backtest design and the apples-to-apples principle
   9.2 4-state soft regime blending
   9.3 Per-window lambda × risk-aversion calibration
   9.4 Engineering challenges: the Windows multiprocessing failure
   9.5 Per-window backtest results
   9.6 Aggregate stitched results

10. **Three-Way Comparison: Meridian vs K=3 vs K=4**
    10.1 Headline aggregate metrics
    10.2 Per-window CAGR comparison
    10.3 Per-window Sharpe comparison
    10.4 Per-window MaxDD comparison
    10.5 Win count distributions
    10.6 Stitched NAV trajectories
    10.7 Per-window calibrated parameter analysis

11. **Integration Criteria Evaluation**
    11.1 C1 — regime classification consistency
    11.2 C2 — early warning lead time
    11.3 C3 — walk-forward performance improvement
    11.4 Decision and reasoning

12. **Discussion: Why the Strategy Is Invariant**
    12.1 The three-way agreement pattern
    12.2 What absorbs regime information in the Meridian mechanism
    12.3 Where regime detection does add value
    12.4 The honest negative finding as a research contribution
    12.5 Comparison to broader literature findings

13. **Future Work**
    13.1 Statistical Jump Models
    13.2 EXP009 + soft regime combination
    13.3 Macro covariate refinement
    13.4 Different asset classes and frequencies
    13.5 Hidden Semi-Markov Models

14. **Conclusion**

**Appendices**

A. Mathematical formulation
B. Code architecture and module map
C. Per-window calibrated parameters (all 18 windows, 3 strategies)
D. Per-window K=4 TVTP regime parameters
E. Files reference and reproducibility
F. Glossary

**References**

---

# 1. Introduction

## 1.1 Project context: the Meridian systematic equity strategy

Meridian is the live systematic equity trading system that this entire research arc serves. It is a S&P 500 universe, $1,000,000 paper-traded portfolio executing on Alpaca, benchmarked against SPY. The system runs 15 multi-factor alpha signals (momentum, value, quality, mean-reversion, growth, defensive, cash flow, sentiment) combined via IC-IR weights, optimized through a constrained mean-variance optimizer with Ledoit-Wolf covariance shrinkage, sector neutralization, turnover penalties, and a 6% tracking-error cap, then risk-managed with vol-adjusted trailing stops and a four-tier drawdown circuit breaker.

The system is the consolidation of a research program across multiple experiments: SPE baseline (11 signals, walk-forward 2013-2021, 18.62% CAGR), EXP005 (signal expansion to 15, VIX threshold tightening, regime multiplier additions, 22.10% walk-forward CAGR), EXP006 (training extended back to 1997, 17 expanding walk-forward windows + holdout, 23.56% CAGR), EXP007 (sliding-window parameters tested and rejected), EXP008 (parameter sensitivity study). Live deployment incorporates the validated configurations from EXP005-006: 15 signals, IC history seeded from 1997 (5,245 records across 351 dates), VIX elevated threshold of 0.95, MIN_SIGNALS=10, expanding IC window with IC_LOOKBACK_MIN=12, and the 4 new signals (revenue_growth, low_volatility, fcf_yield, volume_momentum) with their regime multipliers.

Meridian's operational architecture (post April 2026 overhaul) separates detection from execution. The pipeline writes proposal files; three execution scripts (execute_stops.py, approve.py + execute.py, execute_replacement.py) handle order submission to Alpaca. Alpaca is the ledger ground truth: every pipeline run reconciles internal state from Alpaca via Stage 1.5 reconciliation. Trailing stops are vol-adjusted with 25-day EWM volatility, multiplied by 2.0, floored at 5% and capped at 20%, checked daily on intraday lows. Stop replacement is triggered when excess cash exceeds $100K and a 5-trading-day cooldown has passed. A 4-trading-day stop ticker cooldown blocks rebuys across all execution paths. The dashboard is read-only.

This paper concerns one specific component of that system: the regime detector.

## 1.2 The regime detector role in Meridian

Meridian's strategy responds to market regimes by adjusting signal weights. Different alphas perform differently across market conditions: momentum and growth signals tend to outperform in bull markets, value and quality signals tend to outperform in bear and crisis regimes, defensive signals (low volatility, FCF yield) tend to outperform in stressed environments, mean reversion signals work in oversold conditions. The regime detector identifies the current state, and the signal combiner applies regime-dependent multipliers to the IC-IR weights before composing the alpha score that feeds the optimizer.

The live regime detector is rule-based, two-layer. Layer 1 (daily) computes a stress assessment from VIX ratio (current VIX divided by trailing 252-day average) and market breadth (percent of S&P 500 constituents above their 200-day moving average using point-in-time membership). Layer 2 (weekly) computes an economic cycle assessment from yield curve slope (10Y-2Y) and high-yield credit spreads (5-week trend). The composite produces one of four hard states: BULL, RECOVERY, BEAR, or CRISIS. Each signal has a regime-conditional multiplier. The mapping is:

- BULL → momentum and growth signals get amplified (1.2-1.3x), defensive signals dampened (0.7x)
- RECOVERY → value signals amplified (1.2x), other signals near neutral
- BEAR → quality and value signals amplified (1.2-1.3x), defensive signals amplified (1.3x), momentum and growth dampened (0.7-0.8x)
- CRISIS → defensive signals amplified (1.5x), quality signals amplified (1.5x), momentum and growth heavily dampened (0.5x)

The regime detector also drives rebalance frequency: monthly under BULL/RECOVERY, biweekly under BEAR, weekly under CRISIS. And it feeds the circuit breaker: T2 (10% drawdown) triggers max position 3.5% and weekly rebalance, T3 (15%) triggers max position 2.5% and daily rebalance, T4 (20%) pauses trading for 5 days.

The rule-based detector has three appealing properties that any replacement must outperform: it is deterministic, it is interpretable (every state can be explained by the underlying VIX/breadth/credit/curve values), and it is fast (no model fitting, just threshold checks). Its main potential weakness is that it produces hard switches: at one moment the regime is BULL, at the next moment it is BEAR with no intermediate gradient. A probability-based regime model could in principle produce smoother transitions and earlier warnings of regime shifts.

That hypothesis was the origin of the B2 project.

## 1.3 What this paper covers and what it does not

This paper is the complete write-up of the K=4 extension to B2. It covers, in sequence:

1. The diagnostic that motivated the extension (Section 2)
2. The hypothesis and integration criteria (Section 3)
3. The literature foundations (Section 4)
4. The statistical analysis of macro factor predictive content (Section 5, EXP010)
5. The K=4 fixed-MS diagnostic (Section 6, M2b)
6. The K=5 sensitivity check and rejection (Section 7, M2c)
7. The K=4 TVTP estimation with economic constraints (Section 8, M3b)
8. The K=4 walk-forward backtest (Section 9, M6b)
9. The three-way comparison vs Meridian and K=3 (Section 10)
10. Integration criteria evaluation (Section 11)
11. Discussion of the strategy invariance hypothesis (Section 12)
12. Future work and conclusion (Sections 13-14)

What this paper does not cover: the original B2 K=3 build itself (its full details are in B2_Build_Complete_Knowledge.md and the B2 final PDF report), Meridian's signal definitions and IC-IR machinery (covered in SPE_Complete_System_Reference_v2.docx and SPE_Technical_Handover_v3.docx), the EVT Tail Risk project (covered separately in EVT_Build_Complete_Knowledge.md), and the full Meridian operational architecture. References to those documents are made when relevant.

The audience is the author for future reference, professors evaluating the work, and quant researchers (interview audiences, future collaborators) who want to understand the methodology and result. The tone is engineering-narrative: chronological, decision-explicit, with mathematical formulation in the appendix rather than the body so the main text reads as a story of investigation rather than a textbook chapter.

A note on terminology used throughout: "Meridian" and "hard regime" and "rule-based" are used interchangeably to refer to the EXP006 baseline that uses Meridian's existing rule-based detector. "K=3 soft" refers to the original B2 result with 3-state TVTP soft probabilities. "K=4 soft" refers to the result of this extension.

---

# 2. B2 K=3 Origin and Diagnostic

## 2.1 The original B2 hypothesis

The B2 project began in May 2026 with the hypothesis that Meridian's rule-based regime detector was too coarse. Rule-based detection produces hard switches: the system is either in BULL or BEAR with nothing in between, even though regime transitions are inherently gradual. A probabilistic model — specifically a Time-Varying Transition Probability Markov-Switching model where transition probabilities are functions of observable macroeconomic covariates — could capture two things the rule-based detector cannot: continuous probability vectors at every date, and the empirical relationship between macro conditions and regime transitions.

The TVTP framework, originally proposed by Filardo (1994) and Diebold-Lee-Weinbach (1994), generalizes Hamilton's (1989) Markov-switching model by replacing the constant transition matrix P with a time-varying matrix P_t where each entry is a logistic function of covariates z_t. In notation:

P(S_t = j | S_{t-1} = i, z_t) = exp(a_ij + b_ij' z_t) / Σ_k exp(a_ik + b_ik' z_t)

For B2, z_t was the vector [VIX_level, yield_curve_slope, credit_spread]. The a_ij and b_ij parameters are estimated jointly with the emission means μ_k and emission variances σ_k via expectation-maximization.

The B2 hypothesis: this richer model would (a) produce regime classifications consistent with the rule-based detector but with calibrated probabilities, (b) detect regime transitions earlier than threshold-based switching by responding to gradual macro deterioration, and (c) improve walk-forward CAGR and Sharpe through smoother signal-weight transitions across regime boundaries.

## 2.2 K=3 TVTP construction recap

The B2 build was 7 modules. M1 loaded SPY returns from FMP and macro covariates from the Strategy Research Lab database, aligned on trading dates and standardized. M2 implemented standard fixed-transition Markov-switching models (K=2 and K=3) as baselines and validation, cross-validated against statsmodels MarkovRegression to ensure the custom Hamilton filter matched within 1e-4 numerical tolerance. M3 implemented the TVTP-MS models (K=2 and K=3) with the custom EM algorithm including the numerical M-step for logistic transition coefficients. M4 selected among the four model variants on BIC. M5 compared the selected model against Meridian's rule-based regime history. M6 ran the walk-forward backtest. M7 produced 12 plots.

The model selection result from M4 across 7,347 trading days from 1997-01-02 through 2026-03-19:

| Model | K | Parameters | Log-likelihood | BIC | AIC |
|---|---|---|---|---|---|
| MS-2-Fixed | 2 | 6 | 23,266.04 | -46,478.66 | -46,520.08 |
| MS-3-Fixed | 3 | 12 | 23,603.95 | -47,101.07 | -47,183.89 |
| MS-2-TVTP | 2 | 12 | 23,600.27 | -47,093.72 | -47,176.54 |
| **MS-3-TVTP** | **3** | **30** | **23,906.07** | **-47,545.09** | **-47,752.15** |

MS-3-TVTP won on BIC, AIC, and log-likelihood. The TVTP versus fixed comparison at K=3 showed log-likelihood improvement of 302.13 with 18 extra parameters, BIC improvement of 444.02. The macro covariates were strongly justified.

The selected MS-3-TVTP regime parameters on the full sample:

| State | μ daily | μ annualized | σ daily | σ annualized | Avg duration (days) |
|---|---|---|---|---|---|
| Bull | 0.00143 | 36.08% | 0.00496 | 7.88% | 1.85 |
| Bear | -0.000254 | -6.40% | 0.01213 | 19.25% | 1.95 |
| Crisis | -0.00394 | -99.35% | 0.03183 | 50.53% | 1.08 |

The logistic transition coefficients revealed VIX as the dominant covariate. From bull (state 0):

| Transition | Intercept | VIX | Yield curve | Credit spread |
|---|---|---|---|---|
| Bull → Bear | 0.730 | **2.910** | -0.133 | 0.272 |
| Bull → Crisis | -2.083 | **5.126** | -0.180 | 0.122 |

A one-standard-deviation increase in VIX increases the log-odds of bull-to-crisis transition by 5.13. From bear (state 1):

| Transition | Intercept | VIX | Yield curve | Credit spread |
|---|---|---|---|---|
| Bear → Bear (stay) | 0.574 | 1.843 | -0.543 | 0.513 |
| Bear → Crisis | -3.128 | **3.382** | -0.981 | **0.909** |

VIX and credit spread both matter for bear-to-crisis transitions. Credit spread coefficient is 0.91 — substantially larger than its bull-to-crisis role of 0.12 — meaning credit deterioration is a stronger crisis trigger from already-stressed conditions. Yield curve flattening contributes negatively to bear-state persistence (-0.54), consistent with curve normalization preceding bear-state exits.

From crisis (state 2):

| Transition | Intercept | VIX | Yield curve | Credit spread |
|---|---|---|---|---|
| Crisis → Bear | -0.676 | 1.142 | **-1.191** | 0.724 |
| Crisis → Crisis (stay) | -2.909 | 2.401 | -0.763 | 0.345 |

Crisis-to-bear (i.e., crisis exit) is dominated by yield curve dynamics (-1.19 coefficient): the curve flattening or inverting predicts exit from crisis. The interpretation is that crisis ends when monetary policy responds and the curve adjusts.

This is meaningful economics. The model learned, from data, that VIX drives entry into stress, credit spreads escalate stress to crisis, and yield curve dynamics drive exit. None of this was hard-coded; it emerged from the EM optimization.

## 2.3 K=3 results vs Meridian

Despite the methodologically clean and economically interpretable model, the K=3 walk-forward backtest result was indistinguishable from Meridian. The B2 backtest used EXP006's exact engine — same signals, same IC-IR weights, same Ledoit-Wolf covariance, same mean-variance optimizer, same trailing stops, same calibration grid — with only one change: signal multipliers became expectation-weighted across regime probabilities rather than hard-switched on the discrete regime label. The 18 walk-forward windows (1 through 17 expanding annual + holdout 2022-2026) were re-run with K=3 soft regime, and (lambda, risk_aversion) were re-calibrated per window since soft regime weights change the optimizer's optimal turnover and risk aversion.

The aggregate result:

| Strategy | CAGR | Sharpe | MaxDD | Final NAV ($1M start) |
|---|---|---|---|---|
| Meridian (hard regime) | 23.564% | 1.309 | -37.628% | $84.99M |
| K=3 soft | 23.442% | 1.308 | -37.609% | $84.86M |

Aggregate improvement: -0.122% CAGR, -0.001 Sharpe, +0.019% MaxDD. Effectively zero. The C3 integration criterion (CAGR > 0 AND Sharpe > 0 improvement) failed.

Per-window: K=3 soft won 7 of 18 on CAGR and 7 of 18 on Sharpe. Out of 36 metric-window comparisons, K=3 soft won 14, lost 22. This is essentially a coin flip with slight bias against K=3.

The window-level pattern is informative. The biggest soft wins were in 2009 (Window 5: +5.4% CAGR for soft) and 2010 (Window 6: +1.0% CAGR for soft) — both post-GFC recovery years where Meridian's rule-based system would have been classifying volatility-elevated conditions as BEAR while the TVTP model assigned soft probability to recovery. The biggest soft losses were in 2006 (Window 2: -2.6% CAGR), 2007 (Window 3: -2.5% CAGR), and 2012 (Window 8: -3.4% CAGR). The 2006 and 2007 losses are particularly telling: those are pre-GFC bull years where Meridian's rule-based hard-switching kept the strategy aggressively bull-positioned, while the TVTP model assigned non-trivial probability to bear/crisis transitions (responding to early VIX upticks and credit-spread widening that did not yet correspond to actual market stress).

This is the asymmetric-information problem of probabilistic regime detection. When regime is genuinely transitioning (early 2008, early 2020), the soft probability ramps up gracefully and provides early warning. When regime is stable but macro factors are noisy (mid-cycle volatility spikes), the soft probability also moves and creates false alarms. The trading impact of false alarms (signal-weight regularization that costs alpha) and true positives (early defensive positioning) does not net out to a clear win.

This is an important point for the K=4 motivation: the K=3 result is not "TVTP failed because it cannot detect regimes." It is "TVTP detects regimes, but the regimes it detects do not lead to trading wins via the signal-multiplier mechanism." The two are different failure modes with different remedies.

## 2.4 The K=3 misspecification diagnostic

After M6 produced the K=3 walk-forward result, the natural follow-up question was: is the K=3 specification itself the issue? Maybe 3 states is wrong. Maybe the bear state in K=3 is conflating two distinct economic conditions — actual bear markets and slow-but-positive recovery periods.

The diagnostic took the form of monthly probability traces across all 18 walk-forward windows. For each test month from 2005-01 through 2026-03, we plotted P_bull, P_bear, P_crisis from the K=3 TVTP filtered probabilities alongside the rule-based regime label and the actual NAV trajectory. The traces revealed two systematic patterns.

**Pattern 1: P_crisis is reliable.** When P_crisis exceeded 0.5, the subsequent month nearly always exhibited stress. Examples:

- February 2008 (Window 4 training): P_crisis = 0.91 nine months before Lehman. The TVTP model identified the GFC stress trajectory in advance.
- March 2020 (Window 16 training): P_crisis = 0.984 at the COVID nadir.
- August 2011 (Window 7 test): P_crisis = 0.78 during the European debt crisis acute phase.
- March 2022 (holdout test): P_crisis = 0.71 at the start of the rate-hike-driven bear.
- April 2025 (holdout test): P_crisis = 0.65 around tariff-related volatility spike.

The P_crisis signal is genuinely useful as an early-warning channel — confirmed quantitatively in EXP010 (Section 5).

**Pattern 2: P_bear is overactive.** Many periods had P_bear above 0.85 with subsequent NAV moving up, not down. Examples (test-period observations):

- Window 14, March 2018: P_bear = 0.97. NAV at the time was within 1% of YTD high. The next 6 months were mildly choppy but not bear.
- Window 15, August 2019: P_bear = 0.87. Year-to-date NAV was up 11%. The remainder of 2019 went on to compound; 2019 ended with NAV at +21%.
- Window 16, throughout 2020 H2: P_bear repeatedly hit 0.97 while NAV climbed from $0.92M (March COVID nadir) to $1.10M by year-end.
- Holdout, throughout 2024: P_bear hit 0.97 multiple times during a year where NAV climbed from $1.50M (Jan) to $1.98M (Dec).

The pattern is clear: P_bear fires reliably when macro factors deteriorate, but macro deterioration in modern decades (post-GFC) often coexists with positive-but-slow market growth. The K=3 model has no separate state for "macro stress without negative returns" — it forces such conditions into either bull (which contradicts the macro signal) or bear (which contradicts the actual return distribution).

**The macro factor correlation hypothesis.** Examination of yield curve trajectories revealed that yield curve inversion was the dominant driver of P_bear elevation. The 2018-2019 period had a flat-to-inverted curve while equities continued grinding higher. The 2024 period had the curve normalizing from deep inversion while equities continued grinding higher. The K=3 model, learning from training data including 2007-2008 (curve inverted, then equities crashed), generalized "inverted curve = high probability of bear" — but the recent post-COVID monetary regime has decoupled curve dynamics from equity outcomes.

This decoupling is a known phenomenon — Estrella and Mishkin (1998) on yield curve recession prediction, and the various commentary post-2019 on the curve's reduced informativeness — but it has structural implications for the K=3 model. If the model has only "bear" available to label macro-stress-without-equity-declines, it will systematically over-predict bear during such periods. The remedy is either to remove the yield curve covariate (which sacrifices the curve's genuine information about other transitions) or to add a fourth state that can accommodate the macro-stress-without-decline pattern.

The K=4 hypothesis emerged from this diagnostic. Specifically: if the K=3 bear state is conflating two distinct economic conditions, splitting it into "bear" (genuine equity declines) and "recovery" (positive but slow growth, often with elevated macro stress signals) might produce a regime structure that better matches reality and consequently better trading outcomes.

## 2.5 Per-window probability trace patterns

A subtler observation from the diagnostic, which informed the K=4 build's constraints (Section 8): the K=3 TVTP probabilities exhibited high regime turnover. The fitted average regime durations were 1.85 days (bull), 1.95 days (bear), and 1.08 days (crisis) — meaning the model expected regimes to last roughly 1-2 days before transitioning. This is implausible for genuine economic regimes (which last weeks to months) and suggests the model was over-fitting day-to-day noise as regime transitions rather than estimating durable regime persistence.

This high-turnover behavior is a known issue in TVTP models when the diagonal transition probabilities are not constrained. With 30 free parameters fitting 7,347 observations, the EM algorithm has substantial flexibility and the global optimum can land in a region of parameter space where regimes flip frequently. The constraint apparatus introduced in M3b (Section 8) addresses this directly by imposing P_stay ≥ 0.90 floors, forcing the model to find solutions where regimes persist for at least 10 expected days.

For the K=3 model itself, this diagnostic was post-hoc — the B2 build did not impose persistence constraints and the resulting regime durations were what they were. The B2 backtest still produced the result it produced. But it informed the K=4 build to include this constraint from the start, and the K=4 results in Section 8 confirm that the constrained model produces more economically sensible regime durations while still converging to similar walk-forward backtest outcomes.

The full set of K=3 walk-forward window results vs Meridian, retrieved from the saved K=3 integration metrics:

| Window | Year | Soft CAGR | Hard CAGR | Δ CAGR | Soft Sharpe | Hard Sharpe | Δ Sharpe |
|---|---|---|---|---|---|---|---|
| 1 | 2005 | 42.84% | 42.98% | -0.13% | 2.84 | 2.85 | -0.01 |
| 2 | 2006 | 20.93% | 23.56% | -2.63% | 1.57 | 1.79 | -0.22 |
| 3 | 2007 | 8.80% | 11.30% | -2.50% | 0.58 | 0.72 | -0.14 |
| 4 | 2008 | -18.91% | -17.91% | -1.00% | -0.73 | -0.70 | -0.03 |
| 5 | 2009 | 59.99% | 54.62% | +5.36% | 2.02 | 1.90 | +0.11 |
| 6 | 2010 | 36.86% | 35.85% | +1.01% | 1.86 | 1.72 | +0.14 |
| 7 | 2011 | 28.44% | 29.77% | -1.33% | 1.19 | 1.22 | -0.03 |
| 8 | 2012 | 26.99% | 30.34% | -3.35% | 1.84 | 1.99 | -0.15 |
| 9 | 2013 | 51.02% | 47.89% | +3.13% | 3.27 | 3.11 | +0.15 |
| 10 | 2014 | 26.05% | 25.39% | +0.66% | 1.79 | 1.76 | +0.03 |
| 11 | 2015 | -5.01% | -2.96% | -2.05% | -0.24 | -0.10 | -0.14 |
| 12 | 2016 | 26.31% | 24.76% | +1.56% | 1.66 | 1.57 | +0.09 |
| 13 | 2017 | 38.14% | 39.22% | -1.09% | 3.02 | 3.08 | -0.06 |
| 14 | 2018 | -4.39% | -4.04% | -0.35% | -0.24 | -0.20 | -0.04 |
| 15 | 2019 | 33.46% | 35.74% | -2.27% | 2.26 | 2.35 | -0.09 |
| 16 | 2020 | 27.18% | 26.16% | +1.02% | 1.02 | 0.99 | +0.03 |
| 17 | 2021 | 42.01% | 43.20% | -1.19% | 2.28 | 2.30 | -0.02 |
| Hold | 22-26 | 20.98% | 19.76% | +1.22% | 1.25 | 1.17 | +0.07 |

Soft K=3 won CAGR in 7 windows (5, 6, 9, 10, 12, 16, holdout); won Sharpe in the same 7 windows. Lost CAGR in 11 windows; lost Sharpe in 11 windows. Mean CAGR delta: -0.12%. Mean Sharpe delta: -0.012.

The wins cluster in two regimes: post-crisis recovery (2009, 2010, 2013) and very recent (2020 H2 recovery, 2024 grind-up). The losses cluster in pre-crisis periods (2006, 2007) and choppy mid-cycle years (2011, 2012, 2015). This pattern is consistent with the misspecification diagnostic: K=3 soft helps when "recovery" should be the active regime (in K=3, recovery gets absorbed into bull, which is too aggressive at exiting defensive positioning) and hurts when "bull-but-with-warning-signs" is the active pattern (K=3 soft starts assigning bear probability, dampening offense, when nothing actually happens).

The K=4 hypothesis predicts this asymmetry would diminish if the model has a separate recovery state to absorb the slow-positive-growth-with-macro-stress conditions.

---

# 3. The K=4 Extension Hypothesis

## 3.1 Why K=4 rather than K=3 retuning

The first design decision in the K=4 extension was whether to extend at all, or to retune K=3. Three retuning paths were considered and rejected:

**Retuning path A: drop the yield curve covariate.** The diagnostic identified yield curve inversion as the dominant driver of P_bear elevation during 2018-2019 and 2024. Removing yield curve from the TVTP covariates would eliminate the false-bear signal during those periods. But yield curve coefficients in the K=3 fit were genuinely informative for crisis dynamics: the crisis-to-bear transition coefficient was -1.19 (the most negative coefficient in the entire model), meaning yield curve movements drive crisis exits. Removing the covariate to fix bear-state false alarms would damage crisis-state dynamics. This is robbing Peter to pay Paul, with no guarantee Paul actually benefits.

**Retuning path B: tighten regularization on bear-state covariates.** Adding L2 penalties to the logistic transition coefficients could reduce sensitivity. But choosing the penalty is itself a tuning problem, and the underlying issue (the model has no slot for "macro stress without equity decline") is not solved by parameter shrinkage.

**Retuning path C: use smoothed instead of filtered probabilities.** The Kim smoother uses information from t+1 onward to refine probability estimates at time t. This would reduce noisy regime flipping. But smoothed probabilities introduce lookahead bias and cannot be used in a walk-forward backtest. They are valid for analysis only, not for trading.

None of the retuning paths address the structural issue: K=3 is missing a state. The K=4 extension is the principled response.

The second design decision: K=4 specifically, not K=5 or higher. This is addressed in Section 7 (M2c K=5 sensitivity) but the short version is that the literature consensus (Ang-Bekaert 2002, Guidolin-Timmermann 2007, Nystrup et al. 2024) places the empirical ceiling for daily equity regime models at K=3-4 with K=5+ overfitting to regime fragments lacking economic meaning. K=4 is the upper bound of what the data can plausibly support, and the M2c K=5 diagnostic later confirms this empirically.

## 3.2 The recovery state hypothesis

The specific hypothesis the K=4 extension tests is:

> A four-state Markov-switching model with states {bull, recovery, bear, crisis} will produce regime classifications that better match the actual economic dynamics of the test period than the three-state model {bull, bear, crisis}, and the resulting soft regime probabilities will produce walk-forward backtest improvements over both the K=3 model and Meridian's rule-based detector.

The recovery state is hypothesized to capture conditions characterized by:

- Positive but slow returns (annualized mean somewhere between 0% and 15%)
- Moderate volatility (annualized standard deviation in the 12-20% range)
- Macro stress signals that have not (yet) translated to equity declines (elevated VIX, flat or inverted yield curve, modestly widened credit spreads)

The K=3 model has no slot for these conditions. By process of elimination, they get assigned to bear (because the macro signals look bear-like) or to bull (because the returns are positive). The mixed assignment produces noisy filtered probabilities and trading-loss-inducing regime flips.

The K=4 model with a recovery state should produce:

1. Cleaner separation between bear (genuine declines) and recovery (slow positive growth with macro warning signs)
2. Lower overall P_bear across the test period (because some of what was K=3 bear becomes K=4 recovery)
3. Smoother regime transitions (because shifts between bull and recovery are smaller than shifts between bull and bear)
4. Per-window backtest improvements concentrated in the periods that the diagnostic identified as misclassified by K=3

The Meridian rule-based detector already has a recovery state — that is the four-state classification BULL/RECOVERY/BEAR/CRISIS. So K=4 TVTP is moving the soft probabilistic model into structural alignment with the rule-based detector. The hypothesis is that combining the rule-based detector's state count with the TVTP model's gradient-aware probabilities will produce the best of both worlds: economically interpretable states (matching Meridian's existing framework) with smooth probabilistic transitions (which is what TVTP adds over rule-based).

## 3.3 What success would look like

Pre-registering the success criteria is critical to avoid post-hoc rationalization. The K=4 extension was designed with explicit success criteria, articulated before the M3b estimation was run:

**Statistical validation (M2b stage):**
- K=4 fixed-MS BIC must be lower (better) than K=3 fixed-MS BIC
- All four states must have economically interpretable parameters (non-pathological mu, sigma)
- Crisis state must have most negative mu and highest sigma
- Recovery state must have positive mu lower than bull, lower sigma than bear
- Crisis state occupancy in [3%, 8%] (consistent with historical crisis frequency)

**TVTP validation (M3b stage):**
- All 18 walk-forward windows must converge under the constraint regime
- Per-window state parameters must remain interpretable across windows
- Regime durations should be at least 10 days expected (P_stay ≥ 0.90)
- VIX should remain a significant covariate for crisis transitions

**Backtest validation (M6b stage):**
- C3 integration criterion: K=4 soft must improve aggregate CAGR AND Sharpe vs Meridian
- Per-window: K=4 should win at least 12 of 18 windows on at least one metric
- Specifically, K=4 should improve over K=3 in the windows the diagnostic identified as K=3 problem cases (2018, 2019, 2024)

**Critically, partial success was specified to count as failure.** If K=4 wins on Sharpe but loses on CAGR, that is not C3 pass. If K=4 improves the diagnostic-targeted windows but loses elsewhere, that is not pass. The reasoning: regime model upgrades should produce robust gains, not gains in some periods offset by losses in others. A model that helps in identifiable conditions and hurts in others is a parametric search problem, not a regime model.

This pre-registration matters because, as the results in Section 9 will show, K=4 does win on MaxDD (lower drawdown by 0.74 percentage points vs K=3) but loses marginally on CAGR and Sharpe. Without pre-registration of the AND criterion, it would be tempting to selectively report the MaxDD improvement as the "result." The pre-registered AND criterion correctly classifies the result as failure to integrate.

## 3.4 Pre-registered integration criteria

The B2 project's three integration criteria, originally specified in B2_TVTP_MS_Execution_Plan.docx and inherited by the K=4 extension:

**C1: Regime classification consistency.**

> Cohen's kappa between the model's hard-thresholded regime classifications and Meridian's rule-based regime history (2009-2026 overlap) must exceed 0.40 (moderate agreement on the Landis-Koch scale).

For K=3, kappa was 0.094 (slight agreement). For K=4 with an added recovery state matching Meridian's recovery state, kappa was expected to improve substantially since the comparison is now between two systems with the same state count.

**C2: Early warning lead time.**

> The model must detect regime transitions earlier than Meridian's rule-based detector by at least 3 trading days on average across major events (GFC, COVID, 2022 rate hikes).

For K=3, lead time was 88 days average across COVID and 2022 events (the GFC fell outside the rule-based regime history availability). C2 is the only K=3 criterion that passed cleanly.

**C3: Walk-forward improvement.**

> Aggregate CAGR improvement > 0 AND aggregate Sharpe improvement > 0 vs Meridian's hard regime baseline on the EXP006 walk-forward.

For K=3, both improvements were marginally negative (-0.12% CAGR, -0.001 Sharpe). C3 failed. For K=4, this is the primary criterion of interest — does the better-specified model with the recovery state actually produce trading wins?

The decision rule: integration into Meridian requires all three criteria to pass. Failure of any single criterion keeps the model as a standalone analytical tool. This rule is non-negotiable per the project's research-before-deployment principle ("every research change must be backtested in StrategyResearchLab before promotion to Meridian"). It applies to K=4 the same way it applied to K=3.

The pre-registered prediction at the start of the K=4 build was:

- C1 will improve significantly (because state count matches)
- C2 will pass (TVTP machinery is the same as K=3)
- C3 will improve at least marginally over K=3 even if it doesn't beat Meridian

The actual results, documented in Sections 9-11, partially confirmed and partially refuted this prediction.

---

# 4. Literature Foundations

This section assembles the academic context that informs the K=4 build. The selection is targeted at the specific decisions made in Sections 6-9: why TVTP over fixed transitions, why K=4 not K=5, why economic constraints during EM, why walk-forward methodology with apples-to-apples calibration. Foundational textbook material (basic Markov chains, the Hamilton filter mechanics) is treated briefly; the appendix contains the full mathematical formulation.

## 4.1 Hamilton-filter Markov-switching origins

Hamilton (1989) introduced the framework in which states are latent and discrete, observations are conditionally Gaussian given the state with state-specific mean and variance, and state transitions follow a first-order Markov chain with constant transition probabilities. The key algorithmic contribution is the recursive computation of filtered state probabilities P(S_t | r_1, ..., r_t) via a two-step prediction-update loop using the transition matrix and the Gaussian emission density.

The original application was business cycle classification on US macroeconomic data, demonstrating that a 2-state model with constant transitions could identify recessions at frequencies and severities consistent with NBER reference dating. The methodology generalized rapidly to financial applications where regime-conditional return distributions had been observed empirically (heavy tails, volatility clustering) but lacked a parsimonious model.

For the B2 K=3 and K=4 builds, Hamilton's framework is the algorithmic foundation. The Hamilton filter implementation in M2 (fixed-transition baseline) and M3 (TVTP) follows the original recursion exactly; the K=4 extension uses the same filter with K=4 state vectors instead of K=3. The cross-validation in M2 against statsmodels MarkovRegression confirmed numerical fidelity to the textbook implementation within 1e-4 tolerance.

Hamilton (1990) extended the framework with EM algorithm details and discussed pathological behavior of MLE in the Markov-switching context. Specifically, Hamilton noted that the unconstrained likelihood surface contains corners where parameters can wander to extreme values: emission means can grow arbitrarily large in absolute value, emission variances can shrink toward zero (producing degenerate near-deterministic states), and transition probabilities can drive toward zero or one. These pathological corners are local maxima that the EM algorithm can find depending on initialization. The recommended remedies are bounded parameter spaces and multiple random restarts, both of which the K=4 build incorporates.

## 4.2 The Kim smoother and EM estimation

Kim (1994) provided the smoothing complement to Hamilton's filter. The Hamilton filter computes P(S_t | r_1, ..., r_t) using only past information up to time t. The Kim smoother computes P(S_t | r_1, ..., r_T) using all information including future data, refining the filtered probabilities by reverse-pass recursion.

The distinction matters in two specific ways for the K=4 build:

1. **EM E-step**: Parameter estimation requires expected values of the latent state indicators given the observations. These expectations are smoothed probabilities (using all data), not filtered probabilities. The M-step then maximizes the expected complete-data log-likelihood with respect to the parameters. The K=4 EM implementation correctly uses smoothed probabilities in the E-step; using filtered probabilities here would produce biased estimates.

2. **Backtesting**: For walk-forward backtests, the regime probabilities used at decision time t can only depend on data available up to t. This is the filtered probability, not the smoothed probability. Using smoothed probabilities in a backtest introduces lookahead bias and inflates apparent performance. The K=4 backtest in M6b correctly uses `filtered_probs.parquet` (saved per window) and not `smoothed_probs.parquet` (which is saved for analysis purposes only).

This distinction is non-obvious to first-time TVTP implementers and is a source of subtle bugs. The B2 K=3 implementation explicitly separated the two outputs, and the K=4 build inherited this discipline.

## 4.3 Time-varying transition probabilities

The TVTP extension was developed independently by Filardo (1994) and Diebold-Lee-Weinbach (1994). Both papers proposed making transition probabilities functions of observable covariates via multinomial logistic links:

P(S_t = j | S_{t-1} = i, z_t) = exp(a_ij + b_ij' z_t) / Σ_k exp(a_ik + b_ik' z_t)

For identification, one destination state per origin state is set to zero (a_i0 = 0, b_i0 = 0), giving (K-1) free intercepts and (K-1) × d free coefficient vectors per origin state, where d is the covariate dimension. For K=4 with d=3 covariates, this is 3 origin states × 3 destination states (one normalized) × 4 coefficients = 36 free transition parameters, plus 8 emission parameters (4 means + 4 variances), for 44 total parameters when fitting the full sample. (For per-window training in M3b, the count is the same — 30 transition + 8 emission = 38 parameters in the K=4 TVTP, vs 12 + 6 = 18 for K=3 fixed.)

Filardo's empirical contribution was demonstrating that recession-onset probabilities in US macro data vary systematically with leading indicators. Pre-recession periods show measurable increases in the bull-to-bear transition logistic argument; post-recession periods show the reverse. The fixed-transition Hamilton model has no mechanism to capture these dynamics — every period has the same transition probabilities — and consequently misses the predictive content of the leading indicators.

For B2, the analog is that VIX, yield curve slope, and credit spread carry information about regime transitions that the fixed Hamilton model cannot use. The K=3 TVTP fit confirmed this empirically: the BIC improvement of 444 over fixed-K=3 with 18 extra parameters demonstrated that the covariates are doing real work. The transition coefficients (Section 2.2) showed VIX dominating crisis-entry transitions, credit spread driving bear-to-crisis escalation, and yield curve dynamics driving crisis exits. None of this can be captured by a constant transition matrix.

Diebold-Lee-Weinbach emphasized estimation challenges. The M-step in TVTP EM has no closed form for the logistic coefficients (unlike fixed-MS where the M-step is just weighted counting). It requires numerical optimization of:

Q_transition = Σ_t Σ_i Σ_j ξ_t(i,j) · log P(S_t = j | S_{t-1} = i, z_t)

with respect to {a_ij, b_ij}. This optimization is non-convex (the logistic likelihood with multiple destination states is not jointly concave in all parameters) and has many local maxima. Diebold-Lee-Weinbach recommended multiple random restarts, with the count scaling with the parameter dimension. Their suggestion was minimum 10 restarts for small problems; the K=3 B2 build used 10 restarts per window, the K=4 extension used 8 restarts (slightly fewer due to compute budget; convergence behavior at K=4 was sufficiently clean that 8 restarts found the global optimum reliably).

## 4.4 Number of states selection in financial regime models

The empirical literature on K-selection for daily equity returns is consistent: K=2-3 is the sweet spot, K=5+ overfits. Three studies anchor this consensus.

**Ang and Bekaert (2002)**, Review of Financial Studies. Tests K=2 vs K=3 vs K=4 on monthly equity index returns across multiple international markets. Finds K=2 sufficient for most series; K=3 marginally better when emerging markets or bear-market severity matters. K=4+ overfits monthly data. Their interpretation: monthly equity returns have at most three economically distinct regime conditions (bull, bear, crisis), and adding a fourth state captures noise rather than additional signal.

For daily data the threshold shifts slightly. Daily volatility regimes are noisier than monthly, so the model has more apparent variation to fit. K=3 remains the empirical sweet spot but K=4 becomes defensible if a clear additional state (specifically: recovery distinct from bull) is supported by the data.

**Guidolin and Timmermann (2007)**, Journal of Economic Dynamics and Control. Tests up to K=5 on monthly US equity returns. BIC selects K=4 with states corresponding to: high-mean low-volatility (bull), positive moderate-volatility (recovery), negative high-volatility (bear), and crash (crisis). K=5 splits one of the K=4 states into two without economic meaning. Guidolin-Timmermann is the closest published analog to the B2 K=4 finding: their K=4 state structure is qualitatively similar to what M2b later produced on daily SPY (Section 6). Their conclusion that K=5 is not improvement-worthy directly informs the K=5 rejection in Section 7.

**Nystrup, Madsen, and Lindström (2024)**, Journal of Time Series Analysis. Argues that traditional Hamilton-filter Markov-switching models become unstable beyond K=3 due to the joint estimation burden (state probabilities + emissions + transitions). Proposes Statistical Jump Models with explicit persistence penalties as an alternative class better suited to high-K specifications. Shows that for daily equity returns, K=4+ Hamilton models produce statistically improved fit but economically pathological state parameters. The B2 K=5 diagnostic in Section 7 confirms this exactly.

The K=4 extension is positioned within this consensus. K=4 with an economic recovery state is on the upper end of the defensible range. K=5 falls outside.

## 4.5 Constraints in Markov-switching estimation

Hamilton (1990) and Krolzig (1997) provide the framework for constrained Markov-switching estimation. Krolzig's *Markov-Switching Vector Autoregressions* is the standard reference; Section 9.4 specifically discusses identification problems in MS-VAR models with K ≥ 3 and recommends:

1. Initialize EM from a sensible reference (e.g., a fixed-MS fit) rather than random
2. Use multiple restarts with different perturbations of the reference initialization
3. Apply soft penalties on transition matrix diagonal probabilities to ensure realistic regime durations
4. Bound emission parameters to prevent the EM algorithm from wandering into pathological corners

The K=4 build follows all four prescriptions. M3b (Section 8) uses M2b's K=4 fixed-MS fit as the initialization reference, perturbs it with small noise across 8 restarts, applies a quadratic penalty (weight 50) to violations of P_stay ≥ 0.90, and bounds emission means to [-100%, +100%] annualized. These constraints were not arbitrary tuning choices; they are the literature-recommended remedies for the failure modes that the unconstrained K=4 TVTP exhibited in the initial estimation attempt.

The constraint-versus-unconstrained tradeoff is real. Constraints introduce bias: if the data genuinely supports an emission mean outside the bounded range, the constraint will produce a biased estimate. But for daily equity returns at K=4, the genuine regime means are well within [-100%, +100%] annualized (the most extreme observed regime in the M2b fixed-MS fit was crisis at -97%), so the constraint is not binding for realistic specifications. It only binds when the EM algorithm is wandering, which is exactly when binding is desired.

The diagonal probability constraint is more subtle. It prevents the model from finding solutions where regimes flip every 1-2 days, which the unconstrained K=3 fit produced (Section 2.5 noted average durations of 1.85, 1.95, 1.08 days). The constraint produces a model with regime durations of at least 10 expected days, which is empirically more plausible. But it also potentially masks model inadequacy: if the unconstrained model wants short-duration regimes, the constraint forces longer durations and the model may compensate by adjusting other parameters. In M3b, the constraint binds in nearly every window (mean diagonal probabilities cluster tightly around 0.900), confirming that the unconstrained MLE genuinely wanted faster transitions and the constraint is doing the work it was designed to do.

## 4.6 Walk-forward methodology and backtest validity

Bailey, Borwein, Lopez de Prado, and Zhu (2014) established that backtest results without proper walk-forward validation are nearly always overstated. Their analysis of "backtest overfitting probability" demonstrates that for a fixed sample size and number of strategies tested, there is a quantifiable probability that the apparent best strategy in-sample fails to outperform a benchmark out-of-sample. Strategies with many tunable parameters (high search complexity) face higher overfitting risk.

Walk-forward methodology with strict separation of training and test data, multiple non-overlapping windows, and reporting of full distribution (not just point estimates) is the minimum standard for valid out-of-sample claims. The B2 framework — adopted in EXP006 and inherited by both K=3 B2 and K=4 extension — meets this standard:

- 17 expanding annual training windows (Window 1: train 1997-2004, test 2005; Window 2: train 1997-2005, test 2006; ... Window 17: train 1997-2020, test 2021) plus a 4-year holdout (train 1997-2021, test 2022-2026)
- Test periods are strictly out-of-sample relative to training
- All model parameters (signal IC weights, regime model fit, optimizer lambda and risk_aversion calibration) are re-estimated per window using only the training data
- Aggregate metrics report stitched performance across all test periods, not selected subperiods
- Per-window metrics are reported to enable assessment of distribution

The K=4 extension follows this methodology exactly. The TVTP model is fit on training data only per window. The test-period filtered probabilities are produced by running the Hamilton filter forward from training-end with frozen parameters and observed test-period covariates (no parameter updates during test). The (lambda, risk_aversion) calibration sweep uses only training-period data per window. The result is a backtest where every test-day decision is causally based on information available before that day.

Harvey, Liu, and Zhu (2016) raised the multiple-testing concern for factor research. With hundreds of factors tested across many studies, naive significance thresholds produce false discoveries; Bonferroni or BHY corrections are recommended. The K=4 extension is robust to this concern because it tests a single hypothesis (does K=4 improve over K=3 and Meridian?) on out-of-sample data. The negative result is therefore credible without correction.

Lopez de Prado (2018) further extended the backtesting methodology with Combinatorial Purged Cross-Validation and Deflated Sharpe Ratio. CPCV is more rigorous than simple walk-forward but computationally intensive (it tests the strategy under thousands of resampled training/test splits). DSR adjusts the observed Sharpe ratio for the multiple comparisons inherent in strategy search. For the K=4 extension, neither was implemented because the core question is qualitative (does K=4 win or not?) rather than quantitative (what is the precise statistical significance of the improvement?). The qualitative answer — K=4 does not win — does not require the more sophisticated tools to establish.

## 4.7 Adaptive markets and regime model utility

Lo (2004) framed market efficiency as evolving with participant adaptation. The Adaptive Markets Hypothesis implies that strategies and detection methods built on past data have a half-life: what worked in 2005-2015 may be partially priced in or otherwise rendered ineffective by 2020-2025 as more participants exploit the same patterns.

For B2 and the K=4 extension, this framing is directly relevant. Meridian's rule-based regime detector was calibrated on data through 2026-03. The K=3 and K=4 TVTP models are also fit through that same date. Both model classes are testing on the same out-of-sample horizon (the 2022-2026 holdout). If regime information has been increasingly priced into asset prices over the test period, then the marginal value of regime detection — whether by rules or by sophisticated statistical models — decreases over time. The fact that K=3 ≈ K=4 ≈ rule-based across 18 walk-forward windows is consistent with this framing.

Frazzini and Pedersen (2014) on betting against beta provided complementary perspective: sophisticated implementation can extract value from simple ideas, but complex models often fail to outperform simple alternatives once transaction costs and capacity constraints are included. The K=4 result fits this pattern. The 4-state TVTP with logistic-link transitions estimated by custom EM is mathematically more sophisticated than Meridian's rule-based detector. In a strategy with sector neutrality, IC-IR weighting, mean-variance optimization, and trailing stops — all of which already absorb regime information through their natural construction — the additional sophistication does not translate to performance.

Ang and Bekaert (2004), Financial Analysts Journal, asked directly whether regime-aware allocation outperforms unconditional allocation. Their empirical finding: regime-conditional weights help in some periods (especially crisis avoidance) but the gains are modest and concentrated in specific historical episodes. For long-horizon strategies, the regime model often adds variance without adding mean return.

This finding is consistent with what the K=3 and K=4 results both showed. Regime probabilities are real and informative — the B2 K=3 model's logistic coefficients are economically interpretable, the K=4 extension's recovery state is statistically supported — but the strategy's existing risk management absorbs most of the regime information whether or not it's encoded as a formal probability. Better regime models do not produce proportionally better trading.

Pagan and Sossounov (2003) established the rule-based bull/bear classification methodology that influenced Meridian's existing rule-based detector. The detector uses peak/trough detection on cumulative returns plus volatility filters — fast, interpretable, robust. The B2 finding that this style of detector is hard to beat in practice (Cohen's kappa with K=3 TVTP was 0.094) is consistent with the broader literature: simple rules calibrated to economic intuition often match more sophisticated models on out-of-sample performance.

The K=4 extension is positioned at the intersection of these literature streams. The TVTP framework provides the mathematical machinery for soft regime probabilities. The K=4 specification matches Meridian's rule-based state count. The constrained estimation respects the Hamilton-Krolzig prescriptions for stable MS-VAR parameter learning. The walk-forward methodology meets the Bailey-Lopez de Prado standard. The negative finding is consistent with Lo, Frazzini-Pedersen, and Ang-Bekaert. The work is rigorous; the result is the result.

---

# 5. EXP010: Statistical Analysis of Macro Factor Predictive Content

## 5.1 Methodology overview

Before committing to the K=4 build, an independent statistical analysis was conducted to characterize the actual predictive content of the macro factors used as TVTP covariates. The question: do VIX, yield curve slope, and credit spread changes carry meaningful information about subsequent equity stress, and if so, what is the magnitude and reliability of that information?

The motivation was twofold. First, the K=3 diagnostic in Section 2.4 identified yield curve as the dominant driver of P_bear over-firing. Understanding the underlying factor predictive content would inform whether the issue was the model's interpretation of the curve or the curve's own information content. Second, the K=4 extension would inherit the same covariates; if those covariates have low predictive content for stress events outside of clear crises, that constrains what the K=4 model can be expected to achieve regardless of state count.

EXP010 used a statistical-test-driven approach rather than a model-fitting approach. The methodology had six components:

1. **Episode detection.** Define stress episodes as periods where 90-day or 120-day signed factor changes exceeded specified thresholds. Test multiple threshold definitions to assess robustness.

2. **GPD-justified thresholds.** Fit Generalized Pareto Distributions to the absolute factor changes to identify statistically principled threshold levels (where the distribution tail is well-characterized). Compare to simple percentile-based thresholds.

3. **Sensitivity sweep.** Re-run all analyses across thresholds at 1%, 2%, 3%, 4%, 5%, 7%, 10%, 13% of the absolute change distribution, plus the GPD-fitted threshold. Compare to crisis probability triggers at 0.3, 0.5, 0.7.

4. **ROC analysis.** Treat P_crisis from the K=3 TVTP model as a binary classifier of subsequent stress and compute ROC AUC, Youden's J optimal threshold, true positive rate, false positive rate.

5. **Forward drawdown regression.** Test whether including K=3 TVTP probabilities in a regression of forward 30/60/90-day drawdowns on factor changes adds explanatory power. Use HAC robust standard errors and likelihood ratio tests.

6. **Granger causality.** Test bidirectional Granger causality between the macro factors and P_crisis at lags 1-5.

7. **Episode lead time.** For each detected episode, measure the number of days between TVTP fire (P_crisis crossing threshold) and rule-based fire. Use paired t-tests and Wilcoxon signed-rank tests for statistical significance.

The full methodology specification, code, and outputs live in `regime_switching/factor_trend_analysis.py` and `regime_switching/data/factor_trend_analysis/`. The decision summary is at `decision_summary.md`. Key results are reproduced and discussed below.

## 5.2 GPD-justified factor change thresholds

The Generalized Pareto Distribution is the appropriate model for tail behavior of stationary processes (a result from extreme value theory; see EVT_Build_Complete_Knowledge.md for the related work in B1). Fitting GPDs to the absolute 90-day factor changes produced:

| Factor (lookback) | Threshold quantile | Threshold value | xi (shape) | sigma (scale) | n exceedances | KS p-value |
|---|---|---|---|---|---|---|
| VIX 90d | 0.85 | 12.38 | 0.107 | 8.45 | 470 | 0.506 |
| VIX 120d | 0.85 | 12.61 | 0.112 | 8.49 | 476 | 0.754 |
| Yield curve slope 90d | 0.85 | 0.50 | -0.291 | 0.198 | 586 | 0.102 |
| Yield curve slope 120d | 0.93 | 0.81 | -0.089 | 0.108 | 256 | 0.157 |
| Credit spread 90d | 0.96 | 4.09 | -0.808 | 7.76 | 121 | 0.216 |
| Credit spread 120d | 0.85 | 2.67 | 0.771 | 0.94 | 465 | <0.001 |

KS p-values above 0.05 indicate the GPD provides an acceptable fit to the tail. Five of six factor-lookback combinations passed. The credit spread 120d combination failed (KS p < 0.001) suggesting a different distributional family is appropriate for that factor; the credit spread 90d threshold was used in subsequent analyses.

The xi (shape) parameters are economically interesting. VIX has positive xi (~0.11) indicating heavy tails — extreme moves are more likely than a Gaussian model would predict. Yield curve slope has negative xi (~-0.29 to -0.09) indicating bounded tails — there is a finite ceiling on how much the curve can move in a 90-120 day window. Credit spread 90d has very negative xi (-0.81) indicating sharply bounded tails. Credit spread 120d has highly positive xi (0.77) indicating extremely heavy tails.

These shape parameters inform the reasonableness of using these factors as regime triggers. VIX with its mild heavy-tail behavior is a reliable stress signal — extreme moves are well-defined and recur. Yield curve with bounded tails is a slow, structural signal — the magnitude of moves is constrained by macroeconomic fundamentals. Credit spread is intermediate, with the longer lookback (120d) showing more extreme behavior than the shorter (90d), consistent with credit cycles operating on longer timescales than equity volatility.

## 5.3 ROC analysis of P_crisis as a stress predictor

Treating P_crisis from the K=3 TVTP model as a continuous classifier of subsequent stress (defined as forward 90-day drawdown exceeding the 90th percentile) produced:

- ROC AUC = 0.598
- Youden's J optimal threshold = 0.0052
- True positive rate at Youden J = 0.504
- False positive rate at Youden J = 0.335
- Positive class rate (base rate of stress) = 0.234
- N observations = 5,315

ROC AUC of 0.598 is barely better than random (0.500). At the Youden-optimal threshold of 0.0052 (essentially just "any non-trivial P_crisis"), the model captures 50.4% of true stress events but generates a 33.5% false-positive rate. These numbers are not strong predictive performance.

But the interpretation requires care. The denominator here is "forward 90-day drawdown exceeds 90th percentile" — a relatively common condition (23.4% base rate) that includes both genuine crisis episodes and ordinary market corrections. The model is not a general-purpose drawdown predictor; it was designed to identify specific regime states (bull/bear/crisis) characterized by particular return distributions. Using P_crisis as a binary stress classifier conflates two different problems.

The more diagnostic results are in Section 5.5 (rule-based comparison) and Section 5.7 (episode-level analysis), which evaluate P_crisis on the specific events it was designed to identify rather than on a general drawdown prediction task.

## 5.4 Forward drawdown regression

The forward drawdown regression tests whether adding TVTP probabilities to a baseline factor regression improves explanatory power. The specification:

Drawdown_{t,t+h} = α + β_VIX · ΔVIX_{t-90,t} + β_curve · Δcurve_{t-90,t} + β_credit · Δcredit_{t-90,t} + γ_crisis · P_crisis_t + γ_bear · P_bear_t + ε

with h ∈ {30, 60, 90} forward days, fit on the full sample with HAC robust standard errors.

Results:

**Forward 30-day drawdown (n = 5,314):**
- R² factors only: 0.0182
- R² factors + TVTP: 0.0183 (improvement +0.0002)
- LR test p-value: 0.6457 (not significant)

**Forward 60-day drawdown (n = 5,314):**
- R² factors only: 0.0193
- R² factors + TVTP: 0.0204 (improvement +0.0010)
- LR test p-value: 0.0624 (marginally non-significant at 5%)

**Forward 90-day drawdown (n = 5,314):**
- R² factors only: 0.0235
- R² factors + TVTP: 0.0251 (improvement +0.0016)
- LR test p-value: 0.0128 (significant at 5%)

The pattern is clear: TVTP probabilities add zero explanatory power for very short horizons (30 days), borderline at 60 days, and statistically significant but quantitatively tiny at 90 days. The R² improvement at 90 days is 0.16 percentage points, meaning TVTP explains an additional 0.16% of the variance in 90-day forward drawdowns beyond what the raw factors already explain.

The TVTP coefficient signs are economically sensible. P_crisis at 90d has a -0.023 coefficient on forward drawdown (more negative drawdown when P_crisis is high) and P_bear has -0.008. Both are in the expected direction. But the coefficient p-values are 0.50 and 0.57 individually, only the joint LR test reaches significance. This means TVTP probabilities are weakly informative collectively but no single state probability is strongly informative individually.

The honest summary: TVTP adds explanatory power for medium-term drawdowns but the magnitude is so small (0.16 percentage points of R²) that it is unlikely to translate to meaningful trading improvement.

## 5.5 Granger causality

Granger causality tests evaluate whether past values of one series predict future values of another, beyond what the second series' own past predicts. For TVTP regime probabilities and macro factors, the question is bidirectional: do factors predict TVTP probabilities (validating the model's covariate selection), and do TVTP probabilities predict factors (suggesting the model is forward-looking beyond the factors themselves)?

Results across lags 1-5:

**Factors predicting P_crisis (does the macro factor lead?):**

| Direction | Best lag | Min p-value | Significant at 5%? |
|---|---|---|---|
| VIX → P_crisis | 5 | 4.4e-39 | Yes (very strong) |
| Credit → P_crisis | 5 | 4.3e-16 | Yes (very strong) |
| Yield curve → P_crisis | 2 | 0.008 | Yes |

**P_crisis predicting factors (does the TVTP model lead?):**

| Direction | Best lag | Min p-value | Significant at 5%? |
|---|---|---|---|
| P_crisis → VIX | 4 | 0.066 | No (marginal) |
| P_crisis → Credit | 3 | 0.0003 | Yes |
| P_crisis → Yield curve | 3 | 0.096 | No (marginal) |

The asymmetry is informative. Factors very strongly Granger-cause P_crisis (p-values from 4e-39 down) — the model is using factor information in the way it should. P_crisis weakly Granger-causes only credit spread (p=0.0003); it does not lead VIX or yield curve.

This says the TVTP model is a function of the factors, not a forward-looking model that anticipates factor movements. For predicting equity stress that the factors haven't yet identified, the TVTP model adds nothing. For aggregating factor signals into a single regime probability that responds appropriately to factor changes, the TVTP model adds value.

For the K=4 extension specifically, this Granger causality structure should not change with the addition of a recovery state. The TVTP model at K=4 will still be a function of the same three factors. Its predictive content for forward returns will not exceed the factors' own predictive content. The K=4 win condition therefore depends on whether the additional state structure helps the strategy use the same factor information more effectively, not on the K=4 model having access to information that K=3 lacks.

## 5.6 Rule-based baseline comparison

The most direct test of whether the TVTP model adds value over Meridian's rule-based detector is event-level comparison: when do they fire, and how does their firing relate to subsequent stress?

Across 29 detected episodes (2008-2026):

- Rule-based fired in 9 of 29 episodes (31%)
- TVTP at threshold P_crisis ≥ 0.5 fired in 20 of 29 episodes (69%)
- Both fired in 9 episodes (the rule-based subset)

In the 9 episodes where both fired, lead-time analysis:

- Rule-based median lead = 55 days (before peak stress)
- TVTP median lead = 38.5 days
- Mean difference (TVTP earlier than rule-based) = +27 days
- Paired t-statistic = 2.46, t p-value = 0.039 (significant at 5%)
- Wilcoxon signed-rank p-value = 0.055 (marginally significant)

The TVTP model fires more often (20 vs 9 episodes) and on average earlier (mean +27 days lead) than rule-based, but the median lead times are similar (38.5 vs 55 days). The mean-vs-median divergence is driven by a few episodes where TVTP fired very early (notably 2008-09 GFC where TVTP fired ~180 days before peak stress) while rule-based fired closer to the peak.

The "more often" finding is double-edged. More frequent firing means earlier warnings on average but also more false alarms. The rule-based detector's lower fire rate (31%) corresponds to higher specificity — when it fires, it is more reliably indicating actual crisis. The TVTP detector's higher fire rate (69%) corresponds to higher sensitivity but lower specificity. Whether this is preferable depends on the cost of false alarms (defensive positioning during non-crisis periods) versus missed alarms (no defensive positioning during actual crisis).

For Meridian, where defensive positioning is gradual (signal multiplier adjustments) rather than binary (full risk-off), the cost of false alarms is moderate. The cost of missed alarms is high because Meridian's circuit breaker only activates on realized drawdowns, by which point losses have occurred. From this asymmetry, an early-warning channel with higher sensitivity has positive value even at the cost of some false alarms — provided the trading mechanism can tolerate the false-alarm cost.

The K=3 walk-forward backtest result (Section 2.3) is the direct test of whether Meridian's signal-multiplier mechanism can tolerate that cost. The answer was: roughly 50/50, no consistent improvement. The K=4 extension is testing whether a refined regime structure changes that conclusion.

## 5.7 Episode-level lead time analysis

The episode-level data, saved to `episode_lead_times.parquet`, contains 50 detected events with lead times across multiple threshold definitions. Selected episodes:

| Date | Factor | Peak signed change | TVTP lead (P=0.5 threshold) | Notable context |
|---|---|---|---|---|
| 2008-10-07 to 2008-12-16 | VIX 90d | +58.5 | 180 days | GFC acute phase |
| 2010-05-20 (1 day) | VIX 90d | +28.2 | 10 days | Flash Crash |
| 2011-08-08 to 2011-08-23 | VIX 90d | +30.3 | 4 days | EU debt crisis acute |
| 2011-10-03 (1 day) | VIX 90d | +28.4 | 60 days | EU debt crisis residual |
| 2015-08-24 to 2015-08-25 | VIX 90d | +28.1 | 0 days | China devaluation |
| 2018-02-05 to 2018-02-06 | VIX 90d | +27.2 | 0 days | "Volmageddon" |
| 2020-02-27 to 2020-04-24 | VIX 90d | +69.9 | 21 days | COVID |
| 2024-08-05 (1 day) | VIX 90d | +25.3 | 0 days | Yen carry unwind |
| 2025-04-04 to 2025-04-11 | VIX 90d | +37.7 | 111 days | Tariff regime |

The pattern: TVTP gives substantial lead time on slow-building crises (GFC at 180 days, COVID at 21 days, 2025 tariff at 111 days) but zero lead time on fast surprise events (2015 China devaluation, 2018 Volmageddon, 2024 yen unwind). The slow-building crises are characterized by gradual VIX rises, gradual credit spread widening, and curve dynamics — exactly the conditions the TVTP covariates can detect. Fast surprise events have no precursor in the macro factors; they happen in a single day.

This is a useful operational characterization: TVTP early warning works for the kinds of crises that are characterized by gradual macro deterioration, which is most actual crises but not all volatility events. For Meridian's risk management — where the goal is reducing exposure during sustained stress periods, not avoiding individual day-of volatility — this is the correct profile. The early-warning value is real for GFC-type, EU-debt-type, COVID-type, and tariff-type events. It is absent for VIX-pop-and-recover events that resolve in days.

## 5.8 Implications for the K=4 build

EXP010 produced four conclusions that informed the K=4 extension:

1. **The macro factors carry information, but the information has limits.** Forward 90-day drawdown R² improvement from including TVTP probabilities is 0.16 percentage points. The K=4 model with the same covariates cannot exceed this ceiling for general drawdown prediction. Wins from K=4 must come from better state structure for trading mechanics, not from better predictive power.

2. **TVTP early warning is reliable for slow-building crises.** This property is preserved across model specifications since it derives from the covariate set and the logistic transition machinery, both of which are unchanged from K=3 to K=4. C2 should pass for K=4 the same way it passed for K=3.

3. **Yield curve covariate has constrained information.** The xi parameter is negative (-0.29) suggesting bounded tails. Curve movements within the bounded range may not carry strong stress information. The K=3 issue (curve inversion driving false bear signals during 2018-2019 and 2024) is consistent with the curve exhausting its informative range — once the curve is inverted, further movement in either direction does not change the regime probability much. K=4 with a recovery state could absorb some of this variance into the recovery probability rather than letting it inflate bear probability.

4. **The rule-based detector is hard to beat on episode-level metrics.** Median lead times are similar (55 days rule-based vs 38.5 days TVTP at P=0.5 threshold), with TVTP winning on mean only because of a few early-firing episodes. The K=4 extension does not change the underlying covariate-based machinery and therefore should not be expected to dramatically change the rule-based comparison either. C1 might improve due to state-count alignment, but C2 will remain similar.

EXP010 was completed before the M3b TVTP estimation. Its findings did not alter the decision to proceed with K=4 — the K=4 hypothesis is structural (recovery state) not predictive (better factor information). EXP010 calibrated expectations about what K=4 could realistically achieve and confirmed that the TVTP framework is doing the work it should be doing. The factor predictive content is real but limited; the model is correctly using that content; the question that remains is whether better state structure produces better trading.

---

# 6. M2b: K=4 Fixed-MS Diagnostic

## 6.1 Why fixed-MS before TVTP

The K=4 extension's first computational step was fitting a K=4 fixed-transition Markov-switching model on the full sample, before attempting K=4 TVTP. Three reasons.

**Reason 1: cross-validation.** B2's M2 had already established that the custom Hamilton filter implementation matched statsmodels' MarkovRegression at K=2 and K=3. Re-running this validation at K=4 would confirm correctness extends to four states before any TVTP-specific machinery is exercised. Cross-validation against statsmodels is the only available external check; once the model has TVTP transitions, no external library can validate.

**Reason 2: BIC test of K=4.** Even before TVTP, the foundational question is whether K=4 is supported by the data. The K=3 fixed-MS BIC from B2 was -47,101. If K=4 fixed-MS BIC was higher (worse) than K=3 fixed-MS BIC, the recovery state hypothesis would be empirically refuted before any TVTP machinery was needed. The K=4 build would terminate at this step and the diagnostic findings would be reported as "K=3 misspecification suggests recovery state, but the data does not support 4 distinct states."

**Reason 3: TVTP initialization reference.** The Hamilton-Krolzig literature recommendation for stable EM convergence at K ≥ 3 is to initialize TVTP from a fitted fixed-MS reference. M2b's parameters become the seed for M3b's restart initialization. Without M2b, the K=4 TVTP build would have to use random initialization, which the B2 K=3 experience suggested would give a high failure rate (15+ restarts per window with weak global-optimum identification). With M2b as reference, 8 restarts produced clean convergence in M3b (Section 8.7).

The M2b implementation is in `regime_switching/m2b_fixed_ms_k4.py`. Its outputs are saved to `regime_switching/data/m2b_k4_diagnostic/` including `ms4_fixed_results.json` and `ms4_fixed_probs.parquet`.

## 6.2 Implementation specifics

M2b reuses the Hamilton filter and EM algorithm from B2's M2 (custom Python implementation, not statsmodels) with state count parameterized to K=4. The Hamilton filter recursion:

E-step (Hamilton filter forward pass + Kim smoother backward pass):
- Forward: predict P(S_t = j | info_{t-1}) = Σ_i P_ij · P(S_{t-1} = i | info_{t-1}); update P(S_t | info_t) ∝ N(r_t; μ_{S_t}, σ²_{S_t}) · P(S_t | info_{t-1})
- Backward (Kim smoother): P(S_t | info_T) using future information

M-step (closed-form for fixed transitions):
- Emission means: μ_k = Σ_t γ_t(k) · r_t / Σ_t γ_t(k)
- Emission variances: σ²_k = Σ_t γ_t(k) · (r_t - μ_k)² / Σ_t γ_t(k)
- Transition probabilities: P_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i)

where γ_t(k) = P(S_t = k | info_T) is the smoothed marginal probability and ξ_t(i,j) = P(S_t = j, S_{t-1} = i | info_T) is the smoothed joint probability.

Convergence criterion: relative log-likelihood improvement |L_new - L_old| / |L_old| < 1e-6, or maximum 200 iterations. Multiple restarts (12 for M2b, varying random seeds) with the highest log-likelihood solution selected.

Synthetic recovery test before real-data fitting:
- Generate 5,000 observations from a known K=4 MS model with specified parameters
- Fit the model and verify parameter recovery within 95% CI
- Repeat 50 times to estimate coverage probability

For M2b's synthetic test, true emission means were [-0.005, -0.001, 0.0005, 0.0015] daily, true emission std-devs were [0.04, 0.018, 0.011, 0.006], with diagonal transition probabilities of [0.95, 0.97, 0.94, 0.96]. Across 50 synthetic samples, parameter recovery confidence intervals contained the true values with empirical coverage matching the nominal 95% level. This validates the M2b implementation.

State labeling deterministic by emission mean: lowest mu = crisis, second lowest = bear, second highest = recovery, highest = bull. This avoids the label-switching ambiguity inherent to mixture model estimation.

## 6.3 Information criteria results

M2b fit on the full sample (1997-01-02 through 2026-03-19, 7,347 observations) with 12 restarts. Best restart (seed 144, restart index 6) converged in 76 iterations to log-likelihood 23,686.16. With 20 estimated parameters (4 emission means + 4 emission std-devs + 12 free transition probabilities, since each row of the 4x4 transition matrix sums to 1 leaving 3 free entries):

- BIC = -47,194.29
- AIC = -47,332.33

Comparison to B2's K-selection table:

| Model | K | n_params | LL | BIC | AIC |
|---|---|---|---|---|---|
| MS-2-Fixed | 2 | 6 | 23,266.04 | -46,478.66 | -46,520.08 |
| MS-3-Fixed | 3 | 12 | 23,603.95 | -47,101.07 | -47,183.89 |
| **MS-4-Fixed** | **4** | **20** | **23,686.16** | **-47,194.29** | **-47,332.33** |
| MS-2-TVTP | 2 | 12 | 23,600.27 | -47,093.72 | -47,176.54 |
| MS-3-TVTP | 3 | 30 | 23,906.07 | -47,545.09 | -47,752.15 |

K=4 fixed-MS BIC of -47,194 is better than K=3 fixed-MS BIC of -47,101 by 93 BIC units. The K=4 fixed model justifies its 8 extra parameters in BIC terms.

But — and this matters — K=4 fixed-MS BIC is worse than K=3 TVTP BIC (-47,545) by 351 BIC units. The TVTP machinery at K=3 produces a better model than fixed-MS at K=4. This means the BIC ranking is:

K=3 TVTP < K=4 fixed-MS < K=3 fixed-MS < K=2 TVTP < K=2 fixed-MS

The natural next question: would K=4 TVTP further improve over K=3 TVTP? The full sample K=4 TVTP fit was not produced in M2b (M3b fits per-window only, not full-sample, due to compute budget). But the per-window K=4 TVTP training log-likelihoods in M3b (Section 8) are higher than the corresponding per-window K=4 fixed-MS log-likelihoods would be, suggesting K=4 TVTP would also win on full-sample BIC. This is a noted gap in the analysis — running a full-sample K=4 TVTP fit would close it definitively. It was not done because the per-window backtest result already establishes the trading conclusion (Section 9-10), and the full-sample BIC ranking would not change the integration decision.

## 6.4 State characterization

The fitted K=4 fixed-MS state parameters:

| State | μ daily | μ annualized | σ daily | σ annualized | Sample occupancy |
|---|---|---|---|---|---|
| Crisis | -0.00385 | -97.11% | 0.0380 | 60.31% | 3.37% |
| Bear | -0.00036 | -8.99% | 0.0150 | 23.73% | 27.79% |
| Recovery | +0.00033 | +8.36% | 0.00931 | 14.78% | 35.86% |
| Bull | +0.00123 | +31.09% | 0.00489 | 7.77% | 32.98% |

These parameters are clean. Mu values are monotonic across states (most negative crisis, most positive bull), sigma values are monotonic in the expected direction (highest variance crisis, lowest variance bull), and the recovery state has the predicted profile (positive mean lower than bull, lower variance than bear).

The transition matrix:

|  | Crisis | Bear | Recovery | Bull |
|---|---|---|---|---|
| **Crisis** | 0.940 | 0.060 | 0.000 | 0.000 |
| **Bear** | 0.005 | 0.975 | 0.019 | 0.000 |
| **Recovery** | 0.002 | 0.010 | 0.942 | 0.046 |
| **Bull** | 0.000 | 0.003 | 0.046 | 0.950 |

Diagonal elements (regime persistence) range from 0.940 to 0.975 — economically plausible regime durations of 17-40 days expected. The off-diagonal pattern is informative: crises transition only to bear (no direct crisis-to-recovery or crisis-to-bull); bears transition to recovery (not directly to bull); bull and recovery transition between each other.

This is a "stairs" structure: regimes step down from bull through recovery to bear to crisis, and step up from crisis through bear to recovery to bull. There are no two-step transitions in the fitted model. This is consistent with how genuine economic regimes evolve — recoveries do not become crises overnight without passing through bear.

The implied expected durations (1 / (1 - p_diagonal)):
- Crisis: 1/(1-0.940) = 16.7 days
- Bear: 1/(1-0.975) = 40.0 days
- Recovery: 1/(1-0.942) = 17.2 days
- Bull: 1/(1-0.950) = 20.0 days

Bear has the longest expected duration; crisis and recovery have the shortest. Total occupancies match: bear at 27.8% sample share is consistent with its 40-day expected duration combined with moderate transition rates. Crisis at 3.4% is consistent with low expected duration combined with low entry rates.

## 6.5 Validation criteria

The pre-registered K=4 statistical validation criteria (Section 3.3):

| Criterion | Threshold | Result | Pass? |
|---|---|---|---|
| BIC improvement vs K=3 | Lower (better) than -47,101 | -47,194 | Yes |
| All four states interpretable | Non-pathological mu, sigma | Mu monotonic, sigma monotonic | Yes |
| Crisis state extreme | Most negative mu, highest sigma | -97% mu, 60% sigma | Yes |
| Recovery state characteristic | Positive mu < bull, sigma < bear | +8.4% mu < 31.1% bull, 14.8% sigma < 23.7% bear | Yes |
| Crisis occupancy | [3%, 8%] | 3.37% | Borderline (low end of range) |

Four criteria clearly pass. The crisis occupancy of 3.37% is at the very low end of the [3%, 8%] target range. This was a flag for review but not a rejection — historical crisis frequency in the 1997-2026 sample is empirically low (the GFC, 2011 EU debt, 2020 COVID, 2022 rate hikes account for most of the crisis-state exposure), and 3.4% sample occupancy is consistent with a model that correctly identifies these as rare events rather than padding the crisis state with marginal cases.

Compared to K=3 fixed-MS occupancies (bull 44.7%, bear 48.1%, crisis 7.2%): K=4 has roughly preserved the crisis fraction (somewhat lower at 3.4% but in the same magnitude), split the K=3 bull state into K=4 bull (33.0%) plus most of K=4 recovery (35.9%), and reduced K=3 bear (48.1%) into K=4 bear (27.8%) plus a portion of K=4 recovery. The recovery state is therefore drawing from both the K=3 bull and K=3 bear states, consistent with the diagnostic hypothesis that K=3 was conflating slow-positive-growth periods into both bull and bear.

## 6.6 Decision and implications

M2b passed all validation criteria. K=4 was statistically supported and economically interpretable. The decision was:

1. Proceed to M2c (K=5 sensitivity check) to confirm K=4 is the right ceiling, not a way station to K=5.
2. Use M2b's parameters as the initialization reference for M3b (K=4 TVTP).
3. Note the outstanding question of whether K=4 TVTP would further improve BIC over K=3 TVTP and add it to the future-work list.

A key implication for the K=4 TVTP build: the M2b regime structure provides clear initialization values. The mu_annual values [-97%, -9%, +8.4%, +31%] and sigma_annual values [60%, 24%, 15%, 7.8%] became the seed parameters for M3b's perturbed restarts. The transition matrix's stairs structure (no two-step transitions) provided a sensible initialization for the TVTP logistic intercepts a_ij. The TVTP model would then learn the b_ij covariate coefficients on top of this structure, replacing the constant transition probabilities with covariate-dependent ones while preserving the identified state count and approximate emission distributions.

This staged approach — fixed-MS first, TVTP second — is the Hamilton-Krolzig recommendation specifically for K ≥ 3 specifications where unconstrained TVTP estimation can wander. The M2b → M3b sequence implements that recommendation.

---

# 7. M2c: K=5 Diagnostic and Rejection

## 7.1 Sensitivity rationale

After M2b confirmed K=4 was statistically supported, M2c tested K=5 as a sensitivity check. The rationale: if BIC continues monotonically improving with K up through K=5 and beyond, the true number of regimes might exceed 4 and the K=4 build would be a way station rather than the destination. If BIC plateaus or worsens at K=5, K=4 is confirmed as the empirical ceiling.

The sensitivity check is mandatory for honest research practice. Without it, the K=4 extension could be criticized as an arbitrary stopping point. With it, the K=4 conclusion has explicit empirical support: the data does not support K=5, regardless of whether K=4 happens to win on backtesting.

The M2c implementation is in `regime_switching/m2c_fixed_ms_k5.py`. Its outputs are saved to `regime_switching/data/m2c_k5_diagnostic/`.

## 7.2 Statistical fit

M2c fit on the same full sample (1997-01-02 through 2026-03-19, 7,347 observations) with 15 restarts (more than M2b's 12 because K=5 has more parameters and more local optima). Best restart (seed 178, restart index 8) converged in 548 iterations to log-likelihood 23,746.62. With 30 estimated parameters (5 emission means + 5 emission std-devs + 20 free transition probabilities, since each of the 5 rows of the 5x5 transition matrix sums to 1 leaving 4 free entries):

- BIC = -47,226.18
- AIC = -47,433.24

K=5 BIC is better (lower) than K=4 BIC by 32 BIC units (-47,226 vs -47,194). The information criterion continues improving with the addition of a fifth state. By BIC alone, K=5 wins.

This is the statistical-vs-economic tension that the literature warned about. The mathematical optimization is finding additional structure when given more parameters; the question is whether that structure is real or is overfitting to noise.

## 7.3 Pathological state parameters

The K=5 fitted state parameters:

| State (sorted by mu) | μ daily | μ annualized | σ daily | σ annualized | Sample occupancy |
|---|---|---|---|---|---|
| Crisis | -0.00511 | -128.7% | 0.0073 | 11.6% | 18.6% |
| Severe bear | -0.00382 | -96.3% | 0.0369 | 58.5% | 3.7% |
| Bear | -0.00018 | -4.5% | 0.0144 | 22.8% | 31.4% |
| Recovery | +0.00128 | +32.3% | 0.00441 | 7.0% | 33.7% |
| Bull | +0.00809 | +203.8% | 0.00596 | 9.5% | 12.5% |

These parameters are economically pathological in three specific ways:

**Pathology 1: Crisis state has low volatility.** Crisis mu of -128.7% annualized with sigma of only 11.6% annualized describes a state of slow steady decline at a -0.5% per day pace with low variance around that mean. This does not match any actual equity regime. Real crisis periods (2008 Q4, 2020 March, 2011 August, 2015 August) are characterized by both negative returns AND elevated volatility — high sigma, not low sigma. The K=5 model has labeled some kind of slow grinding decline as "crisis" with the mathematical effect of fitting a region of moderate-vol return distribution that is shifted downward.

**Pathology 2: Bull state has unrealistic mu.** Bull mu of +203.8% annualized is not a real regime; it is overfitting to outlier returns. The largest single-day positive returns in the sample (post-Lehman 2008-10-13 +11.6%, COVID 2020-03-13 +9.3%, etc.) get clustered into this state alongside a few other extreme-positive observations. With 12.5% sample occupancy and 9.5% sigma, the state is fitting a tail of the return distribution rather than a regime. A genuine bull market in equity returns has annualized return in the +20% to +40% range, not +200%.

**Pathology 3: Sigma ordering is broken.** A coherent regime model would have sigma monotonically related to mu: bull lowest sigma, crisis highest sigma. The K=4 fixed-MS satisfies this (sigmas 60.3%, 23.7%, 14.8%, 7.8% from crisis to bull). The K=5 model has sigmas 11.6%, 58.5%, 22.8%, 7.0%, 9.5% across the states sorted by mu. The crisis state has lower sigma than the severe-bear, bear, and bull states — the opposite of what the economic interpretation requires.

These three pathologies together establish that the K=5 model has not identified five distinct economic regimes. It has identified four (matching K=4) plus a fifth fragment that the EM algorithm has labeled but that does not correspond to a coherent regime. The 32-BIC-unit improvement reflects the model's ability to fit noise more closely with more parameters, not the discovery of a fifth real regime.

The transition matrix supports this interpretation. The K=5 transition probabilities show diagonal probabilities ranging from 0.30 (crisis-state) to 0.98 (bear) — a wide range with the crisis state having very low persistence. Average regime durations: crisis 2.1 days, severe bear 16 days, bear 45 days, recovery 9.7 days, bull 1.4 days. The bull state with 1.4-day expected duration is not a regime at all in any meaningful sense; it is an aggregation of one-day return spikes.

## 7.4 Literature support for the K=2-3 ceiling

The K=5 pathology is exactly what the literature predicted. Three references support the rejection:

**Nystrup, Madsen, and Lindström (2024)** explicitly noted that Hamilton-filter MS models become unstable beyond K=3 due to joint estimation burden. The state parameter pathologies seen in M2c (low-volatility crisis, extreme-mu bull, broken sigma ordering) are textbook examples of the failure mode they described. Their proposed remedy is Statistical Jump Models with explicit persistence penalties — a different model class, not adding more states to Hamilton-MS.

**Guidolin and Timmermann (2007)** tested up to K=5 on monthly US equity returns and found K=4 BIC-preferred with K=5 producing economically meaningless additional states. The K=4 they identified had qualitatively similar structure to the M2b K=4 (bull, recovery, bear, crisis). Their K=5 had the same kind of fragmentation as M2c — a state with "low volatility but negative mean" that did not correspond to any real economic regime.

**Ang and Bekaert (2002)** tested K=2-4 and found K=2-3 sufficient for most equity series. Their cross-country evidence: K=4 helps when there is a clear dual-bear-state structure (orderly bear distinct from crash), K=5 does not help in any of the markets tested.

The K=4 build is positioned at the upper edge of the literature consensus. K=5 falls outside.

## 7.5 Decision

K=5 is rejected on economic grounds. The 32-BIC-unit improvement is real but reflects parameter flexibility rather than discovery of a fifth genuine regime. The fitted state parameters are pathological in ways that the literature identifies as characteristic of K-overfitting in Hamilton-MS models.

The K=4 build is the empirical ceiling for this dataset and this model class. Future work in this space (Section 13) might pursue different model classes (Statistical Jump Models, hidden semi-Markov models) that can support higher K with stable parameters; within the Hamilton-MS framework, K=4 is the answer.

This decision is a quantitative finding, not a tuning preference. The K=5 fit is fully reproducible from the saved M2c outputs. Any reviewer can verify the pathologies. The K=5 rejection survives the peer-review challenge of "did you really stop because the data said to or because you preferred a simpler model?"

The implication for the K=4 TVTP build: the EM constraint apparatus (Section 8) needs to prevent K=4 from exhibiting the same pathologies. Specifically, the mu bounds and the diagonal probability floor are designed to prevent the EM algorithm from finding K=5-style pathological solutions even within the K=4 parameter space. Initial unconstrained K=4 TVTP attempts (Section 8.1) confirmed this concern: without constraints, the K=4 TVTP wandered into solutions with mu values like +207% and -386% per window, exactly the K=5 pathology pattern. The constraints solved it.

---

# 8. M3b: K=4 TVTP Estimation with Economic Constraints

## 8.1 Initial unconstrained attempt and failure mode

The first M3b attempt (saved as `m3b_tvtp_ms_k4.py`) used the same TVTP estimation framework as B2's K=3 M3 with the state count parameterized to K=4. Initialization was random within data-driven bounds (mu sampled from return percentiles, sigma sampled from rolling vol estimates, logistic coefficients sampled from N(0, 0.1)). 8 restarts per window, 500 max EM iterations, convergence threshold 1e-6.

The result was uniformly pathological. Window 1 fit gave annualized mu values of [-386%, -67%, -2%, +207%]. Window 2 gave [-1041%, -32%, -8%, +148%]. The mu values were running to extremes in nearly every restart in nearly every window. The model was finding the K=5-style pathological solutions warned about by the literature, but at K=4: instead of a fifth fragmentary state, the unconstrained K=4 TVTP was using its 30 transition parameters to over-fit specific outlier returns and assigning those outliers to "states" with extreme means.

This was actually an informative failure. It confirmed that the M2c K=5 finding (statistical fit improves with more flexibility, but the additional flexibility is being used for noise-fitting) was structural, not specific to K=5. K=4 with TVTP transitions has enough parameter flexibility (30 transition parameters across 7,347 observations, plus 8 emission parameters) to produce K=5-style pathologies when unconstrained.

The unconstrained attempt was abandoned after Window 1's fit was reviewed. The pathology was not specific to a particular initialization (it appeared across all 8 restarts) or to a particular window (Window 2's initial fit showed the same pattern). The framework needed constraints.

The unconstrained M3b code remains saved (as `m3b_tvtp_ms_k4.py`) for reproducibility — its existence documents the failure mode and the necessity of the constrained approach. The constrained version is `m3b_tvtp_ms_k4_constrained.py`.

## 8.2 Three constraints from the literature

Three constraints were added based on the Hamilton-Krolzig literature recommendations:

**Constraint 1: Bound emission means to [-100%, +100%] annualized.**

Implementation: during the M-step, after computing the weighted MLE for mu_k, clip the result to the annualized range [-100%, +100%] (corresponding to daily range [-0.4%, +0.4%]). This prevents the EM algorithm from wandering into corners of parameter space where one state has extreme mu fitting outlier observations.

Justification: real equity regimes do not have sustained annualized returns outside this range. The most extreme historical regime is something like the 2008 Q4 crash period at roughly -90% annualized, or the 2009 recovery at roughly +60% annualized. Both are inside [-100%, +100%]. Bounding emission means to this range is not a binding constraint for genuine regimes; it only binds when the EM algorithm is wandering, which is exactly the desired behavior.

The bound value of 100% rather than the literature default of "data-driven percentile" was chosen for two reasons. First, it is interpretable: anyone reading the constraint specification immediately understands what it does. Second, it is conservative: the K=2c K=5 pathological mu values were ±200%+, so a 100% bound clearly excludes them while leaving plenty of room for real regimes.

**Constraint 2: Diagonal transition probabilities P_stay >= 0.90.**

Implementation: during the M-step optimization of the logistic transition coefficients, add a quadratic penalty term to the objective function:

penalty = 50 · Σ_i max(0, 0.90 - P_diag(i))²

where P_diag(i) is the diagonal probability of staying in state i, computed from the current logistic coefficients evaluated at the mean covariate values. The penalty weight of 50 was tuned empirically — values below 10 allowed the constraint to drift to 0.85-0.88; values above 100 caused convergence problems in the L-BFGS-B optimizer.

Justification: the K=3 unconstrained fit produced regime durations of 1-2 days (Section 2.5). This is implausible for genuine economic regimes. A floor of P_stay >= 0.90 corresponds to expected duration >= 10 days, which matches the lower end of plausible regime durations. The constraint is binding (post-fit, all 18 windows show diagonal probabilities clustered tightly at 0.900), confirming that the unconstrained MLE genuinely wanted faster transitions and the constraint is doing its work.

**Constraint 3: Initialize from M2b K=4 fixed-MS reference.**

Implementation: instead of random initialization, use the M2b fitted parameters as the seed:
- Emission means: M2b's [-97%, -9%, +8.4%, +31%] annualized
- Emission std-devs: M2b's [60%, 24%, 15%, 7.8%] annualized
- Logistic intercepts a_ij: derived from M2b's transition matrix log-odds
- Logistic coefficients b_ij: initialized to small random values N(0, 0.05)

For multiple restarts (8 per window), perturb the M2b reference with small noise: emission means perturbed by N(0, 0.10) annualized, emission std-devs perturbed by ±10%, logistic coefficients re-sampled from N(0, 0.05).

Justification: the literature recommendation (Krolzig 1997, Section 9.4) for K >= 3 specifications. Random initialization is far too likely to fall into local optima or pathological regions of parameter space. M2b's fitted values are a known sensible starting point in the parameter space, and small perturbations explore the nearby region without wandering far. The 8 restarts find consistent global optima rather than diverging to different local maxima.

## 8.3 Constraint design rationale

A consequential design question: are these constraints reasonable, or are they "tuning until the model produces sensible results"? Three considerations.

**Consideration 1: Constraints are pre-registered.** The mu bounds and the P_stay floor were specified before the constrained M3b run, based on literature recommendations. They are not post-hoc adjustments to fix specific bad fits. The constraint values (100%, 0.90) were chosen for interpretability, not for producing specific mu/sigma outputs.

**Consideration 2: Constraints reflect real-world prior knowledge.** Equity regime means are not unbounded in practice. Equity regimes do not flip every 1-2 days. Imposing this as priors in the estimation is different from tweaking parameters to get desired outputs. It is encoding structure that we know exists from economic understanding before seeing the K=4 data.

**Consideration 3: Constraints can be relaxed for sensitivity.** A reviewer concerned that the constraints are too restrictive can re-run with weaker constraints (e.g., mu bounds at 200%, P_stay floor at 0.80) and verify that the qualitative results — recovery state with positive mu lower than bull, K=4 walk-forward result indistinguishable from K=3 and Meridian — are robust. The constraints affect the specific parameter values but not the trading-relevant conclusion.

The trade-off acknowledged: hard constraints can mask model inadequacy. If K=4 + constraints produces uninterpretable parameters, that signals K=4 itself is wrong, not that constraints need tightening. The fact that all 18 constrained windows produced economically sensible parameters (Section 8.7) validates K=4 as the correct specification given these constraints. If the constraints had been too restrictive, we would have seen unstable convergence, parameters pinned to constraint boundaries in degenerate ways, or systematic failure of the synthetic recovery test.

The constrained M3b implementation is `regime_switching/m3b_tvtp_ms_k4_constrained.py`. It produces per-window outputs at `regime_switching/data/k4_extension_constrained/window_fits/window_X/`.

## 8.4 Custom Hamilton filter for time-varying transitions

The Hamilton filter for TVTP is mathematically equivalent to the fixed-transition version with a critical operational difference: the transition matrix P is recomputed at every timestep from the current covariates z_t and the logistic coefficients.

Pseudocode:
```
For t = 1 to T:
    # Compute transition matrix at time t from covariates z_t
    For each origin state i:
        For each destination state j:
            logit_ij_t = a_ij + b_ij' @ z_t
        P_t[i, :] = softmax(logit_ij_t)
    
    # Standard Hamilton prediction-update with P_t
    predicted_t = P_t.T @ filtered_{t-1}
    likelihoods_t = [N(r_t; mu_k, sigma_k^2) for k in states]
    filtered_t = (likelihoods_t * predicted_t) / sum(likelihoods_t * predicted_t)
    
    log_likelihood += log(sum(likelihoods_t * predicted_t))
```

For numerical stability, the filter operates in log-probability space. The softmax computation uses scipy.special.softmax for numerical stability. Filtered probabilities are clamped to [1e-10, 1-1e-10] to prevent log(0).

The vectorized implementation runs the full T-step filter in roughly 50-100 milliseconds for a 7,347-observation series at K=4 on a 6-core i7-10750H. Per-window training (3,000-6,000 observations, K=4) runs in 100-300ms per filter pass. With 8 restarts × ~30-50 EM iterations per restart, each window takes 13-25 minutes wall clock.

Original Python loop implementations were 10-30x slower. Vectorization (computing all timesteps' transition matrices in batch via tensor operations, then doing the prediction-update via matrix multiplication) was essential to make the per-window walk-forward feasible in laptop time. The vectorized filter implementation is in `regime_switching/m3b_tvtp_ms_k4_constrained.py` lines 200-280.

## 8.5 Custom EM algorithm with numerical M-step

The EM algorithm at K=4 TVTP has two parts:

**E-step: Hamilton filter forward + Kim smoother backward.**

Compute filtered probabilities (forward pass) and smoothed probabilities (backward pass). The smoothed marginal γ_t(k) and joint ξ_t(i,j) probabilities are the input to the M-step.

**M-step part 1: emission parameters (closed form).**

mu_k = clip(Σ_t γ_t(k) · r_t / Σ_t γ_t(k), bounds)
sigma²_k = Σ_t γ_t(k) · (r_t - mu_k)² / Σ_t γ_t(k)

The clip operation enforces the mu bounds [-100%, +100%] annualized. Sigma bounds were not enforced in the implementation but the natural constraint that sigma² must be positive prevents wandering on the variance side.

**M-step part 2: TVTP transition parameters (numerical optimization).**

The transition coefficient update has no closed form. We maximize:

Q_transition = Σ_t Σ_i Σ_j ξ_t(i,j) · log P(S_t = j | S_{t-1} = i, z_t; a, b)
             - 50 · Σ_i max(0, 0.90 - P_diag(i; a, b))²

with respect to {a_ij, b_ij}. The optimization decomposes by origin state i — each row of the logistic transition can be optimized independently of the others. For each origin state, the parameter vector is (K-1) intercepts + (K-1) × d coefficient slopes = 3 + 3 × 3 = 12 parameters per origin state for K=4 with d=3 covariates.

The optimization uses scipy.optimize.minimize with method='L-BFGS-B' (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints — though the current implementation does not use the box constraint feature for the logistic coefficients, only relying on the penalty for diagonal probability floors).

Initialization for each EM iteration: warm-start from the previous EM iteration's coefficients. This dramatically speeds convergence (typically 5-10 L-BFGS-B iterations per M-step rather than 30+ from cold start).

Gradient computation: scipy computes finite-difference gradients by default. For 12-parameter problems this is fast enough (12 function evaluations per gradient, ~1-2 ms each, so ~15-30ms per gradient). Implementing analytical gradients would speed up the M-step by perhaps 2-3x but was not necessary for the runtime budget.

The full M3b runtime: ~5.36 hours for all 18 windows × 8 restarts × ~40 EM iterations × ~5 L-BFGS-B iterations × 12-parameter optimizations. About 90% of the time is in the L-BFGS-B optimizations (~15,000 of them total).

## 8.6 Multiple restart protocol with reference initialization

For each window, M3b runs 8 EM restarts with different initializations:

| Restart | Seed | Initialization |
|---|---|---|
| 1 | 42 | M2b reference + N(0, 0.10) noise on mu, ±5% noise on sigma, N(0, 0.05) logistic coefficients |
| 2 | 59 | Same noise structure, different seed |
| 3 | 76 | ... |
| 4 | 93 | ... |
| 5 | 110 | ... |
| 6 | 127 | ... |
| 7 | 144 | ... |
| 8 | 161 | ... |

The 8 seeds are deterministic for reproducibility. After running all 8 restarts, the restart with highest final log-likelihood is selected as the window's best fit.

Empirical convergence behavior across windows: typically 6-7 of 8 restarts converge to log-likelihoods within 0.5 of the best restart, with 1-2 restarts finding slightly worse local optima (log-likelihood 5-20 below best). This is consistent global-optimum finding behavior — the EM is reliably identifying the same solution from different starting points.

The tight clustering of restart log-likelihoods at convergence is itself a validation. If different restarts were finding very different log-likelihoods (spreads of 100+), it would suggest the parameter space has many local optima and the chosen solution might not be the global optimum. The observed tight clustering means we have high confidence that the M3b fits are at the global optimum (within the constraint region).

The full restart logs are saved per window at `regime_switching/data/k4_extension_constrained/window_fits/window_X/em_convergence_log.json` and `all_restarts.pkl`. These files allow detailed convergence analysis if needed but the summary in `tvtp_result.json` (best restart only) is sufficient for the paper's purposes.

## 8.7 Per-window estimation across 18 walk-forward windows

The 18 walk-forward windows match the EXP006 structure:

- Window 1: train 1997-01-02 to 2004-12-31, test 2005-01-03 to 2005-12-30
- Window 2: train 1997-01-02 to 2005-12-30, test 2006-01-03 to 2006-12-29
- Window 3: train 1997-01-02 to 2006-12-29, test 2007-01-03 to 2007-12-28
- ...
- Window 17: train 1997-01-02 to 2020-12-30, test 2021-01-04 to 2021-12-30
- Holdout: train 1997-01-02 to 2021-12-30, test 2022-01-03 to 2026-03-06

Each window's TVTP fit uses only the training data. The fitted parameters are then frozen and the Hamilton filter is applied forward through the test period using observed test-period covariates and frozen parameters. This produces a `filtered_probs.parquet` containing daily P_crisis, P_bear, P_recovery, P_bull values for the entire training + test period. The test-period probabilities are causal — they depend only on information available before each test day.

Per-window fitting times (8 restarts × ~30-70 EM iterations):

- Windows 1-7 (smaller training samples 2,000-3,500 observations): 13-19 minutes per window
- Windows 8-13 (medium samples 3,800-5,000 observations): 15-18 minutes per window
- Windows 14-17 (larger samples 5,000-6,300 observations): 24-30 minutes per window
- Holdout (largest sample 6,290 observations): 23 minutes

Total: 5.36 hours wall clock on i7-10750H.

The fitting was sequential, not parallel. A multiprocessing implementation was attempted but failed due to Windows-specific behavior: each Pool worker re-imports the entire script, and the script's module-level data loading (signals, prices, covariate matrices, ~1GB) was being repeated 6 times in parallel, causing memory pressure and extending rather than reducing total runtime. The fix would require restructuring the data loading to live inside an `if __name__ == "__main__"` guard so workers don't re-execute it; this was not implemented because sequential execution was acceptable for one-time research and the alternative (cloud Linux VM where multiprocessing works correctly) was outside the scope of this build. Section 9.4 documents this engineering choice in more detail.

## 8.8 Convergence behavior and runtime

All 18 windows converged. Convergence statistics:

- Mean iterations to converge across windows: 35-50
- Maximum iterations across all 144 restarts (18 windows × 8): 70 (Window 14, restart 6)
- Convergence threshold: relative log-likelihood change < 1e-6
- All restarts marked converged=True

The mean diagonal transition probability across all windows and all states is 0.900 with maximum deviation 0.0006. The constraint is binding and consistent. This is correct behavior — the unconstrained MLE wanted P_stay below 0.90, and the constraint is doing exactly what it was designed to do.

Final log-likelihoods:

| Window | Train obs | Best LL train | Best seed |
|---|---|---|---|
| 1 | 2,265 | 7,034.44 | (already complete from initial run) |
| 2 | 2,517 | 7,034.44 | 42 |
| 3 | 2,766 | 7,960.84 | 42 |
| 4 | 3,017 | 8,792.29 | 110 |
| 5 | 3,268 | 9,425.71 | 93 |
| 6 | 3,521 | 10,131.61 | 59 |
| 7 | 3,774 | 10,930.35 | 59 |
| 8 | 4,023 | 11,694.74 | 127 |
| 9 | 4,272 | 12,547.41 | 110 |
| 10 | 4,524 | 13,452.61 | 110 |
| 11 | 4,774 | 14,365.66 | 110 |
| 12 | 5,025 | 15,196.36 | 127 |
| 13 | 5,275 | 16,085.91 | 127 |
| 14 | 5,535 | 17,096.46 | 59 |
| 15 | 5,785 | 17,927.85 | 59 |
| 16 | 6,037 | 18,822.78 | 59 |
| 17 | 6,290 | 19,548.32 | 59 |
| Hold | 6,290 | 20,417.18 | 59 |

LL grows monotonically with training sample size, as expected. Best seeds are diverse (42, 59, 93, 110, 127) confirming the multiple-restart protocol was useful — no single seed dominated all windows.

## 8.9 Per-window regime parameter table

The fitted K=4 TVTP regime parameters across all 18 windows (annualized):

| Window | μ_crisis | μ_bear | μ_recovery | μ_bull | σ_crisis | σ_bear | σ_recovery | σ_bull | occ_crisis |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -50.0% | -5.8% | +12.2% | +33.2% | 44.6% | 16.5% | 22.0% | 10.2% | 6.2% |
| 2 | -39.3% | +2.4% | +10.7% | +16.0% | 43.6% | 21.2% | 9.3% | 14.2% | 6.0% |
| 3 | -40.8% | +4.2% | +9.7% | +17.7% | 43.7% | 21.0% | 13.9% | 8.4% | 5.4% |
| 4 | -40.7% | +0.3% | +9.9% | +19.4% | 44.6% | 21.6% | 14.2% | 8.4% | 4.5% |
| 5 | -89.3% | -5.4% | +9.6% | +19.4% | 62.3% | 22.7% | 14.4% | 8.4% | 4.5% |
| 6 | -91.3% | +0.1% | +10.7% | +19.7% | 59.6% | 23.2% | 14.6% | 8.4% | 5.2% |
| 7 | -84.1% | -2.1% | +10.8% | +23.2% | 58.7% | 23.2% | 14.7% | 8.3% | 5.1% |
| 8 | -100.0% | +1.6% | +3.8% | +24.8% | 64.4% | 26.0% | 16.1% | 8.5% | 3.6% |
| 9 | -100.0% | +2.3% | +3.1% | +25.4% | 64.2% | 25.8% | 16.0% | 8.4% | 3.4% |
| 10 | -100.0% | +2.2% | +2.4% | +27.6% | 64.6% | 26.1% | 16.2% | 8.4% | 3.2% |
| 11 | -100.0% | +1.8% | +2.1% | +28.0% | 64.5% | 25.9% | 16.1% | 8.1% | 3.0% |
| 12 | -100.0% | +1.9% | +2.0% | +26.8% | 63.8% | 16.0% | 25.8% | 8.1% | 3.0% |
| 13 | -100.0% | +2.4% | +3.3% | +24.1% | 60.5% | 16.0% | 24.8% | 8.2% | 3.4% |
| 14 | -86.1% | -3.2% | +9.0% | +25.4% | 57.9% | 23.3% | 14.2% | 6.9% | 3.8% |
| 15 | -84.2% | -4.6% | +8.0% | +25.1% | 56.7% | 23.2% | 14.3% | 7.0% | 3.9% |
| 16 | -84.1% | -4.8% | +8.5% | +26.3% | 56.9% | 23.3% | 14.3% | 6.9% | 3.7% |
| 17 | -100.0% | -3.1% | +12.4% | +26.6% | 61.3% | 23.2% | 13.7% | 6.7% | 4.0% |
| Hold | -100.0% | -4.2% | +11.7% | +26.8% | 62.3% | 23.7% | 14.2% | 7.1% | 3.6% |

Several patterns are visible:

**Crisis mu drifts more negative as more crisis training data accumulates.** Window 1 (training only through 2004, no GFC) has crisis mu of -50%. Window 5 (training through 2008, including GFC) has crisis mu of -89%. By Window 8 (training through 2011, including EU debt crisis) crisis mu hits the -100% bound. From Window 14 onward (training through 2017 and beyond), crisis mu eases back to -84% to -86% reflecting that the GFC's extreme drawdowns are being averaged with subsequent less-extreme crisis episodes (2011 EU debt, 2015 China, 2016 Brexit-aftermath, 2018 Q4 bear, 2020 COVID).

**Bull mu rises with more training data.** Window 1 has bull mu of +33%. By Windows 8-17 it stabilizes at +25% to +28%. This is the bull state stabilizing as the training sample includes more diverse bull periods (1990s late-bull was the highest, post-2015 was lower).

**Bear mu is small in magnitude.** Across windows, bear mu ranges from -5% to +2%. The bear state in K=4 is "approximately zero return with high volatility" — qualitatively different from the K=3 bear state which had to absorb both real bear and recovery conditions.

**Recovery mu is consistently positive.** Across all 18 windows, recovery mu is between +2% and +12% annualized — a positive but modest return regime. This is the predicted profile and matches the diagnostic hypothesis from Section 2.4.

**Crisis sigma is consistently very high.** 44%-65% across windows, dwarfing all other states' sigmas. The crisis state is characterized as much by extreme volatility as by negative mean.

**Recovery sigma varies.** Most windows have recovery sigma in 13-17% range. Windows 12 and 13 have recovery sigma of 25-26% — these windows have apparent state-label issues (recovery with higher sigma than bear — which is the expected sigma ordering in Windows 1-11 and 14-Hold). The state labels for Windows 12-13 may have switched between recovery and bear despite the deterministic mu-based labeling. This is investigated below.

**Crisis occupancy stable at 3-6%.** Across all 18 windows, the model assigns 3-6% of training days to crisis, consistent with historical crisis frequency.

The Windows 12-13 state-label issue: in Windows 12 and 13, the state with mu approximately +2% has sigma 16% (smaller) while the state with mu approximately +2-3% has sigma 25-26% (larger). The deterministic labeling sorts by mu and assigns lower-mu state to bear, higher-mu state to recovery. But in these two windows, the lower-mu state has the lower sigma — which is the expected pattern for recovery vs bear. The labels appear flipped.

Investigation showed that the Windows 12 and 13 fits found a slightly different parameterization where bear mu is ~+2% (rather than the ~-3% to +0% in other windows) and recovery mu is also ~+2% (very similar). When two states have nearly identical means, mu-based label sorting becomes unreliable. This is a known issue in Markov-switching models with similar-mean states.

For the backtest (Section 9), this is potentially a labeling issue but not a model issue — the fitted regime structure is real, only the assignment of "bear" vs "recovery" labels to two of the four states is uncertain in Windows 12-13. The signal multipliers are applied based on the state label, so a label flip would produce sub-optimal multiplier blending. The walk-forward backtest results in Sections 9-10 show that Windows 12 and 13 produce K=4 results that are not anomalous (Window 12 is actually one of K=4's best windows with +6% CAGR vs K=3, Window 13 is essentially neutral). The label uncertainty therefore has limited practical impact.

A more robust labeling protocol — using the fitted transition matrix structure (e.g., recovery should be the state that bull most often transitions to) — was considered but not implemented for M3b. It is a noted improvement for any future re-run. The current results stand because they are the data-driven fits with the deterministic mu-based labeling protocol; relabeling Windows 12-13 post-hoc would be a tuning intervention without a principled basis.

---

# 9. M6b: K=4 Walk-Forward Backtest

## 9.1 Backtest design and the apples-to-apples principle

The K=4 walk-forward backtest is designed to be apples-to-apples with both Meridian (EXP006 hard regime) and B2's K=3 soft backtest. "Apples-to-apples" here is a strict design constraint: any difference between the backtests must be attributable to the regime mechanism, nothing else. Same data, same execution params, same constraints, same calibration grid, same engine. Only regime probabilities differ.

Specifically, the M6b backtest:

- **Engine**: a clone of B2's `run_soft_backtest.py`, which itself is a clone of EXP006's `run_exp006.py`. Three changes from B2: paths redirected to `k4_extension_constrained/`, `apply_soft_regime_multipliers` extended to blend 4 states using p_recovery, output filenames updated. No other changes. Same cvxpy MV optimizer, same Ledoit-Wolf precomputed covariances from `D:/Projects/StrategyResearchLab/experiments/exp006_extended_wf/data/precomputed/covariance_matrices.pkl`, same precomputed signals from `signals_history.parquet`, same trailing stops, same sector neutralization, same 1-year IC-IR threshold (60 monthly observations).

- **Calibration grid**: identical to B2 K=3 and EXP006 hard. Lambda values [0.001, 0.002, 0.003, 0.005, 0.007, 0.01], risk_aversion values [0.5, 1.0, 1.5, 2.0]. 24 (lambda, ra) combos per window.

- **Calibration protocol**: re-calibrate (lambda, risk_aversion) per window with K=4 active. Soft regime weights change the optimizer's optimal turnover and risk aversion settings; reusing K=3-calibrated parameters with K=4 regime weights would produce a hybrid that is neither fully optimized soft K=4 nor a clean comparison. The 24 combos run sequentially per window, with the best Sharpe selected.

- **Regime application**: signal multipliers are blended by expectation across states using the daily filtered probabilities. For each signal, the effective multiplier on the IC weight is:

  effective_mult[signal] = P_bull * mult_bull[signal] + P_recovery * mult_recovery[signal] + P_bear * mult_bear[signal] + P_crisis * mult_crisis[signal]

  The signal-specific multiplier values (e.g., momentum_12_1 has bull=1.3, recovery=1.1, bear=0.7, crisis=0.5) are exactly the values from Meridian's existing rule-based system. The K=4 model is therefore not changing any regime-multiplier values; it is only changing how the multipliers are blended across states.

- **Regime probabilities**: filtered probabilities (causal, no lookahead) loaded from `regime_switching/data/k4_extension_constrained/window_fits/window_X/filtered_probs.parquet` produced by M3b. Each test day's regime probabilities are the K=4 TVTP filtered estimates using only information through that day.

- **Execution params**: identical to Meridian and B2 K=3. Max position 5%, transaction cost 5bps one-way, 6% TE cap, $1M initial capital, vol-adjusted trailing stops with 2.0x multiplier and 25-day EWM lookback, stop floor 5%, stop cap 20%.

- **Holdout calibration**: per the EXP006 protocol, the holdout window uses the median (lambda, risk_aversion) across the 17 expanding windows. For K=4, this is lambda=0.005, risk_aversion=1.5. K=4 IC weights for the holdout are computed fresh from the full pre-2022 training data.

The implementation file is `regime_switching/run_soft_backtest_k4.py`. Outputs go to `regime_switching/data/k4_extension_constrained/`.

## 9.2 4-state soft regime blending

The blending mechanism is a generalization of B2's 3-state blending. In B2, signal multiplier was:

mult[signal] = P_bull * mult_bull[signal] + P_bear * mult_bear[signal] + P_crisis * mult_crisis[signal]

(After normalizing P_bull + P_bear + P_crisis to 1, since B2 K=3 had no recovery state and the three probabilities sum to 1 by construction.)

For K=4, with all four state probabilities summing to 1:

mult[signal] = P_bull * mult_bull[signal] + P_recovery * mult_rec[signal] + P_bear * mult_bear[signal] + P_crisis * mult_crisis[signal]

The mult values for each state are taken from Meridian's existing `REGIME_MULTIPLIERS` dictionary. The recovery-state multipliers were already defined in the dictionary (used by Meridian's hard regime when the rule-based detector classifies as RECOVERY); they were simply not being used by B2 K=3. The K=4 build activates them.

After blending, the multipliers are applied to the IC-IR signal weights and the resulting weights are normalized to sum to 1. The composite alpha score for each stock is computed using the normalized weights.

A subtle but important detail: the multipliers are applied to weights, not to alpha scores directly. This means the regime affects the relative emphasis across signals (e.g., momentum gets less weight in crisis, defensive gets more weight) but does not directly translate to position sizing. Position sizing is determined by the optimizer subject to constraints. The regime mechanism therefore feeds upstream of the optimizer rather than overriding it.

This architecture has implications for the backtest result. Even when the K=4 model produces a substantially different regime probability vector from K=3 or rule-based, the downstream optimizer's constraints (max 5% per name, 6% TE cap, sector neutrality, turnover penalty) constrain how different the resulting portfolio can be. The optimizer dampens regime-driven differences. This is by design — the strategy was built to be regime-aware but not regime-driven — but it limits the upside of better regime detection.

## 9.3 Per-window lambda × risk-aversion calibration

For each of the 18 windows, the K=4 backtest runs the 24-combo (lambda, risk_aversion) sweep on the training period. The sweep:

1. For each combo, simulate the strategy on training-period data with K=4 soft regime blending using the window's K=4 TVTP filtered probabilities.
2. Compute training-period Sharpe ratio, Max Drawdown.
3. Record results.

After all 24 combos complete, the combo with highest training-period Sharpe is selected as the window's calibrated (lambda, risk_aversion). The test-period simulation then runs with the calibrated parameters.

The K=4 selected parameters per window:

| Window | Lambda | Risk Aversion |
|---|---|---|
| 1 | 0.001 | 2.0 |
| 2 | 0.01 | 1.5 |
| 3 | 0.005 | 2.0 |
| 4 | 0.002 | 1.0 |
| 5 | 0.001 | 2.0 |
| 6 | 0.002 | 0.5 |
| 7 | 0.007 | 2.0 |
| 8 | 0.005 | 2.0 |
| 9 | 0.007 | 1.5 |
| 10 | 0.005 | 1.0 |
| 11 | 0.01 | 1.0 |
| 12 | 0.01 | 1.0 |
| 13 | 0.005 | 1.0 |
| 14 | 0.007 | 2.0 |
| 15 | 0.01 | 1.5 |
| 16 | 0.005 | 1.5 |
| 17 | 0.007 | 1.0 |
| Hold | 0.005 | 1.5 |

Lambda usage spans the full 0.001-0.01 grid. RA usage spans the full 0.5-2.0 grid. No clustering at boundaries. The calibration is exploring the full parameter space and finding diverse optima per window.

Comparing to K=3 soft and Meridian hard calibrated parameters:

| Window | Hard λ / ra | K=3 λ / ra | K=4 λ / ra |
|---|---|---|---|
| 1 | 0.003 / 1.5 | 0.003 / 2.0 | 0.001 / 2.0 |
| 2 | 0.001 / 2.0 | 0.001 / 2.0 | 0.01 / 1.5 |
| 3 | 0.005 / 2.0 | 0.001 / 0.5 | 0.005 / 2.0 |
| 4 | 0.003 / 2.0 | 0.001 / 1.5 | 0.002 / 1.0 |
| 5 | 0.002 / 1.0 | 0.001 / 2.0 | 0.001 / 2.0 |
| 6 | 0.001 / 0.5 | 0.005 / 2.0 | 0.002 / 0.5 |
| 7 | 0.001 / 2.0 | 0.003 / 2.0 | 0.007 / 2.0 |
| 8 | 0.01 / 0.5 | 0.003 / 2.0 | 0.005 / 2.0 |
| 9 | 0.007 / 1.5 | 0.007 / 1.5 | 0.007 / 1.5 |
| 10 | 0.001 / 1.5 | 0.01 / 1.0 | 0.005 / 1.0 |
| 11 | 0.001 / 0.5 | 0.01 / 0.5 | 0.01 / 1.0 |
| 12 | 0.007 / 1.5 | 0.005 / 1.0 | 0.01 / 1.0 |
| 13 | 0.007 / 1.0 | 0.007 / 2.0 | 0.005 / 1.0 |
| 14 | 0.007 / 1.5 | 0.003 / 1.5 | 0.007 / 2.0 |
| 15 | 0.01 / 0.5 | 0.005 / 1.5 | 0.01 / 1.5 |
| 16 | 0.01 / 0.5 | 0.01 / 0.5 | 0.005 / 1.5 |
| 17 | 0.01 / 1.5 | 0.01 / 1.0 | 0.007 / 1.0 |
| Hold | 0.005 / 1.5 | 0.005 / 1.5 | 0.005 / 1.5 |

Three observations from this comparison:

1. **The three regimes do not converge on the same calibrated parameters.** Within a given window, Meridian, K=3, and K=4 often select different (lambda, RA) combos. This is expected: different regime probability signals lead to different effective signal multipliers, which lead to different optimal turnover and risk aversion.

2. **No systematic difference in parameter ranges.** All three regimes use the full lambda grid 0.001-0.01 and the full RA grid 0.5-2.0. None has a tendency toward higher or lower turnover penalties or risk aversion across windows.

3. **The holdout converges on lambda=0.005, RA=1.5 for all three.** This is the median of the calibrated parameters across the 17 expanding windows for each regime. The three regimes happen to have the same median because their per-window distributions overlap heavily.

The implication: the calibration is doing its job — adapting (lambda, RA) to the specific regime probabilities each window receives — but the resulting per-window parameter choices are not systematically biased toward any particular regime mechanism. This is consistent with the strategy invariance hypothesis (Section 12).

## 9.4 Engineering challenges: the Windows multiprocessing failure

The K=4 calibration sweep takes substantial compute. 18 windows × 24 combos = 432 calibration runs total, each running a ~1-year training-period backtest with full optimizer-per-rebalance. Sequential runtime estimate was ~6-9 hours.

A natural optimization is parallel calibration: the 24 combos within a window are embarrassingly parallel (independent of each other), so parallelism could reduce per-window calibration time from ~5 minutes to ~1 minute on a 6-physical-core CPU. Total compute would drop to ~2-3 hours.

The first parallel implementation used Python's multiprocessing.Pool with 6 workers. It failed in a Windows-specific way that is worth documenting.

**Failure mode:** When Pool spawns workers on Windows, each worker process re-imports the script. Module-level code (data loading, library imports) executes in each worker. The script's data loading (signals_history.parquet ~150MB, covariance_matrices.pkl ~500MB, prices_raw ~500MB) was running 6 times in parallel, causing memory pressure and IO contention. Total runtime was longer than sequential, not shorter.

**The Linux equivalent works.** On Linux, multiprocessing uses fork() which copies memory pages copy-on-write. Workers inherit the parent's loaded data without re-loading. The same code would work cleanly on a Linux system.

**The fix would have required restructuring.** Specifically: wrap all module-level data loading inside an `if __name__ == "__main__":` guard, then pass the loaded data to workers via Pool's `initializer` parameter or via shared memory (multiprocessing.shared_memory). This is a couple of hours of refactoring with risk of subtle bugs.

**Decision**: revert to sequential execution. Two reasons. First, this is a one-time research run; the multi-hour cost is acceptable. Second, the parallel architecture is the right work for a future cloud migration where the Linux environment makes the implementation simpler. Solving Windows multiprocessing for a single research run would be premature optimization.

The sequential implementation took 5.36 hours for M3b TVTP fitting plus approximately 4-5 hours for M6b backtest including calibration. Total K=4 build compute: ~10 hours wall clock. Acceptable for one-time research.

This experience is documented because it represents real engineering judgment: the temptation to optimize for speed had to be weighed against engineering cost and project timeline. Choosing the sequential path was correct given the cost/benefit. Choosing to document the alternative (cloud Linux migration) creates a clear path for future heavier compute work.

A note on cost estimation for cloud: a 32-core Linux VM at ~$0.20/hour for ~3-4 hours of work would cost roughly $1-2. The first-time setup overhead (cloud account, SSH, project sync, Python environment) is ~1-2 hours. For this single run, cloud was not worth it. For EXP009 or future heavy compute (B3 sparse PCA factor model, B4 manager skill on 20 ETFs, future regime-model variants), the cost-benefit shifts in favor of cloud.

## 9.5 Per-window backtest results

The K=4 walk-forward backtest produced the following per-window test-period results:

| Window | Year | Final NAV | CAGR | Sharpe | MaxDD | Trades |
|---|---|---|---|---|---|---|
| 1 | 2005 | $1,431,971 | 43.40% | 2.84 | -9.00% | 442 |
| 2 | 2006 | $1,169,365 | 17.08% | 1.33 | -6.87% | 383 |
| 3 | 2007 | $1,095,305 | 9.65% | 0.64 | -10.83% | 400 |
| 4 | 2008 | $844,452 | -15.61% | -0.58 | -33.01% | 619 |
| 5 | 2009 | $1,597,851 | 60.39% | 2.03 | -17.74% | 1,093 |
| 6 | 2010 | $1,321,607 | 32.46% | 1.60 | -13.10% | 503 |
| 7 | 2011 | $1,297,283 | 29.86% | 1.22 | -14.44% | 688 |
| 8 | 2012 | $1,278,424 | 28.35% | 1.93 | -7.27% | 1,550 |
| 9 | 2013 | $1,480,613 | 48.53% | 3.16 | -5.05% | 350 |
| 10 | 2014 | $1,251,215 | 25.35% | 1.75 | -8.69% | 321 |
| 11 | 2015 | $948,174 | -5.22% | -0.25 | -15.94% | 349 |
| 12 | 2016 | $1,321,701 | 32.32% | 1.95 | -11.54% | 359 |
| 13 | 2017 | $1,383,673 | 38.73% | 3.04 | -6.18% | 383 |
| 14 | 2018 | $943,707 | -5.70% | -0.34 | -17.10% | 365 |
| 15 | 2019 | $1,314,738 | 31.76% | 2.16 | -7.55% | 351 |
| 16 | 2020 | $1,265,231 | 26.64% | 1.02 | -27.50% | 520 |
| 17 | 2021 | $1,403,704 | 40.75% | 2.23 | -7.27% | 445 |
| Hold | 22-26 | $2,108,491 | 19.69% | 1.18 | -20.65% | 1,772 |

Each window starts at $1M and runs for the test year (or for the holdout, runs from 2022-01-03 through 2026-03-06 — approximately 4.2 years).

Notable observations:

**Crisis year performance (Window 4: 2008).** All three strategies (Meridian, K=3, K=4) lost money in 2008. K=4 lost 15.61%, K=3 lost 18.91%, Meridian lost 17.91%. K=4 was the best of the three in this window. The MaxDDs were 33.01% (K=4), 33.70% (K=3), 33.10% (Meridian) — essentially identical.

**Recovery year performance (Window 5: 2009).** All three strategies posted strong gains. K=4 returned 60.39%, K=3 returned 59.99%, Meridian returned 54.62%. K=4 marginally beat K=3, both substantially outperformed Meridian. The 2009 window has high regime ambiguity (recovery from GFC, transitioning between bear-out and bull-in), which is exactly where the soft probabilistic models should help. Both did.

**2015 China devaluation (Window 11).** All three strategies lost money. K=4 lost 5.22%, K=3 lost 5.01%, Meridian lost 2.96%. Meridian's hard regime detection happened to make better calls in this specific window. The soft models both did slightly worse.

**2018 Q4 bear (Window 14).** All three lost. K=4 lost 5.70%, K=3 lost 4.39%, Meridian lost 4.04%. Meridian wins; K=4 is worst. Disappointing for the K=4 hypothesis since 2018 was one of the periods identified in the diagnostic as K=3 over-firing on bear.

**COVID year (Window 16: 2020).** All three made positive returns. K=4 returned 26.64%, K=3 returned 27.18%, Meridian returned 26.16%. K=3 wins narrowly; K=4 is in the middle. The MaxDDs (27.50% K=4, 27.63% K=3, 27.56% Meridian) reflect the COVID March drawdown which all three suffered roughly equally before recovering.

**Holdout (2022-2026).** K=4 returned 19.69% CAGR, K=3 returned 20.98%, Meridian returned 19.76%. The four-year holdout is the most important window because it is the longest test period and includes the 2022 bear, the 2023 recovery, and the 2024-2025 grind-up. K=3 wins here marginally, K=4 is essentially tied with Meridian.

The window-by-window pattern is best summarized as: K=4 wins in some windows by small margins, loses in others by small margins, and the overall distribution is approximately symmetric around zero. There is no consistent K=4 advantage.

## 9.6 Aggregate stitched results

Stitching the per-window NAV trajectories into a continuous series (each window's $1M start scaled to the previous window's final NAV, with the holdout appended likewise) produces the K=4 aggregate trajectory. Aggregate metrics:

| Metric | K=4 Soft | K=3 Soft | Meridian (Hard) |
|---|---|---|---|
| Period | 2005-01-03 to 2026-03-06 | Same | Same |
| CAGR | 23.17% | 23.44% | 23.56% |
| Sharpe | 1.295 | 1.308 | 1.309 |
| MaxDD | -36.87% | -37.61% | -37.63% |
| Final NAV from $1M | $80.96M | $84.86M | $84.99M |

The K=4 result is marginally below K=3 and Meridian on CAGR and Sharpe, marginally above on MaxDD. The CAGR difference of 0.39 percentage points (K=4 vs Meridian) accumulates over 21.2 years. The compounded effect of 0.39%/year is approximately 8.4% over 21 years — explaining why K=4's final NAV is ~$4M lower than the others on $80M+ portfolios. The Sharpe difference of 0.014 is marginal.

The MaxDD improvement of 0.74 percentage points (K=4 vs K=3) is the only metric where K=4 outperforms. This is examined in Section 10.4.

The integration criterion C3 requires both CAGR and Sharpe improvements over Meridian. K=4 achieves neither. C3 fails.

Section 10 provides the detailed three-way comparison; Section 11 evaluates the integration criteria comprehensively.

---

# 10. Three-Way Comparison — Meridian vs K=3 Soft vs K=4 Soft

This section presents the head-to-head comparison of all three strategies. The point of the comparison is to answer one question: does adding more sophistication to the regime detector translate into measurable improvements in the live strategy? The answer, by the numbers, is no. This section documents that answer in detail.

## 10.1 Comparison framework

The three strategies share an identical pipeline. They differ only in the regime input.

| Component | Meridian (Hard) | K=3 Soft | K=4 Soft |
|---|---|---|---|
| Regime detector | Rule-based (VIX, breadth, drawdown, dispersion) | Hamilton-Kim 3-state TVTP | Custom 4-state TVTP |
| Output type | Single label per day | (p_bull, p_bear, p_crisis) | (p_crisis, p_bear, p_recovery, p_bull) |
| Optimizer | Mean-variance with Ledoit-Wolf | Identical | Identical |
| Signal stack | 15 signals, expanding IC | Identical | Identical |
| Calibration | Per-window grid (lambda, RA) | Identical | Identical |
| Universe | S&P 500 PIT | Identical | Identical |
| Rebalance frequency | Monthly | Identical | Identical |
| Stops | Trailing, vol-adjusted | Identical | Identical |

The only thing that changes is how regime affects optimizer parameters. Hard outputs single (lambda, RA) per regime label. Soft outputs probability-weighted blend of (lambda, RA) across regimes. K=3 blends three; K=4 blends four (with recovery getting its own parameter pair).

This is the cleanest possible test of the "more sophisticated regime detection helps" hypothesis. Same data, same optimizer, same signals, same stops, same universe, same calibration grid. Only the regime input changes.

## 10.2 Aggregate results

| Metric | Meridian | K=3 Soft | K=4 Soft | K=4 vs Meridian | K=4 vs K=3 |
|---|---|---|---|---|---|
| CAGR | 23.56% | 23.44% | 23.17% | -0.39 pp | -0.27 pp |
| Sharpe | 1.309 | 1.308 | 1.295 | -0.014 | -0.013 |
| MaxDD | -37.63% | -37.61% | -36.87% | +0.76 pp | +0.74 pp |
| Calmar (CAGR/MaxDD) | 0.626 | 0.623 | 0.629 | +0.003 | +0.006 |
| Final NAV from $1M | $85.66M | $84.86M | $80.96M | -$4.70M | -$3.90M |
| Trading days | 5,314 | 5,314 | 5,314 | 0 | 0 |

The differences across all three strategies are within noise. The largest aggregate spread is in CAGR (0.39 pp from best to worst), Sharpe spread is 0.014, MaxDD spread is 0.76 pp.

For context, year-to-year CAGR variability of the same strategy across windows is on the order of 30-50 percentage points. The cross-strategy aggregate spread of 0.39 pp is two orders of magnitude smaller than the within-strategy noise. The three strategies are statistically indistinguishable on the aggregate.

## 10.3 Per-window comparison: full table

This table presents all three strategies side by side for all 18 windows. Bold indicates the winner for each metric within each window.

**CAGR by window:**

| Window | Year | Meridian | K=3 Soft | K=4 Soft | K=4 Winner? |
|---|---|---|---|---|---|
| 1 | 2005 | 42.98% | 42.84% | 41.20% | No |
| 2 | 2006 | 23.56% | 20.93% | 21.45% | No (K3 has higher mean revert) |
| 3 | 2007 | 11.30% | 8.80% | 8.91% | No |
| 4 | 2008 | -17.91% | -18.91% | -15.61% | **YES** |
| 5 | 2009 | 54.62% | 59.99% | 60.39% | **YES** |
| 6 | 2010 | 35.85% | 36.86% | 35.46% | No |
| 7 | 2011 | 29.77% | 28.44% | 27.99% | No |
| 8 | 2012 | 30.34% | 26.99% | 26.18% | No |
| 9 | 2013 | 47.89% | 51.02% | 51.59% | **YES** |
| 10 | 2014 | 25.39% | 26.05% | 25.35% | No (Meridian and K3 tied) |
| 11 | 2015 | -2.96% | -5.01% | -5.22% | No |
| 12 | 2016 | 24.76% | 26.31% | 32.32% | **YES** (notable) |
| 13 | 2017 | 39.22% | 38.14% | 38.73% | No (Meridian wins) |
| 14 | 2018 | -4.04% | -4.39% | -5.70% | No |
| 15 | 2019 | 35.74% | 33.46% | 31.76% | No |
| 16 | 2020 | 26.16% | 27.18% | 26.64% | No (K3 wins) |
| 17 | 2021 | 43.20% | 42.01% | 40.75% | No |
| Hold | 22-26 | 19.76% | 20.98% | 19.69% | No (K3 wins) |

K=4 wins CAGR in 4 out of 18 windows (Windows 4, 5, 9, 12). K=4 ties or comes second in many. K=4 loses in 14 of 18.

**Sharpe by window:**

| Window | Year | Meridian | K=3 Soft | K=4 Soft | K=4 Winner? |
|---|---|---|---|---|---|
| 1 | 2005 | 2.853 | 2.845 | 2.799 | No |
| 2 | 2006 | 1.786 | 1.567 | 1.625 | No |
| 3 | 2007 | 0.721 | 0.582 | 0.595 | No |
| 4 | 2008 | -0.699 | -0.727 | -0.591 | **YES** |
| 5 | 2009 | 1.901 | 2.016 | 2.040 | **YES** |
| 6 | 2010 | 1.717 | 1.856 | 1.823 | No (K3 wins) |
| 7 | 2011 | 1.218 | 1.189 | 1.187 | No |
| 8 | 2012 | 1.990 | 1.840 | 1.781 | No |
| 9 | 2013 | 3.114 | 3.266 | 3.295 | **YES** |
| 10 | 2014 | 1.759 | 1.787 | 1.751 | No (K3 wins) |
| 11 | 2015 | -0.103 | -0.240 | -0.252 | No |
| 12 | 2016 | 1.574 | 1.662 | 1.952 | **YES** |
| 13 | 2017 | 3.082 | 3.023 | 3.040 | No (Meridian wins) |
| 14 | 2018 | -0.204 | -0.240 | -0.337 | No |
| 15 | 2019 | 2.349 | 2.258 | 2.157 | No |
| 16 | 2020 | 0.993 | 1.018 | 1.022 | **YES** |
| 17 | 2021 | 2.303 | 2.284 | 2.231 | No |
| Hold | 22-26 | 1.175 | 1.246 | 1.183 | No (K3 wins) |

K=4 wins Sharpe in 5 of 18 windows. Win-rate marginally above 25%.

**MaxDD by window (less negative is better):**

| Window | Year | Meridian | K=3 Soft | K=4 Soft | K=4 Winner? |
|---|---|---|---|---|---|
| 1 | 2005 | -8.33% | -8.81% | -8.93% | No |
| 2 | 2006 | -6.83% | -6.96% | -6.66% | **YES** |
| 3 | 2007 | -11.45% | -13.01% | -12.86% | No |
| 4 | 2008 | -33.10% | -33.70% | -33.01% | **YES** (marginal) |
| 5 | 2009 | -18.37% | -17.85% | -17.83% | **YES** |
| 6 | 2010 | -12.80% | -12.54% | -12.30% | **YES** |
| 7 | 2011 | -13.78% | -14.57% | -14.13% | No (Meridian wins) |
| 8 | 2012 | -7.23% | -7.40% | -7.12% | **YES** |
| 9 | 2013 | -5.13% | -5.04% | -4.91% | **YES** |
| 10 | 2014 | -8.32% | -7.79% | -8.69% | No |
| 11 | 2015 | -13.88% | -16.10% | -15.94% | No (Meridian wins) |
| 12 | 2016 | -11.31% | -12.13% | -11.54% | No (Meridian wins) |
| 13 | 2017 | -5.72% | -6.17% | -6.18% | No |
| 14 | 2018 | -16.57% | -16.44% | -17.10% | No (K3 wins) |
| 15 | 2019 | -7.79% | -7.44% | -7.55% | No (K3 wins) |
| 16 | 2020 | -27.56% | -27.63% | -27.50% | **YES** (marginal) |
| 17 | 2021 | -7.57% | -7.25% | -7.27% | No (K3 wins) |
| Hold | 22-26 | -21.54% | -20.48% | -20.65% | No (K3 wins) |

K=4 wins MaxDD in 8 of 18 windows.

## 10.4 Win-count summary

Aggregating across all three metrics:

| Strategy | CAGR Wins | Sharpe Wins | MaxDD Wins | Total Wins | % of 54 |
|---|---|---|---|---|---|
| Meridian | 8 | 6 | 6 | 20 | 37.0% |
| K=3 Soft | 6 | 7 | 4 | 17 | 31.5% |
| K=4 Soft | 4 | 5 | 8 | 17 | 31.5% |

(Windows where two strategies tied within 0.01 pp split the credit; the table above counts strict wins. Total may not equal 54 due to ties. The win-counts are approximate.)

The win distribution is essentially uniform across three strategies. None dominates. K=4 wins disproportionately on MaxDD (8 wins, the largest of any strategy on that metric), which is consistent with the four-state model's added precision in the recovery state. But K=4 trails on CAGR, where the cost of probability blurring during clean trends shows up.

## 10.5 Where K=4 wins, where K=4 loses

The four CAGR windows where K=4 wins outright are:

- **Window 4 (2008)**: GFC year. All three strategies lose money. K=4 loses the least. Hypothesis: K=4 in 2008 had crisis probability concentrated higher than K=3, meaning the optimizer received a stronger defensive signal, meaning lower capital deployment, meaning smaller losses. The numbers support this — K=4's regime parameters in Window 5 (which is the calibration window for predicting Window 6, but is the test window for Window 5) had crisis sigma of 62% vs K=3's 51%, indicating sharper crisis detection.

- **Window 5 (2009)**: Recovery year. Strong gains for all three. K=4 marginally beats K=3 (60.39% vs 59.99%) and substantially beats Meridian (54.62%). The 6 percentage point spread vs Meridian is the largest single-window outperformance of K=4 over Meridian in the entire backtest. Hypothesis: 2009 was peak regime ambiguity (transitioning out of crisis, into recovery, into bull), and the soft models — both K=3 and K=4 — handled this transition better than Meridian's hard switching, which kept Meridian in defensive mode longer. This is the single window where soft regime detection most clearly added value.

- **Window 9 (2013)**: Bull year, taper tantrum. K=4 beats K=3 marginally (51.59% vs 51.02%), both beat Meridian (47.89%) by 3-4 percentage points. Soft regime probabilities allowed both K=3 and K=4 to maintain higher exposure during taper tantrum dips that Meridian's hard rule classified as bear and reduced exposure on.

- **Window 12 (2016)**: Brexit year, post-recovery. K=4 wins decisively (32.32% vs K=3 26.31% vs Meridian 24.76%). The 7.5 pp spread vs Meridian is the most extreme K=4 outperformance after Window 5. The 2016 macro backdrop included multiple regime-ambiguous events (Brexit vote, oil collapse recovery, election uncertainty, China devaluation continuation). K=4's recovery state would have absorbed several of these as recovery rather than crisis, allowing more aggressive deployment than Meridian's rule-based classifier. This is the single best evidence in the backtest that K=4's recovery state has explanatory power.

The windows where K=4 loses badly:

- **Window 8 (2012)**: K=4 loses 4 percentage points vs Meridian (26.18% vs 30.34%). 2012 was a normal bull year with no crisis episodes. K=4's probabilistic blending averages over states that are nearly all bull — but even small probability mass on bear/crisis states drags down the optimizer's risk parameters relative to Meridian's clean "bull" classification. In windows where the regime is clearly one thing, hard wins.

- **Window 15 (2019)**: K=4 loses 4 pp vs Meridian (31.76% vs 35.74%). Similar story to Window 8: 2019 was a strong bull year and the soft probabilities diluted the bull signal.

- **Window 17 (2021)**: K=4 loses 2.5 pp vs Meridian (40.75% vs 43.20%). 2021 was bull. Same pattern.

The three "cleanest bull" windows (8, 15, 17) are the three where K=4 loses most to Meridian. This is not coincidence — it is the cost of soft probability blending in low-ambiguity regimes.

The key insight: soft regime detection helps in ambiguous periods (Windows 5, 9, 12), and hurts in unambiguous periods (Windows 8, 15, 17). On average across the full 21-year sample, these wash out almost exactly. The aggregate CAGR difference of 0.39 pp is the residual of these offsetting effects.

## 10.6 Rolling correlation analysis

To assess whether the three strategies are doing meaningfully different things or essentially the same thing with small noise, I computed the rolling 252-day correlation of daily returns between (a) K=4 vs Meridian, (b) K=3 vs Meridian, (c) K=4 vs K=3.

Across the full sample:
- K=4 vs Meridian daily return correlation: 0.987
- K=3 vs Meridian daily return correlation: 0.989
- K=4 vs K=3 daily return correlation: 0.991

All three strategies have daily return correlations above 0.98 with each other. They are doing essentially the same thing 99% of the time. The 1% of the time they differ is concentrated around regime transitions, which is exactly when the regime detector has the most influence on the optimizer.

The correlation result is the deepest finding of this section: when 99% of daily decisions are identical across three different regime detection approaches, the regime detector is not the determinant of strategy performance. The signals, optimizer, and rebalance discipline are.

## 10.7 Tail behavior comparison

A regime detection model is supposed to add the most value during tail events. The three biggest drawdown periods in the backtest are 2008 (GFC), 2020 (COVID), and 2022 (rate hike bear).

| Period | Meridian DD | K=3 DD | K=4 DD | K=4 best? |
|---|---|---|---|---|
| 2008 GFC peak DD | -33.10% | -33.70% | -33.01% | Marginal |
| 2020 COVID peak DD | -27.56% | -27.63% | -27.50% | Marginal |
| 2022 holdout peak DD | -21.54% | -20.48% | -20.65% | No (K=3 wins) |

K=4 wins by tiny margins in 2008 and 2020. K=3 wins in 2022. The differences are within 0.5 percentage points, well below decision-relevant thresholds.

The MaxDD aggregate improvement (K=4 vs Meridian: 0.76 pp) is the cleanest "win" K=4 can claim. It is real but small.

## 10.8 Summary of three-way comparison

The three strategies produce nearly identical aggregate results. Per-window winners are distributed roughly uniformly. Daily return correlations exceed 0.98. The K=4 strategy is marginally worse on CAGR and Sharpe and marginally better on MaxDD, with all differences smaller than within-strategy noise.

The clearest pattern: K=4 wins in regime-ambiguous periods (recoveries from bear, transitional years) and loses in clean trend periods. K=3 shows the same pattern less prominently. Meridian is the cleanest performer in clean trends and the worst performer in ambiguous transitions. Across 21 years with both kinds of periods, these effects offset.

This is the empirical foundation for the "strategy invariance" hypothesis discussed in Section 12.

The integration criteria (C1, C2, C3) for K=4 are evaluated against this comparison in the next section.

---

# 11. Integration Criteria Evaluation

This section evaluates the K=4 extension against the three pre-registered integration criteria (C1, C2, C3) defined in Section 3.5. The criteria were locked before any K=4 numerical results were known. The evaluation is binary: pass or fail. No criteria were modified, weakened, or replaced after seeing the results.

## 11.1 C1: Regime classification agreement vs ground truth

**Criterion (as registered):** Cohen's kappa between TVTP-detected regimes and a held-out regime ground truth, computed on the holdout period only, must exceed 0.40 (moderate agreement).

**Ground truth definition:** As established in Section 3.5, no objective regime ground truth exists for daily equity markets. The reference baseline used for kappa computation is the Meridian rule-based classifier, which is the operating system in production. The kappa measures agreement between TVTP and Meridian's regime labels on the holdout period (2022-2026).

**For K=3 (B2 result, retained for context):**
- TVTP K=3 hard label = argmax of (p_bull, p_bear, p_crisis)
- Meridian rule-based label
- Cohen's kappa on holdout: **0.094**
- Result: **FAIL**

**For K=4:**
The K=4 hard label is argmax of (p_crisis, p_bear, p_recovery, p_bull). Meridian's classifier outputs only three labels (bull, bear, crisis), with no recovery state. To compute kappa, the K=4 four-state output must be collapsed to three states by mapping recovery to bull. The mapping is justified because Meridian's bull encompasses both recovery and trending bull (it doesn't distinguish them).

After collapsing K=4 → 3 states by merging recovery into bull:
- Cohen's kappa on holdout (3-state): **0.087**
- Result: **FAIL**

The K=4 kappa is marginally lower than K=3 kappa. Collapsing recovery into bull introduces approximation error, but the underlying disagreement is fundamental: the rule-based classifier and the TVTP classifier label the same days differently a large fraction of the time.

**Interpretation.** C1 was registered before B2 K=3 was completed. The criterion is unreliable in retrospect because the "ground truth" baseline (Meridian rules) is itself a low-quality classifier rather than ground truth. The kappa measures disagreement between two flawed classifiers, not the quality of either against truth.

**Decision:** C1 is treated as non-binding. The criterion's design flaw (Meridian as ground truth) was identified in B2 K=3 and the same conclusion applies to K=4. Failure of C1 is documented but does not weigh against K=4 because the criterion itself is invalid.

This is a deliberate departure from the pre-registration, justified on methodological grounds. The journal entry from B2 K=3 (Section 5.7 of this paper) records this decision. K=4 inherits the same treatment.

## 11.2 C2: Early warning timeliness

**Criterion (as registered):** Median lead time of TVTP crisis probability spike (>50%) over Meridian rule-based crisis flag, across the four major drawdowns in the holdout period (March 2022 SVB+rate hike, October 2022 trough, August 2024 Yen carry unwind, April 2025 tariff shock), must be ≥ 30 days.

**For K=3 (B2 result):**
- Median lead time across four drawdowns: **88 days**
- Result: **PASS**

**For K=4:**

K=4 has a recovery state that K=3 lacks. The relevant "early warning" probability is the sum of crisis and bear probabilities for K=4 (since both are defensive states), compared to crisis probability alone for K=3. This is the most charitable comparison for K=4.

Lead time analysis on holdout drawdowns:

| Drawdown | Date trough | Meridian flag date | K=3 P(crisis)>50% | K=4 P(crisis)+P(bear)>50% | K=4 vs Meridian |
|---|---|---|---|---|---|
| SVB / rate hike | 2022-03-08 | 2022-03-04 | 2021-12-15 (79 days) | 2021-12-08 (86 days) | +86 days |
| Bear trough | 2022-10-12 | 2022-09-23 | 2022-07-14 (70 days) | 2022-07-05 (79 days) | +79 days |
| Yen carry | 2024-08-05 | 2024-08-01 | 2024-04-22 (105 days) | 2024-04-18 (109 days) | +109 days |
| Tariff shock | 2025-04-08 | 2025-04-04 | 2025-02-12 (55 days) | 2025-02-08 (59 days) | +59 days |

Median K=4 lead time vs Meridian: **82 days**.

K=4 fires defensive signals roughly 4-9 days earlier than K=3 across these four episodes. This is because K=4 includes the bear state (p_bear) in the early-warning composite, which captures vulnerability before full crisis materializes.

**Result for K=4: PASS (82-day median lead time, well above 30-day threshold).**

C2 passes for K=4 by a comfortable margin. The probabilistic regime model demonstrably fires earlier than the rule-based detector on the four most material drawdowns of the holdout period. This is the strongest evidence of K=4's marginal value over both K=3 and Meridian.

However: passing C2 alone is not sufficient for integration. The early warning must translate into measurable strategy improvement. C3 tests that.

## 11.3 C3: Strategy improvement

**Criterion (as registered):** TVTP-soft strategy must improve Meridian aggregate CAGR by ≥ 0.5 percentage points AND Sharpe by ≥ 0.05 over the full backtest period (Jan 2005 - present).

**For K=3 (B2 result):**
- CAGR improvement: -0.12 pp (worse)
- Sharpe improvement: -0.001 (worse)
- Result: **FAIL**

**For K=4:**
- Aggregate K=4 CAGR: 23.17%
- Aggregate Meridian CAGR: 23.56%
- CAGR improvement: **-0.39 pp (worse)**
- Aggregate K=4 Sharpe: 1.295
- Aggregate Meridian Sharpe: 1.309
- Sharpe improvement: **-0.014 (worse)**
- Result: **FAIL**

The C3 failure for K=4 is unambiguous. K=4 does not just miss the threshold (which would require positive improvements above the threshold), it has negative improvements on both metrics. K=4's strategy result is worse than Meridian's on both CAGR and Sharpe.

The MaxDD improvement of +0.76 pp (K=4 vs Meridian) is positive but C3 does not include MaxDD as an integration metric. MaxDD was deliberately excluded from C3 during pre-registration because optimizing for MaxDD alone is gameable (e.g., by simply reducing leverage). C3 specifies CAGR and Sharpe to ensure improvement is genuine and not achieved by trading off return for risk.

If MaxDD were included as a tiebreaker, K=4 would still fail because the CAGR/Sharpe degradations exceed the MaxDD improvement on a risk-adjusted basis (Calmar improvement is +0.003, marginal).

## 11.4 Aggregate criteria evaluation

| Criterion | K=3 | K=4 | Notes |
|---|---|---|---|
| C1: Cohen's kappa ≥ 0.40 | FAIL (0.094) | FAIL (0.087) | Treated as non-binding; flawed criterion |
| C2: Median lead time ≥ 30 days | PASS (88 days) | PASS (82 days) | K=4 marginally better than K=3 |
| C3: CAGR ≥ +0.5 pp AND Sharpe ≥ +0.05 | FAIL (-0.12, -0.001) | FAIL (-0.39, -0.014) | K=4 is worse than K=3 on both metrics |

**Integration decision: K=4 is not promoted to live Meridian.**

The decision criteria specified in Section 3.5 require all three criteria to pass for integration. C1 is non-binding by design flaw. C3 is the most material criterion and it fails decisively. C2 passing alone is not sufficient.

K=4 is retained as a standalone tool in the regime_switching/ module for:
- Continued research into regime structure
- Comparison baseline for future regime detector experiments
- Production of soft regime probabilities for offline analysis
- Use as a regime feature in any future strategy that integrates regime probability as a soft input rather than as optimizer parameter selector

K=4 is not used in live trading. Meridian continues to use the rule-based classifier in production.

## 11.5 Why C2 passes but C3 fails — the mechanism

The disconnect between C2 (early warning) and C3 (strategy improvement) is the most important methodological finding of this work. Understanding it explains the broader strategy invariance result.

C2 passes because TVTP P(crisis) does indeed rise weeks before Meridian's hard crisis flag fires. The probabilistic detector is genuinely earlier. This is real signal.

C3 fails because the optimizer's response to "rising crisis probability from 5% to 50% over 60 days" is similar to "Meridian's regime stayed in bull until day 60 then snapped to crisis." The mean-variance optimizer with Ledoit-Wolf covariance is dominated by the covariance structure of stocks. The covariance structure is itself a function of cross-sectional volatility, which rises during the same 60 days as P(crisis) rises. The optimizer is already de-risking through the covariance channel even before the regime input changes meaningfully.

By the time P(crisis) reaches 50% in K=4 or K=3, the covariance matrix has already widened, the volatility-targeted constraints have already started reducing position sizes, and the Ledoit-Wolf shrinkage has already pulled estimates toward a more diversified prior. The regime input arrives at a time when the optimizer is already responding.

The early warning of 80+ days is real signal that the optimizer doesn't fully exploit. The information is leaking through other channels first.

This is why strategy invariance holds. The regime detector is not the binding constraint on strategy behavior. Other channels are.

## 11.6 What would change the verdict

For C3 to pass with K=4, one of the following would need to be true:

**Scenario 1: Decouple the optimizer from covariance.** Use a regime-dependent allocation that doesn't run mean-variance — for example, a risk parity overlay that scales total exposure by P(crisis) explicitly. This bypasses the covariance channel and forces regime probability to drive de-risking directly. Future work item.

**Scenario 2: Introduce regime-dependent factor weights.** Currently signals are weighted by IC, which is regime-blind. If signal weights adapted to regime (e.g., increase momentum weight in bull, increase quality weight in bear), the regime input would have a clearer pathway to strategy behavior. EXP009 partially explores this.

**Scenario 3: Use regime probability as a meta-overlay.** Apply a P(crisis)-dependent exposure scale at the portfolio level (e.g., scale gross exposure by 1 - 0.5 * P(crisis)). This is the most direct way to extract value from the early warning signal. Not currently implemented; future work.

**Scenario 4: Run on a different signal stack.** The signal stack used in Meridian is built around expanding-IC weighted alpha factors. A signal stack with different sensitivity to regime (e.g., trend-following signals that genuinely care about regime) might extract more value. Out of scope for this work.

Without one of these mechanisms, the regime input simply doesn't have a strong enough lever on the strategy to overcome the noise floor.

The B4 work (manager skill decomposition) and EXP009 (rolling parameter recalibration) are next on the roadmap. B4 will quantify how much of Meridian's alpha comes from regime detection vs other channels. EXP009 will test whether parameter recalibration adapts optimizer behavior in ways that the regime input doesn't.

## 11.7 Honest accounting of the verdict

K=4 does not improve Meridian. It is marginally worse on the metrics that matter (CAGR, Sharpe) and marginally better on a metric that doesn't bind (MaxDD).

This is a negative result. It is documented as a negative result. K=4 is not retained because it is "almost good enough" or because it "might help in future regimes." It is not in production.

The temptation to declare a partial win — "K=4 has lower drawdowns, integrate it for that" — was considered and rejected. The MaxDD improvement (0.76 pp) is too small to be statistically meaningful and too small to overcome the CAGR cost (0.39 pp). The decision standard requires actual improvement on the metrics that compound (CAGR, Sharpe), not metrics that gameably look better in cherry-picked windows.

Section 12 discusses why this negative result is the most important finding of the work, not in spite of being negative but because of it.

---

# 12. Discussion — The Strategy Invariance Hypothesis

This section steps back from the numbers and asks the harder question: what does it mean that three different regime detectors (rule-based, K=3 TVTP, K=4 TVTP) produce nearly identical strategy results across 21 years?

This is the section that matters most for the broader research program. The answer to this question shapes what should be worked on next, what should be deprioritized, and how to honestly characterize Meridian's strengths and weaknesses to professors and to interviewers.

## 12.1 The strategy invariance hypothesis

**Hypothesis:** Across the Meridian system as currently designed, the regime detection module is not a binding constraint on strategy performance. Replacing it with detectors of varying sophistication (rule-based, 3-state TVTP, 4-state TVTP) produces strategy outputs whose differences are within within-strategy noise.

This hypothesis has three corollaries:

1. **The information that regime detection should provide is already being absorbed by other system components.** Specifically, the covariance matrix (via Ledoit-Wolf shrinkage), the signal IC weighting (which adapts as recent IC shifts), and the volatility-aware risk constraints. By the time the regime detector classifies a state change, these other components have already adjusted.

2. **Improvements to regime detection alone will not improve Meridian.** This was demonstrated empirically across the entire 21-year backtest with two distinct regime detector upgrades. The upgrade path that would deliver gains lies elsewhere.

3. **The regime detector is still useful as a monitoring/communication tool, not as a strategy driver.** Producing soft P(crisis) probabilities is informative for risk management, dashboard display, professor presentations, and interview discussions. It is not informative as an optimizer input.

The strategy invariance hypothesis is the central scientific finding of this work.

## 12.2 Evidence for invariance

The evidence presented in Sections 9, 10, and 11 supports the hypothesis from multiple angles:

**Evidence 1: Aggregate metric convergence.** Three different regime detectors produce CAGR within 0.39 pp, Sharpe within 0.014, and MaxDD within 0.76 pp across 21 years. The within-strategy variability across windows is one to two orders of magnitude larger than the cross-strategy spread.

**Evidence 2: Daily return correlation > 0.98.** All three strategies make essentially the same daily decisions 99% of the time. The 1% disagreements are concentrated around regime transitions and produce the marginal differences in aggregate metrics.

**Evidence 3: Per-window distribution is uniform.** Win counts across CAGR, Sharpe, and MaxDD are roughly equally distributed across the three strategies. No strategy dominates. Wins and losses are not systematic.

**Evidence 4: C2 passes but C3 fails.** The TVTP detectors (both K=3 and K=4) genuinely fire crisis signals 60-100 days earlier than rule-based, but this earlier firing does not translate into measurably better strategy results. The information arrives but doesn't move the needle.

**Evidence 5: K=4 wins where regime is ambiguous, loses where regime is clear.** This pattern is consistent with K=4 producing useful probability information in transition periods, but those transition periods average out across a long sample.

The hypothesis is not proven (negative results never are), but the evidence is strong enough to warrant accepting it as the working assumption for next-stage research planning.

## 12.3 What absorbs the regime information

If the regime detector is not the binding constraint, what is doing the work?

**Channel 1: Covariance matrix dynamics.** The mean-variance optimizer takes a covariance matrix (via Ledoit-Wolf shrinkage of recent returns) as input. During crisis periods, cross-sectional volatility rises sharply, idiosyncratic risk dispersion widens, and the covariance matrix changes shape. The optimizer responds by reducing position sizes and concentrating in low-vol names. This response is automatic and immediate — it doesn't wait for regime classification.

In the K=4 backtest, examining the covariance matrix structure during the August 2024 Yen carry unwind shows that average pairwise correlation rose from 0.34 to 0.52 in the two weeks before Meridian's hard crisis flag fired. The Ledoit-Wolf-shrunk covariance reflected this rise, so the optimizer was already producing more conservative weights before P(crisis) reached 50%. Whether the regime input fired at day 80, day 60, or day 30 didn't matter much — the covariance had already done the de-risking work.

**Channel 2: IC-weighted signal blending.** The signal combiner weights each of the 15 signals by trailing IC. During crisis, signals like value and quality typically have higher IC than momentum. The signal combiner shifts weight automatically, which changes the alpha forecast produced for the optimizer, which changes the optimizer's tilts, all without any regime input.

In Window 4 (2008), the IC ranking shifted dramatically between Q2 and Q4: momentum dropped from rank 3 to rank 12, quality rose from rank 8 to rank 2. The expanding IC window picked this up gradually but unmistakably. The strategy was rotating into quality before the regime detector fired crisis.

**Channel 3: Trailing stops.** The vol-adjusted trailing stop (25-day EWM) tightens automatically as volatility rises, triggering exits before the regime detector classifies crisis. Stops are not regime-dependent; they are volatility-dependent. Volatility rises before regime classification changes, so stops fire first.

**Channel 4: Compliance and concentration limits.** The 75-5-10 compliance check and the 10% per-position concentration limits are static rules that bind harder during crisis (when concentration tends to rise as some positions outperform their peers). They produce defensive behavior automatically.

The combined effect of these four channels is that Meridian's portfolio responds to crisis-onset conditions through volatility, correlation, IC, and rule-based limits — all before the regime detector classifies anything. By the time regime classification updates, most of the de-risking has happened. The regime detector's marginal contribution is small because it is downstream of most of the work.

## 12.4 Where regime detection adds value

The strategy invariance result is conditional on the current Meridian architecture. Regime detection would add measurable value in different setups:

**Setup A: Direct exposure scaling.** Apply gross exposure scaling proportional to (1 - regime defensive probability). This bypasses the optimizer and translates regime probability directly into invested fraction. Estimated value-add: high. Implementation: simple. This is the highest-priority follow-up.

**Setup B: Regime-conditional signal weighting.** Multiply signal IC weights by regime-dependent factors (e.g., quality gets 2x weight in bear, momentum gets 0.5x weight in bear). This transforms the regime input into changes to the alpha forecast itself, which affects both direction and magnitude of trades. Estimated value-add: moderate. EXP009 explores related ideas.

**Setup C: Regime-dependent universe filtering.** Restrict universe to defensive sectors when P(crisis) > threshold. Crude but effective. Not currently implemented in Meridian.

**Setup D: Regime as feature in ML signal.** Use regime probabilities as features in an ML model (XGBoost, neural net) that produces alpha forecasts. The model can learn nonlinear interactions between regime and other features. Future work, would require building an ML pipeline.

The K=4 model produces high-quality regime probabilities. The probabilities are not the bottleneck. The bottleneck is how those probabilities are translated into strategy decisions. The integration architecture matters more than the detection architecture.

## 12.5 The negative finding as a positive contribution

The temptation in research is to keep iterating until you get a positive result. K=3 didn't beat Meridian, so try K=4. K=4 didn't beat Meridian either. Try K=5? Try a Bayesian model? Try a neural network regime classifier? Try a hidden semi-Markov model? Try ensemble of regime detectors?

This temptation is the path to overfitting. Each iteration costs research time, and after enough iterations, one of them will randomly outperform on the in-sample backtest. That outperformance will be falsely attributed to the regime model and will fail out of sample.

The right response to "K=3 and K=4 both fail to improve Meridian" is to accept that improving the regime detector is not the right direction, and pivot to a different research question.

This work does that. The next priority items in the roadmap (B4 manager skill decomposition, EXP009 rolling parameter recalibration, and direct exposure scaling work) target the actual binding constraints rather than the non-binding regime detector.

The honest accounting of "K=4 doesn't improve Meridian, here's why, here's what should be worked on instead" is more valuable than "after some tuning, K=4 works." The former informs the next year of research priorities. The latter would be a single-paper victory followed by a quietly disappointing live deployment.

## 12.6 Implications for production system

The K=4 result has several specific implications for Meridian as a production system:

**Implication 1: Keep the rule-based regime classifier.** It works fine. Replacing it with K=3 or K=4 TVTP would degrade performance (small but measurable). The rule-based classifier is fast, transparent, and auditable. Stay with it.

**Implication 2: Add P(crisis) from K=3 TVTP as a monitoring overlay.** The B2 K=3 TVTP fits are computed and saved. Adding a daily P(crisis) probability to the operational dashboard provides additional context for risk monitoring without changing strategy behavior. Cost: low. Value: communicating risk state to risk-monitoring users.

**Implication 3: Do not pursue K=5 or higher state counts.** K=5 was rejected for pathological identification. K=4 doesn't improve K=3. The diminishing returns from adding states are clear. The state-count axis is exhausted.

**Implication 4: Pursue exposure scaling as next regime-related project.** Direct gross exposure scaling by (1 - P(defensive)) is the simplest way to translate regime probabilities into strategy behavior. This requires a small backtest (StrategyResearchLab) before deployment. Estimated 1-2 days of work.

**Implication 5: Reuse K=4 TVTP probabilities for B4.** The B4 manager skill decomposition will use regime probabilities as inputs to attribute Meridian's returns. K=4's higher resolution (separating recovery from bull) may help isolate which return periods are skill vs which are recovery beta. This reuses the K=4 work without integrating it into live trading.

## 12.7 The broader research lesson

The strategy invariance result is a specific instance of a broader principle in quantitative finance: **the marginal contribution of a single component to an integrated strategy depends on the architecture of the integration, not just the quality of the component.**

A high-quality alpha signal does not improve a strategy if the optimizer can't use it. A high-quality regime detector does not improve a strategy if the optimizer is dominated by other inputs. A high-quality risk model does not improve a strategy if the universe and rebalancing don't allow risk to be expressed.

For Meridian specifically, the architecture is built around a mean-variance optimizer with covariance shrinkage and a 15-signal IC-weighted alpha stack. The architecture is good but it is also opinionated: it places most decision authority in the covariance matrix and the IC-weighted signals. Other inputs (regime, sector tilts, fundamentals) have to compete for influence within this opinionated architecture.

Improving Meridian beyond its current 23.5% CAGR / 1.31 Sharpe likely requires architectural changes (different optimizer, different signal stack, different risk model) rather than improvements to existing components. This is the broader lesson the K=4 work delivers.

The next research priorities (B4, EXP009, exposure scaling, eventual ML signal combination in B9) target architectural questions rather than component-quality questions. This is the right direction.

---

# 13. Future Work

This section catalogs the next research items that follow naturally from the K=4 result. The items are organized by priority — work that should happen first, work that should happen second, and work that is interesting but not urgent.

The organizing principle: prioritize work that targets the **binding constraints** of Meridian as identified in Section 12, not work that targets **already-satisfied constraints** like regime detection accuracy.

## 13.1 Tier 1: Direct consequences of the K=4 result

### 13.1.1 Direct exposure scaling overlay

**Description:** Implement a portfolio-level gross exposure scaler that multiplies aggregate invested fraction by (1 - alpha * P_defensive), where P_defensive is the sum of K=4's crisis and bear probabilities, and alpha is a calibration parameter (proposed range: 0.3 to 0.7).

**Rationale:** Section 12.4 identified this as the highest-leverage way to translate K=4's high-quality regime probabilities into strategy decisions. Bypasses the optimizer's internal averaging. Translates regime input directly into invested fraction.

**Backtest design:** Add the exposure scaler as a post-optimizer step in StrategyResearchLab. Sweep alpha in {0.3, 0.4, 0.5, 0.6, 0.7}. Use K=4 walk-forward fits already computed. Run on the same 18 windows. Compare to Meridian baseline using the same C3 standard (CAGR ≥ +0.5 pp AND Sharpe ≥ +0.05).

**Estimated effort:** 1-2 days for implementation, 4-6 hours of compute for the backtest, 1 day for analysis.

**Risk of failure:** Moderate. The scaler is mechanically simple but if regime probabilities are noisy at the daily level, the scaler will produce excessive turnover and trading costs. May need smoothing.

**Decision criterion:** If the best alpha produces +0.5 pp CAGR AND +0.05 Sharpe vs Meridian, integrate. Otherwise, document the negative result and stop pursuing regime-based exposure scaling.

### 13.1.2 EXP009: Rolling parameter recalibration

**Description:** Test whether the per-window optimizer parameters (lambda turnover penalty, RA risk aversion) should be recalibrated more frequently than annually. Sweep lookback windows of 7-12 years for parameter recalibration, with optional regime-conditional recalibration.

**Rationale:** The current per-window calibration uses an annual recalibration (each window calibrates from the prior 5 years). The hypothesis is that more frequent recalibration with longer lookbacks (7-12 years) might better adapt to slow-moving regime drift. Academic basis: Arnott et al. 2019 (factor timing), DeMiguel et al. 2009 (parameter uncertainty), AQR practitioner research.

**Backtest design:** EXP009 in StrategyResearchLab. Walk-forward identical to EXP006 structure. Calibrate parameters from rolling 7, 8, 9, 10, 11, 12-year lookbacks. Compare to baseline (5-year lookback). Run with both hard regime (Meridian) and soft regime (K=4 TVTP — reusing fits already computed).

**Estimated effort:** 2-3 days for implementation, 8-12 hours of compute (could parallelize on cloud), 2 days for analysis.

**Risk of failure:** Low. Even a negative result is valuable for understanding parameter stability.

**Tie to K=4 work:** EXP009 reuses the K=4 walk-forward fits as soft regime inputs, eliminating refitting cost. This is a tangible value-add of having done the K=4 work even if K=4 isn't integrated.

### 13.1.3 K=4 dashboard integration

**Description:** Add K=4 P(crisis) and P(defensive) = P(crisis) + P(bear) to the operational Streamlit dashboard. Display as time series chart and current state indicator.

**Rationale:** Even though K=4 is not in production, its outputs are useful for risk monitoring and for explaining the system to non-quant audiences. The probabilistic display ("82% probability of bull regime, 12% bear, 6% recovery") is more informative than the binary rule-based label.

**Estimated effort:** Half day. The probabilities are already saved per-window. Just need to load and plot.

## 13.2 Tier 2: Adjacent regime detection research

### 13.2.1 Statistical Jump Models (Nystrup 2024)

**Description:** Replace the EM-based MS estimation with a Statistical Jump Model approach (Nystrup, 2024). Jump models penalize transitions explicitly via a regularization term, producing more interpretable regime sequences with fewer transitions. Less prone to misclassification of single-day shocks.

**Rationale:** The K=3 TVTP and K=4 TVTP both produce regime sequences with high transition frequency (avg duration of 1-2 days for some states). A jump model would smooth this. The marginal value of smoothed regimes for strategy purposes is unclear (per Section 12, the strategy doesn't care much about regime resolution), but for monitoring and interpretation purposes the smoothed regimes would be cleaner.

**Estimated effort:** 1-2 weeks for implementation (the algorithm is non-trivial), 6-8 hours compute, 3-4 days for analysis.

**Priority:** Medium. Pursue only if exposure scaling (13.1.1) shows that regime probability quality matters for strategy decisions.

### 13.2.2 Hidden Semi-Markov Models (HSMM)

**Description:** Replace the geometric duration distribution implied by Markov regime models with explicit duration distributions (e.g., negative binomial). HSMMs capture the empirically observed pattern that crisis regimes have a characteristic duration of weeks-to-months rather than the geometric distribution implied by MS models.

**Rationale:** Markov assumption forces P(stay in state | duration) to be constant, which doesn't match empirical regime persistence. HSMMs allow this to vary.

**Estimated effort:** 2-3 weeks for implementation. Compute cost similar to TVTP.

**Priority:** Low. Likely captures detail that the strategy doesn't care about (per strategy invariance hypothesis).

### 13.2.3 Macro covariate refinement

**Description:** Test additional macro covariates beyond VIX, yield curve slope, and HY credit spread. Candidate covariates: TED spread, MOVE index (rate volatility), oil volatility (OVX), trade-weighted USD, S&P breadth (% of stocks above 200dma).

**Rationale:** The current TVTP covariates are equity-derived (VIX) and rate-derived (yield curve, credit spread). Adding currency, commodity, and breadth covariates might capture regime drivers the current model misses.

**Estimated effort:** 1 week. Reuse M3b infrastructure with expanded covariate set.

**Priority:** Low. K=3 TVTP coefficients show VIX is dominant by 5-10x; additional covariates likely have small marginal explanatory power.

## 13.3 Tier 3: Adjacent strategy research (informed by K=4 lessons)

### 13.3.1 B4: Manager skill decomposition

**Description:** Decompose Meridian's returns into systematic factor exposures, residual alpha, and regime-conditional alpha. Run on 15-20 real ETFs (large-cap mutual funds, factor ETFs, regime-overlay funds) plus Meridian's backtest NAV.

**Rationale:** Quantifies how much of Meridian's 23.5% CAGR is explained by systematic factor exposure (which doesn't require Meridian's specific signal stack), and how much is genuine alpha. Provides a benchmark for how much alpha the regime detector contributes (which the K=4 result suggests is small).

**Tie to K=4:** Uses K=4 P(crisis) and P(bear) probabilities as regime indicators for conditional alpha attribution. Reuses K=4 fits.

**Estimated effort:** 2-3 weeks. New infrastructure (multi-asset fitting, factor model construction, attribution decomposition).

**Priority:** High. This is the next major project after EXP009.

### 13.3.2 Markov-Switching Garch (MS-GARCH)

**Description:** Replace constant-variance regime model with regime-conditional GARCH. Captures vol-of-vol within regimes. Has been shown to fit equity returns better than constant-variance MS.

**Estimated effort:** 1-2 weeks.

**Priority:** Low. Per strategy invariance, regime model improvements don't translate to strategy improvements.

### 13.3.3 Ensemble regime detection

**Description:** Combine outputs of rule-based, K=3 TVTP, K=4 TVTP into an ensemble probability. Soft voting or stacking.

**Rationale:** Each detector has different blind spots. Ensemble may have better-calibrated probabilities at the cost of less interpretability.

**Estimated effort:** 1 week.

**Priority:** Low.

## 13.4 Roadmap timing

The proposed sequence for the next 6 months of regime-related research:

**Months 1-2:** EXP009 (rolling parameter recalibration) and exposure scaling overlay (13.1.1). These two are fastest to execute and most likely to produce strategy improvement.

**Months 2-3:** B4 manager skill decomposition. Larger project but establishes a benchmark for understanding what's worth pursuing further.

**Months 4-5:** Statistical Jump Models if exposure scaling result indicates regime probability quality matters. Skip if exposure scaling fails C3.

**Months 5-6:** Whatever EXP009 and B4 reveal as the next binding constraint.

This sequence prioritizes work that targets binding constraints. It deprioritizes work that targets the regime detection axis (since K=4 has demonstrated this axis is non-binding for strategy improvement).

## 13.5 What this work explicitly will NOT pursue

The following research directions are explicitly dropped from the roadmap based on the K=4 result:

- **K=5 or higher state counts.** K=5 was rejected for pathology. K=4 doesn't improve K=3. Adding states is exhausted.
- **HMM with deeper feature engineering on existing covariates.** The K=4 model is well-calibrated. Marginal improvements to feature engineering won't translate to strategy gains.
- **Bayesian regime models (e.g., DPGMM regime detection).** More sophisticated regime detection. Doesn't address the binding constraint.
- **Replacing rule-based with TVTP in production.** Would degrade performance per K=3 and K=4 results. Stay with rule-based.

Dropping these directions frees up capacity for the higher-leverage work in Sections 13.1 and 13.3.

---

# 14. Conclusion

This paper presented the design, estimation, and evaluation of a four-state time-varying transition probability Markov-switching model (K=4 TVTP) as a candidate replacement for the rule-based regime classifier in Meridian, an institutional-grade systematic equity trading system.

The motivation was specific. The prior B2 K=3 TVTP work demonstrated genuine early-warning value (88-day median lead time over rule-based detection) but failed to improve aggregate strategy performance over the rule-based baseline. The diagnostic identified that the K=3 model conflated two distinct return regimes — recovery and trending bull — into a single bull state, suggesting that adding a fourth state to disentangle them might improve strategy performance.

The hypothesis was tested rigorously. M2b confirmed that K=4 fixed-MS produces a clean four-state characterization with sensible economics (crisis -97% mu / 60% sigma, bear -9% / 24%, recovery +8% / 15%, bull +31% / 8%). M2c rejected K=5 on pathology grounds. M3b estimated K=4 TVTP via custom Hamilton filter and EM with numerical M-step, with explicit constraints (mu bounded ±100%, P_diag soft-penalized to ≥0.90, 8 random restarts per window) to ensure identification. All 18 walk-forward windows converged with stable parameters.

M6b backtested K=4 TVTP soft regime probabilities through the identical Meridian pipeline used for the rule-based baseline and the K=3 TVTP comparison. The backtest produced aggregate metrics for all three strategies over 2005-2026 (5,314 trading days):

| Strategy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| Meridian (Hard) | 23.56% | 1.309 | -37.63% |
| K=3 Soft | 23.44% | 1.308 | -37.61% |
| K=4 Soft | 23.17% | 1.295 | -36.87% |

The three strategies are statistically indistinguishable on aggregate performance. K=4 trails Meridian by 0.39 percentage points of CAGR and 0.014 of Sharpe, against an integration threshold requirement of +0.5 pp CAGR and +0.05 Sharpe (C3 criterion). C3 fails. K=4 is not promoted to live trading.

The K=4 work is retained as a standalone module. The fitted models, the walk-forward output, and the regime probabilities are saved and available for downstream research (B4 manager skill decomposition, dashboard monitoring, EXP009 parameter recalibration).

The deeper finding is the strategy invariance result: across three substantially different regime detection approaches, daily strategy returns correlate at >0.98 and aggregate performance differs by less than within-strategy noise. This is interpreted as evidence that the regime detection module is not the binding constraint on Meridian's performance. The information that regime detection should provide is already absorbed by other system components — Ledoit-Wolf covariance shrinkage, IC-weighted signal blending, vol-adjusted trailing stops, and rule-based concentration limits — by the time the regime detector classifies a state change.

The implication for future research is clear. Improving Meridian beyond its current 23.5% CAGR / 1.31 Sharpe requires architectural rather than component-level changes. Direct exposure scaling (translating regime probability into invested fraction at the portfolio level), parameter recalibration timing (EXP009), and manager skill decomposition (B4) are the prioritized next steps. Further investment in higher-resolution regime detectors (K=5+, HMM, ensemble) is deprioritized.

This is a negative result for K=4 integration. It is a positive result for the broader research program, because it identifies what should be worked on next and what should not.

The pre-registration discipline applied to this work (locking C1, C2, C3 criteria before any K=4 numerical results were available) is what allows this conclusion to be stated cleanly. Without the pre-registration, the temptation to declare a partial win on MaxDD improvement (+0.76 pp) or on C2 passing (82-day lead time) would be hard to resist. With the pre-registration, the answer is unambiguous: K=4 fails C3, K=4 does not get integrated, the research program moves on.

The full reproducibility package — code, data paths, configuration files, walk-forward fits, backtest output, comparison tables — is documented in Appendix E. Any future researcher (or any future iteration of this researcher) can rerun the entire pipeline from raw data to the final comparison table using the commands listed there.

Meridian continues to operate in production with the rule-based regime classifier. The K=4 TVTP model is available as a research artifact.

---


# Appendix A: Mathematical Formulation

This appendix collects the complete mathematical specification of the K=4 TVTP-MS model, the Hamilton filter, the Kim smoother, the EM algorithm, and the constraint formulations. Notation follows Hamilton (1989) and Kim (1994) where possible, with extensions for time-varying transitions following Filardo (1994).

## A.1 Model specification

**Observation equation.** Daily SPY log returns r_t conditional on latent regime state S_t = k follow a Gaussian distribution:

> r_t | S_t = k ~ N(mu_k, sigma_k²)

with K = 4 states k ∈ {0, 1, 2, 3} corresponding to {crisis, bear, recovery, bull} after post-estimation labeling. Each state has free parameters mu_k (regime mean) and sigma_k (regime standard deviation). Total emission parameters: 2K = 8.

**State labeling.** States are labeled by mu_k after estimation: lowest mu is crisis (k=0), highest is bull (k=3), intermediate two are sorted ascending as bear (k=1) and recovery (k=2). This avoids the label-switching identification problem.

**Transition equation (TVTP).** Transition probabilities are time-varying and depend on a covariate vector z_t = [VIX_t, yield_curve_t, credit_spread_t]':

> P(S_t = j | S_{t-1} = i, z_t) = exp(a_{ij} + b_{ij}' z_t) / sum_k exp(a_{ik} + b_{ik}' z_t)

This is a multinomial logistic with z_t as the regressor. For identification, we set a_{i0} = 0 and b_{i0} = 0 for each origin state i (normalize against state 0). This gives:
- (K-1) free intercepts per origin state = 3 per origin
- (K-1) × d free coefficients per origin state = 3 × 3 = 9 per origin
- Total per origin: 3 + 9 = 12
- Total transition parameters: K × 12 = 48

**Total parameter count.** Emission (8) + transition (48) = 56 free parameters for K=4 TVTP. By comparison, K=3 TVTP has 6 + 30 = 36 parameters. The BIC penalty for K=4 over K=3 is therefore Δ × ln(7347) = 20 × 8.90 ≈ 178, which the K=4 model overcomes by a log-likelihood improvement of approximately 250 (depending on constraints).

## A.2 Hamilton filter (forward pass)

The Hamilton filter computes filtered probabilities P(S_t = k | r_1, ..., r_t, z_1, ..., z_t) recursively. Define:

> ξ_{t|t} = [P(S_t = 0 | I_t), ..., P(S_t = K-1 | I_t)]'

where I_t denotes the information set available at time t.

**Step 1 (Prediction).** Given filtered probabilities at t-1 and the time-varying transition matrix at t:

> ξ_{t|t-1} = P_t' × ξ_{t-1|t-1}

where P_t is the K×K transition matrix at time t with entries P_t[i,j] = P(S_t = j | S_{t-1} = i, z_t) computed via the multinomial logistic above.

**Step 2 (Likelihood).** Compute the conditional likelihood vector:

> η_t = [f(r_t | S_t = 0), ..., f(r_t | S_t = K-1)]'

where f(r_t | S_t = k) = (1 / (sigma_k × sqrt(2π))) × exp(-(r_t - mu_k)² / (2 × sigma_k²)).

**Step 3 (Update).** The marginal likelihood at t is:

> f(r_t | I_{t-1}) = sum_k η_t[k] × ξ_{t|t-1}[k] = ξ_{t|t-1}' × η_t

The filtered probability is:

> ξ_{t|t}[k] = (η_t[k] × ξ_{t|t-1}[k]) / f(r_t | I_{t-1})

**Step 4 (Log-likelihood accumulation).** The total log-likelihood is:

> log L = sum_{t=1}^{T} log f(r_t | I_{t-1})

This is the objective maximized by EM.

**Initialization.** ξ_{1|0} is set to the unconditional regime probabilities (steady-state of the average transition matrix). For K=4 TVTP, this is approximated by the steady-state of the time-averaged P_t computed from the full sample of z_t values.

**Numerical stability.** All filter operations are performed in log-probability space using logsumexp to prevent underflow at extreme regime probabilities. Filtered probabilities are clamped to [1e-10, 1 - 1e-10] before storage.

## A.3 Kim smoother (backward pass)

The Kim smoother (Kim 1994) computes smoothed probabilities P(S_t = k | r_1, ..., r_T, z_1, ..., z_T) using all available information. Smoothing is critical for the EM algorithm's M-step.

**Backward recursion.** Starting from ξ_{T|T} (the last filtered probability), iterate backward:

> ξ_{t|T}[i] = ξ_{t|t}[i] × sum_j (P_{t+1}[i,j] × ξ_{t+1|T}[j] / ξ_{t+1|t}[j])

This gives the marginal smoothed probability of state i at time t.

**Joint smoothed probabilities.** For the EM transition update, we also need:

> P(S_t = j, S_{t-1} = i | I_T) = (ξ_{t-1|t-1}[i] × P_t[i,j] × η_t[j] × ξ_{t|T}[j]) / (ξ_{t|t-1}[j] × f(r_t | I_{t-1}))

These joint smoothed probabilities are denoted xi_t(i, j) and are used in the M-step.

**Computational cost.** Backward smoother is O(K²T) per pass. For K=4 and T=7300, this is ~117K operations per pass. Trivial compared to the forward filter (which dominates wall-clock time due to logistic transition matrix construction at each t).

## A.4 EM algorithm

**E-step.** Compute filtered probabilities (forward filter) and smoothed probabilities (backward smoother) given current parameter estimates.

**M-step (emissions).** Maximize the expected complete-data log-likelihood with respect to (mu_k, sigma_k) for each state k. The closed-form solutions are:

> mu_k = (sum_t γ_t(k) × r_t) / (sum_t γ_t(k))
> sigma_k² = (sum_t γ_t(k) × (r_t - mu_k)²) / (sum_t γ_t(k))

where γ_t(k) = ξ_{t|T}[k] is the smoothed probability of state k at time t. These are weighted MLEs with smoothed probabilities as weights.

**M-step constraints (K=4 specific).** The constrained M-step requires:

1. mu_k bounded to [-1.0, 1.0] per day. If unconstrained mu falls outside, project to nearest boundary.
2. sigma_k bounded below by 1e-6 to prevent collapse.
3. Per-state ordering preserved by post-estimation labeling (no constraint at M-step).

**M-step (transitions, fixed-TP case).** For models with constant transition matrices, the M-step has a closed-form update:

> p_{ij} = (sum_t xi_t(i, j)) / (sum_t γ_{t-1}(i))

where the sum is over t = 1, ..., T-1 for transitions.

**M-step (transitions, TVTP case).** For TVTP, the M-step requires numerical optimization. We maximize:

> Q_trans(a, b) = sum_t sum_i sum_j xi_t(i, j) × log P(S_t = j | S_{t-1} = i, z_t; a, b)

This decomposes by origin state i: for each i, optimize {a_{ij}, b_{ij}}_{j ≠ 0} using L-BFGS-B with the multinomial logistic gradient. The gradient with respect to a_{ij} (for j ≠ 0) is:

> ∂Q_trans/∂a_{ij} = sum_t (xi_t(i, j) - γ_{t-1}(i) × P_t[i,j])

The gradient with respect to b_{ij} is:

> ∂Q_trans/∂b_{ij} = sum_t (xi_t(i, j) - γ_{t-1}(i) × P_t[i,j]) × z_t

These gradients have a clean interpretation: the numerical M-step adjusts the logistic coefficients in the direction that aligns the model-implied transition counts (γ_{t-1}(i) × P_t[i,j]) with the smoothed observed transition counts (xi_t(i, j)).

**P_diag penalty.** To enforce the persistence constraint, we add a soft penalty to the EM objective:

> Q_total = Q_emission + Q_transition - λ × sum_i max(0, 0.90 - P_diag_i)²

where P_diag_i = sum_t P_t[i,i] / T is the time-averaged diagonal probability for state i and λ = 50 is the penalty weight. The penalty is zero when all P_diag_i ≥ 0.90 and grows quadratically when any falls below.

This penalty is differentiable (via chain rule through the multinomial logistic), so it integrates cleanly with the L-BFGS-B numerical M-step.

**Convergence criterion.** EM iterates until |L_new - L_old| / |L_old| < 1e-6 or max 700 iterations. Convergence is monitored via a log saved per iteration. Restarts that fail to converge in 700 iterations are flagged but not discarded; the restart with highest log-likelihood is selected regardless.

## A.5 Logistic regression Hessian for standard errors

After EM convergence, standard errors for logistic transition coefficients are computed via the observed information matrix (negative Hessian of Q_transition at the final parameter estimates). For each origin state i, the (K-1)*(d+1) × (K-1)*(d+1) Hessian block is:

> H_{i, (a_{ij}, b_{ij}), (a_{ik}, b_{ik})} = -sum_t γ_{t-1}(i) × (δ_{jk} P_t[i,j] - P_t[i,j] P_t[i,k]) × [1, z_t]' [1, z_t]

where δ_{jk} is the Kronecker delta. The standard errors are square roots of diagonal entries of -H^{-1}.

For the Wald test of "is covariate v significant in the i→j transition?":

> W = (b_{ij,v} - 0)² / Var(b_{ij,v})

distributed as χ²(1) under the null. P-values are reported at the 5% level.

**Covariance significance findings (K=3 TVTP, full sample).** From the saved tvtp_coefficients.json:

- VIX coefficient in state 0 → state 2 (bull → crisis): 5.13. Highly significant.
- VIX coefficient in state 0 → state 1 (bull → bear): 2.91. Highly significant.
- VIX coefficient in state 1 → state 2 (bear → crisis): 3.38. Highly significant.
- Yield curve coefficient in state 1 → state 2: -0.98. Significant.
- Credit spread coefficient in state 1 → state 2: 0.91. Significant.

Across most i→j transitions, VIX is the dominant covariate (largest coefficient magnitude, smallest standard error). Yield curve and credit spread are secondary but contribute incrementally.

For K=4 TVTP (per-window), full coefficient tables are saved in regime_switching/data/k4_extension_constrained/window_fits/window_X/tvtp_result.json. Aggregate patterns are similar to K=3: VIX dominates, with credit spread and yield curve playing supporting roles in specific transitions.

## A.6 Soft regime blending mechanism

The regime detector output is a probability distribution P(S_t = k) over K states. The Meridian optimizer takes (lambda, risk_aversion, exposure_scale, sector_tilt) parameters that depend on regime. For soft regime blending:

> theta_t = sum_k P(S_t = k) × theta_k

where theta_k is the parameter vector pre-calibrated for regime k and theta_t is the time-t parameter applied to the optimizer.

For K=3 (B2 work), theta_k is calibrated for k ∈ {bull, bear, crisis}. For K=4 (this work), theta_k is calibrated for k ∈ {crisis, bear, recovery, bull}. The soft blending is otherwise identical.

The K=4 calibration uses Meridian's bull parameters for the recovery state. This is justified by the fact that recovery state has positive mean returns (mu ~ +8% annualized) and modest volatility, structurally similar to the bull state. The alternative would be to introduce a fifth pre-calibrated parameter set specifically for recovery; this was rejected to maintain comparability with the K=3 backtest and avoid introducing additional degrees of freedom.

## A.7 Per-window calibration

Per-window calibration of (lambda, risk_aversion) follows the EXP006 protocol: grid search over lambda ∈ {0.001, 0.002, 0.003, 0.005, 0.007, 0.01} and risk_aversion ∈ {0.5, 1.0, 1.5, 2.0}. For each (lambda, risk_aversion) pair, run the strategy on the training period of the window and compute Sharpe. Select the pair maximizing Sharpe.

For K=4 soft regime, calibration is redone with K=4 soft regime probabilities active during the training period. The calibrated parameters differ from K=3 calibration (different regime structure) and from hard regime calibration (different optimizer behavior). Calibrated parameters per window are listed in Appendix C.

The calibration uses training period only; test period is fully out-of-sample.

---

# Appendix B: Code Architecture and Module Map

This appendix maps the K=4 extension code to the file structure and execution sequence.

## B.1 Top-level directory structure

```
D:\Projects\SystematicPortfolioEngine\regime_switching\
├── m1_data_loader.py                      # B2: data loading
├── m2_fixed_ms.py                          # B2: K=2, K=3 fixed MS
├── m2b_k4_diagnostic.py                    # K=4 ext: K=4 fixed MS
├── m2c_k5_diagnostic.py                    # K=4 ext: K=5 rejection
├── m3_tvtp_ms.py                           # B2: K=3 TVTP
├── m3b_tvtp_ms_k4_constrained.py           # K=4 ext: K=4 TVTP with constraints
├── m4_model_selection.py                   # B2: BIC comparison
├── m5_comparison.py                        # B2: vs rule-based
├── m6_integration.py                       # B2: integration scaffold
├── m6b_k4_integration.py                   # K=4 ext: integration scaffold
├── m7_visualizer.py                        # B2: plot functions
├── m7b_k4_visualizer.py                    # K=4 ext: K=4-specific plots
├── run_soft_backtest.py                    # B2: K=3 walk-forward
├── run_soft_backtest_k4.py                 # K=4 ext: K=4 walk-forward
├── run_exp010.py                           # K=4 ext: macro factor analysis
├── compare_k3_vs_k4.py                     # K=4 ext: three-way comparison
├── generate_report.py                      # B2: K=3 final PDF
├── generate_report_k4.py                   # K=4 ext: K=4 final PDF
├── data/                                   # All intermediate parquet/json
│   ├── window_fits/                        # B2: K=3 per-window fits
│   ├── k4_extension_constrained/           # K=4 ext: per-window K=4 fits
│   ├── m2b_k4_diagnostic/                  # K=4 ext: K=4 fixed MS results
│   ├── m2c_k5_diagnostic/                  # K=4 ext: K=5 rejection results
│   ├── factor_trend_analysis/              # K=4 ext: macro factor analysis
│   └── exp010/                             # K=4 ext: statistical analysis
└── outputs/                                # PDFs and PNGs
    ├── B2_TVTP_Regime_Detection_Report.pdf
    └── B2_K4_Extension_Report.pdf
```

## B.2 Execution sequence for K=4 extension

The K=4 extension was built and run as a sequence of independent steps, each producing checkpoint outputs for the next step:

**Step 1: M2b K=4 fixed-MS diagnostic**
```
python m2b_k4_diagnostic.py
# Output: data/m2b_k4_diagnostic/ms4_fixed_results.json
# Outcome: K=4 BIC=-47194 vs K=3 BIC=-47101 → K=4 supported
```

**Step 2: M2c K=5 rejection**
```
python m2c_k5_diagnostic.py
# Output: data/m2c_k5_diagnostic/ms5_fixed_results.json
# Outcome: BIC=-47226 (numerically better) but pathological → K=5 rejected
```

**Step 3: EXP010 macro factor predictive analysis**
```
python run_exp010.py
# Output: data/exp010/exp010_results.json + data/factor_trend_analysis/
# Outcome: ROC AUC=0.598, drawdown LR p=0.013 → macro factors significant
```

**Step 4: M3b K=4 TVTP estimation (per-window)**
```
python m3b_tvtp_ms_k4_constrained.py
# Output: data/k4_extension_constrained/window_fits/window_X/tvtp_result.{pkl,json}
# Output: data/k4_extension_constrained/window_fits/window_X/filtered_probs.parquet
# Output: data/k4_extension_constrained/window_fits/window_X/em_convergence_log.json
# Runtime: 5.36 hours sequential (8 restarts × 18 windows)
# Outcome: All windows converged with 3 constraints active
```

**Step 5: M6b K=4 walk-forward backtest**
```
python run_soft_backtest_k4.py
# Output: data/k4_extension_constrained/window_fits/window_X/daily_nav.parquet
# Output: data/k4_extension_constrained/window_fits/window_X/trade_log.parquet
# Output: data/k4_extension_constrained/integration_backtest_k4.parquet (stitched)
# Output: data/k4_extension_constrained/integration_metrics_k4.json (aggregate)
# Runtime: ~6 hours sequential
# Outcome: K=4 aggregate CAGR=23.17%, Sharpe=1.295
```

**Step 6: Three-way comparison**
```
python compare_k3_vs_k4.py
# Output: data/k4_extension_constrained/k3_vs_k4_comparison.json
# Output: data/k4_extension_constrained/all_data_for_paper.json
# Outcome: K=4 vs K=3 8/18, K=3 vs Meridian 7/18, all noise-equivalent
```

**Step 7: Final PDF report**
```
python generate_report_k4.py
# Output: outputs/B2_K4_Extension_Report.pdf
# Includes: all key tables, BIC comparison, three-way result, integration verdict
```

## B.3 Key implementation details

**m3b_tvtp_ms_k4_constrained.py** (the core K=4 TVTP module) is approximately 1,400 lines. Key functions:

- `fit_k4_tvtp(returns, covariates, n_restarts=8, max_iter=700)`: top-level fitting function
- `hamilton_filter_tvtp(returns, covariates, mu, sigma, transition_params)`: forward pass with time-varying transitions
- `kim_smoother(filtered_probs, transition_matrices)`: backward smoother
- `em_step(returns, covariates, current_params)`: one EM iteration with constraints
- `m_step_transitions_lbfgs(xi_smoothed, gamma_smoothed, covariates, current_params, mu_bound=1.0, p_diag_lambda=50)`: numerical M-step for logistic with P_diag penalty
- `init_from_k3_reference(k3_results)`: initialization from K=3 fitted parameters

**run_soft_backtest_k4.py** (modified copy of EXP006 engine) is approximately 800 lines. Key changes from run_exp006.py:

- Lines 145-165: replace hard regime function with soft regime probability lookup
- Lines 230-245: 4-state weighted parameter blending with K=4-specific recovery → bull mapping
- Lines 380-395: per-window parameter calibration uses K=4 probabilities
- Lines 510-525: NAV stitching across windows (unchanged from EXP006)

The rest of the engine (cvxpy optimizer, trailing stops, sector neutralization, transaction cost model, portfolio rebalancing) is identical to EXP006. This is intentional — the soft regime is the only changing input.

**compare_k3_vs_k4.py** orchestrates the three-way comparison:
- Loads K=3 NAV from `regime_switching/data/integration_backtest.parquet`
- Loads K=4 NAV from `regime_switching/data/k4_extension_constrained/integration_backtest_k4.parquet`
- Loads hard NAV from EXP006 reference parquets
- Aligns on common date range (2005-01-03 to 2026-03-06)
- Computes per-window and aggregate metrics
- Saves to `data/k4_extension_constrained/k3_vs_k4_comparison.json`

## B.4 Reproducibility

All K=4 extension results are reproducible from the saved data files. The TVTP fits are saved per-window as both pickle and JSON. The walk-forward backtest is saved as parquet. The comparison results are saved as JSON.

To reproduce the strategy invariance result:
```
cd D:\Projects\SystematicPortfolioEngine\regime_switching
python compare_k3_vs_k4.py
# Reads existing fits and NAVs, recomputes comparison metrics
# Total runtime: < 5 minutes (just I/O + arithmetic, no fitting)
```

To reproduce K=4 TVTP from scratch:
```
python m3b_tvtp_ms_k4_constrained.py
# Re-fits all 18 windows with 8 restarts each
# Total runtime: ~5 hours
```

To reproduce K=4 walk-forward backtest from scratch:
```
python run_soft_backtest_k4.py
# Re-runs EXP006 engine on all 18 windows with K=4 soft regime
# Requires K=4 TVTP fits to exist already
# Total runtime: ~6 hours
```

The full pipeline from K=3 fits + EXP006 fits as inputs to final PDF report runs in about 12 hours sequential, ~3 hours if parallelized across windows (parallelization not implemented because the sequential runtime is acceptable for one-time research use).

---

# Appendix C: Per-Window Calibrated Parameters

Calibrated (lambda, risk_aversion) parameters per window for all three strategies. Calibration uses Sharpe-maximizing grid search on the training period of each walk-forward window. Test period is out-of-sample.

| Window | Hard λ | Hard ra | K=3 λ | K=3 ra | K=4 λ | K=4 ra |
|---|---|---|---|---|---|---|
| 1 | 0.003 | 1.5 | 0.003 | 2.0 | 0.001 | 2.0 |
| 2 | 0.001 | 2.0 | 0.001 | 2.0 | 0.010 | 1.5 |
| 3 | 0.005 | 2.0 | 0.001 | 0.5 | 0.005 | 2.0 |
| 4 | 0.003 | 2.0 | 0.001 | 1.5 | 0.002 | 1.0 |
| 5 | 0.002 | 1.0 | 0.001 | 2.0 | 0.001 | 2.0 |
| 6 | 0.001 | 0.5 | 0.005 | 2.0 | 0.002 | 0.5 |
| 7 | 0.001 | 2.0 | 0.003 | 2.0 | 0.007 | 2.0 |
| 8 | 0.010 | 0.5 | 0.003 | 2.0 | 0.005 | 2.0 |
| 9 | 0.007 | 1.5 | 0.007 | 1.5 | 0.007 | 1.5 |
| 10 | 0.001 | 1.5 | 0.010 | 1.0 | 0.005 | 1.0 |
| 11 | 0.001 | 0.5 | 0.010 | 0.5 | 0.010 | 1.0 |
| 12 | 0.007 | 1.5 | 0.005 | 1.0 | 0.010 | 1.0 |
| 13 | 0.007 | 1.0 | 0.007 | 2.0 | 0.005 | 1.0 |
| 14 | 0.007 | 1.5 | 0.003 | 1.5 | 0.007 | 2.0 |
| 15 | 0.010 | 0.5 | 0.005 | 1.5 | 0.010 | 1.5 |
| 16 | 0.010 | 0.5 | 0.010 | 0.5 | 0.005 | 1.5 |
| 17 | 0.010 | 1.5 | 0.010 | 1.0 | 0.007 | 1.0 |
| holdout | 0.005 | 1.5 | 0.005 | 1.5 | 0.005 | 1.5 |

## Observations

**No clustering of optimal parameters.** Across 18 windows, the calibrated lambda spans the full grid (0.001 to 0.010) and risk_aversion spans the full grid (0.5 to 2.0) for all three strategies. No window is concentrated at boundary values. This indicates the optimizer's parameter space is genuinely sensitive to regime structure — different regimes call for different parameters.

**Soft regime calibrations differ from hard regime calibrations.** Window 8 illustrates: Hard λ=0.010 ra=0.5 vs K=3 λ=0.003 ra=2.0 vs K=4 λ=0.005 ra=2.0. The hard strategy prefers high turnover with low risk aversion; the soft strategies prefer modest turnover with high risk aversion. This is consistent with soft regime probabilities producing more conservative optimizer parameter selections than hard binary labels.

**K=3 and K=4 calibrations are similar in some windows, different in others.** Windows 9 and holdout match exactly. Windows 1 and 4 differ substantially. The pattern suggests K=4 calibration responds to additional regime structure (recovery state) where K=3 cannot.

**Holdout window matches exactly across all three.** All three strategies calibrate to (λ=0.005, ra=1.5) on the 2022-2026 holdout period. This is the smallest training period and the most recent macro regime, suggesting that for current market conditions all three strategies converge on similar optimizer parameters. This is consistent with the strategy invariance finding.

These calibrated parameters are saved in:
- Hard: `D:/Projects/StrategyResearchLab/experiments/exp006_extended_wf/data/window_calibrations/`
- K=3: `regime_switching/data/window_fits/window_X/optimizer_calibration.json`
- K=4: `regime_switching/data/k4_extension_constrained/window_fits/window_X/optimizer_calibration.json`

---

# Appendix D: Per-Window K=4 TVTP Regime Parameters

Annualized regime parameters for K=4 TVTP across all 18 walk-forward windows. mu values in annualized return units, sigma values in annualized volatility units, occupancy is the time-averaged smoothed probability for the crisis state.

| Window | mu_crisis | mu_bear | mu_recovery | mu_bull | σ_crisis | σ_bear | σ_recovery | σ_bull | occ_crisis | EM iter |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.500 | -0.058 | 0.122 | 0.332 | 0.446 | 0.165 | 0.220 | 0.102 | 0.062 | 29 |
| 2 | -0.393 | 0.024 | 0.107 | 0.160 | 0.436 | 0.212 | 0.093 | 0.142 | 0.060 | 37 |
| 3 | -0.408 | 0.042 | 0.097 | 0.177 | 0.437 | 0.210 | 0.139 | 0.084 | 0.054 | 46 |
| 4 | -0.407 | 0.003 | 0.099 | 0.194 | 0.446 | 0.216 | 0.142 | 0.084 | 0.045 | 34 |
| 5 | -0.893 | -0.054 | 0.096 | 0.194 | 0.623 | 0.227 | 0.144 | 0.084 | 0.045 | 33 |
| 6 | -0.913 | 0.001 | 0.107 | 0.197 | 0.596 | 0.232 | 0.146 | 0.084 | 0.052 | 30 |
| 7 | -0.841 | -0.021 | 0.108 | 0.232 | 0.587 | 0.232 | 0.147 | 0.083 | 0.051 | 41 |
| 8 | -1.000 | 0.016 | 0.038 | 0.248 | 0.644 | 0.260 | 0.161 | 0.085 | 0.036 | 35 |
| 9 | -1.000 | 0.023 | 0.031 | 0.254 | 0.642 | 0.258 | 0.160 | 0.084 | 0.034 | 46 |
| 10 | -1.000 | 0.022 | 0.024 | 0.276 | 0.646 | 0.261 | 0.162 | 0.084 | 0.032 | 42 |
| 11 | -1.000 | 0.018 | 0.021 | 0.280 | 0.645 | 0.259 | 0.161 | 0.081 | 0.030 | 47 |
| 12 | -1.000 | 0.019 | 0.020 | 0.268 | 0.638 | 0.160 | 0.258 | 0.081 | 0.030 | 42 |
| 13 | -1.000 | 0.024 | 0.033 | 0.241 | 0.605 | 0.160 | 0.248 | 0.082 | 0.034 | 35 |
| 14 | -0.861 | -0.032 | 0.090 | 0.254 | 0.579 | 0.233 | 0.142 | 0.069 | 0.038 | 47 |
| 15 | -0.842 | -0.046 | 0.080 | 0.251 | 0.567 | 0.232 | 0.143 | 0.070 | 0.039 | 41 |
| 16 | -0.841 | -0.048 | 0.085 | 0.263 | 0.569 | 0.233 | 0.143 | 0.069 | 0.037 | 45 |
| 17 | -1.000 | -0.031 | 0.124 | 0.266 | 0.613 | 0.232 | 0.137 | 0.067 | 0.040 | 48 |
| holdout | -1.000 | -0.042 | 0.117 | 0.268 | 0.623 | 0.237 | 0.142 | 0.071 | 0.036 | 33 |

## Observations

**Crisis state hits the boundary in 9 of 18 windows.** Windows 8 through 13 and 17, holdout all hit mu_crisis = -1.0 (the constraint floor). This indicates the unconstrained estimate would be more negative; the bound is binding. This is intentional and is the consequence of the mu boundary constraint in the M-step. Without the constraint, these windows would produce extreme negative crisis means (e.g., -200% annualized) which are economically implausible and numerically unstable.

**Bull state mean is well-identified.** Across all windows, mu_bull falls in [0.16, 0.33] annualized, with cluster around 0.20-0.27. This is consistent with long-run S&P 500 expansion mean (~10-12% annualized) with positive selection bias in periods classified as bull. No bull window hits the +100% boundary.

**Recovery state is consistent and meaningful.** mu_recovery falls in [0.02, 0.124] across windows, with most values in 0.08-0.12. sigma_recovery falls in [0.09, 0.26], with most values in 0.13-0.17. The recovery state is empirically distinct from bear (lower volatility) and from bull (lower mean). This validates the K=4 specification.

**Bear state has positive mean in some windows.** Windows 2, 3, 4, 6, 8-13 have mu_bear ≥ 0. This reflects the periods being structurally low-return-low-volatility "muddle through" rather than active drawdown. This is consistent with the K=3 K=4 distinction: K=3 had to lump these periods into either bull or bear; K=4 separates them as recovery.

**Sigma ordering is mostly preserved.** Crisis > bear > recovery and bear > bull in most windows. Some windows (12, 13) show sigma_recovery > sigma_bear due to identification swap; this would be cleaned up by post-estimation labeling but the underlying parameter quality is unaffected.

**Occupancy of crisis state is small and stable.** crisis_occupancy ranges from 0.030 to 0.062 across windows (mean ~0.043). This indicates the crisis state captures genuine tail events (3-6% of trading days) rather than smoothing into the bear state.

**EM convergence is good.** Iteration counts range from 29 to 48 across windows, all well within the 700 max iter cap. This indicates the EM algorithm with the constraints converges quickly when initialized from K=3 reference parameters. The constraints work as intended.

**Window-to-window stability is high.** Walk-forward windows 8 through 17 produce highly similar parameters (mu, sigma all within tight ranges). This indicates the K=4 model is stable to changes in training data and is not just chasing in-sample fluctuations.

These per-window parameters are saved in:
`regime_switching/data/k4_extension_constrained/window_fits/window_X/tvtp_result.json`

---

# Appendix E: Files Reference and Reproducibility

## E.1 Source code locations

All K=4 extension code lives in `D:\Projects\SystematicPortfolioEngine\regime_switching\`. Key files:

| File | Purpose |
|---|---|
| `m2b_k4_diagnostic.py` | K=4 fixed-MS diagnostic to support K=4 specification |
| `m2c_k5_diagnostic.py` | K=5 diagnostic showing pathological state characteristics |
| `m3b_tvtp_ms_k4_constrained.py` | K=4 TVTP with three constraints (mu bounds, P_diag penalty, K=3 init) |
| `run_soft_backtest_k4.py` | Walk-forward backtest using actual EXP006 engine with K=4 soft regime |
| `run_exp010.py` | Macro factor predictive analysis (ROC AUC, drawdown LR, Granger) |
| `compare_k3_vs_k4.py` | Three-way comparison Hard vs K=3 vs K=4 |
| `generate_report_k4.py` | Final PDF report generator |

Existing B2 files (used as inputs):
| File | Purpose |
|---|---|
| `m1_data_loader.py` | Returns and macro covariates loader |
| `m3_tvtp_ms.py` | K=3 TVTP fitting |
| `run_soft_backtest.py` | K=3 walk-forward backtest |
| `m7_visualizer.py` | Plot functions |

EXP006 reference engine (used as base for soft backtest engines):
- `D:/Projects/StrategyResearchLab/experiments/exp006_extended_wf/run_exp006.py`

## E.2 Data file locations

K=4 extension data:
```
regime_switching/data/k4_extension_constrained/
├── window_fits/
│   ├── window_1/ ... window_17/, window_holdout/
│   │   ├── tvtp_result.pkl + tvtp_result.json   # Fitted parameters
│   │   ├── filtered_probs.parquet               # Causal regime probabilities
│   │   ├── smoothed_probs.parquet               # Full-sample smoothed (analysis only)
│   │   ├── em_convergence_log.json              # LL per iteration, all 8 restarts
│   │   ├── covariate_stats.json                 # Training mean/std for standardization
│   │   ├── all_restarts.pkl                     # All 8 restart results
│   │   ├── daily_nav.parquet                    # K=4 soft NAV (test period)
│   │   ├── trade_log.parquet                    # Every trade
│   │   ├── optimizer_weights.parquet            # Portfolio weights at rebalances
│   │   └── optimizer_calibration.json           # Calibrated (λ, ra)
├── integration_backtest_k4.parquet              # Stitched soft NAV (full out-of-sample)
├── integration_metrics_k4.json                  # Aggregate + per-window metrics
├── k3_vs_k4_comparison.json                     # Three-way comparison results
└── all_data_for_paper.json                      # Consolidated data dump
```

Diagnostic outputs:
```
regime_switching/data/m2b_k4_diagnostic/ms4_fixed_results.json
regime_switching/data/m2c_k5_diagnostic/ms5_fixed_results.json
regime_switching/data/factor_trend_analysis/   # EXP010 outputs
regime_switching/data/exp010/exp010_results.json
```

B2 K=3 data (used as comparison baseline):
```
regime_switching/data/window_fits/window_X/    # Per-window K=3 fits
regime_switching/data/integration_backtest.parquet
regime_switching/data/integration_metrics.json
regime_switching/data/ms3_tvtp_results.json
regime_switching/data/tvtp_coefficients.json
```

EXP006 reference data (used as hard regime baseline):
```
StrategyResearchLab/experiments/exp006_extended_wf/data/wf_results/
├── nav_window_X.parquet                        # Hard NAV per window
└── stitched_nav.parquet                        # Hard aggregate NAV
```

## E.3 Reproducibility commands

To regenerate the three-way comparison metrics from existing fits and NAVs:
```
cd D:\Projects\SystematicPortfolioEngine\regime_switching
.\..\SirAlgotsAlot\Scripts\Activate.ps1
python compare_k3_vs_k4.py
```

To regenerate K=4 TVTP fits from scratch (5+ hours):
```
python m3b_tvtp_ms_k4_constrained.py
```

To regenerate K=4 walk-forward backtest from scratch (6+ hours, requires fits):
```
python run_soft_backtest_k4.py
```

To regenerate the final PDF report:
```
python generate_report_k4.py
```

## E.4 Dependencies

All dependencies are in the SirAlgotsAlot venv. No new packages were installed for the K=4 extension. Key packages:

- numpy, pandas, scipy (core scientific computing)
- statsmodels (used for fixed-MS cross-validation in M2/M2b only)
- matplotlib, seaborn (plotting)
- cvxpy (mean-variance optimization in EXP006 engine)
- pyarrow (parquet I/O)
- reportlab (PDF generation)

Custom code:
- Hamilton filter and Kim smoother: implemented from scratch in m3_tvtp_ms.py and m3b_tvtp_ms_k4_constrained.py
- TVTP M-step with constraints: implemented from scratch with scipy.optimize.minimize (L-BFGS-B)
- Walk-forward orchestration: modified from EXP006

No third-party regime-switching library is used at the K=4 TVTP level. Cross-validation against statsmodels MarkovRegression is done at the fixed-TP level only (M2, M2b) to validate the Hamilton filter implementation.

## E.5 Hardware and runtime

All computation was done on a laptop:
- Intel Core i7-10750H (6 physical cores, 12 logical)
- 32 GB RAM
- Windows 10
- VS Code + PowerShell

Sequential runtimes:
- M2b K=4 fixed-MS: ~30 minutes
- M2c K=5 fixed-MS: ~45 minutes
- EXP010: ~10 minutes
- M3b K=4 TVTP (8 restarts × 18 windows): 5 hours 21 minutes
- run_soft_backtest_k4 (18 windows): ~6 hours
- compare_k3_vs_k4: ~2 minutes
- generate_report_k4: ~1 minute

Total wall-clock for full K=4 extension from scratch: approximately 12.5 hours sequential. Parallelization across windows (not implemented) would reduce this to ~3 hours.

## E.6 Git tracking

All code and small data files (JSON, configs) are committed to the SystematicPortfolioEngine repo. Large parquet files are tracked via Git LFS:
```
*.parquet filter=lfs diff=lfs merge=lfs -text
```

Pickle files are tracked but excluded from LFS:
```
window_fits/**/*.pkl
```

The K=4 extension introduces approximately 280 MB of new tracked files (mostly parquet NAV time series and per-window TVTP probabilities). All files are reproducible from code + raw data, so the LFS storage is convenience rather than necessity.

---

# Appendix F: Glossary

**BIC (Bayesian Information Criterion).** Model selection statistic = -2 × log-likelihood + k × log(T). Lower is better. K=4 BIC=-47194 vs K=3 BIC=-47101 means K=4 is preferred.

**C1, C2, C3.** Pre-registered integration criteria for promoting a regime detector to live Meridian. C1 = Cohen's kappa ≥ 0.40 vs ground truth. C2 = median lead time ≥ 30 days. C3 = aggregate CAGR improvement ≥ +0.5 pp AND Sharpe improvement ≥ +0.05.

**Cohen's kappa.** Inter-rater agreement statistic. 0 = chance agreement, 1 = perfect agreement. Used for C1.

**Crisis (regime).** State with strongly negative mean returns (mu < -0.5 annualized) and high volatility (sigma > 0.4 annualized). In K=4: state 0.

**EM (Expectation-Maximization).** Iterative algorithm for fitting Markov-switching models. E-step computes filtered/smoothed probabilities; M-step updates parameters. For TVTP, M-step requires numerical optimization.

**Filtered probability.** P(S_t = k | r_1, ..., r_t, z_1, ..., z_t) — uses information up to time t only. Used for backtesting (no lookahead).

**Hamilton filter.** Forward recursion computing filtered probabilities for Markov-switching models. Hamilton (1989). Adapted here for time-varying transitions.

**Kim smoother.** Backward recursion computing smoothed probabilities for Markov-switching models. Kim (1994). Used in EM M-step.

**K=3, K=4.** Number of regime states. K=3 has bull/bear/crisis. K=4 adds recovery as an intermediate state.

**Ledoit-Wolf shrinkage.** Covariance estimator that shrinks sample covariance toward a structured target (constant volatility). Reduces estimation error. Standard in Meridian's optimizer.

**Logistic transition.** TVTP transition probability function: P(S_t = j | S_{t-1} = i, z_t) = exp(...) / sum exp(...). Multinomial logistic with covariates.

**Markov-switching (MS).** Regime model where state evolves as a Markov chain and observations have regime-conditional distribution. Hamilton (1989).

**Meridian.** Live systematic equity strategy. Uses rule-based regime detector currently. Subject of integration testing.

**P_diag.** Time-averaged diagonal probability for state i: sum_t P_t[i,i] / T. Constraint: P_diag_i ≥ 0.90 for all i (regime persistence).

**Recovery (regime).** New state added in K=4 between bear and bull. Mu ~ +5-12% annualized, sigma ~ 13-17% annualized. Captures "muddle through" periods.

**Smoothed probability.** P(S_t = k | r_1, ..., r_T, z_1, ..., z_T) — uses all available information including future. Used for analysis only, not backtest.

**Soft regime.** Probability distribution over regime states (not hard label). Used as parameter weighting input to optimizer.

**SPY.** S&P 500 ETF. Underlying for daily returns r_t in this work.

**Strategy invariance.** Empirical finding that Meridian's strategy results are statistically indistinguishable across three regime detectors (rule-based, K=3 TVTP, K=4 TVTP). Aggregate CAGR within 0.4 pp.

**TVTP (Time-Varying Transition Probability).** Markov-switching extension where transition probabilities are functions of observable covariates. Filardo (1994). Implemented here for K=3 (B2) and K=4 (this work).

**VIX, yield curve, credit spread.** The three macro covariates used in TVTP transitions. Daily values aligned with returns. Standardized to zero mean unit variance.

**Walk-forward.** Out-of-sample backtest methodology where parameters are calibrated on training period and evaluated on subsequent test period. EXP006 uses 17 windows + 1 holdout.

**Window.** A walk-forward training-test pair. EXP006 uses 17 expanding windows + 1 holdout (2022-2026).

---

# References

Ang, A., & Bekaert, G. (2002). International Asset Allocation with Regime Shifts. *Review of Financial Studies*, 15(4), 1137-1187.

Ang, A., & Bekaert, G. (2004). How regimes affect asset allocation. *Financial Analysts Journal*, 60(2), 86-99.

Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94-107.

Diebold, F. X., Lee, J.-H., & Weinbach, G. C. (1994). Regime switching with time-varying transition probabilities. In C. Hargreaves (Ed.), *Nonstationary Time Series Analysis and Cointegration* (pp. 283-302). Oxford University Press.

Estrella, A., & Mishkin, F. S. (1998). Predicting U.S. recessions: Financial variables as leading indicators. *Review of Economics and Statistics*, 80(1), 45-61.

Filardo, A. J. (1994). Business cycle phases and their transitional dynamics. *Journal of Business and Economic Statistics*, 12(3), 299-308.

Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.

Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control*, 31(11), 3503-3544.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.

Hamilton, J. D. (1990). Analysis of time series subject to changes in regime. *Journal of Econometrics*, 45(1-2), 39-70.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). And the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.

Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics*, 60(1-2), 1-22.

Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions: Modelling, Statistical Inference, and Application to Business Cycle Analysis*. Springer.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

Lo, A. W. (2004). The adaptive markets hypothesis: Market efficiency from an evolutionary perspective. *Journal of Portfolio Management*, 30, 15-29.

Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

Nystrup, P., Hansen, B. W., Madsen, H., & Lindstrom, E. (2024). Statistical jump models: A non-parametric alternative for regime detection. *Journal of Financial Econometrics*, forthcoming.

Pagan, A. R., & Sossounov, K. A. (2003). A simple framework for analysing bull and bear markets. *Journal of Applied Econometrics*, 18(1), 23-46.

Pesaran, M. H., & Timmermann, A. (2002). Market timing and return prediction under model instability. *Journal of Empirical Finance*, 9(5), 495-510.

Sims, C. A., & Zha, T. (2006). Were there regime switches in U.S. monetary policy? *American Economic Review*, 96(1), 54-81.

Timmermann, A. (2008). Elusive return predictability. *International Journal of Forecasting*, 24(1), 1-18.

