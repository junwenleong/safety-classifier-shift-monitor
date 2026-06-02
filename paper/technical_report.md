# Technical Report: Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers

**Author:** Jun Wen Leong  
**Date:** June 2026  
**Status:** Pre-registered evaluation complete. arXiv submission pending.

---

## Executive Summary

Safety classifiers deployed at scale degrade silently when input distributions shift. This project builds an online monitoring system that (1) detects distributional shift using sequential statistics on classifier outputs, and (2) adapts decision thresholds post-detection via conformal prediction to preserve coverage guarantees — all without requiring new labeled data.

The system was evaluated in a pre-registered factorial design: 4 classifiers × 5 shift conditions × 20 seeds × 2 window sizes = 800 experimental cells, plus three independent ground-truth regimes. Total compute: ~120 hours across two machines.

**Key findings:**
- 86.6% detection rate across 800 cells (95% CI [84.1%, 88.8%])
- Weighted conformal correction recovers 14 pp of coverage for discriminative classifiers but collapses for generative ones due to density ratio estimation failure
- Classifier × shift interaction explains 18.5% of detection latency variance — monitoring must be tuned per-classifier

---

## 1. Problem Statement

When a safety classifier is deployed in production, the input distribution changes over time: users adapt their language, adversaries optimize attacks, new content categories emerge. The classifier's accuracy degrades, but no error signal is available without ground-truth labels. Current practice relies on periodic offline evaluation (days or weeks of delay) or assumes stationarity (no monitoring at all).

**The operational gap:** deployers need a real-time signal that says "your classifier is no longer in-distribution" *before* accuracy has collapsed. This system provides that signal using only the classifier's own outputs.

---

## 2. Technical Approach

### Detection Channel

A sliding-window KS statistic compares the most recent *w* classifier scores against a frozen reference CDF collected during calibration:

- Window sizes: 100 and 200
- Significance level: α = 0.05
- Alarm threshold: empirically calibrated via 50 negative control streams (97th percentile of maximum observed statistic)
- Warmup period: *w* steps (window must fill before testing)

The KS statistic detects any change in the score distribution — location, scale, or shape shifts all produce signal.

### Adaptation Layer

Upon alarm, a split-conformal prediction layer adapts the decision threshold:

- **Unweighted mode:** Standard conformal prediction (Vovk et al. 2005). Computes nonconformity scores on calibration data, sets threshold at the (1-ε) quantile. Abstains when prediction set ≠ {single class}.
- **Weighted-on-alarm mode:** Estimates density ratios p_target/p_source via logistic regression on classifier embeddings (Tibshirani et al. 2019). Recomputes the conformal quantile as a weighted quantile. Clips weights to [1/10, 10] for stability.

### Factorial Design

Pre-registered before execution (commit `be630f3`):
- 4 classifiers spanning discriminative (DeBERTa 304M, Text-Moderation 86M) and generative (Llama Guard 8B, ShieldGemma 9B)
- 5 shift conditions: paraphrase, code-switch (Singlish), compositional (long-context), temporal (real jailbreaks), adversarial suffix (GCG)
- 20 seeds per cell (increased from initial 5 after pilot showed inflated interaction estimates)
- 2 window sizes (100, 200)
- Parallel negative control for each cell (alarm must not fire on in-distribution data)

---

## 3. Results

### Detection (RQ1)

| Metric | Value |
|---|---|
| Overall detection rate | 86.6% (693/800) |
| Mean latency (w=100) | 39.5 steps [37.0, 42.4] |
| Mean latency (w=200) | 45.4 steps [42.6, 48.3] |
| FAR range | 2.0% – 9.5% |

**Architectural crossover:** Encoders detect paraphrase fast (28-35 steps) but adversarial suffix slow (36.6 steps). Decoders show the opposite (paraphrase 69-85 steps, adversarial suffix 26-28 steps). This crossover is invisible to single-classifier studies.

### Conformal Adaptation (RQ2)

Weighted correction effectiveness degrades monotonically from discriminative to generative classifiers:

| Classifier | Dim | Coverage gap | Recovery | Mechanism |
|---|---|---|---|---|
| DeBERTa | 1024 | 6.5 pp | +14.0 pp | ESS = 79/300, max weight 3.34 |
| ShieldGemma | 3584 | 22.5 pp | +6.0 pp | ESS = 300/300, all at floor (no-op) |
| Llama Guard | 4096 | 33.5 pp | +1.5 pp | ESS = 300/300, all at floor (no-op) |

**Root cause:** Logistic regression achieves perfect separability between calibration and target embeddings in the high-dimensional generative space. All density ratios → 0, all weights clip to 1/C = 0.1 (uniform = no reweighting). The DeBERTa recovery is statistically confirmed: weighted CI [.957, .997] does not overlap unweighted CI [.787, .892].

### Variance Decomposition (RQ3)

| Factor | η² | p-value |
|---|---|---|
| Classifier | 0.243 | < 0.001 |
| Shift type | 0.237 | < 0.001 |
| Interaction | 0.185 | < 0.001 |
| Residual | 0.335 | — |

All three factors contribute substantially. A monitoring system that ignores the interaction will systematically under-alert on hard pairings (DeBERTa × adversarial: +21.6 steps) and over-alert on easy ones (Llama Guard × adversarial: -16.0 steps).

### Robustness

- **Real temporal jailbreaks (Regime B):** 85% detection (17/20 cells)
- **Adversarial success (Regime C):** GCG suffixes that fool DeBERTa appear anomalous to Llama Guard (detected in 14/40 cells) — cross-classifier anomaly detection

---

## 4. Operational Implications

**For teams deploying safety classifiers:**

1. **Monitor per-classifier, not per-system.** The 18.5% interaction variance means a single alarm threshold will be miscalibrated for specific classifier × shift pairings.

2. **Discriminative classifiers can use weighted conformal correction.** DeBERTa-class models (fine-tuned encoders with <1000-d embeddings) benefit from density-ratio reweighting post-alarm.

3. **Generative classifiers need alternative adaptation.** Llama Guard / ShieldGemma class models cannot use standard density ratio estimation. Options: PCA to safety-relevant subspace before estimation, or non-parametric estimators that don't achieve perfect separability.

4. **Window size = 100 is preferred.** 7 steps faster detection at marginal FAR cost.

5. **Cross-classifier monitoring detects evasion attacks.** If an adversarial attack succeeds against one classifier (scores shift toward safe), monitoring a second architecturally-different classifier can detect the anomaly.

---

## 5. Methodology and Reproducibility

**Statistical methods:**
- Bootstrap CIs (1000 resamples, seed 42, percentile method) on means
- Wilson Score intervals on rates
- Clopper-Pearson exact intervals on coverage proportions
- Permutation tests (1000 permutations) for ANOVA significance
- All reported at 95% confidence

**Corpus validation:**
- Paraphrase: 50/500 reviewed, 14-20% refusal contamination (acknowledged as limitation)
- Code-switch: 50/500 reviewed, all confirmed authentic Singlish by native speaker
- Compositional: 20/300 reviewed, 100% placement accuracy
- Temporal: 20/292 reviewed, 100% genuine jailbreaks
- Adversarial suffix: 20/22 reviewed, all score flips confirmed

**Verification:** `scripts/verify_paper_numbers.py` checks 21 statistics against raw data. All pass.

**Code:** [github.com/junwenleong/safety-classifier-shift-monitor](https://github.com/junwenleong/safety-classifier-shift-monitor)

---

## 6. Limitations and Future Work

**Current limitations:**
- Conformal evaluation on temporal shift only (one of five conditions)
- Abrupt shift onset assumption (gradual drift would require CUSUM)
- Binary classifiers only (multi-category taxonomies need different statistics)
- Refusal contamination in paraphrase/code-switch corpora (14-30%)
- No production deployment validation

**Future directions:**
- Dimensionality-reduced density ratio estimation for generative model embeddings
- Online conformal prediction without exchangeability (Gibbs & Candès 2021)
- CUSUM/Bayesian change-point detection for gradual drift
- Multi-channel detection (KS + MMD jointly)
- Production deployment with real traffic streams

---

*All numbers verified against raw experimental data. Pre-registration committed before execution. Full factorial completed without error.*
