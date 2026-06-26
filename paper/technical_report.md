# Technical Report: Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers

> **⚠️ This document describes v1 results only.** The arXiv v2 paper (paper/latex/paper.pdf) supersedes this with: canary detection at n=49, adversarial robustness characterisation, scan martingale, LLM canary evaluation, and monitorability falsification. See FOLLOW_UP_EXPERIMENTS.md for the complete v2 record.

**Author:** Jun Wen Leong  
**Date:** June 2026  
**Status:** Pre-registered evaluation complete. arXiv: [2606.11949](https://arxiv.org/abs/2606.11949)

---

## Executive Summary

Safety classifiers deployed at scale degrade silently when input distributions shift. This project builds an online monitoring system that (1) detects distributional shift using sequential statistics on classifier outputs, and (2) adapts decision thresholds post-detection via conformal prediction to preserve coverage guarantees — all without requiring new labeled data.

The system was evaluated in a pre-registered factorial design: 4 classifiers × 5 shift conditions × 20 seeds × 2 window sizes = 800 experimental cells, plus three independent ground-truth regimes. Total compute: ~120 hours across two machines.

**Key findings:**
- 86.6% detection rate across 800 cells (95% CI [84.1%, 88.8%])
- Weighted conformal correction recovers 14 pp of coverage for DeBERTa but eliminates data-driven reweighting for all other classifiers due to density-ratio estimation failure in high dimensions
- Classifier × shift interaction explains 18.5% of detection latency variance — monitoring must be tuned per-classifier

---

## 1. Problem Statement

When a safety classifier is deployed in production, the input distribution changes over time: users adapt their language, adversaries optimize attacks, new content categories emerge. The classifier's accuracy degrades, but no error signal is available without ground-truth labels. Current practice typically relies on periodic offline evaluation (days or weeks of delay) or assumes stationarity (no monitoring at all).

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
| DeBERTa | 1024 | 6.5 pp | +14.0 pp | ESS = 88/300, max weight 3.02 |
| ShieldGemma | 3584 | 22.5 pp | +6.0 pp | ESS = 300/300, all at floor (no reweighting) |
| Llama Guard | 4096 | 33.5 pp | +1.5 pp | ESS = 300/300, all at floor (no reweighting) |

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
- **Cross-classifier canary architecture (Regime C):** GCG suffixes optimized against DeBERTa shift Llama Guard's scores *toward* unsafe (detected in 14/40 cells). Our results suggest that a second architecturally-different classifier may serve as a distributional canary — detecting evasion attacks not by classifying content, but by flagging anomalous score distributions that the targeted classifier cannot see. (Based on 22-example corpus with a single attack pattern; larger-scale validation needed.)

---

## 4. Operational Implications

**For teams deploying safety classifiers:**

1. **Monitor per-classifier, not per-system.** The 18.5% interaction variance means a single alarm threshold will be miscalibrated for specific classifier × shift pairings.

2. **Discriminative classifiers can use weighted conformal correction.** DeBERTa-class models (fine-tuned encoders with <1000-d embeddings) benefit from density-ratio reweighting post-alarm.

3. **Generative classifiers need alternative adaptation.** Llama Guard / ShieldGemma class models cannot use standard density ratio estimation. Dimensionality reduction before estimation is a known remedy (Stojanov et al. 2019); our diagnostic confirms that PCA to ≤32 dimensions breaks the separability in these specific embeddings. Non-parametric estimators that avoid perfect separability are an alternative.

4. **Window size = 100 is preferred.** 7 steps faster detection at marginal FAR cost.

5. **Architecturally distinct classifiers may serve as distributional canaries.** Our Regime C results suggest that if an adversarial attack succeeds against one classifier (scores shift toward safe), an architecturally-different classifier may detect the anomaly via its own score distribution shift — not by classifying content correctly, but because the attack-perturbed inputs look anomalous from a different representational vantage point. This observation is based on a 22-example corpus with a single attack pattern; validation on larger, more diverse adversarial corpora is needed before operational deployment.

---

## 5. Methodology and Reproducibility

**Statistical methods:**
- BCa bootstrap CIs on means (10,000 resamples, seed 42, bias-corrected and accelerated)
- Wilson Score intervals on rates
- Clopper-Pearson exact intervals on coverage proportions
- Permutation tests (10,000 permutations for pairwise comparisons; 1,000 for ANOVA, sufficient given all p < 0.001)
- Holm-Bonferroni correction on 8 highlighted comparisons (all survive at family-wise α = 0.05)
- All reported at 95% confidence

**Corpus validation:**
- Paraphrase: 50/500 reviewed, 14-20% estimated refusal contamination; automated filtering identifies 9.4% (47/500). Filtered ablation confirms negligible effect on detection (DeBERTa 38.0→37.8, Llama Guard 66.6→60.8 steps).
- Code-switch: 50/500 reviewed, all confirmed authentic Singlish by native speaker
- Compositional: 20/300 reviewed, 100% placement accuracy
- Temporal: 20/292 reviewed, 100% genuine jailbreaks
- Adversarial suffix: 20/22 reviewed, all score flips confirmed

**Verification:** `scripts/verify_paper_numbers.py` checks 90 statistics against raw data. All pass.

**Code:** [github.com/junwenleong/safety-classifier-shift-monitor](https://github.com/junwenleong/safety-classifier-shift-monitor)

---

## 6. Post-Factorial Results

**CS growing-window:** 120/120 detection (100%), 0% FAR, ~2× latency vs KS. At 30% mixing: CS 97% vs KS 43% (Fisher exact p < 0.0001, non-overlapping CIs). Deployment profile: KS for speed at high mixing, CS for reliability at low mixing.

**MMD on embeddings:** 120/120 detection at latency=100 (immediate), FAR 3–10%. Binary alarm with no gradation — complementary to KS's graded severity signal.

**Gradual drift:** Detectable at all ramp rates (50–200 steps) when mixing ≥50%. Detection boundary at ~30% mixing. CS detects at 30% where KS fails.

**Mechanistic hypothesis:** Score std vs latency r=0.97 (n=4, suggestive). Embedding displacement vs latency r=−0.09 (not significant) — detection mediated by score geometry, not embedding geometry.

**Filtered ablation:** 9.4% refusals. Removing them: DeBERTa 38.0→37.8, Llama Guard 66.6→60.8. Negligible effect.

**PCA diagnostic:** ESS reduction at dim=32 confirmed on paraphrase (Llama Guard ESS=32, ShieldGemma ESS=28), confirming the collapse is a dimensionality artifact.

---

## 7. Limitations and Future Work

**Current limitations:**
- Detection boundary at ~30% mixing — below this, neither KS nor CS reliably detects
- MMD provides no latency gradation (immediate binary alarm); Llama Guard FAR at 10% (2× target)
- Binary classifiers only (multi-category taxonomies need different statistics)
- Mechanistic correlation at n=4 classifiers (suggestive, not definitive)
- PCA recovery magnitude depends on calibration split; dimensionality reduction for density-ratio estimation is an established technique (Stojanov et al. 2019), and our contribution is the specific diagnosis in generative safety-classifier embeddings

**Future directions:**
- CUSUM/Bayesian change-point for drift below 30% mixing threshold
- Online conformal prediction without exchangeability (Gibbs & Candès 2021)
- Non-parametric density ratio estimators for high-dimensional embeddings
- Larger MMD reference pools for stable Llama Guard calibration
- Production deployment with real traffic streams

---

*All numbers verified against raw experimental data (90 assertions). Pre-registration committed before execution. Post-factorial additions documented in separate amendment (committed June 8, executed June 9–10).*
