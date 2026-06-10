---
layout: default
title: "Online Shift Detection for Safety Classifiers"
image: https://junwenleong.github.io/safety-classifier-shift-monitor/assets/og-image.png
---

# Online shift detection and conformal adaptation for deployed safety classifiers

*Jun Wen Leong · May 2026*

I built an online monitoring system that detects when a safety classifier has moved out of distribution, then adapts decision thresholds to preserve coverage -- without requiring new labels. Evaluated in a pre-registered factorial design across 800 cells (4 classifiers × 5 shift conditions × 20 seeds × 2 window sizes).

**The three headline findings:**

1. **86.6% detection rate** (693/800 cells, 95% CI [84.1%, 88.8%]) with mean latency of 39.5 steps and empirical false alarm rates of 2–10%.
2. **Weighted conformal correction works for discriminative classifiers but collapses for generative ones.** DeBERTa recovers 14 pp of coverage; ShieldGemma recovers 6 pp; Llama Guard recovers 1.5 pp. The mechanism: logistic regression achieves perfect separability in high-dimensional generative embeddings, driving all density ratios to zero -- the reweighting becomes a no-op.
3. **Classifier × shift interaction explains 18.5% of variance** in detection latency. Neither classifier choice nor shift type alone determines detection difficulty -- per-classifier monitoring profiles are necessary.

---

## Why this matters

Safety classifiers degrade silently. When the input distribution shifts -- through adversarial adaptation, linguistic drift, or emerging attack patterns -- classifier accuracy drops with no error signal until ground-truth labels arrive. In production, labels rarely arrive in real time. By the time periodic offline evaluation detects the problem, the classifier has been making unreliable decisions for days or weeks.

This system tells deployers *when* their classifier has moved out of distribution, before accuracy collapses, using only the classifier's own outputs.

---

## The system

The monitor watches the distribution of classifier scores via a sliding-window KS statistic with empirically calibrated alarm thresholds. When shift is detected, a conformal abstention layer adapts decision thresholds using density-ratio reweighting.

**Classifiers evaluated:**
- DeBERTa-v3-large (304M, discriminative encoder)
- Text-Moderation/KoalaAI (86M, discriminative encoder)
- Llama Guard 3 (8B, generative decoder)
- ShieldGemma (9B, generative decoder)

**Shift conditions:**
- Paraphrase (GPT-4o semantic rewording)
- Code-switch (Singlish transliteration)
- Compositional (harmful content in long-context wrappers)
- Temporal (real jailbreaks from lmsys/toxic-chat)
- Adversarial suffix (GCG-optimized tokens)

---

## Key result: the conformal correction gradient

| Classifier | Embedding dim | Coverage gap | Weighted recovery | Mechanism |
|---|---|---|---|---|
| DeBERTa | 1024-d | 6.5 pp | +14.0 pp | 79/300 calibration points get meaningful weights |
| ShieldGemma | 3584-d | 22.5 pp | +6.0 pp | All weights collapse to floor (ESS = 300/300) |
| Llama Guard 3 | 4096-d | 33.5 pp | +1.5 pp | All weights collapse to floor (ESS = 300/300) |

For both generative models, logistic regression achieves perfect separability between source and target embeddings -- every calibration point gets assigned P(target|x) ≈ 0, all weights clip to 1/C = 0.1, and the weighted quantile degenerates to the unweighted quantile. The apparent 6 pp and 1.5 pp "recovery" derives from the +1 denominator term in the normalized weights, not from meaningful reweighting.

This is the Hughes phenomenon applied to importance weighting: high-dimensional general-purpose embeddings provide trivial separability that defeats density ratio estimation.

---

## Detection latency heatmap

The factorial reveals a crossover interaction between classifier architecture and shift type:

- **Paraphrase** is easy for encoders (28–35 steps) but hard for decoders (69–85 steps)
- **Adversarial suffix** is hardest for DeBERTa (36.6 steps) but easiest for Llama Guard (27.8 steps)

A monitoring system that sets thresholds based on classifier-level or shift-level averages will systematically under-alert on hard pairings and over-alert on easy ones.

---

## Robustness across ground-truth regimes

Detection generalizes beyond synthetic onset:
- **Regime A** (synthetic): 86.6% detection rate (800 cells)
- **Regime B** (real temporal jailbreaks): 85% detection rate (17/20 cells)
- **Regime C** (GCG adversarial): Cross-classifier anomaly detection -- adversarial perturbations optimized against one classifier appear anomalous to architecturally different classifiers

---

## What I built

- Three-channel detection architecture: KS (graded severity), CS (low-mixing sensitivity), MMD (binary backstop)
- Confidence sequence engine with time-uniform coverage guarantees (demonstrated: 100% detection, 0% FAR)
- Sliding-window KS detector on classifier score distributions
- MMD detector on penultimate-layer embeddings (immediate detection, FAR controlled at 3–10%)
- Conformal abstention layer with unweighted and weighted-on-alarm modes
- Density ratio estimation via logistic regression with weight clipping
- Full factorial evaluation harness with negative/positive controls
- Variance decomposition (two-way ANOVA with BCa bootstrap CIs and permutation tests)
- Ramp-rate and mixing-level sensitivity characterization
- 5 shift dataset builders (paraphrase, code-switch, compositional, temporal, adversarial suffix)

---

## Post-factorial findings

**CS vs KS deployment profile:** At 30% mixing (realistic contamination), CS detects 97% while KS detects only 43% (p < 0.0001). At 50%+, both detect but KS is ~2× faster. CS is necessary when signals are weak; KS is preferred when signals are strong.

**MMD as binary alarm:** With proper calibration (1000-permutation bootstrap, pooled reference), MMD fires immediately on all shifts with no latency variation. Provides guaranteed backstop; KS provides severity grading.

**Mechanistic hypothesis:** Null score standard deviation correlates with detection latency (r=0.97, n=4 classifiers). Tighter score boundaries → faster detection. Embedding displacement does NOT predict latency (r=−0.09), ruling out representation-space geometry as the driver.

---

## Links

[arXiv Paper](#) · [GitHub Repository](https://github.com/junwenleong/safety-classifier-shift-monitor) · [Full Results (FINDINGS.md)](https://github.com/junwenleong/safety-classifier-shift-monitor/blob/main/FINDINGS.md) · [Verification Script](https://github.com/junwenleong/safety-classifier-shift-monitor/blob/main/scripts/verify_paper_numbers.py)

All reported statistics were programmatically verified against raw experimental data using `verify_paper_numbers.py` (90 assertions, all passing). Pre-registration committed before execution; post-factorial additions documented in separate amendment.
