---
layout: default
title: "Safety Classifier Shift Monitor"
image: https://junwenleong.github.io/safety-classifier-shift-monitor/assets/og-image-v2.png
---

# Your safety classifier is drifting and you won't know until it's too late

*Jun Wen Leong · June 2026*

I built an online monitoring system that detects distributional shift in deployed safety classifiers using only the classifier's own outputs, without requiring labels. Across 800 pre-registered factorial cells (4 classifiers × 5 shift types × 20 seeds × 2 window sizes), it catches 86.6% of shifts with mean latency of 39.5 steps. It also detects targeted gradient-based evasion attacks via score disagreement with a second classifier, with a formally characterized security boundary.

**The bad news:** the standard fix (weighted conformal prediction) silently fails for 3 of 4 classifiers. The density ratios collapse to floor, and the "adaptation" you think is running is doing nothing.

---

## Why this matters

Safety classifiers degrade silently. When the input distribution shifts (adversarial adaptation, multilingual users, emerging attack patterns) accuracy drops with no error signal. You have no ground-truth labels in real time. By the time periodic offline evaluation catches it, your classifier has been making unreliable decisions for days.

This system watches only the classifier's score distribution and fires an alarm when it changes. No labels, no oracle, no human review of individual predictions.

---

## Detection works

| Classifier | Paraphrase | Code-switch | Compositional | Temporal | Adversarial |
|---|---|---|---|---|---|
| DeBERTa | 28.4 | 32.1 | 24.5 | 23.5 | 36.6 |
| Text-Mod. | 34.6 | 33.2 | 29.6 | 24.8 | 25.3 |
| Llama Guard | 69.4 | 93.4 | 47.0 | 42.1 | 27.8 |
| ShieldGemma | 85.0 | 81.8 | 43.9 | 27.1 | 26.8 |

*Mean detection latency in steps. Lower is faster.*

The critical pattern: **encoders and decoders invert on shift type.** DeBERTa catches paraphrase in 28 steps but adversarial suffix takes 37. Llama Guard catches adversarial in 28 steps but paraphrase takes 69. A single monitoring threshold miscalibrates systematically for one class of shift. The interaction effect explains 18.5% of latency variance, so you need per-classifier monitoring profiles.

---

## Conformal correction silently fails

When shift is detected, weighted conformal prediction should adapt thresholds using density ratios to preserve coverage. For DeBERTa, it works: +14 pp recovery (ESS = 88/300, genuine reweighting).

For the other three classifiers, it does nothing:

| Classifier | Embedding dim | ESS | Recovery | What's happening |
|---|---|---|---|---|
| DeBERTa | 1024 | 88 | +14 pp | ✓ Working: 24 calibration points retain meaningful weights |
| Text-Mod. | 768 | 300 | +2 pp | ✗ Collapsed: all weights at floor |
| Llama Guard | 4096 | 300 | +2 pp | ✗ Collapsed: all weights at floor |
| ShieldGemma | 3584 | 300 | +7.5 pp | ✗ Collapsed: all weights at floor |

*ESS = effective sample size out of 300 calibration points. ESS ≈ 300 means uniform weights, i.e. no adaptation.*

**The mechanism:** Logistic regression achieves perfect separability between source and target embeddings in the high-dimensional space. Every calibration point gets classified as "source" with near-certainty, driving all density ratios to zero. All weights clip to the floor (0.1). The residual +2–7 pp "recoveries" are a formula artifact of the test-point term (w(X_test) = 1.0 raises the quantile from 90.3% to 93.3%), not genuine adaptation.

You think conformal correction is running. Your monitoring dashboard shows "weights applied." But ESS ≈ n_cal means the weights are uniform. You're paying the computational cost of density estimation for zero benefit.

---

## The fix: PCA to 32 dimensions

PCA before density ratio estimation breaks the collapse:

| Classifier | Without PCA | With PCA (d=32) | Recovery gained |
|---|---|---|---|
| Llama Guard | +2 pp (ESS=300) | +33 pp (ESS=19.6) | 31 pp |
| ShieldGemma | +7.5 pp (ESS=300) | +21 pp (ESS=85) | 13.5 pp |

At 32 dimensions, 82–91% of embedding variance is retained but the logistic classifier can no longer achieve perfect separability. At 64 dimensions, ShieldGemma re-collapses; at 128, both re-collapse. The critical threshold is ≤32 dimensions for these embedding spaces.

DeBERTa (1024-d, no baseline collapse) shows no degradation under PCA. The reduction removes noise dimensions rather than safety-relevant signal.

**Practical recommendation:** If you're using weighted conformal with a generative safety classifier, project embeddings to ≤32 dimensions before estimating density ratios. Check ESS: if it's near n_cal, your reweighting is doing nothing.

---

## Variance decomposition

Neither classifier choice nor shift type alone determines detection difficulty:

| Factor | η² | Meaning |
|---|---|---|
| Classifier | 0.243 | Which model you deploy matters |
| Shift type | 0.237 | What kind of shift hits you matters equally |
| Classifier × Shift | 0.185 | The interaction matters almost as much |
| Residual | 0.335 | Random seed variation |

All three systematic factors are significant (p < 0.001, 1000 permutations). A monitoring system tuned for one classifier on one shift type will systematically miscalibrate on other combinations.

---

## Adversarial robustness (v2)

The canary effect — a second, un-targeted classifier detecting when the primary is under gradient-based attack — works under precise conditions:

- **Attack-specific:** GCG divergence >> random noise (p<10⁻¹², n=49). Silent under gibberish.
- **Confident canary:** detection is robust (7% transfer when canary confident)
- **Uncertain canary:** attacker can stealth-evade (100% transfer when uncertain)
- **Phase transition:** a divergence-minimising attacker stalls at gap=1/(2λ) when canary is confident — the defense has a predictable, measurable boundary (predicted 0.250, observed 0.235, within 95% CI)
- **Architecture diversity is NOT required** for detection (η²=0.011), but IS required for transfer robustness (0% cross-family vs 30% within-family)

Deploy k=2 classifiers (one same-family for sensitivity, one cross-family for transfer robustness). A scan martingale provides FAR≤1% without per-classifier threshold tuning.

---

## What I built

- KS detector on sliding-window score distributions with empirically calibrated thresholds (50 negative control streams per classifier)
- Conformal abstention layer with split-conformal prediction sets and optional density-ratio reweighting
- Full factorial evaluation harness with BCa bootstrap CIs, permutation tests, and Holm-Bonferroni correction
- Three ground-truth regimes: synthetic onset, real temporal jailbreaks from public red-team databases, and GCG adversarial demonstrations

---

## Links

[arXiv Paper](https://arxiv.org/abs/2606.11949) · [GitHub Repository](https://github.com/junwenleong/safety-classifier-shift-monitor) · [Full Results (FINDINGS.md)](https://github.com/junwenleong/safety-classifier-shift-monitor/blob/main/FINDINGS.md)
