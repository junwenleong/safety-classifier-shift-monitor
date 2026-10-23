---
layout: default
title: "Safety Classifier Shift Monitor"
image: https://junwenleong.github.io/safety-classifier-shift-monitor/assets/og-image-v2.png
---

# Your safety classifier is drifting and you won't know until it's too late

*Jun Wen Leong · June 2026*

I built an online monitoring system that detects distributional shift in deployed safety classifiers using only the classifier's own outputs, without requiring labels. Across 800 pre-registered factorial cells (4 classifiers × 5 shift types × 20 seeds × 2 window sizes), it catches 86.8% of shifts with mean latency of 39.7 steps. It also flags targeted gradient-based evasion attacks via score disagreement with a second classifier (37 of 49 DeBERTa-evading GCG inputs), with a worked local calculation of where one fixed attack objective stalls. That calculation is not a general security boundary.

**The bad news:** the standard fix (weighted conformal prediction) silently fails for 3 of 4 classifiers. The density ratios collapse to floor, and the "adaptation" you think is running is doing nothing.

---

## Why this matters

Safety classifiers degrade silently. When the input distribution shifts (adversarial adaptation, multilingual users, emerging attack patterns) accuracy drops with no error signal. You have no ground-truth labels in real time. By the time periodic offline evaluation catches it, your classifier has been making unreliable decisions for days.

This system watches only the classifier's score distribution and fires an alarm when it changes. No labels, no oracle, no human review of individual predictions.

---

## Detection works

| Classifier | Paraphrase | Code-switch | Compositional | Temporal | Adversarial |
|---|---|---|---|---|---|
| DeBERTa | 28.4 | 32.2 | 24.5 | 23.5 | 36.5 |
| Text-Mod. | 34.6 | 33.2 | 29.6 | 24.8 | 25.3 |
| Llama Guard | 69.6 | 93.9 | 47.2 | 42.4 | 27.9 |
| ShieldGemma | 85.6 | 83.3 | 43.9 | 27.6 | 27.7 |

*Mean detection latency in steps. Lower is faster.*

The critical pattern: **encoders and decoders invert on shift type.** DeBERTa catches paraphrase in 28 steps but adversarial suffix takes 37. Llama Guard catches adversarial in 28 steps but paraphrase takes 70. A single monitoring threshold miscalibrates systematically for one class of shift. The interaction effect explains 18.6% of latency variance, so you need per-classifier monitoring profiles.

---

## Conformal correction silently fails

When shift is detected, weighted conformal prediction should adapt thresholds using density ratios to preserve coverage. For DeBERTa, it works: +16 pp recovery (ESS = 88/300, genuine reweighting).

For the other three classifiers, it does nothing:

| Classifier | Embedding dim | ESS | Recovery | What's happening |
|---|---|---|---|---|
| DeBERTa | 768 | 88 | +16 pp | ✓ Working: 24 calibration points retain meaningful weights |
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
| ShieldGemma | +7.5 pp (ESS=300) | +20.5 pp (ESS=85) | 13 pp |

At 32 dimensions, 82–91% of embedding variance is retained but the logistic classifier can no longer achieve perfect separability. At 64 dimensions, ShieldGemma re-collapses; at 128, both re-collapse. The critical threshold is ≤32 dimensions for these embedding spaces.

DeBERTa (768-d, no baseline collapse) shows no degradation under PCA. The reduction removes noise dimensions rather than safety-relevant signal.

**Practical recommendation:** If you're using weighted conformal with a generative safety classifier, project embeddings to ≤32 dimensions before estimating density ratios. Check ESS: if it's near n_cal, your reweighting is doing nothing.

---

## Variance decomposition

Neither classifier choice nor shift type alone determines detection difficulty:

| Factor | η² | Meaning |
|---|---|---|
| Classifier | 0.248 | Which model you deploy matters |
| Shift type | 0.239 | What kind of shift hits you matters equally |
| Classifier × Shift | 0.186 | The interaction matters almost as much |
| Residual | 0.328 | Random seed variation |

All three systematic factors are significant (p < 0.001, 1000 permutations). A monitoring system tuned for one classifier on one shift type will systematically miscalibrate on other combinations.

---

## Adversarial robustness

The canary effect (a second, un-targeted classifier detecting when the primary is under gradient-based attack) works under precise conditions:

- **Attack-specific:** GCG divergence >> random noise (p<10⁻¹², n=49). Canary alarms far less often under length-matched random suffixes (12.2%) than under GCG (75.5%, Llama Guard pair; 93.9% for the best single canary).
- **Transfer is mostly pre-existing misses:** of 40 canary observations (10 prompts × 4 epoch-checkpoint canaries), 14 end below 0.5 but 12 of those were already below 0.5, so genuine unsafe-to-safe crossings are 2 of 28. The earlier 4.3% vs 76.5% contrast is withdrawn as a confidence gate.
- **Local stall calculation:** for the fixed objective f_A + λ(f_B - f_A)², a divergence-minimising attacker's final gaps cluster near 1/(2λ)=0.250 among blocked runs (14/20 blocked at λ=2, median 0.250). The blocked frequency does not replicate (70% on the primary checkpoint pair, 30% on a replication pair, 50% in the λ sweep), and the paper withdraws the earlier claim that the stall is confidence-gated.
- **Joint optimisation:** same-family white-box optimisation reaches both scores below 0.5 on 11/20 prompts, but 5 of those started below 0.5; among the 13 prompts both models initially flagged it is 6/13.
- **Architecture diversity:** no clear advantage in this four-model ensemble (η²=0.011, p=0.070); the cause of transfer resistance is unresolved.

A k=2 canary ensemble detects 48/49 at unadjusted thresholds. The scan martingale gave FAR≤1% on the null streams tested without per-classifier threshold tuning; this is empirical, not a proven anytime bound.

---


## The Deployment Configuration Trap

We tested 35 frontier LLMs as safety monitoring canaries via a frontier model service. The headline findings:

**Much of the apparent inverse scaling was a token-budget parsing artefact.** Reasoning models (o3, o4-mini) need a larger completion-token cap for their internal chain-of-thought. At the standard `max_tokens=16`, they produce empty responses that our code parsed as 1.0 (maximum unsafe). Of the 31 models in the Pareto file, 19 give graded scores on both benign and adversarial prompts; the rest mix behaviours (some score once the cap is adequate, some return no score for most adversarial prompts, gpt-5.5 scores everything near 1.0). The paper does not report a split of the 35 models into discriminators and refusers.

**Production recommendation:** Deploy `gpt-4o-mini` ($0.033 per 1,000 calls, at prices current at time of evaluation):
- Wilson 95% CI lower bound of 71% detection on the evaluated corpus
- <1.5% false positive rate (7/1000, N=1000)
- No breach of the 0.5 threshold under black-box coordinate ascent (100 and 500 iterations, small samples)
- Detection mostly tracks the base prompt rather than the suffix, though 59 of 980 prompt-model pairs cross the 0.5 boundary; prefix-rephrasing invariance is not established

For safety-critical deployments, pay 9× for `gpt-5.1` (Wilson 95% lower bound of 83.5% detection).

**Cross-lingual caveat:** Detection degrades 15-25% for non-English prompts. Spanish confirmed at N=49 (gpt-4o-mini): 63.3% detection (Wilson CI [49.3%, 75.3%]) vs 83.7% English (20.4pp drop). Mandarin/Arabic at N=20 with wider CIs. Degradation is vocabulary-driven (explicit harm keywords are English-specific), not structural. Suffix perturbation remains inert across all languages.

---

## Depth results

**Black-box optimisation hit a floor in this budget.** 500-step coordinate-ascent attacks (10 runs on at most 9 distinct high-baseline prompts): 9/10 show zero improvement after step 50; 1/10 has one small reduction still above threshold. Mean final score: 0.90 (threshold: 0.5). No prompt breaches 0.5. No fracture point; increasing attacker budget provides no advantage.

**Temperature sensitivity.** Detection rate varies by <5pp across T∈{0, 0.3, 1.0} (5 models × 20 adv + 20 benign, 3 reps per non-zero T). Within-prompt SD at T=1.0: 0.03–0.11. T=0 recommended for reproducibility.

**CoT Suffocation phase transition.** Response probability follows a sigmoid: $P(\text{response} \mid T_r) = \sigma(k \cdot (T_r - T_{50}))$. For o3: $T_{50}(\text{benign})=46$ tokens, $T_{50}(\text{adversarial})=117$ tokens (95% CI [104, 129], measured on an extended token-budget grid; supersedes an earlier 154-token/3.3× estimate that rested on a single out-of-grid extrapolation point and was withdrawn). The 2.5× gap explains why standard configs fail on complex inputs. Deploy with max_completion_tokens ≥ 200 for adversarial coverage.

**Monitorability law not supported.** The n=4 correlation (r=0.97) between null-score std and detection latency was an encoder/decoder gap artefact. Within-family (n=6 encoder variants): r=0.21, p=0.70; no support for it as a predictable property (n=6, correlated checkpoints).

**Dual-channel cross-lingual mechanism.** Explicit-harm prompts degrade 19-29pp cross-lingually (lexical keyword failure), while ambiguous prompts degrade only 10-15pp (structural intent transfers). For gpt-5.1: 84% of detections are structurally based (hold cross-lingually).

**N=1000 FPR.** gpt-4.1 and claude-haiku-4-5 achieve 0/1000 false positives (95% CI upper ≤ 0.37%). All 10 evaluated canaries maintain FPR < 1.5%.

**Cost-bounded safety routing (illustrative).** Escalation router: screen with gpt-4o-mini, escalate ambiguous/non-English to gpt-5.1. At an assumed 12% escalation rate: $69/1M queries ($33 + 0.12 × $300) for 84.9% detection (vs $300/1M for always-gpt-5.1). The escalation rate depends on production traffic, which we did not measure.

**Honesty note on the scan martingale:** its value is operational simplicity (shared parameters across classifiers, no per-classifier tuning), with an empirical FAR on the tested null streams and no proven anytime bound. For instantaneous step-onset shift with proper per-condition calibrated KS thresholds, KS matches or exceeds scan. For ramped-onset adversaries the scan detected 60% / 93% of streams at 15% / 20% mixing against 33-57% / 67-87% for KS, depending on the KS threshold rule. An earlier, larger gap (37% vs 3%) could not be reproduced and is withdrawn. A claimed advantage for the growing-window confidence sequence at low mixing was also withdrawn: with equal reference information KS detects as often and is faster.

## What I built

- KS detector on sliding-window score distributions with empirically calibrated thresholds (50 negative control streams per classifier)
- Conformal abstention layer with split-conformal prediction sets and optional density-ratio reweighting
- Full factorial evaluation harness with BCa bootstrap CIs, permutation tests, and Holm-Bonferroni correction
- Three ground-truth regimes: synthetic onset, real jailbreak examples from public red-team databases inserted at a fixed onset, and GCG adversarial demonstrations

---

## Links

[arXiv Paper](https://arxiv.org/abs/2606.11949) · [GitHub Repository](https://github.com/junwenleong/safety-classifier-shift-monitor) · [Full Results (FINDINGS.md)](https://github.com/junwenleong/safety-classifier-shift-monitor/blob/main/FINDINGS.md)
