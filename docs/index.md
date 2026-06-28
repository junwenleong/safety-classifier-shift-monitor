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


## The Deployment Configuration Trap (v2)

We tested 35 frontier LLMs as safety monitoring canaries via the frontier-api API. The headline findings:

**All models work when configured correctly.** The apparent 'inverse scaling' (bigger/reasoning models = worse monitors) was entirely a token-budget parsing artifact. Reasoning models (o3, o4-mini) require `max_completion_tokens≥200` for their internal chain-of-thought. At the standard `max_tokens=16`, they produce empty responses that our code parsed as 1.0 (maximum unsafe). With proper configuration, all 33 of 35 models discriminate correctly (benign≈0.0, harmful≈0.8).

**Two genuine failures:** gpt-5-nano and gpt-5.5 refuse to score harmful content entirely (content-aware refusal). They score benign correctly (0.0) but return empty for harmful inputs — a binary detection signal, not a scoring failure.

**Production recommendation:** Deploy `gpt-4o-mini` ($0.15/1M input tokens — approximately $0.000033 per classification call):
- ≥71% guaranteed detection (Wilson 95% CI lower bound)
- <1.5% false positive rate (7/1000, N=1000)
- Robust to black-box adversarial optimization (100 iterations, max Δ=0.2)
- Semantic detection survives prefix rephrasing, language translation, and suffix scrambling

For safety-critical deployments, pay 10× for `gpt-5.1` (≥83.5% guaranteed detection).

**Cross-lingual caveat:** Detection degrades 15-25% for non-English prompts (Spanish, Mandarin, Arabic). The degradation is vocabulary-driven (explicit harm keywords are English-specific), not structural. Suffix perturbation remains inert across all languages.

---

## Depth results (post-v2)

**Black-box optimization hits a hard floor.** 500-step coordinate-ascent attacks reveal a saturating fitness landscape: all improvements occur in the first 50 steps, then the score is completely flat (Δ=0.000 for steps 50–500). Mean final score: 0.88 (threshold: 0.5). No fracture point exists — increasing attacker budget provides zero marginal advantage.

**CoT Suffocation phase transition.** Response probability follows a sigmoid: P(response|T_r) = σ(k·(T_r - T_50)). For o3: T_50(benign)=46 tokens, T_50(adversarial)=154 tokens. The 3.3× gap explains why standard configs fail on complex inputs. Deploy with max_completion_tokens ≥ 200 for adversarial coverage.

**Monitorability law falsified.** The n=4 correlation (r=0.97) between null-score std and detection latency was an encoder/decoder gap artifact. Within-family (n=6 encoder variants): r=0.21, p=0.70 — not a predictable property.

**Dual-channel cross-lingual mechanism.** Explicit-harm prompts degrade 19-29pp cross-lingually (lexical keyword failure), while ambiguous prompts degrade only 10-15pp (structural intent transfers). For gpt-5.1: 84% of detections are structurally based (hold cross-lingually).

**N=1000 FPR.** gpt-4.1 and claude-haiku-4-5 achieve 0/1000 false positives (95% CI upper ≤ 0.37%). All 10 evaluated canaries maintain FPR < 1.5%.

**Cost-bounded safety routing.** Escalation router: screen with gpt-4o-mini, escalate ambiguous/non-English to gpt-5.1. At 12% escalation rate: $65/1M queries for 84.9% detection (vs $300/1M for always-gpt-5.1).

**Honesty note on the scan martingale:** its value is operational simplicity (deploy once, guaranteed FAR ≤ α), not superior detection power. With proper per-condition calibrated KS thresholds, KS matches or exceeds the martingale at all mixing levels.

## What I built

- KS detector on sliding-window score distributions with empirically calibrated thresholds (50 negative control streams per classifier)
- Conformal abstention layer with split-conformal prediction sets and optional density-ratio reweighting
- Full factorial evaluation harness with BCa bootstrap CIs, permutation tests, and Holm-Bonferroni correction
- Three ground-truth regimes: synthetic onset, real temporal jailbreaks from public red-team databases, and GCG adversarial demonstrations

---

## Links

[arXiv Paper](https://arxiv.org/abs/2606.11949) · [GitHub Repository](https://github.com/junwenleong/safety-classifier-shift-monitor) · [Full Results (FINDINGS.md)](https://github.com/junwenleong/safety-classifier-shift-monitor/blob/main/FINDINGS.md)
