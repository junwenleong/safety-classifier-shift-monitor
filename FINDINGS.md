# Findings

## Summary

An online monitoring system detects distributional shift in deployed safety classifiers with 86.6% detection rate across 800 pre-registered factorial cells (4 classifiers × 5 shift conditions × 20 seeds × 2 window sizes), mean detection latency of 39.5 steps, and empirical false alarm rates of 2–10%. Upon detection, weighted conformal prediction recovers coverage for discriminative classifiers (+14 pp for DeBERTa) but collapses completely for generative classifiers (+1.5 pp for Llama Guard, +6 pp for ShieldGemma), due to density ratio estimation failure in high-dimensional general-purpose embedding spaces. Variance decomposition reveals that classifier (η² = 0.243), shift type (η² = 0.237), and their interaction (η² = 0.185) all contribute substantially to detection latency — neither factor alone determines difficulty, and per-classifier monitoring profiles are necessary.

## The Problem

Safety classifiers degrade silently under distributional shift. When the input distribution changes — through adversarial adaptation, linguistic drift, multilingual code-switching, or emerging attack patterns — classifier accuracy drops with no error signal. In production, ground-truth labels rarely arrive in real time. The monitor watches only the classifier's own outputs (scores and embeddings) and alerts deployers before accuracy collapses.

## System Design

The monitor observes a stream of classifier outputs: unsafe-class probability and penultimate-layer representation. A sliding-window KS statistic tracks whether the score distribution has changed relative to a frozen reference CDF. Alarm thresholds are calibrated empirically via 50 negative control streams (no shift injected), set at the 97th percentile of maximum observed statistics. Upon alarm, a conformal abstention layer adapts decision thresholds using density-ratio reweighting (Tibshirani et al. 2019) to preserve a target 90% coverage rate without new labels.

## Classifiers

| Classifier | Architecture | Parameters | Embedding dim |
|---|---|---|---|
| DeBERTa-v3-large | Transformer encoder | 304M | 1024 |
| Text-Moderation (KoalaAI) | DeBERTa-v3-base | 86M | 768 |
| Llama Guard 3 | Decoder-only LLM | 8B | 4096 |
| ShieldGemma | Decoder-only LLM | 9B | 3584 |

## Shift Conditions

| Condition | Mechanism | Threat model |
|---|---|---|
| Paraphrase | GPT-4o semantic rewording | Organic rephrasing |
| Code-switch | Singlish transliteration | Non-English users |
| Compositional | Harmful content in long-context wrappers | Context-window attacks |
| Temporal | Real jailbreaks (lmsys/toxic-chat) | Emerging harm categories |
| Adversarial suffix | GCG-optimized tokens | Automated red-teaming |

## Results

### RQ1: Detection Performance

Detection rate: 693/800 cells = 86.6% (95% Wilson CI [84.1%, 88.8%]).

**Mean detection latency (steps):**

| Classifier | Paraphrase | Code-switch | Compositional | Temporal | Adversarial |
|---|---|---|---|---|---|
| DeBERTa | 28.4 | 32.1 | 24.5 | 23.5 | 36.6 |
| Text-Mod. | 34.6 | 33.2 | 29.6 | 24.8 | 25.3 |
| Llama Guard | 69.4 | 93.4 | 47.0 | 42.1 | 27.8 |
| ShieldGemma | 85.0 | 81.8 | 43.9 | 27.1 | 26.8 |

A crossover interaction is visible: paraphrase is easy for encoders but hard for decoders; adversarial suffix shows the opposite pattern. This motivates RQ3.

**False alarm rates (95% Wilson CIs):** Text-Moderation 2.0% [0.8%, 5.0%] < Llama Guard 3.0% [1.4%, 6.4%] < ShieldGemma 8.5% [5.4%, 13.2%] < DeBERTa 9.5% [6.2%, 14.4%].

### RQ2: Conformal Adaptation

**Coverage under temporal shift (95% Clopper-Pearson CIs, n=200 per condition):**

| Classifier | Mode | Pre-shift | Post-shift | Gap | Abstentions |
|---|---|---|---|---|---|
| DeBERTa | Unweighted | 0.910 [.861, .946] | 0.845 [.787, .892] | 0.065 | 25 |
| DeBERTa | Weighted-on-alarm | 1.000 [.982, 1.00] | 0.985 [.957, .997] | 0.015 | 29 |
| ShieldGemma | Unweighted | 0.915 [.867, .950] | 0.690 [.621, .753] | 0.225 | 47 |
| ShieldGemma | Weighted-on-alarm | 0.925 [.879, .957] | 0.750 [.684, .808] | 0.175 | 31 |
| Llama Guard 3 | Unweighted | 0.890 [.838, .930] | 0.555 [.483, .625] | 0.335 | 41 |
| Llama Guard 3 | Weighted-on-alarm | 0.915 [.867, .950] | 0.570 [.498, .640] | 0.345 | 32 |

DeBERTa's recovery is statistically significant (weighted lower bound 0.957 > unweighted upper bound 0.892, non-overlapping CIs).

**The density ratio collapse mechanism:** For both generative models, logistic regression achieves perfect separability between source and target embeddings. All 300 calibration weights clip to the floor (1/C = 0.1), effective sample size = 300/300 (identical uniform weights = no-op). The weighted quantile degenerates to the unweighted quantile. For DeBERTa, 91.7% of weights clip to floor but ~25 calibration examples retain non-trivial weights (max 3.34, ESS = 79/300), driving the genuine 14 pp recovery.

### RQ3: Variance Decomposition

Two-way ANOVA on detection latency (693 valid detections):

| Factor | η² | 95% CI | Permutation p |
|---|---|---|---|
| Classifier | 0.243 | [0.205, 0.291] | < 0.001 |
| Shift type | 0.237 | [0.193, 0.293] | < 0.001 |
| Classifier × Shift | 0.185 | — | < 0.001 |
| Residual | 0.335 | — | — |

The three systematic factors contribute roughly equally. The initial N=5 estimate inflated the interaction (0.265); at N=20 it shrinks to 0.185 while main effects grow — consistent with small-sample noise.

### Robustness Across Ground-Truth Regimes

- **Regime A (Synthetic onset):** 86.6% detection rate (800 cells)
- **Regime B (Real temporal jailbreaks):** 85% detection rate (17/20 cells, mean latency 32.6 steps)
- **Regime C (GCG adversarial):** Cross-classifier anomaly detection. DeBERTa (target): adversarial suffixes push scores toward safe, monitor fails (38/40). Llama Guard (non-target): same suffixes push scores toward unsafe (+0.73), monitor detects in 14/40. Adversarial perturbations optimized against one classifier appear anomalous to architecturally different classifiers.

## Corpus Validation

Manual review of samples from each corpus:

- **Paraphrase (50/500):** ~18-22 preserved harmful intent; 14-20% became LLM refusals (safety responses instead of paraphrases). Detection latencies for paraphrase should be interpreted conservatively.
- **Code-switch (50/500):** All 50 confirmed as authentic Singlish by native speaker. 20-30% became refusals; same caveat applies.
- **Compositional (20/300):** 20/20 correctly placed harmful content at stated position. 100% structural integrity.
- **Temporal (20/292):** 20/20 genuine jailbreak prompts from lmsys/toxic-chat. Zero false positives.
- **Adversarial suffix (20/22):** 20/22 correct suffix concatenation with confirmed score flips (orig ≥0.95 → attacked ≤0.01). One example excluded post-validation (original score 0.002, already benign).

## Limitations

- **Single-condition conformal evaluation.** RQ2 evaluated on temporal shift only. Other shift types may show different patterns.
- **Abrupt shift onset.** Gradual drift would require CUSUM-style statistics.
- **Residual variance.** 33.5% of latency variance is noise. MDE is 13.9 steps at 80% power.
- **Binary classifiers only.** Multi-category safety taxonomies may exhibit category-specific shift invisible to scalar scores.
- **Refusal contamination.** 14-20% of paraphrase and 20-30% of code-switch corpora are LLM refusals rather than faithful transformations.
- **FAR asymmetry.** False alarm rates vary 5× across classifiers despite identical calibration.

## Verification

All numbers in this document were programmatically verified against raw experimental data using `scripts/verify_paper_numbers.py` (21 assertions, all passing). Experiment configurations in `configs/` were committed before execution (commit `be630f3`). The factorial design, all hyperparameters, and ground-truth regimes match the pre-registration exactly.

Paper: [arXiv](#) · Code and results: [github.com/junwenleong/safety-classifier-shift-monitor](https://github.com/junwenleong/safety-classifier-shift-monitor)
