# Findings

## Summary

An online monitoring system detects distributional shift in deployed safety classifiers with 86.6% detection rate across 800 pre-registered factorial cells (4 classifiers × 5 shift conditions × 20 seeds × 2 window sizes), mean detection latency of 39.5 steps, and empirical false alarm rates of 2–10%. Upon detection, weighted conformal prediction recovers coverage for discriminative classifiers (+14 pp for DeBERTa) but eliminates data-driven reweighting for generative classifiers (residual +1.5–6 pp are a formula artifact of the test-point term, not genuine adaptation), due to density ratio estimation failure in high-dimensional general-purpose embedding spaces. Variance decomposition reveals that classifier (η² = 0.243), shift type (η² = 0.237), and their interaction (η² = 0.185) all contribute substantially to detection latency — neither factor alone determines difficulty, and per-classifier monitoring profiles are necessary.

## The Problem

Safety classifiers degrade silently under distributional shift. When the input distribution changes — through adversarial adaptation, linguistic drift, multilingual code-switching, or emerging attack patterns — classifier accuracy drops with no error signal. In production, ground-truth labels typically do not arrive in real time. The monitor watches only the classifier's own outputs (scores and embeddings) and alerts deployers before the shift accumulates further.

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
| Temporal | Real jailbreaks from public red-team databases | Emerging harm categories |
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

**The density ratio collapse mechanism:** For both generative models, logistic regression achieves perfect separability between source and target embeddings. All 300 calibration weights clip to the floor (1/C = 0.1), eliminating data-driven reweighting. Residual recoveries (+0.02–0.10) at ESS≈300 are a mechanical artifact of the test-point contribution: the implicit weight w(X_test)=1.0 raises the effective quantile level from 90.3% to 93.3% at n_cal=300, ε=0.1 — a formula artifact, not adaptation. For DeBERTa, 92% of weights clip to floor but ~24 calibration examples retain non-trivial weights (max 3.02, ESS = 88/300), driving the genuine 14 pp recovery. (Numbers from the pooled `conformal_full.json` evaluation; an earlier single-classifier exploratory run reported ESS=79 due to a different calibration split.)

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
- **Regime C (GCG adversarial):** Cross-classifier anomaly detection. DeBERTa (target): adversarial suffixes push scores toward safe, monitor fails (38/40). Llama Guard (non-target): same suffixes push scores toward unsafe (+0.73), monitor detects in 14/40. Score direction diagnostics computed at evaluation time via `scripts/check_regime_c_direction.py`; detection counts verified by `verify_paper_numbers.py`. Adversarial perturbations optimized against one classifier appear anomalous to architecturally different classifiers.

## Corpus Validation

Manual review of samples from each corpus:

- **Paraphrase (50/500):** ~18-22 preserved harmful intent; 14-20% became LLM refusals (safety responses instead of paraphrases). Detection latencies for paraphrase should be interpreted conservatively.
- **Code-switch (50/500):** All 50 confirmed as authentic Singlish by native speaker. 20-30% became refusals; same caveat applies.
- **Compositional (20/300):** 20/20 correctly placed harmful content at stated position. 100% structural integrity.
- **Temporal (20/292):** 20/20 reviewed examples were genuine jailbreak prompts; zero false positives. Full corpus draws from three public red-team databases: lmsys/toxic-chat (39%), JailbreakBench (34%), ChatGPT-Jailbreak-Prompts (27%). Subsampled to 300 per factorial cell via repetition of 8 examples.
- **Adversarial suffix (20/22):** 20/22 correct suffix concatenation with confirmed score flips (orig ≥0.95 → attacked ≤0.01). One example excluded post-validation (original score 0.002, already benign).

## Post-Factorial Results

### Detection Channel Comparison

**CS growing-window:** 120/120 detection (100%), 0/40 FAR (0%), ~2× latency vs KS. At 30% mixing: CS 29/30 (97%) vs KS 13/30 (43%), Fisher exact p < 0.0001, non-overlapping Wilson CIs [0.83, 0.99] vs [0.27, 0.61].

**Deployment profile:** KS is preferred at high mixing (fast); CS is necessary at low mixing (reliable). Real drift is rarely 100% contamination — the CS advantage at low mixing is operationally significant.

**MMD on embeddings:** 120/120 detection at latency=100 (immediate). FAR controlled: DeBERTa 3.3%, Text-Mod 3.3%, ShieldGemma 0%, Llama Guard 10%. MMD provides binary alarm with no latency gradation — KS grades severity, MMD provides guaranteed backstop.

### Gradual Drift Sensitivity

Ramp-rate sweep (DeBERTa × paraphrase, cached scores, n=10):
- 50-step ramp: KS 10/10 (mean 94), CS 10/10 (mean 151)
- 200-step ramp: KS 9/10 (mean 210), CS 9/10 (mean 223)

Mixing-level sweep (50-step ramp):
- 30%: KS 4/10, CS 9/10
- 50%: KS 10/10, CS 10/10
- 100%: KS 10/10 (mean 64), CS 10/10 (mean 83)

### Mechanistic Hypothesis (Exploratory, n=4)

Null score std correlates with mean detection latency: r=0.97, p=0.032 (n=4 classifiers). However, within-family evaluation (6 encoder variants at different training epochs) yields r=0.21, p=0.70 — the original correlation was an artifact of the encoder/decoder architectural gap, not an intrinsic monitorability property. Pattern is shift-specific: paraphrase/temporal/compositional show wider→slower (r=0.70–0.97); adversarial suffix reverses (r=−0.20), producing the crossover.

Embedding displacement does NOT mirror this pattern (overall r=−0.09, p=0.78). Detection is mediated by score-boundary geometry, not representation-space distance.

### Filtered Paraphrase Ablation

Refusal rate: 47/500 = 9.4% (lower than 14–20% manual estimate). Removing refusals has negligible effect: DeBERTa 38.0→37.8 steps, Llama Guard 66.6→60.8 steps. Both 5/5 detected in both conditions.

### PCA Diagnostic

ESS reduction at dim=32 generalizes to paraphrase shift: Llama Guard ESS=32, ShieldGemma ESS=28 (both breaking separability). This confirms the collapse is driven by the curse of dimensionality (d ≫ n), consistent with established high-dimensional density-ratio instability (Stojanov et al. 2019; Sugiyama et al. 2011); the specific contribution is diagnosing the failure mode in generative safety-classifier embeddings. Coverage recovery magnitude is split-dependent; primary result (temporal: +33pp Llama Guard, +20.5pp ShieldGemma) uses fresh inference with proper conformal framework.

## Limitations

- **Gradual drift detection boundary.** At ≤30% mixing, KS detects only 43%. CS detects 97% but requires growing memory. Below 30%, neither channel reliably detects.
- **MMD provides no latency gradation.** Fires immediately (latency=100) on any shift; useful as binary backstop but not for severity assessment.
- **Residual variance.** 33.5% of latency variance is noise. MDE is 13.9 steps at 80% power.
- **Binary classifiers only.** Multi-category safety taxonomies may exhibit category-specific shift invisible to scalar scores.
- **Refusal contamination.** 9.4% of paraphrase corpus are LLM refusals (lower than the 14–20% manual estimate). Filtered ablation confirms negligible effect on detection.
- **FAR asymmetry.** Empirical KS false alarm rates vary 5× across classifiers (Text-Moderation 2.0% vs DeBERTa 9.5%). Resolved in v2 by a scan martingale that achieves FAR≤1% uniformly with no per-classifier calibration.
- **PCA diagnostic validated on temporal + paraphrase.** ESS reduction generalizes but coverage recovery magnitude depends on calibration split.

## Verification

All numbers in this document were programmatically verified against raw experimental data using `scripts/verify_paper_numbers.py` (90 assertions, all passing). Experiment configurations in `configs/` were committed before execution (commit `be630f3`). Post-factorial additions pre-registered in `docs/pre_registration_amendment_2.md` (committed June 8, executed June 9–10).

Paper: [arXiv](https://arxiv.org/abs/2606.11949) · Code and results: [github.com/junwenleong/safety-classifier-shift-monitor](https://github.com/junwenleong/safety-classifier-shift-monitor)

## v2 Additions (arXiv v2)

### Canary Detection (§7)
- Score disagreement detects GCG evasion (p<10⁻¹², n=49)
- Target-specific, not architecture-diverse (η²=0.011)
- k=2 ensemble achieves 98% detection
- Tokeniser barrier: 1.73× fragmentation ratio blocks cross-family transfer

### Adversarial Robustness (§8)
- 4-tier threat model with confidence-gated routing
- Within-family transfer: 30% passive, 70% with joint optimization
- Divergence-minimisation stall at gap=1/(2λ)=0.250 (empirical: 0.235-0.247)
- Black-box coordinate ascent fails against both Pareto-optimal canaries (max Δ=0.2)

### Frontier LLMs as Semantic Canaries (§7.8)
- 35 models tested via frontier-api API (api.openai.com)
- The 'ceiling-clipping' phenomenon was a deployment configuration artifact (max_tokens=16 + empty→1.0 parsing)
- With max_tokens≥200: ALL models discriminate correctly (33 discriminating + 2 content-aware refusers)
- N=1000 FPR: gpt-4.1 and claude-haiku-4-5 achieve 0/1000 (CI ≤ 0.37%)
- Semantic mechanism: detection operates on base-prompt intent, ignoring suffix tokens (BF01>3 for 16/20 models, TOST ±0.15 for 16/20)
- Ghost suffix: GCG suffixes inert on benign content (Δ=+0.005 for gpt-5.1)
- Prefix mutation: 92% detection survives rephrasing (gpt-5.1)
- Cross-lingual: 15-25% sensitivity degradation (vocabulary-driven), suffix inert across languages. Spanish confirmed N=49 (gpt-4o-mini): 63.3% (Wilson CI [49.3%, 75.3%]) vs 83.7% English — 20.4pp drop
- Cost-scaling: r=0.51 (flat), cost determines optimal choice, not capability
- Pareto frontier (Wilson CI): gpt-4o-mini ≥71% at $0.033/1k calls, gpt-5.1 ≥83.5% at $0.30/1k calls

### Scan Martingale (§3.7)
- Conformal test martingale replaces empirical KS calibration
- FAR≤1% uniformly across all 4 classifiers (vs 2-9.5% under empirical KS)
- Zero per-model calibration required
- **Honesty note**: with proper per-condition calibrated KS (97th-percentile threshold), KS matches or exceeds scan detection power at all tested mixing levels. The martingale's contribution is *operational simplicity* (deploy once, guaranteed FAR), not superior detection power.

### Depth Results (post-v2 characterization)

**CoT Suffocation phase transition.** Response probability follows a sigmoid: P(response|T_r) = σ(k·(T_r - T_50)). For o3: T_50(benign)=46 tokens (k=0.173), T_50(adversarial)=154 tokens (k=0.030). The 3.3× gap means standard API configs (max_tokens=16) cause total failure on adversarial inputs while benign queries occasionally succeed. Deploy with T_r ≥ T_50(hardest class) + 2.2/k for >90% coverage.

**500-step black-box hard floor.** Extended coordinate ascent (500 steps, n=10 prompts with baseline ≥0.8) reveals a saturating fitness landscape: 9/10 prompts show zero improvement after step 50; 1/10 shows one additional reduction (still well above 0.5). Mean final score: 0.90. No prompt breaches 0.5. No fracture point — increasing attacker budget provides no advantage.

**Temperature sensitivity.** Detection rate varies by <5pp across T∈{0, 0.3, 1.0} (5 models × 20 adv + 20 benign, 3 reps per non-zero T). Within-prompt SD at T=1.0: 0.03–0.11. T=0 recommended for reproducibility; temperature does not materially affect detection.

**Dual-channel cross-lingual mechanism.** The 15-25% cross-lingual degradation is asymmetric: explicit-harm prompts (direct violence, slurs) degrade 19-29pp, while ambiguous/roleplay prompts degrade only 10-15pp. Two safety channels: (i) lexical keyword matching (language-specific, fails under translation), (ii) structural intent recognition (language-invariant, transfers natively). For gpt-5.1: 84% of detections are structural (hold cross-lingually), 16% are lexical (leak when translated). Multilingual suffix transfer: mean Δ = +0.022 (suffix completely inert across languages).

**Cross-lingual N=49 (Spanish, gpt-4o-mini).** Confirmed at N=49: 63.3% detection (31/49, Wilson CI [49.3%, 75.3%]) vs 83.7% English (41/49, CI [71.0%, 91.5%]). Drop: 20.4pp. Tightens CI from ±20pp (N=20) to ±13pp. Mandarin/Arabic remain at N=20.

**Cost-bounded safety routing (CBSE).** The Pareto frontier admits a dynamic escalation router: screen all inputs with gpt-4o-mini ($0.033/1k); escalate ambiguous (score ∈ [0.3, 0.7]) and non-English inputs to gpt-5.1 ($0.30/1k). At ~12% escalation rate: expected cost $65/1M queries, 84.9% detection — vs $300/1M for always-gpt-5.1 (93.9%) and $33/1M for always-gpt-4o-mini (83.7%). Near-optimal detection at 78% cost reduction.

**Monitorability law falsified (Track C).** The n=4 correlation (r=0.97) between null-score std and detection latency was an encoder/decoder gap artifact. Within-family evaluation (n=6 encoder variants, epoch-{1,3,5,10} + 2 originals) yields r=0.21, p=0.70. Monitorability is not a predictable intrinsic property of score-distribution geometry.

**Llama Guard 3 surrogate GCG transfer.** Script prepared (`scripts/run_gcg_llama_guard_transfer.py`), model downloaded. Pending execution on Mac Studio (requires ≥20GB for 8B fp16 + GCG gradients). Expected outcome: non-transfer (Δ≈0 on API canaries), closing the architecture-matched white-box attack vector.