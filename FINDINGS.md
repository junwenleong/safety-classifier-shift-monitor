# Findings

## Summary

Reasoning models deployed as safety monitors suffer **reasoning-token budget starvation**: on o3, GCG-appended harmful inputs needed a 2.5x larger completion-token cap than simple benign inputs to produce valid safety scores (T50_adv=117 vs T50_benign=46; harmfulness, length and suffix are not separated), so standard low-budget deployments silently starve the monitor on exactly the inputs it must catch. This compounds a second threat we characterize: gradient-based evasion against deployed safety classifiers. Score-disagreement monitoring detects much of it (75.5% of DeBERTa-evading GCG inputs, 37/49), but not all. The stall of a monitor-aware attacker near gap=1/(2*lambda) is a local stationary-score calculation for one fixed attack objective, with the final gaps clustering near it at high lambda (location replicates among blocked runs; the blocked frequency of 70% / 30% / 50% does not). It is not a general security boundary.

An online monitoring system detects distributional shift in deployed safety classifiers with 86.8% control-validated detection rate across 800 pre-registered factorial cells (4 classifiers x 5 shift conditions x 20 seeds x 2 window sizes). A scan martingale gave FAR<=1% on the null streams we tested for all classifiers without per-model calibration; this is an empirical result, not a proven anytime guarantee. Weighted conformal prediction recovers coverage for discriminative classifiers (+16 pp for DeBERTa) but fails for generative classifiers due to density-ratio collapse in high-dimensional embedding spaces; a PCA diagnostic (projection to <=32 dimensions) breaks the separability and restores coverage on temporal shift (+33pp Llama Guard, +20.5pp ShieldGemma), consistent with a dimensionality mechanism.

## The Problem

Safety classifiers degrade silently under distributional shift. When the input distribution changes (through adversarial adaptation, linguistic drift, multilingual code-switching, or emerging attack patterns), classifier accuracy drops with no error signal. In production, ground-truth labels typically do not arrive in real time. The monitor watches only the classifier's own outputs (scores and embeddings) and alerts deployers before the shift accumulates further.

## System Design

The monitor observes a stream of classifier outputs: unsafe-class probability and penultimate-layer representation. A sliding-window KS statistic tracks whether the score distribution has changed relative to a frozen reference CDF. Alarm thresholds are calibrated empirically via 50 negative control streams (no shift injected), set at the 97th percentile of maximum observed statistics. Upon alarm, a conformal abstention layer adapts decision thresholds using density-ratio reweighting (Tibshirani et al. 2019) to preserve a target 90% coverage rate without new labels.

## Classifiers

| Classifier | Architecture | Parameters | Embedding dim |
|---|---|---|---|
| DeBERTa-v3-base (fine-tuned) | Transformer encoder | 86M | 768 |
| Text-Moderation (KoalaAI) | DeBERTa-v3-base | 86M | 768 |
| Llama Guard 3 | Decoder-only LLM | 8B | 4096 |
| ShieldGemma | Decoder-only LLM | 9B | 3584 |

## Shift Conditions

| Condition | Mechanism | Threat model |
|---|---|---|
| Paraphrase | GPT-4o semantic rewording | Organic rephrasing |
| Code-switch | Singlish transliteration | Non-English users |
| Compositional | Harmful content in long-context wrappers (after the harness's 1,000-character truncation, 200 of 300 examples hold only benign wrapper text, so cells measure the wrapper) | Context-window attacks |
| Temporal | Real jailbreaks from public red-team databases | Emerging harm categories |
| Adversarial suffix | GCG-optimized tokens | Automated red-teaming |

## Results

### RQ1: Detection Performance

Detection rate: 694/800 cells = 86.8% (95% Wilson CI [84.2%, 88.9%]). Values are from the regeneration with the corrected KS detector.

**Mean detection latency (steps):**

| Classifier | Paraphrase | Code-switch | Compositional | Temporal | Adversarial |
|---|---|---|---|---|---|
| DeBERTa | 28.4 | 32.2 | 24.5 | 23.5 | 36.5 |
| Text-Mod. | 34.6 | 33.2 | 29.6 | 24.8 | 25.3 |
| Llama Guard | 69.6 | 93.9 | 47.2 | 42.4 | 27.9 |
| ShieldGemma | 85.6 | 83.3 | 43.9 | 27.6 | 27.7 |

A crossover interaction is visible: paraphrase is easy for encoders but hard for decoders; adversarial suffix shows the opposite pattern. This motivates RQ3.

**False alarm rates (95% Wilson CIs):** Text-Moderation 2.0% [0.8%, 5.0%] < Llama Guard 3.0% [1.4%, 6.4%] < ShieldGemma 8.0% [5.0%, 12.6%] < DeBERTa 9.5% [6.2%, 14.4%].

### RQ2: Conformal Adaptation

**Coverage under temporal shift (95% Clopper-Pearson CIs, n=200 per condition):**

| Classifier | Mode | Pre-shift | Post-shift | Gap | Abstentions |
|---|---|---|---|---|---|
| DeBERTa | Unweighted | 0.910 [.861, .946] | 0.825 [.765, .878] | 0.085 | 25 |
| DeBERTa | Weighted-on-alarm | 1.000 [.982, 1.00] | 0.985 [.957, .997] | 0.015 | 29 |
| ShieldGemma | Unweighted | 0.915 [.867, .950] | 0.690 [.621, .753] | 0.225 | 47 |
| ShieldGemma | Weighted-on-alarm | 0.925 [.879, .957] | 0.750 [.684, .808] | 0.175 | 31 |
| Llama Guard 3 | Unweighted | 0.890 [.838, .930] | 0.555 [.483, .625] | 0.335 | 41 |
| Llama Guard 3 | Weighted-on-alarm | 0.915 [.867, .950] | 0.570 [.498, .640] | 0.345 | 32 |

DeBERTa's recovery is statistically significant (weighted lower bound 0.957 > unweighted upper bound 0.892, non-overlapping CIs).

**The density ratio collapse mechanism:** For both generative models, logistic regression achieves perfect separability between source and target embeddings. All 300 calibration weights clip to the floor (1/C = 0.1), eliminating data-driven reweighting. Residual recoveries (+0.02-0.10) at ESS~300 are a mechanical artifact of the test-point contribution: the implicit weight w(X_test)=1.0 raises the effective quantile level from 90.3% to 93.3% at n_cal=300, epsilon=0.1: a formula artifact, not adaptation. For DeBERTa, 92% of weights clip to floor but ~24 calibration examples retain non-trivial weights (max 3.02, ESS = 88/300), driving the genuine 16 pp recovery under temporal shift. (Numbers from the pooled `conformal_full.json` evaluation.)

### RQ3: Variance Decomposition

Two-way ANOVA on detection latency (694 valid detections):

| Factor | eta-squared | 95% CI | Permutation p |
|---|---|---|---|
| Classifier | 0.248 | [0.210, 0.297] | < 0.001 |
| Shift type | 0.239 | [0.196, 0.296] | < 0.001 |
| Classifier x Shift | 0.186 | [0.166, 0.221] | < 0.001 |
| Residual | 0.328 | n/a | n/a |

The three systematic factors contribute roughly equally.

### Robustness Across Ground-Truth Regimes

- **Regime A (Synthetic onset):** 86.8% detection rate (800 cells)
- **Regime B (Real jailbreak examples in a synthetic-onset stream):** 85% detection rate (17/20 cells, mean latency 32.6 steps). The stream is built with a fixed onset and the temporal corpus used in Regime A; it is not a chronological temporal split.
- **Regime C (GCG adversarial):** Cross-classifier anomaly detection. DeBERTa (target): adversarial suffixes push scores toward safe, monitor fails (38/40). Llama Guard (non-target): same suffixes push scores toward unsafe (+0.73), monitor detects in 14/40. Score direction diagnostics computed at evaluation time via `scripts/check_regime_c_direction.py`; detection counts verified by `verify_paper_numbers.py`. Adversarial perturbations optimized against one classifier appear anomalous to architecturally different classifiers.

## Frontier LLMs as Semantic Canaries

35 models were screened through an OpenAI-compatible frontier model service; the detailed analyses cover 20. Score disagreement between a primary classifier and an LLM canary detects GCG evasion (p<10^-12, n=49). The detection is target-specific; architectural diversity gave no clear advantage in this four-model ensemble (eta-squared=0.011, p=0.070), which is weaker than showing architecture is irrelevant. A k=2 ensemble detects 98% (48/49) at unadjusted per-canary thresholds. The 1.73x tokeniser fragmentation ratio was measured on random DeBERTa token sequences; it is a plausible explanation for the lack of cross-family transfer, not a demonstrated barrier.

### The Deployment Configuration Trap

The apparent "ceiling-clipping" phenomenon (flagship/reasoning models scoring 1.0 on all inputs) is largely a deployment configuration artefact: reasoning models need a larger completion-token cap for their internal chain-of-thought, and at the standard max_tokens=16 they produce empty responses that the scoring code parsed as maximum-unsafe. Of the 31 models in the Pareto file, 19 give graded scores on both benign and adversarial prompts and 12 were ceiling-clipped at the 16-token screen. A re-check shows the clipped group mixes behaviours: some models return graded scores once the cap is adequate, others return no score for most adversarial prompts, gpt-5-nano returned no score for 19 of 20 benign prompts, and gpt-5.5 scored every input near 1.0. The paper therefore does not report a 33/2 split of the 35 models (see its model-taxonomy table).

### Detection Mechanism

- **Semantic mechanism:** Detection operates on base-prompt intent, ignoring suffix tokens (BF01>3 for 16/20 models, TOST +/-0.15 for 16/20). Mean equivalence does not mean per-prompt invariance: 59/980 prompt-model pairs (6.0%) cross the 0.5 boundary.
- **Ghost suffix:** GCG suffixes inert on benign content (delta=+0.005 for gpt-5.1)
- **Prefix mutation:** not established. The paraphrasing model refused to rephrase 8 of 20 prompts; on the 12 real rephrasings gpt-5.1 scores 11/12 for the original and 11/12 for the mutated prompt. The sample is small and the excluded prompts are those the paraphraser declined.
- **N=1000 FPR:** gpt-4.1 and claude-haiku-4-5 achieve 0/1000 (CI <= 0.37%)
- **Newer-model re-run (claude-sonnet-5, gemini-3.6/3.7-flash):** withdrawn. The update script scored the same truncated original prompt in both the adversarial and clean arms, so its detection and "FPR" figures are not a benign-FPR measurement. They are not cited.

### Pareto Frontier and Cost-Scaling

Pareto frontier (Wilson CI): gpt-4o-mini >=71% at $0.033/1k calls, gpt-5.1 >=83.5% at $0.30/1k calls. The earlier cost-detection correlation (r=0.51) mixed mean scores with detection rates for some models and is being recomputed; see the paper for the current value.

**Cost-bounded safety routing (illustrative).** The Pareto frontier admits an escalation router: screen all inputs with gpt-4o-mini ($0.033/1k); escalate ambiguous (score in [0.3, 0.7]) and non-English inputs to gpt-5.1 ($0.30/1k). At an illustrative ~12% escalation rate: expected cost $69/1M queries ($33 + 0.12 x $300, since every query is screened), 84.9% detection, compared with $300/1M for always-gpt-5.1 (93.9%) and $33/1M for always-gpt-4o-mini (83.7%). The escalation rate depends on production traffic, which we did not measure.

## Adversarial Robustness

A 4-tier threat model organises the attacker's knowledge and access. It is a descriptive framework, not a proven security boundary.

### Stall Condition

For the fixed objective L = f_A + lambda (f_B - f_A)^2, a canary score held fixed gives a stationary target score at gap 1/(2 lambda) (0.250 at lambda=2). This is a local stationary-score calculation specific to that objective and to the search procedure used (candidates are proposed from the target model's gradient and ranked by the combined objective). Empirically, 14 of 20 prompts were blocked at lambda=2 (mean final gap 0.218, median 0.250). The gap location replicates among blocked runs at high lambda; the blocked frequency does not (70% on the primary checkpoint pair, 30% on a replication pair, 50% in the lambda sweep), and the data do not identify what determines it. The paper withdraws the earlier claim that the stall is confidence-gated: saturated canaries appear among both stealth and blocked cases.

### Transfer Properties

- Passive transfer, recomputed on eligible observations (canary originally scoring >=0.5): the 40 observations are 10 prompts x 4 epoch-checkpoint canaries. 14 end below 0.5, but 12 of those were already below 0.5, so genuine unsafe-to-safe crossings are 1/23 (canary original >=0.99) and 1/5 (below 0.99), 2/28 overall. The earlier 4.3% vs 76.5% contrast is withdrawn as evidence of a confidence gate.
- Joint GCG optimisation (same-family canary, n=20): 11/20 prompts end with both scores below 0.5, but 5 of those started below 0.5. Among the 13 prompts where both models initially scored the input unsafe, 6/13 (46.2%) reach joint evasion. The passive 3/10 includes 2 inputs already below 0.5 for both models; among 6 eligible it is 1/6. The two runs differ in prompts and size and are not compared with a test.
- Architectural diversity was not necessary for strong disagreement in this four-model ensemble (eta-squared=0.011, p=0.070); the causal determinants of transfer resistance are unresolved.

### Black-Box Hard Floor

Extended coordinate ascent (500 steps, 10 runs on at most 9 distinct prompts, baseline >=0.8, upper tail of the 49-prompt corpus) shows a saturating fitness landscape: 9/10 prompts show zero improvement after step 50; 1/10 shows one additional reduction (still well above 0.5). Mean final score: 0.90. No prompt breaches 0.5. No fracture point within the tested 500-step budget; increasing attacker budget provided no advantage across these runs. With n=10 and zero breaches, the exact two-sided upper bound on the breach rate is about 31%, so this is not strong robustness evidence.

### White-Box Transfer

Llama Guard 3 surrogate GCG transfer (n=10): white-box suffixes optimised against Llama Guard 3 (8B decoder) produce delta=0.000 on gpt-4o-mini and delta=-0.010 on gpt-5.1. Even architecture-matched decoder-to-decoder suffixes are inert on frontier canaries, consistent with a cross-family transfer barrier. Llama Guard 3 did not initially score these prompts as unsafe (mean 0.057, 0/10 above 0.5), so this is a check that the suffixes are inert on frontier canaries, not a test of successful decoder evasion.

## Scan Martingale

A conformal test martingale replaces empirical KS calibration. On the null streams tested, FAR was <=1% for all 4 classifiers (vs 2-9.5% under empirical KS), with no per-model calibration. This is an empirical result: conformal p-values computed against one frozen reference are only marginally valid, and the scan keeps more start points over a long deployment than its active-window cap, so there is no proven anytime or infinite-horizon false-alarm bound for this implementation.

The value is operational simplicity: shared parameters across classifiers rather than per-model tuning. For ramped-onset adversaries (gradual contamination injection over 50 steps, DeBERTa x paraphrase, 30 streams per cell), scan detects 60% / 93% / 100% / 100% of streams at 15 / 20 / 25 / 30% mixing against 33-57% / 67-87% / 93-100% / 100% for KS, depending on how the KS threshold is calibrated (AD1 table in the paper). The earlier figures (37% vs 3% at 15% mixing) could not be reproduced and are withdrawn. For instantaneous step-onset shift with a per-condition calibrated KS threshold (97th percentile), KS matches or exceeds scan detection power.

## Post-Factorial Results

### Detection Channel Comparison

**CS growing-window:** 120/120 detection (100%) with 0/40 false alarms, an empirical count only (its 95% Clopper-Pearson upper bound is 8.8%). The reference mean is estimated in-sample from the same null scores, which erodes the time-uniform guarantee. Counts are from a re-run under the corrected predictable-betting update.

**CS versus KS at low mixing: the earlier advantage is withdrawn.** The earlier comparison gave KS a reference frozen from the first 100 scores and CS the mean of 500 null scores. With a KS reference built from the same 500 scores, KS detects 10/10 at 30% mixing (30/30 on the extended test) and has the shorter mean latency at every mixing level (for example 71.5 vs 122.6 steps for CS at 30%). Given equal reference information, KS is at least as sensitive as the growing-window betting test on these streams. This is one classifier x shift pair.

**MMD on embeddings:** 120/120 shifted streams detected at the first possible evaluation (latency=100, the window size). FAR on null windows: DeBERTa 3.3%, Text-Mod 3.3%, ShieldGemma 0%, Llama Guard 10% (30 null windows). MMD is evaluated per full window, so this is a batch-window protocol and its FAR is not comparable with KS's stream-level FAR.

### Gradual Drift Sensitivity

Ramp-rate sweep (DeBERTa x paraphrase, cached scores, n=10), with KS using a 100-score reference, the weaker configuration discussed above:
- 50-step ramp: KS 10/10 (mean 94 steps), CS 10/10 (mean 84)
- 200-step ramp: KS 9/10 (mean 210), CS 10/10 (mean 155)

Mixing-level sweep (50-step ramp, 100-score KS reference): at 30% mixing KS detects 4/10 and CS 10/10; at 50% and above both detect 10/10. These KS figures are those of the weaker configuration; the equal-reference rerun above reverses the low-mixing ordering.

### Mechanistic Hypothesis (Exploratory, n=4)

The n=4 correlation (r=0.97, p=0.032) between null score std and detection latency is an artefact of the encoder/decoder architectural gap, not an intrinsic monitorability property. Within-family evaluation (6 encoder variants at different training epochs) yields r=0.21, p=0.70, which gives no support for monitorability as a predictable property of score-distribution geometry (n=6, correlated checkpoints). The pattern is shift-specific: paraphrase/temporal/compositional show wider-to-slower (r=0.70-0.97); adversarial suffix reverses (r=-0.20), producing the crossover.

Embedding displacement does NOT mirror this pattern (overall r=-0.09, p=0.78). Detection is mediated by score-boundary geometry, not representation-space distance.

### Filtered Paraphrase Ablation

Refusal rate: 47/500 = 9.4% (lower than 14-20% manual estimate). Removing refusals has negligible effect: DeBERTa 38.0->37.8 steps, Llama Guard 66.6->60.8 steps. Both 5/5 detected in both conditions.

### PCA Diagnostic

ESS reduction at dim=32 generalizes to paraphrase shift: Llama Guard ESS=32, ShieldGemma ESS=28 (both breaking separability). This confirms the collapse is driven by the curse of dimensionality (d >> n), consistent with established high-dimensional density-ratio instability (Stojanov et al. 2019; Sugiyama et al. 2011); the specific contribution is diagnosing the failure mode in generative safety-classifier embeddings. Coverage recovery magnitude is split-dependent; primary result (temporal: +33pp Llama Guard, +20.5pp ShieldGemma) uses fresh inference with proper conformal framework.

## Depth Results

**CoT Suffocation phase transition.** Response probability follows a sigmoid: P(response|T_r) = sigma(k*(T_r - T_50)). For o3: T_50(benign)=46 tokens (k=0.173), T_50(adversarial)=117 tokens (k=0.047), measured on an extended 10-300 token grid. The 2.5x gap means standard service configs (max_tokens=16) cause total failure on adversarial inputs while benign queries occasionally succeed. Deploy with T_r >= T_50(hardest class) + 2.2/k for >90% coverage.

**Temperature sensitivity.** Detection rate varies by <5pp across T in {0, 0.3, 1.0} (5 models x 20 adv + 20 benign, 3 reps per non-zero T). Within-prompt SD at T=1.0: 0.03-0.11. T=0 recommended for reproducibility; temperature does not materially affect detection.

**Dual-channel cross-lingual mechanism.** The 15-25% cross-lingual degradation is asymmetric: explicit-harm prompts (direct violence, slurs) degrade 19-29pp, while ambiguous/roleplay prompts degrade only 10-15pp. Two safety channels: (i) lexical keyword matching (language-specific, fails under translation), (ii) structural intent recognition (language-invariant, transfers natively). For gpt-5.1: 84% of detections are structural (hold cross-lingually), 16% are lexical (leak when translated). Multilingual suffix transfer: mean delta = +0.022 (suffix completely inert across languages).

**Cross-lingual N=49 (all three languages, 3 canary models).** Confirmed at N=44-49 per language:
- Spanish (gpt-4o-mini): 63.3% detection (31/49) vs 83.7% English (41/49). Drop: 20.4pp. Wilson CI [49.3%, 75.3%].
- Mandarin (claude-sonnet-4-5): 61.4% detection (27/44) vs 81.1% English (36/44). Drop: 19.7pp.
- Arabic (claude-sonnet-4-5): 73.3% detection (33/45) vs 81.6% English (37/45). Drop: 8.3pp.
Arabic shows notably less cross-lingual degradation than Spanish/Mandarin, consistent with structural-intent recognition dominating over lexical matching for Arabic script (fewer shared subwords with English training corpus). All CIs now +/-13pp or tighter (previously +/-20pp at N=20).

**Monitorability law not supported (Track C).** The n=4 correlation (r=0.97) between null-score std and detection latency was an encoder/decoder gap artefact. Within-family evaluation (n=6 encoder variants, epoch-{1,3,5,10} + 2 originals) yields r=0.21, p=0.70. We find no support for monitorability as a predictable intrinsic property of score-distribution geometry (n=6, correlated checkpoints; the result rests on the absence of a trend).

## Corpus Validation

Manual review of samples from each corpus:

- **Paraphrase (50/500):** ~18-22 preserved harmful intent; 14-20% became LLM refusals (safety responses instead of paraphrases). Detection latencies for paraphrase should be interpreted conservatively.
- **Code-switch (50/500):** All 50 confirmed as authentic Singlish by native speaker. 20-30% became refusals; same caveat applies.
- **Compositional (20/300):** 20/20 correctly placed harmful content at stated position. 100% structural integrity.
- **Temporal (20/292):** 20/20 reviewed examples were genuine jailbreak prompts; zero false positives. Full corpus draws from three public red-team databases: lmsys/toxic-chat (39%), JailbreakBench (34%), ChatGPT-Jailbreak-Prompts (27%). Subsampled to 300 per factorial cell via repetition of 8 examples.
- **Adversarial suffix (20/22):** 20/22 correct suffix concatenation with confirmed score flips (orig >=0.95, attacked <=0.01). One example excluded post-validation (original score 0.002, already benign).

## Limitations

- **Gradual drift detection boundary.** With a 100-score reference, KS detects 4/10 ramped streams at 30% mixing. With equal reference information KS detects 10/10 and is faster than the growing-window CS, so no low-mixing advantage for CS is claimed. Below roughly 12% mixing the sliding-window detectors are blind; the sub-threshold experiment (synthetic shifted scores with a known onset, DeBERTa only, exploratory) shows CUSUM and EWMA detecting down to 4% mixing at a latency cost.
- **MMD provides no latency gradation.** It evaluates full shifted windows, so it reports latency=100 on any shift; it is a batch-window check, not directly comparable with the stream-level KS protocol.
- **Residual variance.** 33.5% of latency variance is noise. MDE is 13.9 steps at 80% power.
- **Binary classifiers only.** Multi-category safety taxonomies may exhibit category-specific shift invisible to scalar scores.
- **Refusal contamination.** 9.4% of paraphrase corpus are LLM refusals (lower than the 14-20% manual estimate). Filtered ablation confirms negligible effect on detection.
- **FAR asymmetry.** Empirical KS false alarm rates vary 5x across classifiers (Text-Moderation 2.0% vs DeBERTa 9.5%). The scan martingale gave FAR<=1% on the null streams we tested with no per-classifier calibration, without a proven long-horizon bound.
- **PCA diagnostic validated on temporal + paraphrase.** ESS reduction generalizes but coverage recovery magnitude depends on calibration split.
- **Conformal setup.** The coverage experiments use safe-only calibration data and treat every shifted example as unsafe, with a fixed test-point weight of 1.0 and no class-prior correction in the density-ratio estimate. They diagnose density-ratio collapse; they are not a clean covariate-shift coverage guarantee.
- **Invalid factorial cells.** The 106 invalid cells are early alarms (68) and dirty negative controls (45), 7 both; none failed to alarm. They are specificity failures, so constant right-censoring imputation (which gives interaction eta-squared 0.02-0.03) is a stress test, not an estimate of the interaction.
- **Known limitations of this repository.** The shifted corpora under `data/shifted/` are git-ignored in the public repository, and encoder checkpoints are not committed. `smsr_connection.tex` is a withdrawn appendix and is not part of the paper.

## Reproducibility Notes

Some results are not reproducible from this repository's committed artefacts.

**Canary detection with a locally run open-weight canary.** An earlier version reported 40 of 49 (81.6%) DeBERTa-evading inputs flagged by a Qwen2.5-14B canary. This repository's case-study script (`scripts/run_primary_fooled_case.py`) uses a different, API-based canary, and its committed results file holds no valid canary scores (every canary call in that run failed; the failures had been recorded as valid negative results before an audit-pass fix). No script, checkpoint or result file for a Qwen2.5-14B evaluation exists here, and the paper no longer reports the figure. The reproducible primary-fooled evidence is the 37/49 (75.5%) disagreement detection in `results/gate_a_divergence.json`.

**Newer frontier models as canaries.** The post-hoc re-run of models released after the primary evaluation (`scripts/exp_pareto_update.py`, `scripts/run_llm_canary_v2.py`) is withdrawn. The update script loads each prompt from the `orig` field, which `run_llm_canary_v2.py` stores as the truncated original prompt in every arm, so the adversarial and clean conditions were scored on the same text and the resulting detection and false-positive figures do not measure benign false positives. A rerun needs the full GCG inputs and a genuinely benign corpus.

The primary 800-cell factorial evaluation is covered by `scripts/verify_paper_numbers.py` and by the regeneration in the paper's appendix. Not every experiment is re-runnable: the May adversarial-suffix corpus was not retained, and some scripts need API access or the encoder checkpoints, which are not committed.

## Verification

`scripts/verify_paper_numbers.py` checks 225 selected reported values against committed result files (all passing); it does not re-run experiments or validate methodology, and some checks read stored summaries rather than per-example data. Some figures in this document (for example the cost-routing numbers) are illustrative and are not covered by it. Experiment configurations in `configs/` were committed before execution (commit `0380167`; see `HISTORY_REWRITE.md` for the mapping from earlier identifiers). Post-factorial additions pre-registered in `docs/pre_registration_amendment_2.md` (committed June 8, executed June 9-10).

Paper: [arXiv](https://arxiv.org/abs/2606.11949) · Code and results: [github.com/junwenleong/safety-classifier-shift-monitor](https://github.com/junwenleong/safety-classifier-shift-monitor)
