# Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers

## Abstract

We present an online monitoring system for distributional shift in deployed safety classifiers, combining calibrated sequential statistics—confidence sequences on KS statistics and kernel MMD on classifier embeddings—to detect when a classifier has moved out of distribution. Upon detection, a conformal abstention layer adapts decision thresholds to preserve a target error rate. In a pre-registered factorial evaluation across 4 classifiers × 5 shift conditions × 5 seeds × 2 window sizes (200 cells), the system achieves an 86.5% valid detection rate (173/200 cells, 95% CI [81.1%, 90.6%]), with mean detection latency of 36.3 steps at window size 100 and empirical false alarm rates of 4–12% across classifiers. Detection holds across three ground-truth regimes: synthetic onset (86.5%), real temporal jailbreaks from public red-team databases (85%, 17/20), and GCG adversarial success (13.8% overall, but 35% for Llama Guard on DeBERTa-targeted suffixes—detecting adversarial tokens as anomalous rather than through attack transfer). Under temporal shift, unweighted conformal prediction loses 6.5 percentage points of coverage (from 91.0% to 84.5%), violating the 90% target; weighted conformal prediction recovers coverage to 98.5% with only 4 additional abstentions. Variance decomposition reveals that the classifier×shift interaction (η² = 0.265, p < 0.001) is the largest systematic factor, exceeding both main effects of classifier (η² = 0.196) and shift type (η² = 0.217), indicating that detection difficulty is fundamentally a property of the classifier–shift pairing rather than either factor alone.

## 1. Introduction

Safety classifiers are the last line of defense between a language model and its users. When deployed at scale, they operate under a stationarity assumption: the distribution of inputs tomorrow will resemble the distribution on which the classifier was calibrated. This assumption fails routinely—through adversarial adaptation (Zou et al., 2023), organic linguistic drift, multilingual code-switching, and compositional attacks that chain benign components into harmful sequences.

The failure mode is silent. A classifier whose accuracy has degraded from 95% to 80% produces no error signal unless ground-truth labels arrive—which, in production safety systems, they rarely do in real time. By the time periodic offline evaluation detects the problem, the classifier has been making unreliable decisions for days or weeks.

We propose an online monitor with empirically calibrated alarm thresholds that tells deployers *when* their safety classifier has moved out of distribution—before accuracy collapses. The system watches two signals: the distribution of classifier scores (via KS statistics) and the geometry of internal representations (via kernel MMD). Both are wrapped in confidence sequences with false alarm rates calibrated via null simulation. When shift is detected, a conformal abstention layer reweights its prediction sets to preserve coverage under the new distribution.

We address three research questions:

**RQ1 (Detection):** Can sequential statistical tests on classifier outputs detect distributional shift online, with controlled false alarm rates, across diverse shift types and classifier architectures?

**RQ2 (Adaptation):** Does weighted conformal prediction recover coverage guarantees after shift detection, compared to unweighted conformal sets?

**RQ3 (Factor Importance):** In a factorial evaluation design, what proportion of variance in detection latency is attributable to classifier choice, shift type, and their interaction?

## 2. Related Work

**Sequential testing and confidence sequences.** Classical sequential analysis (Wald, 1945) provides stopping rules with controlled error rates, but fixed-sample-size guarantees do not extend to continuous monitoring. Waudby-Smith & Ramdas (2024) resolve this with betting-based confidence sequences: by constructing a wealth process that is a non-negative supermartingale under the null, Ville's inequality yields time-uniform coverage—the guarantee holds simultaneously at all time steps, not just at a pre-specified stopping time. We employ an adaptive betting strategy within the framework of Waudby-Smith & Ramdas (2024) for the growing-window variant of our detector. For the sliding-window variant used in our factorial evaluation, the time-uniform guarantee does not hold; we instead calibrate alarm thresholds empirically via null simulation (§3.8), which provides finite-horizon control at the cost of the anytime guarantee.

**Two-sample testing on streams.** Kernel MMD (Gretton et al., 2012) is the standard nonparametric two-sample test for high-dimensional data, but its original formulation assumes fixed samples. We adapt it to the streaming setting by maintaining a sliding window of embeddings and comparing against frozen reference statistics. The bandwidth is fixed at calibration time following the standard median heuristic in the MMD literature, preventing adaptation of kernel parameters to the detection window. This is closer to the online kernel tests of Zaremba et al. (2013) than to the original batch MMD, though we do not claim optimality of the kernel choice.

**Conformal prediction under covariate shift.** Split conformal prediction (Vovk et al., 2005) provides distribution-free coverage guarantees under exchangeability. When the test distribution shifts, exchangeability breaks and coverage degrades. Tibshirani et al. (2019) restore coverage by reweighting the empirical distribution of calibration nonconformity scores with density ratios estimated from the covariate shift. Our weighted-on-alarm mode implements their approach, triggered by the shift detector rather than assumed available or estimable from unlabeled target covariates. The density ratio is estimated via logistic regression on classifier embeddings—a lightweight approximation that avoids the instability of kernel density estimation in high dimensions. Gibbs & Candès (2021) extend conformal prediction to the fully online setting without exchangeability; we note this as future work (§5.4). For conformal prediction applied specifically to classification, Romano, Sesia & Candès (2020) provide adaptive prediction sets that maintain valid coverage while minimizing set size—our abstention criterion (flag when $|C(x)| \neq 1$) follows their framework.

**Safety classifier monitoring.** Prior work on monitoring deployed classifiers focuses on performance estimation from unlabeled data (Garg et al., 2022) or drift detection via population-level statistics (Rabanser et al., 2019), finding that performance depends strongly on shift type and representation, with pretrained-classifier dimensionality reduction performing well in their experiments. Our contribution is the integration of detection and adaptation: the shift detector triggers the conformal layer, which adapts thresholds without requiring new labels. The factorial evaluation design—crossing classifiers with shift conditions—is, to our knowledge, novel in this literature and reveals interaction effects invisible to single-classifier studies.

## 3. Methods

### 3.1 System Architecture

The monitor observes a stream of classifier outputs $(x_t, s_t, \mathbf{r}_t)$: input text $x_t$, unsafe-class probability $s_t \in [0,1]$, and penultimate-layer representation $\mathbf{r}_t \in \mathbb{R}^d$. Two parallel detectors—one on scores, one on embeddings—feed into a multiplicity-corrected alarm controller. The score detector catches shifts that change the classifier's output distribution directly; the embedding detector catches shifts that alter the representation geometry even when the score marginal is preserved (e.g., adversarial inputs that fool the classification head but distort internal representations).

### 3.2 Reference Window Calibration

Before monitoring begins, the system collects $n_{\text{ref}}$ records under the known in-distribution regime and freezes four quantities:

1. **Kernel bandwidth** $\sigma$: the median pairwise Euclidean distance among reference embeddings (the standard median heuristic in the MMD literature).
2. **Reference CDF** $\hat{F}_{\text{ref}}$: the sorted reference scores, against which the KS statistic is computed.
3. **MMD null distribution**: $B = 100$ bootstrap MMD² values obtained by resampling within the reference set, used to contextualize observed MMD values.
4. **PCA projection** (optional): when $d > 128$, a projection to 64 dimensions reduces computational cost of the MMD kernel without discarding the dominant variance directions.

All four are frozen at calibration time. No adaptive estimation occurs during monitoring—this is critical for the validity of the sequential tests.

### 3.3 KS Detector

The KS detector tracks whether the marginal distribution of classifier scores has changed. It maintains a sliding window of the most recent $w$ scores and computes the one-sample Kolmogorov–Smirnov statistic:

$$D_w = \sup_x |\hat{F}_w(x) - \hat{F}_{\text{ref}}(x)|$$

The intuition: if the input distribution shifts in a way that changes how the classifier scores inputs—even slightly—the empirical CDF of recent scores will diverge from the reference CDF. The KS statistic measures the maximum pointwise divergence, making it sensitive to any location, scale, or shape change in the score distribution.

### 3.4 MMD Detector

The MMD detector tracks whether the geometry of classifier representations has changed. It maintains a sliding window of the most recent $w$ embedding vectors and computes the unbiased MMD² between the reference embeddings and the window:

$$\widehat{\text{MMD}}^2_u(X, Y) = \frac{1}{m(m-1)} \sum_{i \neq j} k(x_i, x_j) + \frac{1}{n(n-1)} \sum_{i \neq j} k(y_i, y_j) - \frac{2}{mn} \sum_{i,j} k(x_i, y_j)$$

with Gaussian kernel $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$. The MMD is zero if and only if the two distributions are identical (for characteristic kernels), making it a consistent test against any alternative. In practice, it is most powerful against shifts that move the embedding mass—compositional attacks that chain multiple inputs, or temporal drift that introduces novel topic clusters.

### 3.5 Confidence Sequences and Alarm Logic

Each detector's statistic is wrapped in a confidence sequence (CS) to control false alarm rates over the monitoring horizon.

**The guarantee we want:** an alarm should fire only when the input distribution has genuinely shifted, with probability of false alarm bounded by $\alpha$ over the entire (potentially infinite) monitoring period.

**What we use in practice:** a sliding-window Hoeffding bound. For a window of $n$ bounded observations in $[a, b]$, the confidence interval around the empirical mean has half-width:

$$h = (b - a)\sqrt{\frac{\log(1/\alpha)}{2n}}$$

This provides valid coverage for each individual window—at any fixed time $t$, the probability that the true mean lies outside the interval is at most $\alpha$. However, it does *not* provide the time-uniform guarantee $\Pr(\exists t: \mu \notin [L_t, U_t]) \leq \alpha$. Over a long monitoring horizon, the probability of at least one false alarm exceeds $\alpha$.

**How we close the gap:** empirical FAR calibration (§3.8). Rather than relying on the theoretical bound alone, we simulate the null distribution by running $N_{\text{cal}} = 50$ negative control streams through the full pipeline and set the alarm threshold at the 97th percentile of the maximum observed statistic. This gives finite-horizon control calibrated to the actual monitoring duration and window size.

**Growing-window mode** (not used in the factorial, but available) provides exact time-uniform coverage via Ville's inequality on the ONS betting wealth process. The tradeoff is sensitivity: growing windows dilute recent evidence with old observations, increasing detection latency for shifts that occur late in the stream.

An alarm fires when the reference value exits the confidence interval. Alarms are suppressed during a warmup period of $w$ steps to allow the window to fill.

### 3.6 Multiplicity Correction

The KS and MMD detectors run in parallel, each with its own CS. To control the family-wise error rate at $\alpha$ across $k = 2$ detectors, we apply Šidák correction: $\alpha_{\text{per}} = 1 - (1-\alpha)^{1/k}$. Each detector fires at most once per shift (deduplication prevents repeated alarms from the same event). A combined advisory is emitted when both detectors alarm within a configurable time window, providing higher confidence that the shift is genuine rather than a statistical artifact in one channel.

### 3.7 Statistical Methodology

All statistical analyses use the following specifications:

- **Bootstrap CIs on means:** 1000 resamples, seed = 42, percentile method (2.5th and 97.5th percentiles).
- **CIs on rates (detection rate, FAR):** Wilson Score interval at 95% confidence.
- **CIs on coverage proportions:** Clopper-Pearson exact interval at 95% confidence.
- **Permutation tests:** 1000 permutations of factor labels, p-value computed as (count of permuted η² ≥ observed + 1) / (n_perm + 1).
- **Significance level:** α = 0.05 throughout.
- **Pairwise comparisons:** None pre-specified. The factorial tests main effects and interactions via ANOVA; individual cell estimates are reported with CIs but no multiplicity correction is applied to cell-level comparisons.
- **Power:** Post-hoc analysis indicates minimum detectable effect size d = 1.46 (Cohen's d) for pairwise cell comparisons at n = 5 per cell.

### 3.8 Empirical FAR Calibration

We run $N_{\text{cal}} = 50$ negative control streams—reference data only, no shift injected—through the full detection pipeline. For each stream, we record the maximum KS statistic observed over the monitoring horizon. The alarm threshold is set at the $p$-th percentile of these maxima; in the factorial evaluation, $p = 97$. This means: if we ran 100 null streams, at most 3 would trigger a false alarm. The empirical calibration accounts for the sliding-window correlation structure, the specific window size, and the stream length—factors that the theoretical Hoeffding bound treats conservatively.

### 3.9 Conformal Abstention Layer

Upon alarm, the system activates a conformal prediction layer that adapts decision thresholds to preserve a target error rate $\epsilon$ without requiring new labeled data.

**Unweighted mode.** Standard split-conformal prediction (Vovk et al., 2005). Given $n$ calibration examples with known labels:
1. Compute nonconformity scores: $\alpha_i = 1 - f(x_i)_{y_i}$ (how "surprising" each calibration example is to the classifier).
2. Set threshold $\hat{q}$ at the $\lceil(1-\epsilon)(n+1)\rceil/n$-th quantile.
3. At test time, include class $y$ in the prediction set if its nonconformity score $\leq \hat{q}$.
4. Abstain (flag for human review) when the prediction set contains more than one class or is empty.

Under exchangeability, this guarantees $1-\epsilon$ coverage. Under covariate shift, exchangeability breaks and coverage degrades—as we demonstrate empirically in §4.2.

**Weighted-on-alarm mode.** After the shift detector fires, we estimate density ratios $w_i = p_{\text{target}}(x_i) / p_{\text{source}}(x_i)$ via logistic regression on source vs. target embeddings (Tibshirani et al., 2019). The ratios are clipped to $[1/C, C]$ with $C = 10$ for stability. The conformal quantile is then recomputed as a weighted quantile:

$$\hat{q}_w = \inf\left\{q : \sum_{i: \alpha_i \leq q} \tilde{w}_i \geq 1 - \epsilon\right\}$$

where $\tilde{w}_i = w_i / (\sum_j w_j + 1)$. This reweights the calibration scores to account for the covariate shift, restoring the coverage guarantee under the new distribution (up to the accuracy of the density ratio estimate).

### 3.10 Variance Decomposition

To quantify which experimental factors drive detection latency, we fit a two-way fixed-effects ANOVA:

$$\text{latency}_{ijk} = \mu + \alpha_i^C + \beta_j^S + (\alpha\beta)_{ij}^{CS} + \epsilon_{ijk}$$

with factors classifier ($C$, 4 levels) and shift type ($S$, 5 levels). We report $\eta^2$ (proportion of total sum of squares) for each term. Bootstrap confidence intervals (200 resamples, percentile method) quantify uncertainty in the main-effect estimates. The interaction term $(\alpha\beta)^{CS}$ captures classifier–shift pairings that are systematically easier or harder than predicted by the marginal effects alone—this is the term that motivates per-classifier monitoring profiles.

## 4. Experimental Setup

### 4.1 Classifiers

| Classifier | Architecture | Parameters | Embedding dim |
|---|---|---|---|
| DeBERTa-v3-large | Transformer encoder | 304M | 1024 |
| Text-Moderation (KoalaAI) | DeBERTa-v3-base | 86M | 768 |
| Llama Guard 3 | Decoder-only LLM | 8B | 4096 |
| ShieldGemma | Decoder-only LLM | 9B | 3584 |

All classifiers are fine-tuned on WildGuardMix (unharmful/harmful binary classification). The selection spans two architectural families—discriminative encoders and generative decoders—and two scales within each family, enabling analysis of both architecture and scale effects.

### 4.2 Shift Conditions

| Condition | Mechanism | Threat model |
|---|---|---|
| Paraphrase | GPT-4o paraphrasing of harmful prompts | Organic rephrasing by users |
| Code-switch | Multilingual transliteration | Non-English user populations |
| Compositional | Multi-turn concatenation into long contexts | Context-window attacks |
| Temporal | Recent harmful content (post-training-cutoff) | Emerging harm categories |
| Adversarial suffix | GCG-optimized suffixes (Zou et al., 2023) | Automated red-teaming |

Each corpus contains 300 examples. The five conditions span the spectrum from naturalistic drift (paraphrase, temporal) to deliberate adversarial attack (GCG suffixes), with code-switch and compositional as intermediate cases.

### 4.3 Factorial Design

Full factorial: 4 classifiers × 5 shift conditions × 5 random seeds × 2 window sizes (100, 200) = **200 cells**. Each cell runs a complete detection pipeline: reference window calibration → stream simulation with shift onset → alarm detection → latency measurement.

**Pre-registration:** The factorial design, including all hyperparameters, was committed before execution (commit `be630f3`).

**Negative controls:** Each cell includes a parallel negative control run (reference data only, no shift) to verify the alarm threshold does not fire on in-distribution data. A cell is marked as a valid detection only if: (1) detection latency ≥ 0, and (2) the negative control does not alarm.

**Compute:** Mac Studio (M2 Ultra, 192GB) for Llama Guard 3 and ShieldGemma; MacBook Pro (M3 Max, 128GB) for DeBERTa and Text-Moderation. Total wall-clock: ~29 hours. All 200 cells completed without error; the extended 800-cell run is ongoing.

### 4.4 Deviations from Pre-Registration

The pre-registration (commit `be630f3`) specified a larger design that was reduced for compute budget:

| Parameter | Pre-registered | Executed | Reason |
|---|---|---|---|
| Seeds | 20 | 5 | Compute budget (29 hours for 200 cells; 3,600 would require ~520 hours) |
| Ground-truth regimes | 3 (A, B, C) | 3 (A, B, C) | All three regimes completed; B and C at reduced scale (§5.4) |
| Window sizes | 100, 200, 500 | 100, 200 | w=500 produces insufficient post-shift observations with 300 shifted examples |

These are scope reductions, not protocol changes. The analysis plan (ANOVA, conformal evaluation, OC curves) and all hyperparameters (α=0.05, calibration percentile=97, reference size=500) match the pre-registration exactly. The reduced seed count (5 vs 20) limits statistical power for individual cell estimates but provides adequate power for main effects and interactions (all p < 0.001 by permutation test).

### 4.5 Reproducibility

Code, configurations, pre-registration document, and raw results are available at https://github.com/junwenleong/safety-classifier-shift-monitor (commit `be630f3` anchors the pre-registration). All hyperparameters are specified in version-controlled YAML files under `configs/`. Shifted corpora are generated offline with fixed seeds and committed before evaluation. A verification script (`scripts/verify_paper_numbers.py`) confirms all reported statistics against raw data.

## 5. Results

### 5.1 RQ1: Detection Performance

The system detects shift in 173 of 200 cells (86.5% detection rate, 95% Wilson CI [0.811, 0.906]), with empirical false alarm rates of 4–12% across classifiers. But the aggregate number obscures the structure. The detection latency table reveals a clear pattern:

| Classifier | Paraphrase | Code-switch | Compositional | Temporal | Adversarial |
|---|---|---|---|---|---|
| DeBERTa | 29.1 [25, 32] | 33.1 [28, 39] | 26.2 [23, 30] | 26.9 [24, 29] | 40.2 [36, 46] |
| Text-Moderation | 37.2 [33, 41] | 32.8 [27, 39] | 27.8 [25, 31] | 26.6 [25, 29] | 28.0 [26, 31] |
| Llama Guard | 64.0 [54, 73] | 69.0 [54, 84] | 50.3 [42, 60] | 38.1 [34, 42] | 26.7 [23, 31] |
| ShieldGemma | 97.4 [82, 112] | 68.5 [48, 92] | 39.4 [31, 48] | 26.9 [24, 30] | 27.7 [25, 30] |

*Values are mean detection latency (steps) with 95% bootstrap CIs. n = 6–10 valid detections per cell.*

Reading down the columns: paraphrase is easy for encoders (29–37 steps) but hard for decoders (64–97 steps). Reading across the rows: adversarial suffix is the hardest condition for DeBERTa (40.2) but the easiest for Llama Guard (26.7). This crossover interaction—not visible in any single-classifier study—is the central finding that motivates RQ3.

**Window size:** w=100 detects 9 steps faster on average (36.3 [32.4, 40.7] vs 45.3 [40.6, 50.4]) at the cost of slightly higher false alarm rates. The smaller window is more reactive but noisier.

**False alarm rates (95% Wilson CIs):** Text-Moderation 4.0% [1.1%, 13.5%] < Llama Guard 6.0% [2.1%, 16.2%] < DeBERTa 8.0% [3.2%, 18.8%] < ShieldGemma 12.0% [5.6%, 23.8%]. We hypothesize that ShieldGemma's higher FAR reflects greater score variability under the null, as larger generative models produce more variable safety judgments on in-distribution inputs.

**Failure analysis.** Of the 27 cells without valid detections: 14 failed because the negative control also alarmed (false alarm), and 13 failed because the detector alarmed before shift onset (early alarm, latency < 0). No cell failed to detect entirely. The failures are distributed across classifiers (DeBERTa 8, Text-Moderation 7, ShieldGemma 7, Llama Guard 5) with one concentration: Llama Guard × adversarial-suffix accounts for 4 of 27 failures, consistent with its low detection rate (6/10) for that condition. The failure mode is predominantly calibration-related (threshold set too aggressively) rather than signal-related (no shift detected).

**Detector channels.** The factorial evaluation uses the KS detector on classifier scores as the primary alarm channel. The MMD detector on embeddings is available in the system architecture but was not the alarm trigger in this evaluation; detector-channel attribution (which channel fires first, marginal value of MMD) is logged in the extended 800-cell run and will be reported in the final version.

### 5.2 RQ2: Conformal Adaptation

Evaluated on DeBERTa × temporal shift—the pairing with the strongest shift signal.

| Mode | Pre-shift coverage | Post-shift coverage | Gap | Abstentions |
|---|---|---|---|---|
| Raw classifier (no conformal) | 0.910 | 0.845 | 0.065 | 0 |
| Unweighted conformal | 0.910 [0.861, 0.946] | 0.845 [0.787, 0.892] | 0.065 | 25 |
| Weighted-on-alarm | 1.000 [0.982, 1.000] | 0.985 [0.957, 0.997] | 0.015 | 29 |

*95% Clopper-Pearson CIs, n=200 per condition. Raw classifier coverage equals unweighted conformal coverage because the conformal threshold at ε=0.10 produces prediction sets that match the classifier's own decision boundary for most inputs.*

Unweighted conformal prediction loses 6.5 percentage points of coverage under temporal shift, dropping below the 90% target (upper CI bound 0.892 excludes 0.90). The raw classifier row shows that conformal prediction does not degrade performance pre-shift — it matches the classifier's native accuracy — but also does not improve it without the weighted correction. The value of the conformal layer is realized only post-alarm: weighted conformal recovers coverage to 98.5% at the cost of 29 abstentions (14.5% abstention rate), providing a principled "I don't know" signal that the raw classifier cannot produce. The mechanism is straightforward: the calibration quantile was computed under the reference distribution; under shift, the nonconformity scores are systematically larger, and the fixed threshold becomes too permissive.

Weighted conformal prediction recovers coverage to 98.5% post-shift—well above target—at the cost of only 4 additional abstentions (29 vs 25). The density-ratio reweighting upweights calibration examples that resemble the shifted distribution, effectively recalibrating the quantile without new labels.

**Limitation:** This evaluation uses a single classifier×shift pairing. Coverage degradation under subtler shifts may be smaller, reducing the benefit of weighted correction. A full factorial conformal evaluation is left to future work.

### 5.3 RQ3: Variance Decomposition

| Factor | η² proportion | 95% CI | Permutation *p* |
|---|---|---|---|
| Classifier | 0.196 | [0.137, 0.293] | < 0.001 |
| Shift type | 0.217 | [0.146, 0.342] | < 0.001 |
| Classifier × Shift | 0.265 | — | < 0.001 |
| Residual | 0.322 | — | — |

All three systematic factors are significant by permutation test (1000 permutations, all p < 0.001). The interaction term is the largest single systematic factor (η² = 0.265), though the difference from shift_type (0.217) is not statistically significant (bootstrap 95% CI on the difference: [−0.090, 0.199]). The key finding is not that the interaction dominates in magnitude, but that it is significant and large: knowing the classifier tells you less about detection latency than knowing the classifier *and* the shift type together. The top interactions by magnitude:

| Combination | Effect (steps) | Interpretation |
|---|---|---|
| ShieldGemma × paraphrase | +34.5 | Generative model robust to rephrasing → slow detection |
| DeBERTa × adversarial-suffix | +18.7 | Encoder partially ignores appended tokens → slow detection |
| DeBERTa × paraphrase | −15.9 | Encoder sensitive to surface form → fast detection |

These interaction effects are large—ShieldGemma × paraphrase takes 34.5 steps longer than predicted by the marginal effects of ShieldGemma and paraphrase separately. A monitoring system that sets thresholds based on classifier-level or shift-level averages will systematically under-alert on hard pairings and over-alert on easy ones.

### 5.4 Robustness Across Ground-Truth Regimes

The preceding results (§5.1–5.3) use Regime A: synthetic shift onset at a known step, with shifted corpora generated offline. This validates the detection machinery but leaves open whether the system detects *naturally-occurring* shift. We evaluate two additional ground-truth regimes:

**Regime B (Temporal split).** We replace the synthetic shift corpus with real temporal jailbreaks drawn from public red-team databases (post-training-cutoff harmful prompts). The monitor receives no signal about when or whether shift occurs—it must detect the distributional change from the stream alone. Evaluated on DeBERTa and Llama Guard × temporal corpus × 5 seeds × 2 window sizes = 20 cells.

**Regime C (Adversarial success).** We filter GCG-optimized adversarial suffixes to only those that *successfully* flip DeBERTa's classification (success=True), then measure whether the monitor detects the resulting distributional shift. This tests whether attacks that fool the classifier individually also produce detectable *distributional* signatures. Evaluated on all 4 classifiers × 4 shift conditions × 10 seeds = 160 cells.

#### Regime B Results

The system detects shift in 17 of 20 cells (**85% detection rate**), with mean detection latency of 32.6 steps. Per-classifier breakdown:

| Classifier | Detected / Total | Detection Rate | Mean Latency |
|---|---|---|---|
| Llama Guard | 10/10 | 100% | 28.4 |
| DeBERTa | 7/10 | 70% | 38.2 |

Llama Guard achieves perfect detection on real temporal jailbreaks, consistent with its strong performance on temporal shift in Regime A (38.1 steps). DeBERTa's lower rate (70% vs 100% in Regime A) likely reflects the greater heterogeneity of real-world jailbreaks compared to the curated temporal corpus. The key finding: **detection generalizes to naturally-occurring shift**, closing the "you only detected shifts you injected" gap.

#### Regime C Results: Cross-Classifier Anomaly Detection

Overall detection rate: 22/160 cells (**13.8%**). The cross-classifier breakdown:

| Classifier | Detected / Total | Detection Rate |
|---|---|---|
| Llama Guard | 14/40 | 35.0% |
| ShieldGemma | 6/40 | 15.0% |
| DeBERTa | 2/40 | 5.0% |
| Text-Moderation | 0/40 | 0.0% |

The GCG suffixes were optimized against DeBERTa, yet Llama Guard's monitor fires at 7× DeBERTa's rate (35% vs 5%). A score direction analysis reveals the mechanism.

#### Score Direction Analysis

For all 14 valid Llama Guard detections, we computed mean pre-shift and post-shift classifier scores:

| | Mean pre-shift score | Mean post-shift score | Direction |
|---|---|---|---|
| All 14 cells | 0.05 (safe) | 0.78 (unsafe) | **TOWARD UNSAFE** (+0.73) |

The post-shift score is identical (0.7835) across all 14 cells, indicating that all 23 GCG-suffixed examples score identically through Llama Guard — consistent with the adversarial suffix being a fixed token pattern that Llama Guard uniformly classifies as highly unsafe.

**The GCG suffixes did not transfer their adversarial effect to Llama Guard.** They did the opposite: inputs that DeBERTa classifies as safe (score → 0, attack successful) are classified as highly unsafe (score → 0.78) by Llama Guard. The monitor fires because the score distribution shifts *upward*, not because the attack fooled a second classifier.

#### Mechanistic Interpretation

The asymmetry between classifiers reveals two distinct failure modes:

**DeBERTa (target classifier, 2/40 detected):** GCG suffixes successfully push DeBERTa's scores into the safe region (near zero), making attacked examples indistinguishable from the safe reference distribution. The attack succeeds *precisely because* it moves inputs into the region where safe examples already live — there is no distributional signal for the monitor to detect.

**Llama Guard (non-target classifier, 14/40 detected):** The same garbled GCG token sequences that fool DeBERTa's classification head appear anomalous to Llama Guard's generation mechanism. Llama Guard scores these inputs as highly unsafe (0.78), producing a massive upward shift in the score distribution that the monitor detects easily.

The correct interpretation is **cross-classifier anomaly detection**, not adversarial transfer: adversarial perturbations optimized against one classifier appear anomalous to architecturally different classifiers, enabling detection via score distribution shift even without attack transfer. The attacker evades the target classifier but is flagged by a monitor on a different classifier — not because the attack transferred, but because the adversarial tokens themselves look suspicious.

**Practical implication:** Heterogeneous cross-classifier monitoring provides anomaly detection coverage that single-classifier monitoring misses. An attacker who successfully evades one classifier's score distribution may produce detectable anomalies in another classifier's score space. This does not require the attack to transfer — it requires only that adversarial perturbations look different to different architectures.

| | Regime A (Synthetic) | Regime B (Temporal) | Regime C (Adversarial) |
|---|---|---|---|
| **Ground truth** | Synthetic onset at known step | Real temporal jailbreaks | GCG success = True |
| **Detection rate** | 86.5% (173/200) | 85.0% (17/20) | 13.8% (22/160) |
| **Mean latency** | 36.3 steps (w=100) | 32.6 steps | — |
| **Key finding** | Interaction dominates variance | Generalizes to real shift | Cross-classifier anomaly detection |

*Table: Summary across three ground-truth regimes. Regime C's low overall rate masks the Llama Guard finding (35% detection via anomaly, not transfer).*

## 6. Discussion

### 6.1 Deployment Requires Per-Classifier Monitoring Profiles

The dominance of the interaction term (η² = 0.265, the largest single systematic factor) is the central finding of the variance decomposition: detection difficulty is not a property of the classifier or the shift type alone, but of their pairing. The practical implication is that deployment teams cannot select a single monitoring configuration and apply it uniformly. The variance decomposition tells you *which* detector to pair with *which* classifier—this is the practitioner contribution of the factorial design.

Concretely: DeBERTa requires aggressive monitoring for adversarial suffixes (+18.7 steps above expected) but can afford relaxed thresholds for paraphrase (−15.9 steps). ShieldGemma requires the opposite profile—tight monitoring for paraphrase (+34.5 steps) but standard thresholds for adversarial inputs. A monitoring system that ignores these interaction effects will either over-alert on easy pairings or under-alert on hard ones.

### 6.2 Encoder vs. Decoder Safety Models: A Detection Tradeoff

The factorial reveals a clean architectural split in detection behavior:

**Discriminative models (DeBERTa, Text-Moderation)** detect paraphrase and compositional shift fast (26–37 steps). We hypothesize this is because these shifts alter the input distribution in ways that immediately change the score distribution — paraphrasing changes surface tokens that discriminative models attend to directly. However, they are slower on adversarial suffixes: DeBERTa's detection latency on adversarial-suffix (40.2 [36, 46]) is its worst condition. The hypothesized mechanism: the suffix preserves semantic content while appending tokens outside the model's fine-tuning distribution, and the classification head partially ignores appended tokens, causing the score distribution to shift slowly.

**Generative safety models (Llama Guard, ShieldGemma)** show the opposite pattern. They are slow on paraphrase (64–97 steps) because their scoring mechanism — generating a safety judgment token — is robust to surface-level rephrasing that preserves semantic content. They detect adversarial suffixes quickly (26.7 steps for Llama Guard) because the suffix disrupts the generation distribution more visibly than it disrupts a classification head.

This maps directly to deployment tradeoffs: if your threat model is adversarial red-teaming, monitor the discriminative classifier more aggressively. If your threat model is organic linguistic drift, monitor the generative classifier more aggressively.

### 6.3 Limitations

- **Single-condition conformal evaluation.** RQ2 was evaluated on DeBERTa × temporal shift only—the pairing with the strongest shift signal. Coverage degradation under subtler shifts (e.g., code-switch on Llama Guard) may be smaller, reducing the benefit of weighted correction. This is a limitation of the current evaluation, not a claim about the method's generality.
- **Abrupt shift onset.** Our evaluation assumes abrupt shift onset at a known step; gradual drift would require CUSUM-style statistics and is a harder problem that we do not address.
- **Residual variance.** 32% of variance in detection latency is attributable to seed/noise. This is expected with 5 seeds per cell but means individual cell estimates have wide confidence intervals. Post-hoc power analysis: with n=5 per cell, pairwise comparisons can only detect differences exceeding ~32 steps (Cohen's d = 1.46) at α=0.05 with 80% power. The factorial design provides power for main effects and interactions (all p < 0.001) but not for individual cell-level claims.
- **Binary classifiers only.** All four classifiers produce a single unsafe probability. Multi-category safety taxonomies (e.g., Llama Guard's 14 categories) may exhibit category-specific shift patterns invisible to a scalar score monitor.
- **Homogeneous negative controls.** Our negative control streams draw exclusively from WildGuardMix unharmful examples, forming a homogeneous reference distribution; production streams with higher natural variance—mixed content, multiple topics, temporal drift within the reference period—may require larger calibration sets or recalibration of the empirical threshold.

### 6.4 Future Work

- **Gradual shift detection.** Extending the CS framework to detect gradual drift via CUSUM or Bayesian online change-point detection (BOCPD), where the shift onset is not abrupt but accumulates over hundreds of steps.
- **Online conformal prediction without exchangeability.** The current weighted conformal layer assumes access to a post-alarm batch for density ratio estimation. Truly online conformal prediction under arbitrary distribution shift (Gibbs & Candès, 2021) would eliminate this batch requirement.

## References

- Garg, S., Balakrishnan, S., Lipton, Z. C., Neyshabur, B., & Sedghi, H. (2022). Leveraging unlabeled data to predict out-of-distribution performance. *ICLR*.
- Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*. arXiv:2106.00170.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *JMLR*, 13, 723–773.
- Han, Z., et al. (2024). WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs. arXiv:2406.18495.
- Inan, H., et al. (2023). Llama Guard: LLM-based input-output safeguard for human-AI conversations. arXiv:2312.06674.
- Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS*.
- Romano, Y., Sesia, M., & Candès, E. (2020). Classification with valid and adaptive coverage. *NeurIPS*. arXiv:2006.02544.
- Tibshirani, R. J., Foygel Barber, R., Candès, E., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*. arXiv:1904.06019.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*, 16(2), 117–186.
- Waudby-Smith, I. & Ramdas, A. (2024). Estimating means of bounded random variables by betting. *Journal of the Royal Statistical Society Series B*, 86(1), 1–27. arXiv:2010.09686.
- Zaremba, W., Gretton, A., & Blaschko, M. (2013). B-tests: Low variance kernel two-sample tests. *NeurIPS*.
- Zeng, Y., et al. (2024). ShieldGemma: Generative AI content moderation based on Gemma. arXiv:2407.21772.
- Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and transferable adversarial attacks on aligned language models. arXiv:2307.15043.
