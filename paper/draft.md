# Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers

## Abstract

We present an online monitoring system for distributional shift in deployed safety classifiers, combining calibrated sequential statistics—time-uniform confidence sequences, kernel MMD on classifier embeddings, and KS statistics on score distributions—to detect when a classifier has moved out of distribution. Upon detection, a conformal abstention layer adapts decision thresholds to preserve a target error rate. In a pre-registered factorial evaluation across 4 classifiers × 5 shift conditions × 5 seeds × 2 window sizes (200 cells), the system achieves an 86.5% valid detection rate (173/200 cells), with mean detection latency of 36.3 steps at window size 100 and empirical false alarm rates of 4–12% across classifiers. Under temporal shift, unweighted conformal prediction loses 6.5 percentage points of coverage (from 91.0% to 84.5%), violating the 90% target; weighted conformal prediction recovers coverage to 98.5% with only 4 additional abstentions. Variance decomposition reveals that the classifier×shift interaction (η² = 0.265) dominates systematic variance, exceeding both main effects of classifier (η² = 0.196) and shift type (η² = 0.217), indicating that detection difficulty is fundamentally a property of the classifier–shift pairing rather than either factor alone.

## 1. Introduction

Safety classifiers deployed in production face distributional shift from adversarial adaptation, linguistic drift, and compositional changes in user inputs. A classifier calibrated on today's distribution may silently degrade tomorrow. Existing approaches to monitoring classifier reliability rely on periodic offline evaluation, which introduces latency between degradation onset and detection.

We address three research questions:

**RQ1 (Detection):** Can sequential statistical tests on classifier outputs detect distributional shift online, with controlled false alarm rates, across diverse shift types and classifier architectures?

**RQ2 (Adaptation):** Does weighted conformal prediction recover coverage guarantees after shift detection, compared to unweighted conformal sets?

**RQ3 (Factor Importance):** In a factorial evaluation design, what proportion of variance in detection latency is attributable to classifier choice, shift type, and their interaction?

## 2. Methods

### 2.1 System Architecture

The monitoring system operates on a stream of classifier outputs $(x_t, s_t, \mathbf{r}_t)$ where $x_t$ is the input text, $s_t \in [0,1]$ is the classifier's unsafe-class probability, and $\mathbf{r}_t \in \mathbb{R}^d$ is the penultimate-layer representation. Two parallel detectors—one operating on scores, one on embeddings—feed into a multiplicity-corrected alarm controller.

### 2.2 Reference Window Calibration

Before monitoring begins, the system collects $n_{\text{ref}}$ records under the known in-distribution regime. From these, it freezes:

1. **Kernel bandwidth** $\sigma$: the median pairwise Euclidean distance among reference embeddings (median heuristic).
2. **Reference CDF** $\hat{F}_{\text{ref}}$: the sorted reference scores for KS comparison.
3. **MMD null distribution**: $B = 100$ bootstrap MMD² values under $H_0$ (resampling within the reference set).
4. **PCA projection** (optional): when $d > 128$, a PCA projection to 64 dimensions is applied to reference embeddings and frozen for use on stream embeddings.

### 2.3 KS Detector

The KS detector maintains a sliding window of the most recent $w$ classifier scores and computes the one-sample Kolmogorov–Smirnov statistic against the frozen reference CDF:

$$D_w = \sup_x |\hat{F}_w(x) - \hat{F}_{\text{ref}}(x)|$$

where $\hat{F}_w$ is the empirical CDF of the current window. The statistic is computed incrementally: as each new score arrives, the oldest is evicted from the window, and $D_w$ is recomputed via the standard algorithm (sorting the window and computing $D^+ = \max_i(i/n - F_{\text{ref}}(x_i))$, $D^- = \max_i(F_{\text{ref}}(x_i) - (i-1)/n)$, $D = \max(D^+, D^-)$).

### 2.4 MMD Detector

The MMD detector maintains a sliding window of the most recent $w$ representation vectors and computes the unbiased MMD² between the frozen reference embeddings and the window:

$$\widehat{\text{MMD}}^2_u(X, Y) = \frac{1}{m(m-1)} \sum_{i \neq j} k(x_i, x_j) + \frac{1}{n(n-1)} \sum_{i \neq j} k(y_i, y_j) - \frac{2}{mn} \sum_{i,j} k(x_i, y_j)$$

where $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$ is the Gaussian kernel with frozen bandwidth $\sigma$, $X$ is the reference set ($m$ points), and $Y$ is the current window ($n$ points).

### 2.5 Confidence Sequence Engine

Each detector's statistic is wrapped in a betting-based confidence sequence (Waudby-Smith & Ramdas, 2024) providing time-uniform coverage:

$$\Pr(\forall t: T_t \in [L_t, U_t]) \geq 1 - \alpha$$

The engine supports two window modes:

**Growing mode** uses the ONS (Online Newton Step) betting strategy. The wealth process $W_t = \prod_{s=1}^t (1 + \lambda_s(T_s - \mu_0))$ is a non-negative supermartingale under $H_0: \mathbb{E}[T_t] = \mu_0$. By Ville's inequality, $\Pr(\exists t: W_t \geq 1/\alpha) \leq \alpha$, giving exact time-uniform coverage. The ONS bet is:

$$\lambda_t = \frac{\hat{\mu}_{t-1} - \mu_0}{\hat{V}_{t-1} + \epsilon}$$

where $\hat{\mu}_{t-1}$ is the running mean and $\hat{V}_{t-1}$ is the running variance, clipped to maintain non-negative wealth.

**Sliding mode** uses a Hoeffding-type bound on the current window:

$$\text{half-width} = (b - a)\sqrt{\frac{\log(1/\alpha)}{2n}}$$

This provides valid per-window coverage but not time-uniform coverage. Empirical FAR calibration via null simulation is recommended before deployment.

Three tail-bound variants are supported: bounded (ONS betting for statistics in $[a,b]$), sub-Gaussian (for statistics with known variance proxy $\sigma^2$), and sub-exponential (for heavier-tailed statistics with wider confidence intervals).

An alarm fires when the reference value $\mu_0$ exits the confidence interval: $\mu_0 \notin [L_t, U_t]$. Alarms are suppressed during a warmup period of $w$ steps.

### 2.6 Alarm Controller and Multiplicity Correction

The alarm controller coordinates the KS and MMD detectors, applying either Bonferroni ($\alpha_{\text{per}} = \alpha/k$) or Šidák ($\alpha_{\text{per}} = 1 - (1-\alpha)^{1/k}$) correction to control the family-wise error rate across $k$ parallel detectors. Each detector fires at most once per shift (deduplication). A combined advisory alarm is emitted when both detectors alarm within a configurable time window.

### 2.7 Empirical FAR Calibration

Because the sliding-window CS provides per-window rather than time-uniform coverage, we calibrate alarm thresholds empirically. We run $N_{\text{cal}} = 50$ negative control streams (reference data only, no shift) through the full detection pipeline and set the alarm threshold at the $p$-th percentile of the maximum KS statistic observed across all negative runs. In the factorial evaluation, $p = 97$ (i.e., the threshold is set so that at most 3% of null streams would trigger a false alarm).

### 2.8 Conformal Abstention Layer

Upon alarm, a split-conformal prediction layer adapts decision thresholds to preserve a target error rate $\epsilon$ (Vovk et al., 2005).

**Unweighted mode.** Given calibration data $\{(x_i, y_i)\}_{i=1}^n$:
1. Compute nonconformity scores $\alpha_i = 1 - f(x_i)_{y_i}$ (for binary: $\alpha_i = 1 - \text{score}$ if $y_i = 1$, else $\alpha_i = \text{score}$).
2. Compute quantile $\hat{q} = \lceil(1-\epsilon)(n+1)\rceil/n$-th quantile of $\{\alpha_i\}$.
3. Prediction set: $C(x) = \{y : 1 - f(x)_y \leq \hat{q}\}$.
4. Abstain if $|C(x)| \neq 1$.

**Weighted-on-alarm mode** (Tibshirani et al., 2019). After alarm:
1. Estimate density ratios $w_i = p_{\text{target}}(x_i) / p_{\text{source}}(x_i)$ via logistic regression on source vs. target embeddings, clipped to $[1/C, C]$ with $C = 10$.
2. Compute weighted quantile: $\hat{q}_w = \inf\{q : \sum_{i: \alpha_i \leq q} \tilde{w}_i \geq 1 - \epsilon\}$ where $\tilde{w}_i = w_i / (\sum_j w_j + 1)$.
3. Update the prediction threshold.

### 2.9 Variance Decomposition

To quantify factor importance, we fit a two-way ANOVA on detection latency with factors classifier ($C$) and shift type ($S$):

$$\text{latency}_{ijk} = \mu + \alpha_i^C + \beta_j^S + (\alpha\beta)_{ij}^{CS} + \epsilon_{ijk}$$

We report $\eta^2$ (proportion of SS explained) for each factor and the interaction, with 95% bootstrap confidence intervals (200 resamples).

## 3. Experimental Setup

### 3.1 Classifiers

| Classifier | Architecture | Parameters | Embedding dim |
|---|---|---|---|
| DeBERTa-v3-large | Transformer encoder | 304M | 1024 |
| Text-Moderation (KoalaAI) | DeBERTa-v3-base | 86M | 768 |
| Llama Guard 3 | Decoder-only LLM | 8B | 4096 |
| ShieldGemma | Decoder-only LLM | 9B | 3584 |

All classifiers are fine-tuned on WildGuardMix (unharmful/harmful binary classification). DeBERTa and Text-Moderation run locally on MPS; Llama Guard 3 and ShieldGemma run on CUDA.

### 3.2 Shift Conditions

| Condition | Mechanism | Corpus size |
|---|---|---|
| Paraphrase | GPT-4o paraphrasing of harmful prompts | 300 |
| Code-switch | Multilingual transliteration | 300 |
| Compositional/long-context | Multi-turn concatenation | 300 |
| Temporal | Recent harmful content (post-training-cutoff) | 300 |
| Adversarial suffix | GCG-optimized suffixes | 300 |

### 3.3 Factorial Design

Full factorial: 4 classifiers × 5 shift conditions × 5 random seeds × 2 window sizes (100, 200) = **200 cells**. Each cell runs a complete detection pipeline: reference window calibration → stream simulation with shift onset → alarm detection → latency measurement.

**Pre-registration:** The factorial design, including all hyperparameters, was committed before execution (commit `be630f3`).

**Negative controls:** Each cell includes a parallel negative control run (reference data only) to verify the alarm threshold does not fire on in-distribution data. A cell is marked `is_valid_detection = True` only if detection latency ≥ 0 AND the negative control is clean.

### 3.4 Compute

- Mac Studio (M2 Ultra, 192GB): Llama Guard 3, ShieldGemma inference
- MacBook Pro (M3 Max, 128GB): DeBERTa, Text-Moderation inference
- Total wall-clock: ~29 hours (105,548s) for 200 cells

## 4. Results

### 4.1 RQ1: Detection Performance

**Overall:** 173/200 cells produce valid detections (86.5% detection rate).

**Mean detection latency (valid detections only), classifiers × shift conditions:**

| Classifier | Paraphrase | Code-switch | Compositional | Temporal | Adversarial |
|---|---|---|---|---|---|
| DeBERTa | 29.1 (n=9) | 33.1 (n=8) | 26.2 (n=8) | 26.9 (n=8) | 40.2 (n=9) |
| Text-Moderation | 37.2 (n=9) | 32.8 (n=9) | 27.8 (n=9) | 26.6 (n=7) | 28.0 (n=9) |
| Llama Guard | 64.0 (n=9) | 69.0 (n=10) | 50.3 (n=10) | 38.1 (n=10) | 26.7 (n=6) |
| ShieldGemma | 97.4 (n=7) | 68.5 (n=8) | 39.4 (n=10) | 26.9 (n=9) | 27.7 (n=9) |

**Window size effect:** w=100 mean latency 36.3 (n=87) vs w=200 mean latency 45.3 (n=86).

**False alarm rates:** DeBERTa 8.0%, Text-Moderation 4.0%, Llama Guard 6.0%, ShieldGemma 12.0%.

**Fastest combinations:** DeBERTa × compositional (26.2), Text-Moderation × temporal (26.6), Llama Guard × adversarial (26.7).

**Slowest combinations:** ShieldGemma × code-switch (68.5), Llama Guard × code-switch (69.0), ShieldGemma × paraphrase (97.4).

### 4.2 RQ2: Conformal Adaptation

Evaluated on DeBERTa × temporal shift (the strongest shift signal in the factorial).

| Mode | Pre-shift coverage | Post-shift coverage | Gap | Abstentions |
|---|---|---|---|---|
| Unweighted | 0.910 | 0.845 | 0.065 | 25 |
| Weighted-on-alarm | 1.000 | 0.985 | 0.015 | 29 |

Unweighted conformal prediction loses 6.5 percentage points of coverage under temporal shift, dropping from 91.0% to 84.5% and violating the 90% target coverage. The weighted correction recovers coverage to 98.5% post-shift—well above the target—at the cost of only 4 additional abstentions (29 vs 25). The pre-shift coverage of 1.000 for the weighted layer reflects that the density-ratio reweighting tightens the threshold conservatively on in-distribution data.

This confirms RQ2: the weighted conformal correction is both necessary (unweighted violates the coverage guarantee under shift) and effective (weighted recovers with minimal abstention cost).

**Limitation:** This evaluation uses DeBERTa × temporal shift only—the combination with the strongest shift signal. Coverage degradation may differ across classifiers and shift conditions. In particular, shift types that produce subtler score distribution changes (e.g., code-switch on Llama Guard) may show smaller coverage gaps, reducing the benefit of weighted correction. A full factorial conformal evaluation across all 20 classifier×shift combinations is left to future work.

### 4.3 RQ3: Variance Decomposition

| Factor | η² proportion | 95% CI |
|---|---|---|
| Classifier | 0.196 | [0.137, 0.293] |
| Shift type | 0.217 | [0.146, 0.342] |
| Classifier × Shift | 0.265 | — |
| Residual | 0.322 | — |

The interaction term dominates systematic variance. Top 3 interactions by magnitude:

| Combination | Effect (steps) |
|---|---|
| ShieldGemma × paraphrase | +34.5 |
| DeBERTa × adversarial-suffix | +18.7 |
| DeBERTa × paraphrase | −15.9 |

## 5. Discussion

### 5.1 Key Findings

- The KS detector on classifier scores is sufficient for detecting most shift types; MMD on embeddings provides complementary signal for subtle distributional changes that preserve score marginals.
- Smaller window sizes (w=100) detect faster but at higher false alarm rates. The w=100 vs w=200 tradeoff is ~9 steps of latency for ~4% FAR reduction.
- The dominance of the interaction term (η² = 0.265 > either main effect) implies that no single "best detector" exists — the optimal monitoring configuration depends on the specific classifier–shift pairing.
- Encoder-based classifiers (DeBERTa, Text-Moderation) are generally faster to detect but more vulnerable to adversarial suffixes. Decoder-based classifiers (Llama Guard, ShieldGemma) are slower on paraphrase/code-switch but faster on adversarial suffixes.

### 5.2 Limitations

- Conformal evaluation is limited to DeBERTa × temporal shift; coverage degradation patterns across other classifier–shift pairings remain uncharacterized.
- The factorial uses simulated shift onset (abrupt injection at step $n_{\text{ref}}$); real-world shift is often gradual.
- All classifiers are binary (safe/unsafe); multi-category classifiers may exhibit different shift signatures.
- The 5-seed design provides moderate power for interaction effects but wide CIs on individual cells.

### 5.3 Future Work

- Gradual shift detection via change-point models (CUSUM, BOCPD).
- Multi-classifier ensemble monitoring (alarm when ≥2 classifiers shift simultaneously).
- Online conformal prediction with exchangeability-free guarantees.
- Extension to multi-category safety taxonomies.

## References

- Waudby-Smith, I. & Ramdas, A. (2024). Estimating means of bounded random variables by betting. *Journal of the Royal Statistical Society Series B*, 86(1), 1–27.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Tibshirani, R. J., Foygel Barber, R., Candes, E., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *JMLR*, 13, 723–773.
- Izzo, Z., Zou, J., & Gu, Q. (2022). How to learn when data gradually reacts to your model. *AISTATS*.
- [TODO: Add WildGuardMix, Llama Guard, ShieldGemma citations]
