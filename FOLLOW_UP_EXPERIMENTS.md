# Follow-Up Experiment Plan — Post-arXiv Scope Extension

**Status:** Gates in progress.

## Current Status (2026-06-23 21:09 SGT)

| Gate/Test | Status | Where | Result |
|------|--------|-------|--------|
| **B (Gate)** | ✅ Complete | MacBook | **GO** — Scan martingale 83–100% vs KS 43–47%, FAR 0/200 |
| **B (AV2)** | ✅ Complete | MacBook | **Confirmed** — uniform FAR ≤0.5% across all 4 classifiers (vs 2–9.5% KS spread). Zero calibration. |
| **A** | 🔄 Running | Mac Studio (PID 13014) | GCG 27/100 done (14 flipped, 52%). Check `wc -l data/shifted/adversarial_suffix/deberta_suffixes_gate_a.jsonl` |
| **C** | 🔄 Queued | Mac Studio (after Gate A) | 10 additional classifiers. Check `results/gate_c_monitorability.json` |

**Monitor progress:** `tail -f results/gates_ac.log` on Mac Studio.

---
**Parent work:** arXiv:2606.11949 (Shift Detection Monitor). The 980-cell factorial + post-factorial additions (CS growing-window, MMD, PCA-conformal, gradual drift, mechanistic n=4) are *complete and submitted*. This document plans the next phase.
**Compute available:** Mac Studio M3 Ultra (96 GB) for local inference + GCG gradients; AWS Bedrock for breadth. No time/budget constraint — the binding constraint is research risk, so every track gates on a cheap replication before the full build.
**Convention:** Matches `docs/pre_registration.md` + `docs/pre_registration_amendment_2.md` — hypothesis IDs with directional predictions, pre-specified success criteria, α = 0.05, reference size 500, onset 500, windows {100, 200}, 97th-percentile empirical FAR calibration, Wilson / Clopper–Pearson CIs, η² with bootstrap CIs. Commit this file before executing any gate.

---

## 0. Scope at a Glance

Three independent deliverables, each a potential standalone paper. They do **not** belong in one paper — they tell three different stories.

| Track | Working title | Builds on | Novelty ceiling | Risk |
|---|---|---|---|---|
| **A** | Heterogeneous Monitoring Ensembles: cross-architecture divergence as an attack detector | Regime C canary (currently 22-example PoC) | Highest | Medium |
| **B** | Anytime-valid shift monitoring: conformal test martingales + low-rank conformal | CS engine (Ville/ONS) + density-ratio collapse + PCA diagnostic | Rigor-defining | Low |
| **C** | A monitorability law: predicting detection latency from score-distribution geometry | Mechanistic n=4 (r=0.97) | High (or clean negative) | High |

Cross-cutting threat experiment (the **monitor-aware adversary**) bridges A and B.

**Sequencing rule (from the Agentic POST_RUN playbook):** run the three validation gates first — they barely compete for resources (one GPU, one CPU-only, one Bedrock). Kill dead arms early; scope-decide A/B/C placement only after gate data is in.

---

## Track A — Heterogeneous Monitoring Ensembles

### A.0 Reframing (the key idea)

`scripts/check_regime_c_direction.py` already established the *honest* mechanism behind the README's headline #3:

- A GCG suffix optimized against **DeBERTa** pushes DeBERTa's score **toward safe** → DeBERTa is fooled, monitor misses (38/40).
- The *same* suffix pushes **Llama Guard's** score **toward unsafe** (Δ ≈ +0.73) → Llama Guard is *not* fooled; it sees the suffix as anomalous, monitor detects (14/40).

So the signal is **not** transfer (the attack does not fool the second model). The signal is **cross-architecture score divergence**: an evasion tuned against one architecture leaves a disagreement signature that a second, architecturally different classifier exposes. The original paper states this on a 22-example PoC. Track A turns the observation into a **method + threat model**: deliberately co-deploy ≥2 architecturally diverse classifiers and monitor the *distribution of their disagreement* as a first-class detection channel.

### A.1 Pre-registered hypotheses

| ID | Statement | Direction | Success criterion |
|---|---|---|---|
| **CA1** | For an attack optimized against target T, the cross-architecture divergence D = \|score_T − score_C\| (C = non-target, different architecture) is larger under attack than under the null reference stream. | one-sided, D_attack > D_null | Divergence-channel detection rate ≥ 80%, Wilson lower bound > null 97th-pct FAR, non-overlapping CIs |
| **CA2** | The divergence channel detects attacks in cells where **neither** T nor C individually alarms ("jointly-invisible" cells). | count > 0 | ≥ 1 jointly-invisible cell detected by divergence with FAR ≤ α; report rate with Wilson CI |
| **CA3** | The divergence signal is not GCG-specific: it replicates for ≥ 2 attack families. | replication | CA1 holds for GCG **and** (AutoDAN or PAIR) at the same threshold |
| **CA4** | Architecture diversity is the active ingredient: cross-family divergence (encoder↔decoder) exceeds within-family divergence (encoder↔encoder or decoder↔decoder). | cross > within | η² of architecture-pair on divergence > 0.10, CI excludes 0 |
| **CA5 (multi-cat)** | A taxonomy-resolved divergence (per-hazard-category vector, not scalar) detects category-localized evasion invisible to the scalar channel. | count > 0 | ≥ 1 category where vector divergence alarms while scalar does not |
| **CA6 (gibberish control — MAKE-OR-BREAK)** | GCG suffixes produce *more* cross-architecture divergence than length-matched random-token suffixes. If random gibberish diverges equally, the canary is an OOD/gibberish detector, not an attack detector. | GCG > random | GCG divergence Wilson LB > random-suffix divergence Wilson UB (non-overlapping). If overlapping → reframe as "anomalous-input detector," not "attack detector." |
| **CA7 (latent divergence)** | Cross-architecture divergence is *stronger and harder to evade in penultimate representations than in output scores* — an attacker can align two models' output scores but not their internal activation trajectories. | repr-divergence > score-divergence under joint evasion | Under the CA8 joint-evasion adversary, representation-space divergence (Wasserstein on PCA-projected penultimate, + CKA as secondary) stays detectable while score divergence collapses |
| **CA8 (joint-evasion adversary — the crux)** | An adversary minimizing *both* the target's unsafe score *and* the cross-classifier score variance can keep two architectures in lockstep agreement. | existence test | If such suffixes exist at scale → canary is best-effort, not a guarantee (still publishable). If they do not (architectures cannot be jointly aligned) → that *is* the headline. |

### A.2 Validation gate (do this first)

**Gate A — divergence replication at n≈100.** Current corpus is ~22 GCG suffixes (`data/shifted/adversarial_suffix/deberta_suffixes.jsonl`). Scale to ~100 via `scripts/run_gcg.py` (DeBERTa target) and re-measure DeBERTa↔Llama-Guard divergence with `scripts/check_regime_c_direction.py`-style direction logic.

- **GO** if the +0.73-style divergence replicates with a Wilson lower bound above the null FAR at n≈100.
- **NO-GO** → write up as "PoC does not scale beyond 22 examples; cross-architecture divergence is corpus-specific" (honest negative, still informative). Stop Track A.

**Two checks to run on the Gate A output before declaring GO (added from review):**
1. **Score variance, not just mean (CA6 precursor).** Confirm the canary (Llama Guard) scores are *spread*, not a constant offset. If all 100 suffixes push Llama Guard to the same value, the "divergence distribution" is a 1-D shift and the distributional framing is overbuilt. `gate_a` already records the delta std — inspect it.
2. **Gibberish control (CA6).** Score length-matched random-token suffixes on both models. If random gibberish diverges as much as GCG, the claim downgrades from "attack detector" to "anomalous-input detector." Cheap — reuses the scoring pipeline.

⚠️ **Tokenizer-artifact confound:** GCG suffixes are optimized against the DeBERTa tokenizer. When fed to Llama Guard (different tokenizer), they may fragment differently. Part of the observed divergence could be text-tokenization breakage, not semantic anomaly. The gibberish control (CA6) partially isolates this; for the full build, also test suffixes optimized in a *shared* token space where possible.

Compute: **Mac Studio** (GCG gradients on DeBERTa-v3-large 304M are cheap; Llama Guard 8B scoring is forward-only).

### A.3 Full build (conditional on Gate A)

1. **Divergence detector** — new `shift_detection_monitor/detection/divergence_detector.py`: monitors the **joint 2-D distribution** of (score_T, score_C) against a frozen reference, not just the scalar |score_T − score_C|. The 2-D joint captures both margin shifts and correlation shifts; alarm on a 2-D KS / kernel two-sample test of the joint series. Calibrate FAR on null streams (97th pct).
2. **Internal latent divergence (CA7)** — add a representation-space divergence channel: Wasserstein distance on PCA-projected penultimate embeddings (primary; the MMD detector already extracts these), CKA as a secondary diagnostic (with the caveat that cross-dimension CKA is invariant to orthogonal transforms — interpret carefully). Thesis: output scores are alignable by an attacker, internal trajectories across architectures are not.
3. **Joint-evasion adversary (CA8 — the crux experiment, promoted from a footnote)** — optimize suffixes against `L_target(x+δ) + λ·Var[score_T, score_C]` to test whether two architectures can be held in lockstep. This is the decisive test of the whole thesis, not a robustness afterthought. Report whether such suffixes exist, and if so at what λ and attack cost. Bridges to Track B's monitor-aware adversary.
4. **Second attack family** — AutoDAN or PAIR generator alongside GCG (CA3). New builder under `scripts/` + corpus under `data/shifted/adversarial_suffix/`.
5. **Architecture-pair sweep** (CA4) — all 6 pairs from {DeBERTa, Text-Moderation (encoders), Llama Guard, ShieldGemma (decoders)}; variance-decompose divergence by pair type. Reuse `run_variance_decomposition.py`.
6. **Natural-shift generality (review addition)** — test whether the heterogeneous ensemble also detects *non-adversarial* shifts (paraphrase, code-switch) earlier than single classifiers, using existing factorial data. If yes, the canary is a general-purpose monitor, not just an adversarial detector.
7. **Multi-category channel** (CA5) — see §A.4. Start binary; add categories as secondary analysis to avoid drowning the clean signal in dimensionality.

### A.4 Multi-category enablement (shared dependency)

`ClassifierOutput` (in `types.py`) is currently `score: float` + `representation` + `metadata: dict[str,str]`. The Llama Guard adapter extracts a single safe/unsafe scalar from `softmax(logits[[safe_id, unsafe_id]])`; ShieldGemma a single Yes/No. To get per-category vectors:

- Extend `ClassifierOutput` with optional `category_scores: dict[str, float] | None` (non-breaking; defaults None).
- Llama Guard: parse the generated hazard-category tokens (S1–S14) into a probability vector.
- ShieldGemma: run the per-policy prompt variants to get a policy-resolved vector.
- New `MultivariateKSDetector` / kernel test on the category simplex.

This is the largest engineering item; it also feeds CA5 and is reusable by Track C.

### A.5 Deliverable & scope decision
Standalone paper: *"Heterogeneous Monitoring Ensembles."* Decision after Gate A + CA3: if cross-family generality (CA4) holds, lead with the architecture-diversity principle; if only GCG+DeBERTa↔Llama-Guard holds, frame as a focused case study (mirroring the original paper's careful "22-example, larger-scale validation needed" framing).

---

## Track B — Anytime-Valid Shift Monitoring

### B.0 The gap (grounded in code)

`detection/confidence_sequence.py` already implements a betting wealth supermartingale:
- **Growing mode** = exact time-uniform coverage via Ville's inequality (ONS betting). `P(∀t: T_t ∈ [L_t,U_t]) ≥ 1−α`.
- **Sliding mode** = Hoeffding per-window. The docstring explicitly states this is **not** time-uniform and *"empirical FAR calibration via null simulation is recommended before deployment."*

So the production detector (sliding KS, empirically calibrated at the 97th percentile) has no anytime-valid guarantee, and FAR varies **5×** across classifiers (Text-Mod 2.0% → DeBERTa 9.5%; Llama Guard MMD 10%). Two stated limitations — the **<30% mixing blind spot** (KS 43% vs CS 97%) and the **FAR asymmetry** — are both symptoms of empirical thresholding. A conformal test martingale (Vovk) replaces the empirical knob with a guarantee, and the betting machinery to do it is already in-house.

**Prior-art positioning (review addition — required for novelty survival).** The scan/window-limited martingale is **not** a new statistical object: scan statistics, MOSUM charts, CUSUM, and window-limited anytime-valid tests already exist (Howard, Ramdas, McAuliffe & Sejdinovic 2021; Vovk conformal test martingales). Our contribution is **not** the estimator — it is (a) applying conformal test martingales to *safety-classifier shift monitoring*, and (b) the empirical finding that they beat empirically-calibrated KS at low mixing with no per-classifier tuning. Frame it exactly this narrowly; cite the sequential-analysis literature up front. **Verify every reference against the actual paper before citing** (see Reproducibility § — citation hygiene).

### B.1 Pre-registered hypotheses

| ID | Statement | Direction | Success criterion |
|---|---|---|---|
| **AV1** | A conformal test martingale (betting on conformal p-values of the score stream) detects low-mixing drift (≤30%) at a higher rate than sliding-window KS. | martingale > KS | Detection rate at 30% mixing: martingale Wilson LB > KS Wilson UB (replicate the CS 97% vs KS 43% gap *with* a formal guarantee) |
| **AV2** | The martingale's e-value threshold (reject when wealth ≥ 1/α) controls FAR ≤ α **uniformly** across all 4 classifiers — eliminating the 5× empirical-FAR spread. | FAR ≤ α ∀ classifier | All 4 classifiers' null-stream FAR ≤ 0.05, no calibration; spread < 2× |
| **AV3** | A bounded-memory martingale variant retains the low-mixing advantage of growing-window CS without unbounded memory. | ≈ growing | Detection rate within 5 pp of growing-window CS at ≤30% mixing |
| **AV4 (PCA method)** | Low-rank projection (PCA to d ≤ 32) before density-ratio estimation restores effective sample size and conformal coverage; a data-driven dimension rule (from the ESS/separability curve) generalizes across shift types. | ESS↑, coverage↑ | Coverage recovery ≥ original temporal result (+33 pp Llama Guard, +20.5 pp ShieldGemma) on ≥2 unseen shift types; ESS > 50 |
| **AD1 (bridge)** | A **monitor-aware adversary** doing slow sub-threshold drift evades sliding KS but is still caught by the anytime-valid martingale. | martingale robust | KS misses (latency > horizon) while martingale alarms, on ≥80% of adversarial drift seeds. **Define adversary precisely:** gradual concept injection holding the windowed KS distance just below the calibrated threshold; report the drift slope at which the crossover occurs (the "killer chart"). Caveat: drift sub-threshold for a window-100 KS may also be sub-threshold for a window-50 scan — the martingale's edge comes from time-uniform accumulation over a longer horizon, so test slopes where that horizon advantage actually bites. |
| **AV5 (exchangeability stress — review addition, CRITICAL)** | The martingale's FAR guarantee holds only under an **exchangeable** null. Benign domain drift (new topics, dialect shifts) breaks exchangeability *without* being adversarial. | FAR characterization | Measure FAR on a **non-iid benign** null stream (real temporal corpus, benign-only) — not just iid-resampled reference. Report honestly: if FAR inflates, that is the method's boundary, and the monitor cannot distinguish benign from adversarial drift (both are distribution shift). |
| **AV6 (ε robustness — review addition)** | A single betting parameter (ε≈0.3, w=50) works across all 4 classifiers and shift types, OR a principled default derived from the reference window's score variability does. | invariance | One (ε, w) achieves ≥ KS detection on all 4 classifiers without per-case tuning; else provide the variability-based default. Protects the "no calibration" selling point. |

### B.2 Validation gate (**✅ COMPLETE — GO**)

**Gate B — martingale on simulated streams from null score distributions.** Tested scan martingale (w=50, ε=0.3) against KS at matched difficulty (KS ~43% detection). **Zero new inference — used `results/null_scores.json` only.**

**Result (2026-06-23):**

| Condition (calibrated to KS ≈ 43%) | KS | Scan Martingale |
|---|---|---|
| Small per-sample shift, 30% mixing | 47% (14/30) | **100%** (30/30), μ=95 steps |
| Large per-sample shift, 20% mixing | 43% (13/30) | **83%** (25/30), μ=140 steps |
| FAR (200 null streams) | 0% | **0%** (provable ≤5% by Ville + union bound) |

**Method:** Union of W=50 sub-martingales, each betting ε·p^(ε−1) on two-sided conformal p-values derived from frozen reference CDF. Threshold = log(W/α). No empirical calibration needed.

**Key insight:** The point martingale (single accumulator from t=0) fails because 500 pre-shift observations dilute the post-shift signal. The scan martingale succeeds by starting fresh sub-martingales at every step — whichever one begins near the changepoint accumulates evidence fastest.

**GO criterion met:** ≥70% at 30% mixing. Actual: 83–100%. Proceed to full build.

### B.2b AV2 — Cross-classifier FAR uniformity (**✅ CONFIRMED**)

Tested the same scan martingale (w=50, ε=0.3) on 200 null streams per classifier:

| Classifier | Martingale FAR | Old KS FAR (arxiv) |
|---|---|---|
| DeBERTa | 0.5% (1/200) | 9.5% |
| Text-Moderation | 0.5% (1/200) | 2.0% |
| Llama Guard | 0.0% (0/200) | 3.0% |
| ShieldGemma | 0.0% (0/200) | 8.5% |

**Spread: 0.5 pp** (vs 7.5 pp under empirical KS). All ≤ α with zero calibration. The 5× FAR asymmetry is eliminated.

### B.2c AV6 — Epsilon robustness (**⚠️ PARTIAL — classifier-dependent**)

At matched difficulty (KS ≈ 45% detection per classifier), (ε=0.3, w=50) works excellently for encoders (DeBERTa 100%, Text-Mod 100%) and Llama Guard (100%), but **fails on ShieldGemma (10%)** because ShieldGemma's null distribution is already widely spread (std=0.14, only 28% of scores below 0.01) — conformal p-values stay near-uniform even under shift.

**Implication:** a single fixed (ε, w) is not universal. The paper must (a) acknowledge this, (b) propose a variability-based default (e.g., scale ε inversely with null-score entropy/IQR), and (c) recommend pairing the martingale with KS for wide-distribution classifiers (the hybrid architecture from AV6 review feedback).

### B.2d AV5 — Exchangeability stress test (**⚠️ HONEST BOUNDARY FOUND**)

| Test | Description | Alarms? |
|---|---|---|
| 1. Original order | Reference scores in sequential order (mild autocorrelation) | **0/4** — safe ✅ |
| 2. Block-structured | Bottom-half then top-half of reference (simulates topic drift) | **4/4** — all alarm ⚠️ |
| 3. Gradual drift | Mean walks +0.5×std over 800 steps (benign concept drift) | **2/4** (encoders alarm, decoders don't) |

**The guarantee holds only under exchangeability.** The martingale detects *any* distributional departure — adversarial or benign. It cannot distinguish topic drift from attack. State this honestly; recommend: alarm triggers investigation, not automatic response.

### B.2e AD1 — Monitor-aware adversary (**✅ CONFIRMED for ramped onset**)

Against a DeBERTa small-shift adversary with ramped onset (50-step ramp to target mixing):

| Mixing | KS | Scan Martingale | Gap |
|---|---|---|---|
| 15% | 3% | **37%** | +10 |
| 20% | 7% | **67%** | +18 |
| 25% | 10% | **87%** | +23 |
| 30% | 47% | **100%** | +16 |

At 20% mixing: KS is effectively blind (7%), martingale catches two-thirds. **This is the "killer chart" for the paper.**

**Caveat (honest):** against a *constant low-rate* adversary (no ramp, sustained trickle at ≤12%), neither KS nor the scan martingale reliably detects. The martingale's advantage is at *ramped* onsets — the fresh sub-martingale started near the changepoint accumulates evidence during the ramp. Constant sub-threshold injection evades both methods. State this as the residual threat model.

### B.3 Full build (conditional on Gate B)

1. `detection/conformal_martingale.py` — conformal p-values from the frozen reference CDF, betting martingale (reuse the `_log_wealth` ONS accumulation from `ConfidenceSequenceEngine`), alarm at wealth ≥ 1/α.
2. FAR-uniformity evaluation across 4 classifiers (AV2) — reuse `run_cs_evaluation.py` harness.
3. **Exchangeability stress test (AV5)** — run FAR on a non-iid benign null stream (real temporal corpus, benign-only) to characterize the guarantee's real boundary. State honestly.
4. **ε robustness sweep (AV6)** — confirm one (ε, w) works across all classifiers/shifts, or derive a variability-based default.
5. **Latency at high mixing (review addition)** — confirm the bounded-memory scan closes the ~2× latency gap that growing-window CS had vs KS at high mixing (be competitive at high mixing, dominant at low).
6. Bounded-memory variant (AV3) — windowed/decayed wealth; compare to growing-window CS.
7. PCA-conformal as a *method*, not a diagnostic (AV4): formalize dimension selection from the ESS-vs-d curve; extend `run_pca_conformal_sweep.py` to emit the rule and validate on held-out shift types.
8. Monitor-aware adversary (AD1, also a Track-A bridge): an attacker that ramps mixing to stay under the calibrated KS threshold; new `scripts/run_monitor_evasion.py`.

### B.4 Deliverable
Methods paper: *"Anytime-valid monitoring of safety classifiers."* Strongest rigor story; closes two named limitations with theory already owned. AV4 is a guaranteed-positive sub-result even if AV1 fails.

---

## Track C — A Monitorability Law

### C.0 The hint

`scripts/mechanistic_analysis.py` finds null-score std correlates with mean detection latency at **Pearson r = 0.97 (p = 0.032, n = 4)**: DeBERTa std 0.087, Text-Mod 0.066, Llama Guard 0.144, ShieldGemma 0.141. Embedding displacement does **not** predict latency (r = −0.09) — a clean negative control. If the score-geometry law holds at larger n, you can predict a classifier's monitorability *offline, with zero attack data* — a deployable "monitorability score."

### C.1 Pre-registered hypotheses

| ID | Statement | Direction | Success criterion |
|---|---|---|---|
| **ML1** | Null-score std predicts mean detection latency across N ≥ 15 classifiers. | positive | Pearson r > 0.6, p < 0.01 at N ≥ 15 (pre-registered threshold; r=0.97 at n=4 is not assumed to hold) |
| **ML1b (confounding test — CRITICAL, review addition)** | The correlation holds **within** architecture family, not just across the encoder/decoder gap. | within-family r > 0 | Report Pearson r *separately* within encoders and within decoders. If it vanishes within families, the "law" reduces to "discriminative models are easier to monitor" — a weaker but honest finding. The n=4 r=0.97 is almost certainly the 2-cluster gap in disguise; Gate C must include multiple classifiers *within* each family. |
| **ML2** | A **simple 1–2 feature** linear model (not a multi-feature black box) predicts latency with honest CIs. | predictive | At N≈15, LOOCV on >2 features is too high-variance to trust. Prefer a 1-feature linear fit reported with bootstrap CI. No "monitorability score product" until N is much larger. |
| **ML2b (boundary curvature — review addition, better candidate for a "law")** | Local decision-boundary sharpness (first-step adversarial perturbation size at benign inputs) predicts latency *better and more family-invariantly* than static score moments. | predictive, invariant | We have gradient access to the encoders (GCG pipeline). A model whose benign inputs sit near a sharp boundary should detect fast. Test whether this breaks the family confound that std cannot. |
| **ML3** | The shift-specific sign reversal replicates: surface shifts give wider→slower; adversarial suffix reverses. | reversal | Sign(r) flips for adversarial-suffix vs paraphrase at N ≥ 15. If it persists, hypothesize mechanism (adversarial = sharp concentrated score shift; surface = diffuse). |
| **ML4** | Embedding-displacement remains a non-predictor (negative control). | null | \|r\| < 0.3, p > 0.05 |

### C.2 Validation gate (Bedrock breadth — run in parallel)

**Gate C — extend beyond n=4.** Run the existing latency pipeline (`run_factorial.py` Regime A) on ~10–12 additional safety classifiers and recompute the correlation.

- **GO** if r > 0.6 (p < 0.05) at N ≈ 12–16 **and** the correlation does not entirely vanish within families (ML1b).
- **NO-GO** → "monitorability is not an intrinsic, predictable property" — a clean, publishable negative that corrects an over-strong reading of the n=4 result.

**Sampling requirement (review addition):** the classifier set must include multiple models *within* each family (≥3 encoders, ≥3 decoders), or the result is uninterpretable — a 2-cluster correlation is not a law. Bias selection toward within-family spread, not just more models.

**Taxonomy normalization is NOT required (answers a reviewer question).** Monitorability is computed *per-classifier on its own score distribution* relative to its *own* frozen reference. We never compare harm-label definitions across models. Each model's "P(unsafe)" by its own taxonomy is just a scalar stream; the law concerns the *geometry* of that stream, not cross-model label agreement. The only normalization is score orientation (unsafe = high), handled by label-name mapping in `gate_c`.

Compute: **Mac Studio** (local fine-tuned encoders + HF safety heads, mostly forward-only inference, overnight) + **Bedrock** (gpt-oss-safeguard and other hosted safety classifiers for breadth). Candidate additions: Llama Guard 2 / 1, ShieldGemma 2B/27B, Aegis/other WildGuard variants, OpenAI moderation-style heads, additional fine-tuned DeBERTa/Text-Moderation checkpoints at varying temperatures.

**Optional secondary finding (review addition):** with N≈15 you can test whether *more accurate* safety classifiers are systematically *harder to monitor* — a fundamental accuracy↔monitorability tension worth naming if it appears.

### C.3 Full build (conditional on Gate C)
Add classifier adapters (reuse the `ClassifierInterface` Protocol), cache null scores into `results/null_scores.json` (already keyed by classifier), extend `mechanistic_analysis.py` to fit ML2's regression with LOOCV. Pre-register the feature set before fitting.

### C.4 Deliverable
Focused paper: *"A monitorability law for safety classifiers"* (or the honest negative). Highest variance in outcome; cheapest gate, so resolve early.

---

## Compute Allocation

| Workload | Resource | Why |
|---|---|---|
| GCG / AutoDAN / PAIR gradient optimization (Track A) | **Mac Studio M3 Ultra** | needs gradients; DeBERTa 304M trivial, decoder GCG heavier but fits in 96 GB |
| Local 4-classifier scoring (all tracks) | **Mac Studio** | Llama Guard 8B + ShieldGemma 9B fp16 on MPS, fits comfortably |
| Conformal-martingale prototyping (Track B gate) | **Mac Studio CPU** | pure NumPy on cached score arrays |
| Classifier breadth for the monitorability law (Track C) | **Bedrock** | many hosted safety classifiers without local weights; gpt-oss-safeguard etc. |
| Multi-category taxonomy signals (A.4) | **Mac Studio** (local logits) + **Bedrock** (hosted Llama Guard variants) | needs token-level category logits |

GCG/attack optimization is gradient-bound → never on Bedrock. Bedrock is the *breadth* arm (Track C, hosted variants), not the *attack* arm.

---

## Execution Order

**Phase 0 — Gates (parallel, ~days):**
1. Gate B (cached scores, CPU, fastest) → is Track B's low-mixing claim real?
2. Gate C (overnight inference, Bedrock breadth) → does the monitorability law survive n=4?
3. Gate A (GCG corpus 22→100, Mac Studio GPU) → does cross-architecture divergence replicate?

**Phase 1 — Full builds** on whichever gates pass. Dead arms get written up as honest negatives (per the Agentic dead-arm protocol), not silently dropped.

**Phase 2 — Scope decisions** (§A.5, §B.4, §C.4): how many papers, and how each finding is framed.

---

## Reproducibility (lesson carried from the Agentic study)

The Agentic project's qwq non-replication traced to **unlogged engine/host state**. Avoid the same gap here:
- Log model digest/revision (HF commit hash), library versions (torch, transformers, scikit-learn), and device (MPS/Bedrock region) into every result record.
- For Bedrock classifiers, log `model_id`, region, and inference profile.
- Commit each track's pre-registration section (this file) before its gate executes; track deviations in an amendment, exactly as `pre_registration.md` + `amendment_2.md` do.
- All headline numbers verified programmatically (extend `scripts/verify_paper_numbers.py`) before any paper claim.
- **Citation hygiene (hard gate).** Verify every reference against the actual paper before it enters any draft — Vovk conformal test martingales, Howard & Ramdas / McAuliffe & Sejdinovic 2021, Waudby-Smith & Ramdas 2024, Sugiyama et al., Stojanov et al. For any 2026 pre-print, confirm it actually exists on arXiv before citing. This is non-negotiable given prior fabricated-citation incidents; treat it like the reproducibility-logging lesson.

---

## Framing Guardrails (don't overclaim)

- **Track A:** the canary detects *anomaly via divergence*, not *transferred attack success*. The second classifier is **not** fooled — say "cross-architecture divergence," never "the attack transfers." Gate cross-family generality (CA4) before any general "architecture diversity defeats evasion" claim.
- **Track B:** anytime-valid guarantees hold for the **growing/martingale** construction; the sliding window remains per-window only. State the memory/guarantee trade-off explicitly (AV3).
- **Track C:** r=0.97 at n=4 is a *hint*, not a result. The pre-registered bar is r>0.6 at N≥15. A null result is a deliverable, not a failure.
- **Multi-category:** scalar-invisible category shift is a *new phenomenon claim* — require ≥1 concrete category where the vector channel alarms and the scalar does not, with FAR controlled.
