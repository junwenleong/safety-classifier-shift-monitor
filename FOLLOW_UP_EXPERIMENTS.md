# Follow-Up Experiment Plan — Post-arXiv Scope Extension

**Status:** Planning (pre-registration draft). Nothing executed yet.
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

### A.2 Validation gate (do this first)

**Gate A — divergence replication at n≈100.** Current corpus is ~22 GCG suffixes (`data/shifted/adversarial_suffix/deberta_suffixes.jsonl`). Scale to ~100 via `scripts/run_gcg.py` (DeBERTa target) and re-measure DeBERTa↔Llama-Guard divergence with `scripts/check_regime_c_direction.py`-style direction logic.

- **GO** if the +0.73-style divergence replicates with a Wilson lower bound above the null FAR at n≈100.
- **NO-GO** → write up as "PoC does not scale beyond 22 examples; cross-architecture divergence is corpus-specific" (honest negative, still informative). Stop Track A.

Compute: **Mac Studio** (GCG gradients on DeBERTa-v3-large 304M are cheap; Llama Guard 8B scoring is forward-only).

### A.3 Full build (conditional on Gate A)

1. **Divergence detector** — new `shift_detection_monitor/detection/divergence_detector.py`: maintains the bivariate (or k-variate) score-disagreement distribution against a frozen reference, alarms on KS/MMD of the disagreement series. Calibrate FAR on null streams (97th pct), same as KS.
2. **Second attack family** — AutoDAN or PAIR generator alongside GCG (CA3). New builder under `scripts/` + corpus under `data/shifted/adversarial_suffix/`.
3. **Architecture-pair sweep** (CA4) — all 6 pairs from {DeBERTa, Text-Moderation (encoders), Llama Guard, ShieldGemma (decoders)}; variance-decompose divergence by pair type. Reuse `run_variance_decomposition.py`.
4. **Multi-category channel** (CA5) — see §A.4.

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

### B.1 Pre-registered hypotheses

| ID | Statement | Direction | Success criterion |
|---|---|---|---|
| **AV1** | A conformal test martingale (betting on conformal p-values of the score stream) detects low-mixing drift (≤30%) at a higher rate than sliding-window KS. | martingale > KS | Detection rate at 30% mixing: martingale Wilson LB > KS Wilson UB (replicate the CS 97% vs KS 43% gap *with* a formal guarantee) |
| **AV2** | The martingale's e-value threshold (reject when wealth ≥ 1/α) controls FAR ≤ α **uniformly** across all 4 classifiers — eliminating the 5× empirical-FAR spread. | FAR ≤ α ∀ classifier | All 4 classifiers' null-stream FAR ≤ 0.05, no calibration; spread < 2× |
| **AV3** | A bounded-memory martingale variant retains the low-mixing advantage of growing-window CS without unbounded memory. | ≈ growing | Detection rate within 5 pp of growing-window CS at ≤30% mixing |
| **AV4 (PCA method)** | Low-rank projection (PCA to d ≤ 32) before density-ratio estimation restores effective sample size and conformal coverage; a data-driven dimension rule (from the ESS/separability curve) generalizes across shift types. | ESS↑, coverage↑ | Coverage recovery ≥ original temporal result (+33 pp Llama Guard, +20.5 pp ShieldGemma) on ≥2 unseen shift types; ESS > 50 |
| **AD1 (bridge)** | A **monitor-aware adversary** doing slow sub-threshold drift evades sliding KS but is still caught by the anytime-valid martingale. | martingale robust | KS misses (latency > horizon) while martingale alarms, on ≥80% of adversarial drift seeds |

### B.2 Validation gate (cheapest — do this first overall)

**Gate B — martingale on cached scores.** Prototype the conformal test martingale on **already-cached** score series (`results/regime_c_ks_series.json`, `results/cs_growing_window_results.json`, ramp-sweep caches). **Zero new inference.** Check whether it beats KS's 43% detection at 30% mixing.

- **GO** if martingale ≥ 70% at 30% mixing on cached data.
- **NO-GO** → martingale offers no low-mixing advantage; downgrade Track B to the PCA-method paper (AV4) only.

Compute: **Mac Studio CPU** (pure NumPy on cached arrays; minutes).

### B.3 Full build (conditional on Gate B)

1. `detection/conformal_martingale.py` — conformal p-values from the frozen reference CDF, betting martingale (reuse the `_log_wealth` ONS accumulation from `ConfidenceSequenceEngine`), alarm at wealth ≥ 1/α.
2. FAR-uniformity evaluation across 4 classifiers (AV2) — reuse `run_cs_evaluation.py` harness.
3. Bounded-memory variant (AV3) — windowed/decayed wealth; compare to growing-window CS.
4. PCA-conformal as a *method*, not a diagnostic (AV4): formalize dimension selection from the ESS-vs-d curve; extend `run_pca_conformal_sweep.py` to emit the rule and validate on held-out shift types.
5. Monitor-aware adversary (AD1, also a Track-A bridge): an attacker that ramps mixing to stay under the calibrated KS threshold; new `scripts/run_monitor_evasion.py`.

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
| **ML2** | A multi-feature monitorability score (std, IQR, kurtosis, boundary sharpness, calibration temperature) predicts latency on held-out classifiers. | predictive | LOOCV R² > 0.5 |
| **ML3** | The shift-specific sign reversal replicates: surface shifts give wider→slower; adversarial suffix reverses. | reversal | Sign(r) flips for adversarial-suffix vs paraphrase at N ≥ 15 |
| **ML4** | Embedding-displacement remains a non-predictor (negative control). | null | \|r\| < 0.3, p > 0.05 |

### C.2 Validation gate (Bedrock breadth — run in parallel)

**Gate C — extend beyond n=4.** Run the existing latency pipeline (`run_factorial.py` Regime A) on ~10–12 additional safety classifiers and recompute the correlation.

- **GO** if r > 0.6 (p < 0.05) at N ≈ 12–16.
- **NO-GO** → "monitorability is not an intrinsic, predictable property" — a clean, publishable negative that corrects an over-strong reading of the n=4 result.

Compute: **Mac Studio** (local fine-tuned encoders + HF safety heads, mostly forward-only inference, overnight) + **Bedrock** (gpt-oss-safeguard and other hosted safety classifiers for breadth). Candidate additions: Llama Guard 2 / 1, ShieldGemma 2B/27B, Aegis/other WildGuard variants, OpenAI moderation-style heads, additional fine-tuned DeBERTa/Text-Moderation checkpoints at varying temperatures.

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

---

## Framing Guardrails (don't overclaim)

- **Track A:** the canary detects *anomaly via divergence*, not *transferred attack success*. The second classifier is **not** fooled — say "cross-architecture divergence," never "the attack transfers." Gate cross-family generality (CA4) before any general "architecture diversity defeats evasion" claim.
- **Track B:** anytime-valid guarantees hold for the **growing/martingale** construction; the sliding window remains per-window only. State the memory/guarantee trade-off explicitly (AV3).
- **Track C:** r=0.97 at n=4 is a *hint*, not a result. The pre-registered bar is r>0.6 at N≥15. A null result is a deliverable, not a failure.
- **Multi-category:** scalar-invisible category shift is a *new phenomenon claim* — require ≥1 concrete category where the vector channel alarms and the scalar does not, with FAR controlled.
