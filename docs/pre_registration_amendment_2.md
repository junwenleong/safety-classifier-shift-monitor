# Pre-Registration Amendment #2: Post-Factorial Additions

**Date:** 2026-06-08  
**Status:** Committed before execution of additional experiments  
**Relationship to original:** The original pre-registration (commit `be630f3`) and Amendment #1 (actual evaluation grid) cover the 980-cell factorial. This amendment documents experiments added after the factorial was complete, motivated by reviewer-anticipation analysis.

---

## Motivation

After completing the pre-registered factorial evaluation, external review identified gaps between the paper's theoretical framing and empirical demonstration. These additions close those gaps without re-running the main evaluation. All additions are clearly labeled as post-hoc.

---

## Addition 1: Holm-Bonferroni Multiplicity Correction

**Rationale:** The factorial results discuss 8 specific pairwise comparisons. Without multiplicity correction, the familywise error rate exceeds α.

**Method:** Holm-Bonferroni correction applied to 8 pre-specified comparisons:
1. Decoder vs Encoder on paraphrase (latency)
2. Encoder vs Decoder on adversarial suffix (latency)
3. DeBERTa: adversarial vs paraphrase (within-classifier crossover)
4. Llama Guard: paraphrase vs adversarial (within-classifier crossover)
5. Window size 100 vs 200 (paired difference)
6. Llama Guard × code-switch vs grand mean (slowest cell)
7. ShieldGemma: paraphrase vs adversarial (within-classifier crossover)
8. FAR: DeBERTa vs Text-Moderation (5× spread)

**Result:** All 8 survive at familywise α = 0.05. Weakest: comparison #2 (adjusted p = 0.044).

---

## Addition 2: CS Growing-Window Evaluation (Mac Studio)

**Rationale:** Paper cites Waudby-Smith & Ramdas (2024) confidence sequences. The factorial uses empirically calibrated sliding-window KS. The growing-window CS exists in code but has no empirical demonstration.

**Design:**
- 4 classifiers × 3 shift conditions (paraphrase, temporal, adversarial-suffix) × 10 seeds
- Growing-window CS with ONS betting strategy, bounded mode, α = 0.05
- Report: detection latency, FAR, comparison table vs sliding-window KS

**Hypothesis:** CS-driven detection will have comparable detection rate with formally controlled FAR (≤ α over infinite horizon).

---

## Addition 3: MMD Subset Evaluation (Mac Studio)

**Rationale:** Original design positioned MMD as a complementary channel catching shifts invisible to the score marginal. The factorial evaluates KS only.

**Design:**
- 4 classifiers × 3 shift conditions × 10 seeds
- Embeddings cached once, reused for both MMD and PCA-conformal
- Report: detection latency, KS-vs-MMD comparison, any conditions where MMD detects and KS does not

**Hypothesis:** At least one shift condition will produce detectable embedding-space shift while leaving the score distribution unchanged.

---

## Addition 4: PCA-Reduced Conformal (Mac Studio)

**Rationale:** Density-ratio collapse was identified as the dominant failure mode in RQ2. PCA to ≤32 dimensions breaks collapse on temporal shift. Need to verify generalization across shift types.

**Design:**
- 2 classifiers (Llama Guard, ShieldGemma) × 2 shift conditions (temporal, paraphrase) × PCA sweep (4, 8, 16, 32 dims)
- Report: coverage recovery, ESS, whether fix generalizes

**Hypothesis:** PCA-32 fix will generalize across shift types (mechanism is preventing linear separability, which is shift-agnostic).

---

## Addition 5: Gradual-Drift Experiment (MacBook Pro)

**Rationale:** All factorial cells use abrupt shift onset (step function). Real production drift is typically gradual.

**Design:**
- 1 classifier (DeBERTa) × 1 shift condition (paraphrase)
- Mixing proportion ramps linearly from 0% to 50% over 200 steps (starting at step 500)
- 20 seeds
- Report: detection latency compared to abrupt case, characterize sensitivity to drift rate

**Hypothesis:** Detection will be delayed relative to abrupt onset but still occur before mixing reaches 75% of max (i.e., before step 650, which corresponds to 75% of the 200-step ramp completing). This criterion means the monitor detects shift before the majority of production traffic is affected.

**Success criterion (pre-specified):** Detection latency < 150 steps (i.e., alarm fires before ramp reaches 75% of max mixing proportion) in ≥80% of seeds.

---

## Addition 6: Filtered Paraphrase Ablation (Mac Studio)

**Rationale:** 14-30% refusal contamination in paraphrase corpus may inflate detection signals.

**Design:**
- Manually filter refusals from paraphrase corpus (50-100 clean examples)
- 2 classifiers (DeBERTa, Llama Guard) × 5 seeds
- Report: detection latency on clean vs contaminated corpus

**Hypothesis:** Detection latency will increase slightly (less distributional signal) but detection will still occur.

---

## Addition 7: Lightweight Mechanistic Analysis (MacBook Pro)

**Rationale:** Paper documents the encoder/decoder crossover but does not explain why.

**Method:**
- Score distribution spread analysis (std of null scores by classifier)
- Detection latency vs embedding displacement scatter plot (using cached reference embeddings)
- No new experiments -- analysis of existing data

**Hypothesis:** Discriminative classifiers have sharper score boundaries (lower score spread under null) making them more sensitive to lexical perturbation. Generative classifiers smooth the boundary, delaying detection.

---

## Unchanged from Original Pre-Registration

- All hyperparameters (α=0.05, calibration percentile=97, reference size=500, shift onset=500)
- Statistical methodology (now BCa instead of percentile bootstrap, Holm-Bonferroni added)
- Primary metrics (detection latency, FAR, conformal coverage)
- Success criteria for RQ1, RQ2, RQ3
- All 980 factorial cells remain as the primary evidence

---

*This amendment is committed before execution of additions 2-7. Addition 1 (Holm-Bonferroni) was completed prior to this amendment using only existing data.*
