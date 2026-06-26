# Paper Framing & Key Section Language (Final)

## Context: arXiv v2 update (not a new paper)

This is an update to arXiv:2606.11949 ("Online Shift Detection and Conformal
Adaptation for Deployed Safety Classifiers"). The original paper established
the factorial evaluation (800 cells), KS detection, conformal adaptation, and
a 22-example canary PoC. v2 replaces the weak KS calibration with the scan
martingale, expands the canary PoC into a full adversarial robustness
characterization, and corrects the monitorability claim.

Same arXiv ID. LinkedIn/GitHub/GitHub Pages auto-update.

---

## Abstract (~150 words)

Three assumptions underpin current safety classifier monitoring: (1)
architectural diversity strengthens ensembles, (2) out-of-distribution
detectors catch adversarial inputs, and (3) score-disagreement monitoring can
be suppressed by an adaptive attacker. We empirically test all three.
Assumption (1) is false: within-family canaries produce the strongest detection
signal (η²=0.011 for architecture type). Assumption (2) is false:
random-token perturbations produce far less divergence than gradient-optimised
attacks (p<10⁻¹², Mann-Whitney), showing the signal is attack-specific, not
generic anomaly. Assumption (3) is partially false: when the canary is
confident, a divergence-minimising attacker stalls at a predicted equilibrium
(gap=1/(2λ), validated within 95% CI), but succeeds via joint-flip when the
canary is uncertain. We characterize the exact security boundary across a
4-tier threat model and provide a calibration-free sequential monitor (scan
martingale, FAR≤1%) requiring no per-classifier tuning.

---

## One-sentence pitch

"We characterize the exact conditions under which classifier-disagreement
monitoring detects gradient-based evasion, identifying a confidence-gated phase
transition that quantitatively predicts when the defense holds (gap stalls at
1/(2λ)) and when it fails (attacker succeeds via joint-flip)."

---

## §1 Introduction — "Ladder of Misconceptions" structure

**Opening hook (architecture diversity falsification):**

> The prevailing assumption in multi-model safety monitoring is that
> architectural diversity fortifies ensembles against evasion attacks — that
> deploying an encoder alongside a decoder creates a fundamentally harder target
> for an adversary. We systematically falsify this assumption. Across 49
> gradient-optimised attacks and 6 classifier pairs, architectural heterogeneity
> is completely orthogonal to attack detection (η² = 0.011). The true
> determinant of monitoring robustness is not model family, but the canary
> classifier's baseline confidence on the input.

**Problem statement:**

> Safety classifiers deployed at scale face gradient-based evasion (GCG, Zou et
> al. 2023). Fine-tuned classifiers are robust to template-based jailbreaks (1%
> success rate in our evaluation), making gradient attacks the remaining threat
> class. We ask: can a second, un-targeted classifier detect when the primary
> classifier is under attack — and under what conditions does this detection
> fail?

**Contributions (numbered):**

> 1. We prove that score disagreement between a targeted and un-targeted
>    classifier is attack-specific (p<10⁻¹², n=49; random-token controls show
>    significantly less divergence).
> 2. We identify a confidence-gated phase transition: when the canary
>    classifier is confident (B_orig ≥ 0.5), adversarial optimization stalls at
>    a predictable equilibrium (gap = 1/(2λ), empirically validated within 95%
>    CI). When uncertain, the attacker succeeds via joint-flip.
> 3. We map the complete security boundary across a 4-tier threat model, from
>    monitor-ignorant attackers (98% detection) to omniscient joint optimisers
>    (70% success within-family, structurally penalised cross-family).
> 4. We provide a calibration-free sequential monitor (scan martingale,
>    FAR ≤ 1% across all classifiers) that requires no per-classifier threshold
>    tuning.

---

## Section-level framing language (use verbatim or adapt)

### CA4 (§4.3 or wherever the architecture result goes)

> We demonstrate that contrary to prevailing assumptions in multi-model defense
> literature, architectural heterogeneity is completely orthogonal to passive
> attack detection (η² = 0.011, F = 3.30, p = 0.07). The true active ingredient
> is weight-configuration divergence: any non-targeted model — even a
> homogeneous checkpoint from an earlier training epoch — generates a robust
> divergence signature under single-target optimisation. The highest divergence
> in our evaluation (0.899) occurs within-family (DeBERTa ↔ Text-Moderation),
> not cross-family.
>
> Architecture diversity IS relevant for a different property: transfer
> robustness. Within-family canaries suffer 30–37% passive transfer (the attack
> accidentally fools both). Cross-family canaries (different tokeniser) show 0%
> transfer. The deployment recommendation is therefore: deploy any second
> classifier for detection, but prefer cross-family for robustness against
> adaptive attackers.

### Track C (Appendix or §7.2)

> We prove that "monitorability" cannot be reduced to a scalar property of a
> classifier's null-distribution score geometry. The correlation between
> null-score standard deviation and detection latency — reported as r = 0.97 at
> n = 4 in our original evaluation — collapses to r = 0.21 (p = 0.70) when
> tested within the encoder family (n = 6). The original correlation was an
> artifact of the encoder/decoder architectural gap, not an intrinsic
> monitorability law. This falsification serves as a caution against using
> static score-distribution metrics to predict runtime detection difficulty.

### Scan Martingale (§6)

> The practical utility of the scan martingale lies entirely in its
> parameter-free deployment profile. While sliding-window KS detection requires
> per-classifier empirical calibration — and produces a 5× FAR spread across
> our 4 classifiers (2% to 9.5%) — the scan martingale provides a strict,
> anytime-valid guarantee (FAR ≤ 1%) out-of-the-box via Ville's inequality and
> union-bound correction. It does not outperform a perfectly calibrated KS test
> in detection power; its contribution is operational simplicity for deployment
> teams who cannot afford per-model threshold tuning.

### Divergence-minimisation (§5.3)

> [See docs/paper/section5_3_draft.md for full section]

---

## What the paper does NOT claim (state in §7 Limitations)

- We do NOT claim the defense is absolute or universally robust.
- We do NOT claim architecture diversity is useless (it provides transfer
  robustness, just not detection signal).
- We do NOT claim the scan martingale outperforms calibrated KS detection.
- We do NOT claim the 1/(2λ) stall is a formal theorem (it is a continuous
  relaxation that matches discrete data within CI).
- We do NOT claim our findings generalise beyond the 4 classifiers and 49
  GCG suffixes evaluated (state scale as limitation).

---

## Title options (ranked)

1. "The Confidence-Gating Frontier: When Score Disagreement Monitoring Detects
   Gradient-Based Evasion and When It Cannot"
2. "Cheap Canaries: Detecting Targeted Evasion Attacks via Classifier Score
   Disagreement"
3. "Characterizing the Security Boundary of Multi-Classifier Safety Monitoring
   Under Adaptive Adversaries"

Recommendation: #2 for accessibility (workshop), #1 for precision (SaTML main).

---

## Operational framing (weave into Discussion §7)

**Alert fatigue:** Standard OOD detectors flag any statistical anomaly,
drowning analysts in false positives. Score disagreement is surgically
specific: silent under random/gibberish noise, triggers only under directed
gradient optimisation (CA6, p<10⁻¹²). This directly addresses the
alert-fatigue problem that kills production monitoring deployments.

**Work factor inflation:** A naive single-target GCG attack takes ~50 steps.
Adding a canary forces the attacker into multi-objective joint optimisation
with conflicting gradient vectors and (for cross-family) a 1.73× tokeniser
fragmentation penalty. We do not claim an unbreakable defense — we quantify
the computational tax imposed on the adversary (from trivial single-model GCG
to expensive, constrained multi-objective search).

---

## arXiv v2: what changes from v1

| Section | v1 (current) | v2 (update) |
|---------|-------------|-------------|
| Abstract | Monitoring + detection | + three falsified assumptions, phase transition |
| §1 Intro | Drift detection gap | + "triple disillusionment" hook, gradient evasion as primary threat |
| §2 Approach | KS + conformal | + §2.4 Scan martingale |
| §3 Results | 800-cell factorial | Unchanged (still valid) |
| §4 Operational | 5 recommendations | Rewrite #5 → full threat model + decision matrix |
| §5 Methodology | Stats, corpus | + methodology for new experiments |
| §6 Post-Factorial | CS, MMD, mechanistic | Replace "r=0.97 mechanistic" with honest negative. Add martingale eval (AV2/AV5/AV6). |
| **§7 NEW** | — | **Canary Detection** (CA6, k-scaling, target-specificity, AutoDAN) |
| **§8 NEW** | — | **Adversarial Robustness** (threat model, transfer, confidence-gating, joint evasion, divergence-min, tokeniser barrier) |
| §9 Limitations | Original list | Update with new honest limits + what we don't claim |

**v1 corrections:**
- ~~"scores shift toward unsafe"~~ → "target collapses while non-target holds"
- ~~"architecturally-different classifier"~~ → "any un-targeted classifier"
- ~~"22 examples... validation needed"~~ → full n=49 characterisation
- ~~"r=0.97 mechanistic"~~ → "falsified at n=8 (r=0.21, p=0.70)"
- ~~"empirically calibrated at 97th percentile"~~ → scan martingale (no calibration)

**Stays unchanged:** §3 factorial results (800 cells), corpus validation, post-factorial CS/MMD/PCA

---

## Venue & timeline

**Target:** SaTML 2026 (check deadline) or AISec @ CCS 2025
**Backup:** NeurIPS SafeGenAI workshop 2025
**arXiv v2:** push after paper draft is complete (same ID: 2606.11949)
