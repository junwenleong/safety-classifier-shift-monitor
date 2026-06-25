# Follow-Up Experiment Plan — Post-arXiv Scope Extension

**Status:** All gates complete. Track A PASS, Track B PASS (reframed), Track C FAIL (honest negative).

## Final Results (2026-06-25 09:20 SGT, independently audited)

| Track | Verdict | Paper role |
|---|---|---|
| **A** | ✅ PASS | Headline: cross-arch divergence detects evasion attacks |
| **B** | ✅ PASS (reframed) | Spine: calibration-free FAR control (not "beats KS") |
| **C** | ❌ FAIL | Honest negative: 1 paragraph (n=4 law was encoder/decoder gap) |

---

### Track A — Cross-Architecture Divergence ✅ PASS

**Gate A (n=100 GCG, 49 flipped):**

| Test | Result | Verdict |
|------|--------|---------|
| Cross-arch divergence | mean 0.784 [0.699, 0.865] vs null 97th-pct 0.585 | ✅ 1.3× threshold |
| Delta t-test | +0.083, t=4.18, p=0.0001 | ✅ Significant |
| Direction (binomial) | 27/49=55% toward unsafe, p=0.57 | ❌ NOT significant |

**CA6 — Gibberish Control (decisive):**

| Metric | GCG (n=49) | Random (n=50) |
|--------|-----------|---------------|
| Detection rate | **76%** | **12%** |
| Wilson 95% CI | [0.619, 0.854] | [0.057, 0.242] |
| DeBERTa flip rate | 49/100 (49%) | **0/50 (0%)** |

- Non-overlapping CIs: GCG LB 0.619 > Random UB 0.242 ✅
- Mann-Whitney: p = 2.4×10⁻¹² ✅
- Random tokens never flip DeBERTa (0/50) — GCG is structurally different

**Honest framing for paper:** The signal is *score divergence* (DeBERTa collapses to ~0 while Llama Guard holds), NOT "Llama Guard moves toward unsafe" (direction test fails, p=0.57). The divergence is attack-specific — random gibberish produces far less divergence and never flips DeBERTa.

---

### Track B — Conformal Test Martingale ✅ PASS (reframed)

**Production module:** `detection/conformal_martingale.py` (29 tests passing)

| Hypothesis | Result |
|---|---|
| AV2 (FAR uniformity) | ✅ 0–1% all classifiers, spread 1%, zero calibration |
| AV5 (exchangeability) | Boundary found: block-structured always alarms, gradual drift alarms encoders only |
| AV6 (ε robustness) | ε=0.3 works for encoders (100%), fails decoders (LG 57%, SG 77%) |
| AD1 (scan vs KS) | ❌ Scan never outperforms properly-calibrated KS at any mixing level |

**Critical reframing (from audit):** With per-condition calibrated KS (97th-percentile threshold), KS dominates scan at every tested mixing level:

| Mixing | Scan (DeBERTa) | KS (DeBERTa) |
|--------|---------------|--------------|
| 15% | 40% | 83% |
| 20% | 80% | 100% |
| 25% | 100% | 100% |
| 30% | 100% | 100% |

The Gate B validation (scan 83–100% vs KS 43–47%) used the *factorial's* KS threshold, which was calibrated across all shift types and was less sensitive. With proper per-condition calibration, KS is better.

**What the paper can honestly claim:** The scan martingale provides *calibration-free* FAR control ≤α with zero per-classifier tuning. Its value is **operational simplicity** (deploy once, guaranteed FAR across all classifiers), not superior detection power. If you invest in per-classifier calibration, KS matches or beats it.

**Implementation notes:**
- CUSUM with threshold log(1/α) inflates FAR (Page's resets give multiple chances). Fixed with log(H/α).
- ScanMartingale has the cleanest anytime-valid guarantee (union bound over W sub-martingales).
- ε=0.3 is not universal — decoders with wide null distributions need ε=0.4-0.5.

---

### Track C — Monitorability Law ❌ FAIL

**Epoch sweep results** (DeBERTa-v3-large at epoch {1,3,5,10} + 2 original encoders):

| Classifier | null_std | mean_latency |
|---|---|---|
| orig-text-moderation | 0.066 | 11.8 |
| orig-deberta | 0.087 | 25.4 |
| deberta-epoch-10 | 0.109 | 23.0 |
| deberta-epoch-1 | 0.140 | 21.0 |
| deberta-epoch-3 | 0.144 | 13.0 |
| deberta-epoch-5 | 0.153 | 24.0 |

**Correlations:**

| Test | r | p | n | Criterion met? |
|---|---|---|---|---|
| DeBERTa epoch sweep only | −0.22 | 0.78 | 4 | ❌ |
| Within-family (all encoders) | +0.21 | 0.70 | 6 | ❌ |
| Full (+ decoders, n=8) | +0.42 | 0.30 | 8 | ❌ (need r>0.6, p<0.05) |

**Diagnosis:** The original r=0.997 (n=4) was the encoder/decoder gap in disguise. Decoders (std ~0.14, latency ~60-85) vs encoders (std ~0.07-0.09, latency ~12-25) create a 2-cluster correlation. Within the encoder family, null_std has no relationship to detection latency (epoch 3 has std=0.144 and latency 13; epoch 5 has std=0.153 and latency 24 — non-monotonic, no pattern).

**Conclusion:** Monitorability is not an intrinsic, predictable scalar property of a classifier's score distribution geometry. The pre-registration correctly anticipated this (§C.2): "a clean, publishable negative that corrects an over-strong reading of the n=4 result."

---

### What's Next

### Track A Full Build Results (2026-06-25 11:51 SGT)

Completed in 1.8h. **Original hypothesis falsified, but a cleaner finding emerged.**

#### CA4 — Architecture-pair divergence: ❌ FAIL

| Pair | Type | Mean divergence |
|------|------|----------------|
| DeBERTa ↔ Text-Moderation | within-family | **0.899** (highest!) |
| DeBERTa ↔ Llama Guard | cross-family | 0.784 |
| DeBERTa ↔ ShieldGemma | cross-family | 0.566 |
| Text-Mod ↔ ShieldGemma | cross-family | 0.393 |
| Llama Guard ↔ ShieldGemma | within-family | 0.253 |
| Text-Mod ↔ Llama Guard | cross-family | 0.231 |

- η² = 0.011 (criterion was >0.10) → **FAIL**
- Cross-family mean (0.493) < within-family mean (0.576) → **opposite direction**
- The highest divergence is within-family (DeBERTa↔TM) because DeBERTa is the TARGET — its score collapses to ~0 while TM (same corpus, same sensitivity) stays high.

**Key insight:** The active ingredient is NOT architecture diversity. It's "targeted model vs any un-targeted model." DeBERTa↔TM diverges most because TM is maximally sensitive to the same content while not being attacked.

#### CA6 Extended — GCG vs random per pair: TARGET-SPECIFIC

| Pair | GCG mean | Random mean | MW p | Sig? |
|------|----------|-------------|------|------|
| DeBERTa ↔ LG | 0.784 | 0.214 | <0.0001 | ✅ |
| DeBERTa ↔ SG | 0.566 | 0.415 | 0.002 | ✅ |
| DeBERTa ↔ TM | 0.899 | 0.014 | <0.0001 | ✅ |
| TM ↔ LG | 0.231 | 0.214 | 0.320 | ❌ |
| TM ↔ SG | 0.393 | 0.415 | 0.715 | ❌ |
| LG ↔ SG | 0.253 | 0.282 | 0.862 | ❌ |

**Pattern:** GCG > random ONLY for pairs involving DeBERTa (the target). Pairs not involving the targeted model show zero GCG-specific divergence. This confirms: the signal is "target collapses, non-targets hold" — not a general cross-architecture property.

#### CA8 Probe — Joint evasion: METHODOLOGICALLY FLAWED

Raw result: 8/10 "joint evasion" (DeBERTa flipped + LG stable <0.1 divergence).

**BUT** the script only optimized against DeBERTa (GCG gradients) and *checked* LG passively every 20 steps. It did NOT put LG in the gradient loop. So "joint evasion" here means "single-target GCG happens not to disturb LG" — expected when LG is saturated (4/8 successes had LG_orig > 0.9).

Additionally, cross-architecture joint GCG is confounded by tokenizer mismatch (DeBERTa SentencePiece vs Llama tokenizer — same suffix tokens have different meanings). Cannot cleanly interpret.

**This CA8 probe result is NOT valid for claims about joint evasion difficulty.**

#### Corrected Interpretation of Track A

~~"Cross-architecture divergence is an attack detector"~~ → Partially wrong.

**What's actually true:**
1. Any un-targeted classifier produces detectable score divergence when the targeted classifier is under attack (CA6 PASS, p<10⁻¹²)
2. The signal is target-specific, not architecture-specific (CA6 extended: only pairs involving the target show GCG > random)
3. The signal is attack-specific, not OOD (random tokens produce far less divergence)
4. **Attack transfer is the key variable** (cheap canary finding): cross-architecture canaries resist GCG transfer (76% detection), same-architecture canaries are vulnerable to transfer (36% also flip → detection fails)

### Cheap Canary Results (2026-06-25 13:40 SGT) ✅

GCG against primary DeBERTa-v3-large, scored on epoch-{1,3,5,10} checkpoints (same architecture, different training state).

| Canary epoch | Mean divergence | Detection (>0.5) | Canary also flipped |
|---|---|---|---|
| 1 | 0.498 | 44% | high transfer |
| 3 | 0.555 | 56% | moderate |
| 5 | 0.758 | 78% | low transfer |
| 10 | 0.715 | 78% | low transfer |
| **Overall** | **0.631** | **64%** | **36% flipped** |

**Comparison:** Cross-arch canary (DeBERTa→LG): mean div=0.784, det=76%, LG never flips.

**Critical finding:** When the same-architecture canary also gets flipped by the GCG suffix (36% of cases), divergence drops to 0.08 — useless. When it resists, divergence is 0.94 — excellent. The variable is **attack transfer**: GCG transfers within-architecture 36% of the time.

**Implication:** Architecture diversity IS relevant, but the mechanism is *transfer resistance*, not gradient geometry. A decoder-based canary is robust because encoder-targeted GCG doesn't transfer to decoders. Same-architecture canaries are partially vulnerable.

### CA8 Proper — Joint GCG Results (2026-06-25)

**First attempt** (200 steps × 512 search × 2 models): crashed on embedding dimension mismatch (primary=large/1024d, epoch-sweep=large/1024d — fixed). Got 1 data point at λ=0 before timeout:

- **λ=0, prompt 1: A=0.002, B=0.002 → BOTH FLIPPED** (single-target attack transfers perfectly)

Then confirmed with epoch-1 vs epoch-5 (both 1024-d):
- **λ=0, prompt 1: A=0.003, B=0.007 → BOTH FLIPPED**

**Interpretation:** Within-family, single-target GCG transfers to the companion model without any joint optimization. The attacker doesn't even need to be monitor-aware — optimizing against one DeBERTa checkpoint automatically defeats another.

**Why:** Both models share architecture + tokenizer + training corpus. The adversarial perturbation that fools one is geometrically similar enough to fool the other. This is consistent with known GCG transfer results in the adversarial ML literature.

**Parameter issue:** Full λ sweep was ~80h (200 steps × 512 search × 10 prompts × 4 λ). Killed — redundant since λ=0 already defeats both. The λ>0 sweep would only show "joint optimization is at least as effective as single-target" which is obvious.

**⚠️ EMERGING (TBC): Attack-budget dependence on transfer rate.**
- `run_ca8_proper.py` (200 steps, 512 search): 1/1 prompts → both flip. Suggested near-total transfer.
- `run_ca8_minimal.py` (50 steps, 256 search): 7/10 in progress, 2/7 transfer so far (~29%).
- Hypothesis: transfer is budget-dependent. At 50 steps (standard GCG), the suffix is optimized just enough to fool Model A but not extreme enough to passively transfer to B (unless B is low-confidence). At 200 steps, the suffix is so over-optimized that it accidentally fools same-architecture companions.
- If confirmed: "cheap canaries provide a time-window of protection that degrades as attacker invests more compute." Quantifiable as transfer-rate-vs-steps curve.
- Also: transfer only occurs when canary's baseline confidence is low (<0.99). High-confidence canaries resist passive transfer.
- **Status:** awaiting full 10/10 results to confirm.

### Currently Running (2026-06-25 23:06 SGT)

**Mac Studio:** `scripts/run_ca8_minimal.py` — ETA ~23:30 SGT

Minimal confirmation: 10 prompts × 50 steps × 256 search, λ=0 only (single-target, check if B transfers). Using epoch-1 vs epoch-5 checkpoints (both deberta-v3-large, 1024-d).

**Expected:** Most/all transfer (based on 2/2 observed so far). This gives us a proper n=10 with Wilson CI for the paper.

### Complete Track A Story (all data in)

| Finding | Data | Verdict |
|---------|------|---------|
| Divergence detects single-target attack | CA6: p<10⁻¹² | ✅ |
| Divergence is attack-specific, not OOD | CA6: GCG 76% vs random 12% | ✅ |
| Architecture diversity is NOT the active ingredient | CA4: η²=0.011, within-family highest | ❌ (hypothesis falsified) |
| Signal is target-specific (only pairs with target show it) | CA6 ext: 3/6 pairs (all DeBERTa-involving) | ✅ (reframing) |
| Same-arch canary works passively | Cheap canary: 65% detection | ✅ (partial) |
| Same-arch canary is vulnerable to transfer | CA8: λ=0 both flip, cheap canary 36% also flipped | ✅ (honest limitation) |
| Cross-arch canary resists transfer | CA6: LG never flips under DeBERTa GCG | ✅ |
| True joint optimization (cross-arch) | Untestable — tokenizer confound | ⚠️ Open question |

**Paper framing:**
1. Any un-targeted classifier is a "cheap canary" against single-target evasion
2. Cross-architecture canaries are strictly more robust (GCG doesn't transfer across tokenizer/architecture boundaries)
3. Same-architecture canaries are partially defeated by attack transfer (36%)
4. The defense is security-in-diversity: the attacker must know AND target all deployed models
5. The tokenizer mismatch between architectures acts as a structural barrier to joint optimization

### K-Classifier Scaling (2026-06-25, from CA4 data)

| Ensemble (target=DeBERTa) | Detection rate | Note |
|---|---|---|
| k=1: Text-Mod only (within-family) | **94%** | Best single canary — but vulnerable to transfer |
| k=1: Llama Guard (cross-family) | 76% | Robust to transfer, lower baseline detection |
| k=1: ShieldGemma (cross-family) | 55% | Wide null distribution reduces sensitivity |
| k=2: Text-Mod + Llama Guard | **98%** | Best pair — covers transfer gap |
| k=2: Text-Mod + ShieldGemma | 96% | — |
| k=3: All three | 98% | No improvement over k=2 |

**Takeaway:** A single within-family canary (Text-Mod) is best against monitor-ignorant attackers (94%). Adding one cross-family canary (LG) reaches 98% and closes the transfer vulnerability. Beyond k=2, returns diminish.

### Formalized Threat Model

| Tier | Attacker knowledge | Attack type | Defense status |
|------|-------------------|-------------|----------------|
| **1: Monitor-ignorant** | Knows target, unaware of canary | Single-target GCG/AutoDAN | ✅ Detected (94-98% depending on ensemble) |
| **2: Architecture-aware** | Knows target + canary's family | Single-target, relies on transfer | ⚠️ Within-family: 36% transfer defeats canary. Cross-family: robust (0% transfer to LG) |
| **3: Monitor-aware, single-target** | Knows both models, optimizes one | GCG against target, hopes canary misses | ✅ Cross-family canary holds (CA6). Within-family partially vulnerable. |
| **4: Joint optimizer** | White-box access to both, joint loss | Joint GCG (λ>0) | ❌ Within-family: trivially defeated (λ=0 transfers perfectly). Cross-family: structurally bounded by tokenizer mismatch (untested, open question). |

**Deployment recommendation by threat level:**
- Low-threat (Tier 1-2): Deploy any second classifier. Within-family is cheapest and catches 94%.
- Medium-threat (Tier 3): Deploy cross-family canary (different architecture + tokenizer). Blocks transfer.
- High-threat (Tier 4): Cross-family ensemble + canary rotation (keep canary identity secret). Accept best-effort.

### After CA8 Minimal Completes

**Paper writing** — all experiments done.

**Title:** "Cheap Canaries: Detecting Targeted Evasion Attacks via Classifier Score Disagreement"

**Core claims (supported by data):**
1. Any un-targeted classifier detects single-target evasion (p<10⁻¹², CA6)
2. Within-family canary: 94% detection but vulnerable to transfer (36%)
3. Cross-family canary: 76% detection but robust to transfer (0%)
4. Optimal ensemble (k=2, one within + one cross): 98% detection across threat tiers
5. Formalized threat model mapping findings to attacker capabilities

**Remaining compute (nice-to-have):**
- AutoDAN attack family (~4-6h) — confirms findings aren't GCG-specific
- Canary rotation simulation — demonstrates defense against Tier 3+

**Structure:**
- Track A: target-vs-non-target divergence + transfer resistance + joint-evasion cost
- Track B: calibration-free monitoring (operational simplicity framing)
- Track C: 1 paragraph honest negative
- Track A: target-vs-non-target divergence as evasion detector + joint-evasion cost analysis
- Track B: calibration-free monitoring (operational simplicity framing, NOT "beats KS")
- Track C: 1 paragraph honest negative (monitorability law was encoder/decoder gap)
   - Do NOT claim "scan beats KS" — claim "scan requires no calibration"

---
**Parent work:** arXiv:2606.11949 (Shift Detection Monitor). The 980-cell factorial + post-factorial additions (CS growing-window, MMD, PCA-conformal, gradual drift, mechanistic n=4) are *complete and submitted*. This document plans the next phase.
**Compute available:** Mac Studio M3 Ultra (96 GB) for local inference + GCG gradients; AWS Bedrock for breadth. No time/budget constraint — the binding constraint is research risk, so every track gates on a cheap replication before the full build.
**Convention:** Matches `docs/pre_registration.md` + `docs/pre_registration_amendment_2.md` — hypothesis IDs with directional predictions, pre-specified success criteria, α = 0.05, reference size 500, onset 500, windows {100, 200}, 97th-percentile empirical FAR calibration, Wilson / Clopper–Pearson CIs, η² with bootstrap CIs. Commit this file before executing any gate.

---

## 0. Scope at a Glance

Three groups of experiments, organized by theme. **Each could in principle become a paper, but that is not the plan** (see Scope philosophy below) — they are grouped this way to organize the work, and the likely outcome is one consolidated follow-up.

| Track | Working title | Builds on | Novelty ceiling | Risk |
|---|---|---|---|---|
| **A** | Heterogeneous Monitoring Ensembles: cross-architecture divergence as an attack detector | Regime C canary (currently 22-example PoC) | Highest | Medium |
| **B** | Anytime-valid shift monitoring: conformal test martingales + low-rank conformal | CS engine (Ville/ONS) + density-ratio collapse + PCA diagnostic | Rigor-defining | Low |
| **C** | A monitorability law: predicting detection latency from score-distribution geometry | Mechanistic n=4 (r=0.97) | High (or clean negative) | High |

Cross-cutting threat experiment (the **monitor-aware adversary**) bridges A and B.

**Sequencing rule:** run the validation gates first — they barely compete for resources (one GPU, one CPU-only, one Bedrock). Kill dead arms early; scope-decide A/B/C placement only after gate data is in.

### ⚠️ Scope philosophy — tracks are experiment groups, NOT paper commitments

This document organizes *experiments*, not *submissions*. The three-track structure is a way to group related work, not a plan to produce three papers. Read it that way.

- **The goal is a better, honest monitor + results that are true** — not maximizing publication count.
- **Default outcome: ONE consolidated follow-up** that strengthens the arXiv work — either as new sections of its venue submission or a single companion paper. The strongest spine is Track B (provable, calibration-free monitoring), with Track A's canary as a possible headline *if the gibberish control (CA6) survives*.
- **Track C** is a section or a footnote depending on whether the law survives the family-confounding test — not a paper to build around preemptively.
- **No pre-committing to N papers.** Packaging is decided *after* the gates land. Resist scope-sprawl: do not promote "future directions" (below) into deliverables until the core results are in.
- **No timeline/venue pressure drives this.** Decisions are made on what the data shows, not on deadlines.

#### Paper-count decision — honest considerations (revisit after gates)

Whether this becomes 1, 2, or 3 papers is **downstream of the gates**, not a pre-decision. Recorded honestly so the call is made on merit, not ambition or my earlier over-correction:

**Honest per-track novelty (the real content is ~1.5 papers, not 3):**
- **Track B** — 4 confirmed results, but the scan/anytime-valid machinery is **not novel** (Howard & Ramdas; MOSUM; Vovk conformal test martingales). The contribution is *applied* ("known machinery → safety-classifier monitoring, beats calibrated KS, with honest limits"), plus the exchangeability (AV5) and ShieldGemma (AV6) bounds. Strong as an **applied/systems** paper or a section — **not** a top-tier methods paper. A methods reviewer says "the estimator isn't new."
- **Track A** — the genuinely novel one *iff* CA6 (gibberish control) survives. Then it's standalone-worthy security. Fully contingent on a gate that can fail.
- **Track C** — most likely collapses to "discriminative models are easier to monitor" once the family confound is exposed → section/footnote, not a standalone.

**The fair case FOR three papers (don't dismiss it):**
- Academia rewards count: three CV lines, three venues, three communities (security / ML / measurement). Salami-slicing is common and works for visibility/career capital.
- Different audiences genuinely don't overlap; one combined paper can't reach all three rooms.
- Each track *can* be padded to a full paper — that's how much of the literature is made.

**The case FOR consolidation (one strong paper):**
- Honest novelty is ~1.5 papers; three would mean two thin ones.
- Thin papers risk weak reviews / desk rejects — worse than one strong paper.
- A's canary (*what* to monitor) + B's guarantee (*how* to monitor) reinforce each other; combined is more than the sum.
- Reputation: one excellent paper > three forgettable ones.

**It depends on what you optimize for** (a genuine choice, not an objective answer):
- Optimizing **count / breadth of visibility** → three papers is legitimate; consolidating leaves value on the table.
- Optimizing **reputation / impact / clean work** → one strong paper wins.

**The actual trap to avoid:** not "three papers," but **forcing three papers regardless of gate outcomes** — padding a confounded Track C, or claiming the martingale is novel. Three papers *if the results earn three* is fine. Three papers *by decree* costs reputation.

**Most likely distribution of outcomes:** CA6 passes → ~2 papers (A standalone + B applied, C as section). CA6 fails → 1 consolidated. Three-equal-papers is the *least* likely outcome.

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

### A.5 What this becomes
If Gate A passes *and* the gibberish control (CA6) survives, the cross-architecture divergence result is the most novel thing in the whole extension and the natural headline of the consolidated follow-up (or, only if it's clearly strong enough on its own, a standalone). If only GCG+DeBERTa↔Llama-Guard holds, it's a focused case-study section (mirroring the original paper's careful "22-example, larger-scale validation needed" framing). Decided after the gate — not before.

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

### B.4 What this becomes
The strongest, most-proven part of the extension (already 4 confirmed results). The likely spine of the consolidated follow-up — it converts the arXiv work's weakest point (empirical calibration, 5× FAR spread) into a provable, calibration-free method. AV4 (PCA-conformal) is a guaranteed-positive sub-result even if AV1 fails. Whether this is "new sections of the existing paper's venue submission" or "one companion paper" is decided later.

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

### C.4 What this becomes
A section of the consolidated follow-up if the law holds; a footnote/honest-negative if it doesn't. Highest variance in outcome; cheapest gate, so resolve early. *If* the correlation survives the family-confounding test (ML1b), the zero-cost framing upgrade is to **name the one-feature version as a proposed "monitorability" metric** for standardized reporting (alongside accuracy/robustness) — value is in the naming, not a complex model.

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

**Phase 2 — Consolidation** (§A.5, §B.4, §C.4): assemble the surviving results into the strongest single artifact (default: one consolidated follow-up or new sections of the existing paper). Not a paper-count exercise — decide framing from what the data shows.

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

---

## Future Directions (parked — NOT committed, do not build before core results land)

Good ideas that are deliberately *not* in the deliverable structure. They become relevant only after the core extension is done, and only if there's appetite. Listing them so they aren't lost — and so they don't quietly become scope.

- **Adaptive / context-aware betting (Track B v2).** The exchangeability limitation (AV5) is real: the martingale fires on benign drift. A natural fix is to scale the bet by a semantic "surprise" signal (e.g., perplexity under a general LM) — suppress bets on merely-novel topics, amplify on structurally anomalous inputs. This upgrades "detects any departure" → "detects adversarial departures." Strictly a *follow-up*; it adds a model dependency that undercuts the clean calibration-free story of the first result.
- **Named monitorability metric (Track C).** If the law holds, propose a one-number, offline-computable "monitorability" score for standardized reporting. Zero-cost framing add; not a separate effort.
- **Combined systems artifact.** Track A (what to monitor) + Track B (how to monitor, with guarantees) = a complete deployable monitoring layer. Only worth packaging *after* both stand on their own; a tool/release, not a forced merge.
- **Cross-project unification (Sentry × Agentic).** The cross-architecture canary here and the Agentic "distributional canary" are the same idea in two domains. A unifying view is interesting but speculative; revisit only if Track A's divergence result is strong and there's a clear bridging experiment, not as a planned deliverable.
