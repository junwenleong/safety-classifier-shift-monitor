# §5.3 (draft) — The Divergence Phase Transition: When Can an Attacker Suppress the Canary?

**Framing: This section characterizes the exact boundary between detectable and
undetectable evasion. It does NOT claim the defense is absolute — it maps the
conditions under which it holds and fails, with a quantitative prediction.**

---

## The stealth objective

The strongest adversary (Tier 4) tries to flip A while keeping the canary signal
low. We formalise this as:

    L(δ) = f_A(x+δ) + λ · (f_B(x+δ) − f_A(x+δ))²

where λ=2.0 controls the stealth pressure.

## Empirical result

| Pair | Stealth (flip + div<0.5) | Blocked (A stuck >0.5) | Pattern |
|------|--------------------------|------------------------|---------|
| v1: epoch-1 vs epoch-5 | 4/10 | 6/10 | Blocked when B confident |
| v2: epoch-3 vs epoch-10 | [PENDING] | [PENDING] | Same pattern expected |

**Two distinct outcomes on the same loss function:**
- **Blocked cases:** B is confident (B≈1.0). A stalls at ~0.75. Gap = 0.25.
- **Stealth cases:** B is also uncertain. Both models flip together (joint-flip path). Div ≈ 0.

This is NOT "defense works" vs "defense fails." It is a **phase transition**
governed by a measurable input property: the canary's baseline confidence.

## The mechanism: a confidence-gated phase transition

The combined gradient:

    ∇L = [1 − 2λ(f_B − f_A)] · ∇f_A  +  2λ(f_B − f_A) · ∇f_B

**The coefficient [1 − 2λ(f_B − f_A)] inverts sign at gap = 1/(2λ).**

At λ=2: the critical gap is 0.25. Beyond this point, the optimizer is driven to
INCREASE f_A — the defense hijacks the attacker's optimization path.

Empirical validation: all 6 blocked cases in v1 stall at gap ∈ [0.165, 0.251],
mean 0.235 ≈ predicted 0.250. The theory quantitatively predicts where the
attack stalls.

**The gradient-norm effect (why confident canaries are immune):**
When B is deep in its confident basin, ‖∇f_B‖ → 0. The condition for the descent
step to reduce f_A requires:

    cos(θ) ≥ (1 − 1/[2λ(f_B−f_A)]) · ‖∇f_A‖ / ‖∇f_B‖

As ‖∇f_B‖ → 0, the RHS → ∞. No angle satisfies this — the optimizer is
completely paralyzed. The defense is strongest exactly when the canary is
most confident.

## What the paper claims (honest, bounded)

NOT: "Divergence is unavoidable."
NOT: "The canary cannot be defeated."

INSTEAD: "We identify a confidence-gated phase transition in the adversarial
optimization landscape. When the canary's baseline confidence exceeds a threshold
(empirically: B_orig ≥ 0.5, theoretically: gap stalls at 1/(2λ)), the
divergence-minimisation attack is geometrically blocked — the optimizer cannot
simultaneously reduce the target's score and suppress disagreement. Below this
threshold, the attacker succeeds via joint-flip. This characterizes the exact
security boundary of score-disagreement monitoring."

## Deployment implication

The phase transition directly yields a deployment rule:
- **Confident canary (B ≥ 0.99):** trust the divergence signal. Attack is blocked.
- **Uncertain canary (B < 0.5):** divergence signal is unreliable. Route to human
  review regardless of divergence.

This is not a limitation — it is a **measurable, input-time security indicator**
that tells the deployer exactly which inputs are protected and which are not.

## Honest limitations

- n=10 per pair, one λ value. Small sample (but theory match is quantitative).
- GCG is discrete; continuous analysis is a guide, not a proof.
- Higher λ would block more aggressively but also prevent the attack entirely
  (attacker gives up evasion to satisfy divergence penalty).
- Only within-family tested (shared tokenizer). Cross-family has additional
  tokenizer barrier (§5.5) making stealth harder.
- An attacker who doesn't care about stealth (joint-flip, §5.2) succeeds 70%
  regardless. This section only addresses the stealth-aware attacker.
