# §5.3 (draft) — Divergence-Minimisation: Can a Stealthy Attacker Suppress the Canary?

**STATUS: experiment in progress (divergence-min run). Numbers below marked
[PARTIAL] are from the first 3 prompts and MUST be replaced with full n=10
results before this goes in the paper. Lead with the experiment, not the theory.**

---

## Threat model for this section

The strongest adversary in our hierarchy (Tier 4) does not merely try to flip the
target classifier A — they additionally try to keep the canary B from looking
anomalous. We formalise this as a stealth objective: minimise A's unsafe score
while keeping the two classifiers' scores close (so the divergence detector sees
nothing). The attacker minimises:

    L(δ) = f_A(x+δ) + λ · (f_B(x+δ) − f_A(x+δ))²

where f_A, f_B ∈ [0,1] are unsafe-class probabilities and λ controls how hard the
attacker tries to suppress divergence. We use λ=2.0, w=50 steps.

**Caveat (state honestly):** this is one formalisation of stealth. An attacker
who also wants B's *absolute* score low would use a different penalty; that case
reduces to the joint-flip attack of §5.2 (which succeeds 70% within-family). The
divergence-suppression objective here is the specific case "flip A, keep B from
diverging" — the case that directly attacks the canary signal.

## Empirical result

[PARTIAL — replace with full run] On the first 3 prompts (epoch-1 target,
epoch-5 canary), the attack failed to flip A on all 3: final f_A ∈ {0.75, 0.76,
0.84}, never below 0.5. Under single-target GCG (λ=0) at the same 50-step budget,
these same prompts flip to f_A ≈ 0.002. Adding the divergence penalty does not
just reduce stealth — it prevents evasion entirely.

## Why: gradient conflict (mechanism, not theorem)

The combined-loss gradient (continuous relaxation) is:

    ∇L = [1 − 2λ(f_B − f_A)] · ∇f_A  +  2λ(f_B − f_A) · ∇f_B

Two observations explain the failure. We present these as **geometric intuition
for attack difficulty**, not as a convergence proof (GCG is discrete; the
continuous relaxation is only a guide).

**(1) Coefficient inversion.** The coefficient on ∇f_A is [1 − 2λ(f_B − f_A)].
When the gap (f_B − f_A) exceeds 1/(2λ), this coefficient turns **negative**: the
optimiser is now driven to *increase* f_A to reduce the divergence penalty,
directly opposing the flip objective. At λ=2 this boundary is a gap of 0.25.

  *Striking match to data:* prompts 1 and 2 stall at gap = 0.250 and 0.242 —
  essentially exactly the predicted 1/(2λ) = 0.25 equilibrium. The optimiser
  cannot push past the point where reducing f_A further costs more divergence
  penalty than it gains in flip loss.

**(2) The confident-canary basin (gradient-norm effect).** The condition for a
descent step to fail to decrease f_A (i.e. ∇L·∇f_A ≥ 0), keeping the norms, is:

    cos(∇f_A, ∇f_B) ≥ (1 − 1/[2λ(f_B−f_A)]) · ‖∇f_A‖ / ‖∇f_B‖

When B is deep in its high-confidence region (f_B ≈ 1.0), it sits on a flat part
of its loss surface, so ‖∇f_B‖ → 0. The ratio ‖∇f_A‖/‖∇f_B‖ blows up, making the
condition impossible to satisfy for any angle: B offers almost no gradient signal
to help reduce the divergence penalty, so the optimiser is paralysed. This is why
the result is strongest exactly when the canary is confident — consistent with
the confidence-gating finding of §5.2 (transfer succeeds only when B_orig < 0.5).

## What this means

Evasion and divergence-suppression are conflicting objectives for an ensemble
whose members have non-identical decision boundaries. Moving toward A's boundary
increases divergence (B does not follow); suppressing divergence prevents reaching
A's boundary. The conflict does not depend on architecture family — only on
training-induced boundary separation. This is the mechanism behind the canary's
robustness: the divergence is not an incidental side-effect an attacker can
optimise away, it is an inherent consequence of single-target evasion.

## Honest limitations

- Continuous-relaxation analysis; GCG is discrete coordinate search. The
  inequality is a heuristic, not a theorem. At much higher step counts the
  discrete search might find paths the continuous view misses.
- The gap (f_B − f_A) and the gradients change every step, so the condition is
  dynamic, not static.
- Only the "suppress divergence" stealth objective is tested; other stealth
  formulations reduce to joint-flip (§5.2).
- n=10 prompts, one λ, within-family (shared tokenizer). Cross-family is harder
  for the attacker (tokenizer barrier, §5.5), so within-family is the easier
  case for the attacker — failure here is a lower bound on canary robustness.

## TODO before paper

1. ✅/⏳ Finish full n=10 divergence-min run → replace [PARTIAL] numbers.
2. ⏳ Run v2 (epoch-3 vs epoch-10) → generality across pairs.
3. Decide: keep the cos/‖·‖ inequality in an appendix, or cut to prose only
   (per reviewer feedback — math risks looking like window-dressing unless the
   gap=1/(2λ) prediction match is shown, which it is).
4. Reframe "sufficient condition" → "geometric intuition for difficulty"
   everywhere. No theorem claims.
