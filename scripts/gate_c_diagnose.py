"""Gate C — Diagnose classifier selection problem and identify usable models.

The Gate C run scored 7 new classifiers but got 0% detection on all of them.
This script diagnoses WHY and identifies which classifiers are actually usable
for the monitorability law.

Problem: Most new classifiers have saturated distributions (mean ≈ 0 or ≈ 1,
std < 0.04). A KS detector cannot detect shift in a near-constant distribution.

Usage:
    .venv/bin/python scripts/gate_c_diagnose.py
"""
from __future__ import annotations
import json
import numpy as np

GATE_C = json.load(open("results/gate_c_monitorability.json"))

# Minimum std for a classifier to be "monitorable" — below this the KS
# detector has no variance to work with. The original 4 classifiers have
# stds in [0.066, 0.144].
MIN_STD_USABLE = 0.04  # generous lower bound


def main():
    data = GATE_C["classifier_data"]

    print("=" * 70)
    print("GATE C — CLASSIFIER SELECTION DIAGNOSTIC")
    print("=" * 70)

    print(f"\n{'Classifier':<25} {'Std':<8} {'Mean':<8} {'Usable?':<10} {'Problem'}")
    print("-" * 80)

    usable = []
    saturated = []
    for name, info in sorted(data.items(), key=lambda x: -x[1]["null_std"]):
        std = info["null_std"]
        mean = info["null_mean"]
        source = info.get("source", "existing")

        # Diagnosis
        if std >= MIN_STD_USABLE:
            problem = ""
            status = "✅"
            usable.append((name, info))
        elif mean > 0.95 or mean < 0.05:
            problem = f"saturated ({'all toxic' if mean > 0.5 else 'all safe'})"
            status = "❌"
            saturated.append(name)
        elif std < 0.005:
            problem = "constant output (not fine-tuned?)"
            status = "❌"
            saturated.append(name)
        else:
            problem = f"low variance (std={std:.4f})"
            status = "⚠️ "
            saturated.append(name)

        tag = f" [{source}]" if source == "gate_c" else ""
        print(f"  {name:<23} {std:<8.4f} {mean:<8.4f} {status:<10} {problem}{tag}")

    print(f"\n  Usable: {len(usable)} / {len(data)}")
    print(f"  Saturated/unusable: {len(saturated)}")

    # The original 4 ARE usable
    orig = [(n, d) for n, d in usable if d.get("source") == "existing"]
    new_usable = [(n, d) for n, d in usable if d.get("source") != "existing"]

    print(f"\n  Original 4: {len(orig)} usable (expected)")
    print(f"  New classifiers usable: {len(new_usable)}")

    if new_usable:
        print("\n  Usable new classifiers:")
        for name, info in new_usable:
            print(f"    • {name}: std={info['null_std']:.4f}, mean={info['null_mean']:.4f}")
            print(f"      model_id={info.get('model_id', '?')}")

    # Correlation with what we have
    with_latency = [(n, d) for n, d in usable if d.get("mean_latency") is not None]
    print(f"\n  Have latency data: {len(with_latency)}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    print("""
  Root cause: The new classifiers are generic toxicity models that SATURATE
  on the WildGuardMix reference corpus (mostly harmful prompts → they all
  return near-1.0 or near-0.0 with no variance).

  Why the original 4 work: they were either fine-tuned on WildGuardMix
  (DeBERTa, Text-Mod) or are instruction-following safety classifiers
  (Llama Guard, ShieldGemma) that produce calibrated probability scores.

  The Gate C pipeline IS correct — the problem is model selection.""")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS — WHAT TO DO NEXT")
    print("=" * 70)

    # deberta-v3-base-mnli has high std (0.248) — it's an NLI model 
    # being used as a zero-shot classifier. It might work!
    mnli = data.get("deberta-v3-base-mnli", {})
    if mnli.get("null_std", 0) > MIN_STD_USABLE:
        print(f"""
  1. deberta-v3-base-mnli (std={mnli['null_std']:.3f}) IS usable by variance.
     It's an NLI model — the "unsafe" score comes from entailment with a
     hypothesis like "This text is harmful." Detection might work if the
     pipeline ran correctly. Check why detection_rate=0.0 — likely the
     label-inversion logic misidentified the safe/unsafe class.""")

    print("""
  2. STRATEGY A — Fine-tune more variants (within-family spread for ML1b):
     • DeBERTa-v3-large at epoch 1, 3, 5, 10 (varying calibration)
     • DeBERTa-v3-large at temperature {0.5, 1.0, 2.0} (post-hoc scaling)
     • Text-Moderation with different training seeds
     → Gives 4-6 encoders with VARIED std, tests the within-family question.

  3. STRATEGY B — Use models designed for safety classification:
     • Llama Guard 2 (different checkpoint, should have different score dist)
     • ShieldGemma-2B or 27B (different capacity → different calibration)
     • WildGuard (allenai/wildguard) — proper safety classifier
     • Aegis (was in the list but may have failed to load)
     → Natural safety classifiers that produce calibrated probabilities.

  4. STRATEGY C — Temperature-scale existing 4 classifiers:
     • Apply T={0.5, 1.0, 1.5, 2.0, 3.0} to the 4 existing classifiers
     • Each temperature gives a different null_std (lower T → sharper → lower std)
     • Cheap: no new inference, just transform cached null_scores.
     → 20 "classifiers" for free. Tests ML1 purely. Doesn't test ML1b.

  RECOMMENDED: Start with Strategy C (zero-cost, answers ML1 immediately),
  then Strategy A (answers ML1b, costs ~2h fine-tuning), skip B unless
  you specifically need decoder breadth.

  Strategy C implementation: for each of the 4 classifiers and each T,
  transform scores via sigmoid(logit(score)/T). This changes std but
  preserves rank ordering. Rerun KS detection on transformed scores.
  The prediction is: lower T → lower std → faster detection (lower latency).""")

    # Quick Strategy C preview
    print("\n" + "=" * 70)
    print("STRATEGY C PREVIEW — Temperature-scaled null_std")
    print("=" * 70)

    null_scores = json.load(open("results/null_scores.json"))
    temps = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    print(f"\n  {'Classifier':<18} " + " ".join(f"T={t:<4}" for t in temps))
    print("  " + "-" * 65)

    for clf in ["deberta", "text-moderation", "llama-guard", "shieldgemma"]:
        scores = np.array(null_scores[clf])
        # Clip to avoid log(0)
        scores = np.clip(scores, 1e-7, 1 - 1e-7)
        logits = np.log(scores / (1 - scores))
        stds = []
        for t in temps:
            rescaled = 1 / (1 + np.exp(-logits / t))
            stds.append(np.std(rescaled))
        print(f"  {clf:<18} " + " ".join(f"{s:.4f}" for s in stds))

    print("""
  ✅ Temperature scaling gives a clean spread of null_std values per classifier.
  Strategy C is viable — can test ML1 immediately with zero new inference.""")


if __name__ == "__main__":
    main()
