"""Check score direction for Regime C valid detections.

Replays only the valid detection cells using thresholds already stored in
regime_c_results.jsonl. No calibration needed.

Critical question: do Llama Guard scores shift toward unsafe (GCG tokens
look dangerous) or toward safe (GCG suffixes fool Llama Guard too)?

Usage:
    python scripts/check_regime_c_direction.py --classifier llama-guard
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_full_canary import run_detection_with_threshold
from scripts.run_factorial import (
    CLASSIFIERS, N_REFERENCE, get_classifier, load_reference, load_adversarial_success,
)

REGIME_C_RESULTS = Path("results/regime_c_results.jsonl")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="llama-guard")
    args = parser.parse_args()

    # Load valid detections with their stored thresholds
    results = [json.loads(l) for l in open(REGIME_C_RESULTS) if l.strip()]
    valid = [r for r in results
             if r.get("is_valid_detection") and r["classifier"] == args.classifier]

    if not valid:
        print(f"No valid detections for {args.classifier}")
        return

    print(f"Replaying {len(valid)} valid Regime C detections for {args.classifier}")
    print("=" * 70)

    reference, _ = load_reference()
    shifted = load_adversarial_success()

    print(f"Loading {args.classifier}...")
    classifier = get_classifier(args.classifier)

    directions = []
    for cell in valid:
        seed, ws = cell["seed"], cell["window_size"]
        threshold = cell["threshold"]
        cond_seed = seed * 1000 + 999

        pos = run_detection_with_threshold(
            classifier, reference, shifted, N_REFERENCE, ws, cond_seed, threshold
        )

        mean_pre = pos["mean_pre"]
        mean_post = pos["mean_post"]
        if mean_pre is not None and mean_post is not None:
            delta = mean_post - mean_pre
            direction = "TOWARD_UNSAFE" if delta > 0 else "TOWARD_SAFE"
        else:
            delta = None
            direction = "UNKNOWN"

        directions.append(direction)
        pre_s = f"{mean_pre:.4f}" if mean_pre is not None else "?"
        post_s = f"{mean_post:.4f}" if mean_post is not None else "?"
        delta_s = f"{delta:+.4f}" if delta is not None else "?"
        print(f"  s{seed}/w{ws}: pre={pre_s} post={post_s} Δ={delta_s} → {direction}")

    # Summary
    toward_unsafe = directions.count("TOWARD_UNSAFE")
    toward_safe = directions.count("TOWARD_SAFE")
    print(f"\n{'=' * 70}")
    print(f"RESULT: {toward_unsafe} TOWARD_UNSAFE, {toward_safe} TOWARD_SAFE")
    print("-" * 70)
    if toward_unsafe > toward_safe:
        print("Llama Guard scores shift TOWARD UNSAFE.")
        print("→ GCG tokens look more dangerous to Llama Guard, not less.")
        print("→ 'Cross-classifier transfer detection' claim is WRONG.")
        print("→ Rewrite §5.4: monitor detects anomalous tokens, not transferred attack.")
    else:
        print("Llama Guard scores shift TOWARD SAFE.")
        print("→ GCG suffixes fool Llama Guard too (true transfer).")
        print("→ 'Cross-classifier transfer detection' claim is SUPPORTED.")


if __name__ == "__main__":
    main()
