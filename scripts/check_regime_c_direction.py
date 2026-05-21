"""Check score direction for Regime C valid detections.

For each valid detection, replays the stream and computes mean pre-shift vs
post-shift scores. Reports whether the shift is toward unsafe (higher scores)
or toward safe (lower scores).

Critical for interpreting the Llama Guard "transfer detection" finding:
- If post > pre: Llama Guard sees GCG tokens as MORE unsafe → not transfer
- If post < pre: Llama Guard was fooled (scores toward safe) → true transfer

Must be run on a machine with classifier models available (Mac Studio).

Usage:
    python scripts/check_regime_c_direction.py
    python scripts/check_regime_c_direction.py --classifier llama-guard
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_full_canary import run_detection_with_threshold, run_stream_ks
from scripts.run_factorial import (
    CLASSIFIERS, SEEDS, WINDOW_SIZES, N_REFERENCE, N_CALIBRATION, CAL_PCT,
    get_classifier, load_reference, load_adversarial_success,
)

REGIME_C_RESULTS = Path("results/regime_c_results.jsonl")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default=None, help="Only check this classifier")
    args = parser.parse_args()

    # Load regime C results, filter to valid detections
    results = [json.loads(l) for l in open(REGIME_C_RESULTS) if l.strip()]
    valid = [r for r in results if r.get("is_valid_detection")]
    if args.classifier:
        valid = [r for r in valid if r["classifier"] == args.classifier]

    if not valid:
        print("No valid detections to check.")
        return

    print(f"Checking score direction for {len(valid)} valid Regime C detections")
    print("=" * 70)

    # Load data
    reference, neg_pool = load_reference()
    shifted = load_adversarial_success()

    # Group by classifier to avoid reloading models
    from itertools import groupby
    valid.sort(key=lambda r: r["classifier"])

    directions = []
    for clf_name, cells in groupby(valid, key=lambda r: r["classifier"]):
        cells = list(cells)
        print(f"\n[{clf_name}] {len(cells)} valid detections, loading classifier...")
        classifier = get_classifier(clf_name)

        # Calibrate thresholds
        thresholds = {}
        for ws in WINDOW_SIZES:
            if ws not in thresholds:
                max_ks = []
                for i in range(N_CALIBRATION):
                    start = (i * 50) % len(neg_pool)
                    examples = neg_pool[start:start + N_REFERENCE]
                    if len(examples) < ws:
                        examples = neg_pool[:N_REFERENCE]
                    mk, _ = run_stream_ks(classifier, examples, ws, seed=42 + i * 7)
                    max_ks.append(mk)
                thresholds[ws] = float(np.percentile(max_ks, CAL_PCT))

        for cell in cells:
            seed, ws = cell["seed"], cell["window_size"]
            threshold = thresholds[ws]
            cond_seed = seed * 1000 + 999

            pos = run_detection_with_threshold(
                classifier, reference, shifted, N_REFERENCE, ws, cond_seed, threshold
            )

            mean_pre = pos["mean_pre"]
            mean_post = pos["mean_post"]
            delta = mean_post - mean_pre if (mean_pre is not None and mean_post is not None) else None
            direction = "TOWARD_UNSAFE" if delta and delta > 0 else "TOWARD_SAFE" if delta and delta < 0 else "UNKNOWN"

            directions.append({
                "classifier": clf_name,
                "seed": seed,
                "window_size": ws,
                "mean_pre": mean_pre,
                "mean_post": mean_post,
                "delta": delta,
                "direction": direction,
            })
            print(f"  s{seed}/w{ws}: pre={mean_pre:.4f} post={mean_post:.4f} Δ={delta:+.4f} → {direction}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    for clf_name in CLASSIFIERS:
        clf_dirs = [d for d in directions if d["classifier"] == clf_name]
        if not clf_dirs:
            continue
        toward_unsafe = sum(1 for d in clf_dirs if d["direction"] == "TOWARD_UNSAFE")
        toward_safe = sum(1 for d in clf_dirs if d["direction"] == "TOWARD_SAFE")
        mean_delta = np.mean([d["delta"] for d in clf_dirs if d["delta"] is not None])
        print(f"  {clf_name}: {toward_unsafe} toward_unsafe, {toward_safe} toward_safe, mean Δ={mean_delta:+.4f}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    llama_dirs = [d for d in directions if d["classifier"] == "llama-guard"]
    if llama_dirs:
        toward_unsafe = sum(1 for d in llama_dirs if d["direction"] == "TOWARD_UNSAFE")
        toward_safe = sum(1 for d in llama_dirs if d["direction"] == "TOWARD_SAFE")
        if toward_unsafe > toward_safe:
            print("  Llama Guard scores shift TOWARD UNSAFE → GCG tokens look more dangerous")
            print("  → NOT transfer detection. Llama Guard flags garbled tokens as suspicious.")
            print("  → Paper claim 'cross-classifier transfer detection' is INCORRECT.")
        else:
            print("  Llama Guard scores shift TOWARD SAFE → GCG suffixes fool Llama Guard too")
            print("  → TRUE transfer detection. Suffixes optimized against DeBERTa also fool Llama Guard.")
            print("  → Paper claim 'cross-classifier transfer detection' is SUPPORTED.")


if __name__ == "__main__":
    main()
