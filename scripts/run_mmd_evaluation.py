"""Step 7: MMD evaluation on cached embeddings — KS vs MMD comparison.

Includes MMD null calibration check before running the full evaluation.

Requires: results/cached_streams/ from cache_embeddings.py

Usage:
    .venv/bin/python scripts/run_mmd_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.detection.mmd_detector import compute_mmd_squared

CACHE_DIR = Path("results/cached_streams")
FACTORIAL = Path("results/factorial_results.jsonl")
OUTPUT = Path("results/mmd_evaluation.json")

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "temporal", "adversarial-suffix"]
SEEDS = list(range(10))
SHIFT_ONSET = 500
WINDOW_SIZE = 100
N_CALIBRATION_PERMUTATIONS = 1000  # bootstrap permutations for null distribution
TARGET_FAR_ALPHA = 0.05  # target false alarm rate


def median_bandwidth(X: np.ndarray) -> float:
    """Median heuristic for Gaussian kernel bandwidth."""
    from scipy.spatial.distance import pdist
    dists = pdist(X[:min(500, len(X))], "euclidean")
    return float(np.median(dists)) + 1e-10


def calibrate_mmd_threshold(ref_embeddings: np.ndarray, bandwidth: float,
                            window_size: int = WINDOW_SIZE,
                            n_perms: int = N_CALIBRATION_PERMUTATIONS,
                            alpha: float = TARGET_FAR_ALPHA) -> float:
    """Calibrate MMD threshold via permutation null at target FAR α.
    
    Uses full pooled reference. Draws two disjoint random subsets for each
    permutation to estimate the null distribution of MMD under no shift.
    """
    n = len(ref_embeddings)
    rng = np.random.default_rng(42)
    null_mmds = []

    for _ in range(n_perms):
        # Draw two disjoint subsets from the pooled reference
        indices = rng.permutation(n)
        ref_sample = ref_embeddings[indices[:200]]
        window = ref_embeddings[indices[200:200 + window_size]]
        mmd = compute_mmd_squared(ref_sample, window, bandwidth)
        null_mmds.append(mmd)

    threshold = float(np.percentile(null_mmds, 100 * (1 - alpha)))
    return threshold


def run_mmd_detection(embeddings: np.ndarray, is_shifted: np.ndarray,
                      pooled_ref: np.ndarray, bandwidth: float, threshold: float) -> dict:
    """Run sliding-window MMD detection on a stream of embeddings.
    
    Compares each post-onset window against a random 200-sample from pooled reference.
    """
    rng = np.random.default_rng(123)
    ref_sample = pooled_ref[rng.choice(len(pooled_ref), size=200, replace=False)]

    alarm_step = None
    for t in range(SHIFT_ONSET + WINDOW_SIZE, len(embeddings)):
        window = embeddings[t - WINDOW_SIZE:t]
        mmd = compute_mmd_squared(ref_sample, window, bandwidth)
        if mmd > threshold and alarm_step is None:
            alarm_step = t

    latency = (alarm_step - SHIFT_ONSET) if alarm_step is not None else None
    return {"alarm_step": alarm_step, "detection_latency": latency,
            "is_valid_detection": latency is not None and latency >= 0}


def main():
    print("=" * 60)
    print("MMD EVALUATION")
    print("=" * 60)

    # Step 1: Pool reference embeddings per classifier (all seeds, all shifts)
    print("\n--- Pooling reference embeddings ---")
    pooled_refs = {}
    for clf in CLASSIFIERS:
        all_ref = []
        for shift in SHIFTS:
            for seed in SEEDS:
                path = CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                if "embeddings" not in data:
                    break
                all_ref.append(data["embeddings"][:SHIFT_ONSET])
        if all_ref:
            pooled_refs[clf] = np.concatenate(all_ref)
            print(f"  {clf}: {len(pooled_refs[clf])} reference embeddings pooled")
        else:
            print(f"  {clf}: no embeddings available")

    if not pooled_refs:
        print("  No embeddings found. Run cache_embeddings.py first.")
        return

    # Step 2: Calibrate per-classifier using pooled reference
    print("\n--- Calibration (1000 permutations, α=0.05) ---")
    thresholds = {}
    bandwidths = {}
    for clf, ref_embs in pooled_refs.items():
        bw = median_bandwidth(ref_embs)
        threshold = calibrate_mmd_threshold(ref_embs, bw)
        thresholds[clf] = threshold
        bandwidths[clf] = bw
        print(f"  {clf}: bandwidth={bw:.3f}, threshold={threshold:.6f}")

    # Step 3: FAR validation on null windows
    print("\n--- FAR validation (null streams) ---")
    for clf in CLASSIFIERS:
        if clf not in pooled_refs:
            continue
        ref_sample = pooled_refs[clf][np.random.default_rng(99).choice(len(pooled_refs[clf]), 200, replace=False)]
        n_false = 0
        n_windows = 0
        for seed in SEEDS:
            path = CACHE_DIR / clf / SHIFTS[0] / f"seed_{seed}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            if "embeddings" not in data:
                continue
            ref_portion = data["embeddings"][:SHIFT_ONSET]
            for start in range(WINDOW_SIZE, len(ref_portion) - WINDOW_SIZE, WINDOW_SIZE):
                window = ref_portion[start:start + WINDOW_SIZE]
                mmd = compute_mmd_squared(ref_sample, window, bandwidths[clf])
                n_windows += 1
                if mmd > thresholds[clf]:
                    n_false += 1
        far = n_false / max(n_windows, 1)
        print(f"  {clf}: FAR = {n_false}/{n_windows} = {far:.3f} (target: {TARGET_FAR_ALPHA})")

    # Step 4: Full evaluation
    print("\n--- Full MMD Evaluation ---")
    results = []

    for clf in CLASSIFIERS:
        if clf not in pooled_refs:
            continue
        print(f"\n  {clf} (bandwidth={bandwidths[clf]:.3f}, threshold={thresholds[clf]:.6f}):")

        for shift in SHIFTS:
            latencies = []
            for seed in SEEDS:
                path = CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                if "embeddings" not in data:
                    continue
                res = run_mmd_detection(data["embeddings"], data["is_shifted"],
                                       pooled_refs[clf], bandwidths[clf], thresholds[clf])
                latencies.append(res["detection_latency"])
                results.append({"classifier": clf, "shift_condition": shift, "seed": seed, **res})

            valid = [l for l in latencies if l is not None and l >= 0]
            if valid:
                print(f"    {shift}: detect {len(valid)}/{len(latencies)}, mean={np.mean(valid):.1f}")
            else:
                print(f"    {shift}: no detections ({len(latencies)} seeds)")

    # Step 3: Comparison table
    print("\n" + "=" * 60)
    print("KS vs MMD COMPARISON")
    print("=" * 60)

    ks_rows = [json.loads(l) for l in open(FACTORIAL) if l.strip()]
    for r in ks_rows:
        r["is_valid_detection"] = (r.get("detection_latency") is not None
                                    and r["detection_latency"] >= 0 and r.get("neg_clean") is True)

    print(f"\n  {'Classifier':<16} {'Shift':<20} {'MMD det':<10} {'KS det':<10} {'MMD lat':<10} {'KS lat'}")
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            mmd_lats = [r["detection_latency"] for r in results
                        if r["classifier"] == clf and r["shift_condition"] == shift and r["is_valid_detection"]]
            ks_lats = [r["detection_latency"] for r in ks_rows
                       if r["classifier"] == clf and r["shift_condition"] == shift
                       and r["is_valid_detection"] and r["window_size"] == 100]

            mmd_det = f"{len(mmd_lats)}/10"
            ks_det = f"{len(ks_lats)}/20"
            mmd_lat = f"{np.mean(mmd_lats):.1f}" if mmd_lats else "—"
            ks_lat = f"{np.mean(ks_lats):.1f}" if ks_lats else "—"
            print(f"  {clf:<16} {shift:<20} {mmd_det:<10} {ks_det:<10} {mmd_lat:<10} {ks_lat}")

    # Save
    with open(OUTPUT, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
