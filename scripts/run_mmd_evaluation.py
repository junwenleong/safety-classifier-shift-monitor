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
N_CALIBRATION_WINDOWS = 50  # null windows for threshold calibration


def median_bandwidth(X: np.ndarray) -> float:
    """Median heuristic for Gaussian kernel bandwidth."""
    from scipy.spatial.distance import pdist
    dists = pdist(X[:min(500, len(X))], "euclidean")
    return float(np.median(dists)) + 1e-10


def calibrate_mmd_threshold(ref_embeddings: np.ndarray, bandwidth: float,
                            window_size: int = WINDOW_SIZE, n_cal: int = N_CALIBRATION_WINDOWS,
                            percentile: float = 97) -> float:
    """Calibrate MMD threshold from null distribution (reference vs reference windows)."""
    n = len(ref_embeddings)
    rng = np.random.default_rng(42)
    null_mmds = []

    for _ in range(n_cal):
        # Random window from reference
        start = rng.integers(0, n - window_size)
        window = ref_embeddings[start:start + window_size]
        # Compare against full reference (excluding the window)
        ref_excl = np.concatenate([ref_embeddings[:start], ref_embeddings[start + window_size:]])
        ref_sample = ref_excl[rng.choice(len(ref_excl), size=min(200, len(ref_excl)), replace=False)]
        mmd = compute_mmd_squared(ref_sample, window, bandwidth)
        null_mmds.append(mmd)

    return float(np.percentile(null_mmds, percentile))


def run_mmd_detection(embeddings: np.ndarray, is_shifted: np.ndarray,
                      bandwidth: float, threshold: float) -> dict:
    """Run sliding-window MMD detection on a stream of embeddings."""
    ref_embs = embeddings[:SHIFT_ONSET]

    alarm_step = None
    for t in range(SHIFT_ONSET + WINDOW_SIZE, len(embeddings)):
        window = embeddings[t - WINDOW_SIZE:t]
        mmd = compute_mmd_squared(ref_embs[:200], window, bandwidth)
        if mmd > threshold and alarm_step is None:
            alarm_step = t

    latency = (alarm_step - SHIFT_ONSET) if alarm_step is not None else None
    return {"alarm_step": alarm_step, "detection_latency": latency,
            "is_valid_detection": latency is not None and latency >= 0}


def main():
    print("=" * 60)
    print("MMD EVALUATION")
    print("=" * 60)

    # Step 1: Null calibration check
    print("\n--- MMD Null Calibration Check (DeBERTa) ---")
    check_path = CACHE_DIR / "deberta" / "paraphrase" / "seed_0.npz"
    if check_path.exists():
        data = np.load(check_path)
        if "embeddings" in data:
            ref_embs = data["embeddings"][:SHIFT_ONSET]
            bw = median_bandwidth(ref_embs)
            threshold = calibrate_mmd_threshold(ref_embs, bw)
            print(f"  Bandwidth: {bw:.4f}")
            print(f"  Threshold (97th pct): {threshold:.6f}")

            # Run on reference-only to check FAR
            n_false = 0
            for start in range(0, SHIFT_ONSET - WINDOW_SIZE, WINDOW_SIZE):
                window = ref_embs[start:start + WINDOW_SIZE]
                mmd = compute_mmd_squared(ref_embs[:200], window, bw)
                if mmd > threshold:
                    n_false += 1
            n_windows = (SHIFT_ONSET - WINDOW_SIZE) // WINDOW_SIZE
            print(f"  Null FAR check: {n_false}/{n_windows} windows exceed threshold")
            if n_false / max(n_windows, 1) > 0.1:
                print("  ⚠ WARNING: FAR > 10% on null. Threshold may need adjustment.")
            else:
                print("  ✓ FAR controlled on null stream.")
        else:
            print("  ⚠ No embeddings in cache. Cannot run MMD.")
            return
    else:
        print(f"  ⚠ Cache not found at {check_path}. Run cache_embeddings.py first.")
        return

    # Step 2: Full evaluation
    print("\n--- Full MMD Evaluation ---")
    results = []

    for clf in CLASSIFIERS:
        # Calibrate per-classifier
        cal_path = CACHE_DIR / clf / SHIFTS[0] / "seed_0.npz"
        if not cal_path.exists():
            print(f"  {clf}: no cache, skipping")
            continue
        cal_data = np.load(cal_path)
        if "embeddings" not in cal_data:
            print(f"  {clf}: no embeddings, skipping")
            continue

        ref_embs = cal_data["embeddings"][:SHIFT_ONSET]
        bw = median_bandwidth(ref_embs)
        threshold = calibrate_mmd_threshold(ref_embs, bw)
        print(f"\n  {clf} (bandwidth={bw:.3f}, threshold={threshold:.6f}):")

        for shift in SHIFTS:
            latencies = []
            for seed in SEEDS:
                path = CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                if "embeddings" not in data:
                    continue
                res = run_mmd_detection(data["embeddings"], data["is_shifted"], bw, threshold)
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
