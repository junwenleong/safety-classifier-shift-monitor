"""Faster ramp experiment: characterize the ramp-rate detection threshold.

Runs DeBERTa + paraphrase with ramp durations of 50, 100, 150, 200 steps
to identify the threshold at which the KS detector can/cannot detect gradual drift.

Usage:
    .venv/bin/python scripts/run_ramp_rate_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse the gradual-drift infrastructure
from scripts.run_gradual_drift import (
    get_classifier, load_reference, load_shifted,
    calibrate_threshold, SHIFT_ONSET, WINDOW_SIZE, N_CALIBRATION, CAL_PCT,
    N_REFERENCE,
)
from scripts.run_gradual_drift import run_stream

OUTPUT = Path("results/ramp_rate_sweep.json")
RAMP_DURATIONS = [50, 100, 150, 200]
MAX_MIXING = 0.5
N_SEEDS = 20


def run_stream_with_ramp(classifier, reference, shifted, seed, threshold, ramp_duration):
    """Wrapper that patches RAMP_DURATION for a single run."""
    import scripts.run_gradual_drift as gd
    # Temporarily override the module constants
    orig_ramp = gd.RAMP_DURATION
    orig_mix = gd.MAX_MIXING
    gd.RAMP_DURATION = ramp_duration
    gd.MAX_MIXING = MAX_MIXING
    result = run_stream(classifier, reference, shifted, seed, threshold, mode="gradual")
    gd.RAMP_DURATION = orig_ramp
    gd.MAX_MIXING = orig_mix
    return result


def main():
    wall_start = time.time()
    print("=" * 60)
    print("RAMP-RATE SWEEP: DeBERTa + paraphrase")
    print(f"  Ramp durations: {RAMP_DURATIONS} steps")
    print(f"  Max mixing: {MAX_MIXING}")
    print(f"  Seeds: {N_SEEDS}")
    print("=" * 60)

    classifier = get_classifier()
    reference, neg_pool = load_reference()
    shifted = load_shifted()

    print("\nCalibrating threshold...")
    threshold = calibrate_threshold(classifier, reference, neg_pool)
    print(f"  Threshold: {threshold:.4f}")

    results = {"threshold": threshold, "ramp_durations": {}}

    for ramp_dur in RAMP_DURATIONS:
        print(f"\n  Ramp duration: {ramp_dur} steps (0→{MAX_MIXING*100:.0f}% over {ramp_dur} steps)")
        latencies = []
        for seed in range(N_SEEDS):
            res = run_stream_with_ramp(classifier, reference, shifted, seed, threshold, ramp_dur)
            latencies.append(res["detection_latency"])

        valid = [l for l in latencies if l is not None and l >= 0]
        detect_rate = len(valid) / N_SEEDS
        mean_lat = float(np.mean(valid)) if valid else None

        results["ramp_durations"][str(ramp_dur)] = {
            "detection_rate": detect_rate,
            "mean_latency": mean_lat,
            "latencies": latencies,
            "n_detected": len(valid),
        }
        print(f"    Detection: {len(valid)}/{N_SEEDS} ({detect_rate*100:.0f}%)")
        if valid:
            print(f"    Mean latency: {mean_lat:.1f} steps")

    # Summary
    print("\n" + "=" * 60)
    print("RAMP-RATE THRESHOLD SUMMARY")
    print(f"  {'Ramp (steps)':<15} {'Detect rate':<15} {'Mean latency'}")
    for ramp_dur in RAMP_DURATIONS:
        r = results["ramp_durations"][str(ramp_dur)]
        lat_str = f"{r['mean_latency']:.1f}" if r['mean_latency'] else "—"
        print(f"  {ramp_dur:<15} {r['n_detected']}/{N_SEEDS:<12} {lat_str}")

    wall_time = time.time() - wall_start
    print(f"\n  Wall time: {wall_time/60:.1f} min")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
