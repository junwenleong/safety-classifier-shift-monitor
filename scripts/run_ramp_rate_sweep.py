"""Faster ramp experiment: characterize the ramp-rate detection threshold.

Runs DeBERTa + paraphrase with ramp durations of 50, 100, 150, 200 steps
to identify the threshold at which the KS detector can/cannot detect gradual drift.
Also tests the CS growing-window detector on the same streams.

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
from shift_detection_monitor.detection.confidence_sequence import ConfidenceSequenceEngine

OUTPUT = Path("results/ramp_rate_sweep.json")
RAMP_DURATIONS = [50, 100, 150, 200]
MAX_MIXING = 0.5
N_SEEDS = 20
CS_ALPHA = 0.05


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


def run_stream_with_cs(classifier, reference, shifted, seed, ramp_duration, ref_mean):
    """Run gradual-drift stream and detect with CS growing-window."""
    import random
    from shift_detection_monitor.types import StreamRecord

    rng = random.Random(seed)
    ref_pool = list(reference[:N_REFERENCE])
    shift_pool = list(shifted)
    rng.shuffle(ref_pool)
    rng.shuffle(shift_pool)

    # Collect all scores
    scores = []
    total_steps = N_REFERENCE + len(shift_pool)
    shift_idx = 0

    for t in range(total_steps):
        if t < SHIFT_ONSET:
            mix_prob = 0.0
        else:
            steps_since = t - SHIFT_ONSET
            mix_prob = min(MAX_MIXING, MAX_MIXING * steps_since / ramp_duration)

        use_shifted = (mix_prob > 0) and (rng.random() < mix_prob) and (shift_idx < len(shift_pool))

        if use_shifted:
            ex = shift_pool[shift_idx]
            shift_idx += 1
        elif t < len(ref_pool):
            ex = ref_pool[t]
        else:
            break

        output = classifier.predict(ex.get("text", ""))
        scores.append(output.score)

    # Run CS growing-window on collected scores
    engine = ConfidenceSequenceEngine(
        alpha=CS_ALPHA,
        reference_value=ref_mean,
        window_mode="growing",
        tail_bound="bounded",
        lower_bound=0.0,
        upper_bound=1.0,
        min_warmup_steps=WINDOW_SIZE,
    )

    alarm_step = None
    for t, score in enumerate(scores):
        result = engine.update(score)
        if result.alarm and alarm_step is None:
            alarm_step = t

    latency = (alarm_step - SHIFT_ONSET) if alarm_step is not None else None
    return {"alarm_step": alarm_step, "detection_latency": latency}


def main():
    wall_start = time.time()
    print("=" * 60)
    print("RAMP-RATE SWEEP: DeBERTa + paraphrase")
    print(f"  Ramp durations: {RAMP_DURATIONS} steps")
    print(f"  Max mixing: {MAX_MIXING}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Detectors: KS sliding-window + CS growing-window")
    print("=" * 60)

    classifier = get_classifier()
    reference, neg_pool = load_reference()
    shifted = load_shifted()

    print("\nCalibrating KS threshold...")
    threshold = calibrate_threshold(classifier, reference, neg_pool)
    print(f"  KS Threshold: {threshold:.4f}")

    # Compute reference mean for CS detector
    null_scores_path = Path("results/null_scores.json")
    if null_scores_path.exists():
        null_scores = json.load(open(null_scores_path))
        ref_mean = float(np.mean(null_scores["deberta"]))
    else:
        # Fallback: compute from first few reference examples
        ref_mean = 0.015
    print(f"  CS reference value: {ref_mean:.4f}")

    results = {"threshold": threshold, "ref_mean": ref_mean, "ramp_durations": {}}

    for ramp_dur in RAMP_DURATIONS:
        print(f"\n  Ramp duration: {ramp_dur} steps (0→{MAX_MIXING*100:.0f}% over {ramp_dur} steps)")
        ks_latencies = []
        cs_latencies = []

        for seed in range(N_SEEDS):
            # KS detector
            res_ks = run_stream_with_ramp(classifier, reference, shifted, seed, threshold, ramp_dur)
            ks_latencies.append(res_ks["detection_latency"])

            # CS detector
            res_cs = run_stream_with_cs(classifier, reference, shifted, seed, ramp_dur, ref_mean)
            cs_latencies.append(res_cs["detection_latency"])

        ks_valid = [l for l in ks_latencies if l is not None and l >= 0]
        cs_valid = [l for l in cs_latencies if l is not None and l >= 0]

        results["ramp_durations"][str(ramp_dur)] = {
            "ks": {
                "detection_rate": len(ks_valid) / N_SEEDS,
                "mean_latency": float(np.mean(ks_valid)) if ks_valid else None,
                "n_detected": len(ks_valid),
                "latencies": ks_latencies,
            },
            "cs": {
                "detection_rate": len(cs_valid) / N_SEEDS,
                "mean_latency": float(np.mean(cs_valid)) if cs_valid else None,
                "n_detected": len(cs_valid),
                "latencies": cs_latencies,
            },
        }
        print(f"    KS: {len(ks_valid)}/{N_SEEDS} detected" + (f", mean={np.mean(ks_valid):.1f}" if ks_valid else ""))
        print(f"    CS: {len(cs_valid)}/{N_SEEDS} detected" + (f", mean={np.mean(cs_valid):.1f}" if cs_valid else ""))

    # Summary
    print("\n" + "=" * 60)
    print("RAMP-RATE THRESHOLD SUMMARY")
    print(f"  {'Ramp':<8} {'KS det':<12} {'KS lat':<10} {'CS det':<12} {'CS lat'}")
    for ramp_dur in RAMP_DURATIONS:
        r = results["ramp_durations"][str(ramp_dur)]
        ks_lat = f"{r['ks']['mean_latency']:.1f}" if r['ks']['mean_latency'] else "—"
        cs_lat = f"{r['cs']['mean_latency']:.1f}" if r['cs']['mean_latency'] else "—"
        print(f"  {ramp_dur:<8} {r['ks']['n_detected']}/{N_SEEDS:<9} {ks_lat:<10} {r['cs']['n_detected']}/{N_SEEDS:<9} {cs_lat}")

    wall_time = time.time() - wall_start
    print(f"\n  Wall time: {wall_time/60:.1f} min")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
