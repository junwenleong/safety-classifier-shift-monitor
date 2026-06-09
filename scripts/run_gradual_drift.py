"""Gradual-drift experiment: DeBERTa + paraphrase, ramp 0%→50% over 200 steps.

Compares detection latency between abrupt shift (mixing=1.0 at step 500) and
gradual ramp (mixing linearly increases from 0% to 50% over steps 500-700).

Usage:
    .venv/bin/python scripts/run_gradual_drift.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.detection.ks_detector import KSDetector
from shift_detection_monitor.detection.reference_window import ReferenceWindow
from shift_detection_monitor.types import StreamRecord

SHIFT_ONSET = 500
RAMP_DURATION = 200  # steps to go from 0% to 50%
MAX_MIXING = 0.5
N_REFERENCE = 500
WINDOW_SIZE = 100
N_SEEDS = 20
N_CALIBRATION = 50
CAL_PCT = 97
OUTPUT = Path("results/gradual_drift_results.json")


def get_classifier():
    from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
    return DeBERTaAdapter()


def load_reference():
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=42)
    prompts = ds["prompt"]
    ref = [{"text": prompts[i]} for i in range(N_REFERENCE)]
    neg = [{"text": prompts[N_REFERENCE + i]} for i in range(N_REFERENCE * 2)]
    return ref, neg


def load_shifted():
    path = Path("data/shifted/paraphrase/output.jsonl")
    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    for r in raw:
        if "shifted" in r and "text" not in r:
            r["text"] = r["shifted"]
        if len(r.get("text", "")) > 1000:
            r["text"] = r["text"][:1000]
    return raw[:300]


def run_stream(classifier, reference, shifted, seed, threshold, mode="abrupt"):
    """Run a single stream with either abrupt or gradual drift.

    mode='abrupt': mixing=1.0 at step 500
    mode='gradual': mixing ramps from 0.0 to 0.5 over steps 500-700
    """
    rng = random.Random(seed)

    ref_pool = list(reference)
    shift_pool = list(shifted)
    rng.shuffle(ref_pool)
    rng.shuffle(shift_pool)

    ref_idx = 0
    shift_idx = 0

    # Calibrate reference window
    ref_window = ReferenceWindow(min_size=WINDOW_SIZE, n_bootstrap=100)
    ref_records = []

    # Generate stream records
    all_records = []
    total_steps = N_REFERENCE + len(shifted)

    for t in range(total_steps):
        # Determine mixing proportion at this step
        if t < SHIFT_ONSET:
            mix_prob = 0.0
        elif mode == "abrupt":
            mix_prob = 1.0
        else:  # gradual
            steps_since_onset = t - SHIFT_ONSET
            mix_prob = min(MAX_MIXING, MAX_MIXING * steps_since_onset / RAMP_DURATION)

        # Decide whether this step draws from shifted pool
        use_shifted = (mix_prob > 0) and (rng.random() < mix_prob) and (shift_idx < len(shift_pool))

        if use_shifted:
            ex = shift_pool[shift_idx]
            shift_idx += 1
        elif ref_idx < len(ref_pool):
            ex = ref_pool[ref_idx]
            ref_idx += 1
        else:
            break

        text = ex.get("text", "")
        output = classifier.predict(text)

        record = StreamRecord(
            time_step=t,
            text=text,
            score=output.score,
            representation=output.representation,
            ground_truth_label=None,
            is_shifted=use_shifted,
            source_dataset="shifted" if use_shifted else "reference",
            shift_condition="paraphrase" if use_shifted else None,
        )
        all_records.append(record)

    # Now run detection
    # First WINDOW_SIZE records are the reference window
    for rec in all_records[:WINDOW_SIZE]:
        ref_window.add(rec)
    frozen = ref_window.freeze()
    ks_det = KSDetector(frozen_stats=frozen, window_size=WINDOW_SIZE)

    # Warm up KS on reference records
    for rec in all_records[:WINDOW_SIZE]:
        ks_det.update(rec)

    # Continue and detect
    alarm_step = None
    for rec in all_records[WINDOW_SIZE:]:
        val = ks_det.update(rec)
        if val > threshold and rec.time_step > 2 * WINDOW_SIZE and alarm_step is None:
            alarm_step = rec.time_step

    latency = (alarm_step - SHIFT_ONSET) if alarm_step is not None else None
    return {"alarm_step": alarm_step, "detection_latency": latency}


def calibrate_threshold(classifier, reference, neg_pool):
    """Calibrate threshold from null streams (no shift).
    
    Matches the factorial's approach: freeze reference from first WINDOW_SIZE
    examples, then track max KS only after 2*WINDOW_SIZE steps (warmup).
    """
    max_ks_values = []
    for cal_seed in range(N_CALIBRATION):
        rng = random.Random(cal_seed + 1000)
        pool = list(neg_pool)
        rng.shuffle(pool)

        ref_window = ReferenceWindow(min_size=WINDOW_SIZE, n_bootstrap=100)
        records = []
        for i, ex in enumerate(pool[:N_REFERENCE]):
            output = classifier.predict(ex["text"])
            rec = StreamRecord(
                time_step=i, text=ex["text"], score=output.score,
                representation=output.representation, ground_truth_label=None,
                is_shifted=False, source_dataset="reference", shift_condition=None,
            )
            records.append(rec)
            if i < WINDOW_SIZE:
                ref_window.add(rec)

        frozen = ref_window.freeze()
        ks_det = KSDetector(frozen_stats=frozen, window_size=WINDOW_SIZE)

        max_ks = 0.0
        for i, rec in enumerate(records):
            val = ks_det.update(rec)
            # Only track max after warmup, matching detection logic
            if i >= 2 * WINDOW_SIZE and val > max_ks:
                max_ks = val
        max_ks_values.append(max_ks)

    threshold = float(np.percentile(max_ks_values, CAL_PCT))
    return threshold


def main():
    wall_start = time.time()
    print("Gradual-Drift Experiment: DeBERTa + paraphrase")
    print(f"  Ramp: 0% → {MAX_MIXING*100:.0f}% over {RAMP_DURATION} steps")
    print(f"  Seeds: {N_SEEDS}, Window: {WINDOW_SIZE}")
    print("=" * 60)

    classifier = get_classifier()
    reference, neg_pool = load_reference()
    shifted = load_shifted()

    print(f"\nCalibrating threshold ({N_CALIBRATION} null streams)...")
    threshold = calibrate_threshold(classifier, reference, neg_pool)
    print(f"  Threshold: {threshold:.4f}")

    abrupt_results = []
    gradual_results = []

    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed}/19...", end=" ", flush=True)

        # Abrupt
        res_a = run_stream(classifier, reference, shifted, seed, threshold, mode="abrupt")
        abrupt_results.append(res_a)

        # Gradual
        res_g = run_stream(classifier, reference, shifted, seed, threshold, mode="gradual")
        gradual_results.append(res_g)

        a_lat = res_a["detection_latency"]
        g_lat = res_g["detection_latency"]
        print(f"abrupt={a_lat}, gradual={g_lat}")

    # Summarize
    abrupt_latencies = [r["detection_latency"] for r in abrupt_results if r["detection_latency"] is not None and r["detection_latency"] >= 0]
    gradual_latencies = [r["detection_latency"] for r in gradual_results if r["detection_latency"] is not None and r["detection_latency"] >= 0]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Abrupt (mixing=1.0 at onset):")
    print(f"    Detection rate: {len(abrupt_latencies)}/{N_SEEDS}")
    if abrupt_latencies:
        print(f"    Mean latency: {np.mean(abrupt_latencies):.1f} steps")
        print(f"    Median latency: {np.median(abrupt_latencies):.1f} steps")

    print(f"\n  Gradual (0%→50% over {RAMP_DURATION} steps):")
    print(f"    Detection rate: {len(gradual_latencies)}/{N_SEEDS}")
    if gradual_latencies:
        print(f"    Mean latency: {np.mean(gradual_latencies):.1f} steps")
        print(f"    Median latency: {np.median(gradual_latencies):.1f} steps")

    if abrupt_latencies and gradual_latencies:
        diff = np.mean(gradual_latencies) - np.mean(abrupt_latencies)
        print(f"\n  Difference (gradual - abrupt): {diff:+.1f} steps")

    wall_time = time.time() - wall_start
    print(f"\n  Wall time: {wall_time/60:.1f} minutes")

    # Save results
    output_data = {
        "experiment": "gradual_drift",
        "classifier": "deberta",
        "shift_condition": "paraphrase",
        "ramp_duration": RAMP_DURATION,
        "max_mixing": MAX_MIXING,
        "window_size": WINDOW_SIZE,
        "n_seeds": N_SEEDS,
        "threshold": threshold,
        "abrupt": {
            "detection_rate": len(abrupt_latencies) / N_SEEDS,
            "mean_latency": float(np.mean(abrupt_latencies)) if abrupt_latencies else None,
            "median_latency": float(np.median(abrupt_latencies)) if abrupt_latencies else None,
            "latencies": abrupt_latencies,
            "all_results": abrupt_results,
        },
        "gradual": {
            "detection_rate": len(gradual_latencies) / N_SEEDS,
            "mean_latency": float(np.mean(gradual_latencies)) if gradual_latencies else None,
            "median_latency": float(np.median(gradual_latencies)) if gradual_latencies else None,
            "latencies": gradual_latencies,
            "all_results": gradual_results,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
