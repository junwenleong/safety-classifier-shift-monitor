"""Step 4: CS growing-window evaluation on cached streams.

Feeds cached per-step scores into the ConfidenceSequenceEngine (growing mode)
and reports detection latency + FAR alongside the KS factorial results.

Requires: results/cached_streams/ from cache_embeddings.py

Usage:
    .venv/bin/python scripts/run_cs_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.detection.confidence_sequence import ConfidenceSequenceEngine

CACHE_DIR = Path("results/cached_streams")
FACTORIAL = Path("results/factorial_results.jsonl")
NULL_SCORES = Path("results/null_scores.json")
OUTPUT = Path("results/cs_growing_window_results.json")

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "temporal", "adversarial-suffix"]
SEEDS = list(range(10))
SHIFT_ONSET = 500
ALPHA = 0.05
WINDOW_SIZE = 100  # warmup period


def run_cs_on_scores(scores: np.ndarray, reference_value: float) -> dict:
    """Run growing-window CS on a score stream. Return detection result."""
    engine = ConfidenceSequenceEngine(
        alpha=ALPHA,
        reference_value=reference_value,
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
    return {
        "alarm_step": alarm_step,
        "detection_latency": latency,
        "is_valid_detection": latency is not None and latency >= 0,
    }


def main():
    print("=" * 60)
    print("CS GROWING-WINDOW EVALUATION")
    print("=" * 60)

    # Load null scores to get reference values (mean under null)
    null_scores = json.load(open(NULL_SCORES))

    # Also run on null streams for FAR estimation
    results = []
    far_results = {clf: [] for clf in CLASSIFIERS}

    for clf in CLASSIFIERS:
        ref_value = float(np.mean(null_scores[clf]))
        print(f"\n  {clf} (reference_value={ref_value:.4f}):")

        for shift in SHIFTS:
            latencies = []
            for seed in SEEDS:
                path = CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if not path.exists():
                    print(f"    {shift}/seed_{seed}: not cached, skipping")
                    continue
                data = np.load(path)
                scores = data["scores"]

                res = run_cs_on_scores(scores, ref_value)
                latencies.append(res["detection_latency"])
                results.append({
                    "classifier": clf,
                    "shift_condition": shift,
                    "seed": seed,
                    **res,
                })

            valid = [l for l in latencies if l is not None and l >= 0]
            if valid:
                print(f"    {shift}: detect {len(valid)}/{len(latencies)}, mean latency={np.mean(valid):.1f}")
            else:
                print(f"    {shift}: no valid detections")

        # FAR: run CS on reference-only portion of any cached stream
        for seed in SEEDS:
            # Use any shift's cache — first SHIFT_ONSET steps are reference
            for shift in SHIFTS:
                path = CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if path.exists():
                    data = np.load(path)
                    ref_scores = data["scores"][:SHIFT_ONSET]
                    res = run_cs_on_scores(ref_scores, ref_value)
                    far_results[clf].append(1 if res["alarm_step"] is not None else 0)
                    break

    # Summary comparison with KS
    print("\n" + "=" * 60)
    print("COMPARISON: CS growing-window vs KS sliding-window")
    print("=" * 60)

    # Load KS results from factorial
    ks_rows = [json.loads(l) for l in open(FACTORIAL) if l.strip()]
    for r in ks_rows:
        r["is_valid_detection"] = (
            r.get("detection_latency") is not None
            and r["detection_latency"] >= 0
            and r.get("neg_clean") is True
        )

    print(f"\n  {'Classifier':<16} {'Shift':<20} {'CS latency':<14} {'KS latency':<14} {'CS det%':<10} {'KS det%'}")
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            cs_lats = [r["detection_latency"] for r in results
                       if r["classifier"] == clf and r["shift_condition"] == shift
                       and r["is_valid_detection"]]
            ks_lats = [r["detection_latency"] for r in ks_rows
                       if r["classifier"] == clf and r["shift_condition"] == shift
                       and r["is_valid_detection"] and r["window_size"] == 100]

            cs_mean = f"{np.mean(cs_lats):.1f}" if cs_lats else "—"
            ks_mean = f"{np.mean(ks_lats):.1f}" if ks_lats else "—"
            cs_rate = f"{len(cs_lats)}/{len(SEEDS)}" if cs_lats else "0/10"
            ks_n = len([r for r in ks_rows if r["classifier"] == clf and r["shift_condition"] == shift and r["window_size"] == 100])
            ks_valid = len(ks_lats)
            ks_rate = f"{ks_valid}/{ks_n}"

            print(f"  {clf:<16} {shift:<20} {cs_mean:<14} {ks_mean:<14} {cs_rate:<10} {ks_rate}")

    # FAR summary
    print(f"\n  FAR (CS growing-window on reference-only streams):")
    for clf in CLASSIFIERS:
        alarms = far_results[clf]
        if alarms:
            far = sum(alarms) / len(alarms)
            print(f"    {clf}: {sum(alarms)}/{len(alarms)} = {far:.3f}")

    # Save
    output_data = {
        "results": results,
        "far": {clf: {"alarms": sum(v), "total": len(v)} for clf, v in far_results.items()},
    }
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
