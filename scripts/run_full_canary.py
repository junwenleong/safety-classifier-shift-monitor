"""Full canary: run monitor across all 5 shift conditions with DeBERTa.

Uses empirical FAR calibration: runs negative controls to set alarm threshold,
then tests each shift condition against that threshold.

Usage:
    export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
    python scripts/run_full_canary.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.config import StreamConfig
from shift_detection_monitor.detection.ks_detector import KSDetector
from shift_detection_monitor.detection.reference_window import ReferenceWindow
from shift_detection_monitor.stream.simulator import StreamSimulator

SHIFT_CORPORA = {
    "paraphrase": Path("data/shifted/paraphrase/output.jsonl"),
    "code-switch": Path("data/shifted/code-switch/output.jsonl"),
    "compositional-long-context": Path("data/shifted/compositional/output.jsonl"),
    "temporal": Path("data/shifted/temporal/output.jsonl"),
    "adversarial-suffix": Path("data/shifted/adversarial_suffix/deberta_suffixes.jsonl"),
}

WINDOW_SIZE = 100
SHIFT_ONSET = 500
N_REFERENCE = 500
SYNTHETIC_SHIFT = 0.5
SEED = 42
N_CALIBRATION_RUNS = 50


def load_shifted(path: Path | None, n: int = 300) -> list[dict]:
    if path is None or not path.exists():
        return []
    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    if not raw:
        return []
    # Adversarial suffix corpus uses 'combined' (original + suffix) as the shifted text
    for r in raw:
        if "combined" in r and "text" not in r:
            r["text"] = r["combined"]
        elif "shifted" in r and "text" not in r:
            r["text"] = r["shifted"]
        # Truncate very long texts to keep inference fast
        if len(r.get("text", "")) > 1000:
            r["text"] = r["text"][:1000]
    if len(raw) >= n:
        return raw[:n]
    return (raw * ((n // len(raw)) + 1))[:n]


def run_stream_ks(classifier, examples, window_size, seed):
    """Run a stream through KS detector, return max KS statistic and per-step values."""
    config = StreamConfig(shift_condition=None, shift_onset_step=len(examples), mixing_proportion=0.0, seed=seed)
    sim = StreamSimulator(config=config, classifier=classifier, seed=seed, reference_examples=examples, shifted_examples=[])

    ref_window = ReferenceWindow(min_size=window_size, n_bootstrap=100)
    ref_records = []
    stream_iter = iter(sim)
    for i, record in enumerate(stream_iter):
        ref_window.add(record)
        ref_records.append(record)
        if i + 1 >= window_size:
            break

    frozen = ref_window.freeze()
    ks_det = KSDetector(frozen_stats=frozen, window_size=window_size)

    # Warm up KS on reference records
    for rec in ref_records:
        ks_det.update(rec)

    # Continue and track max KS
    max_ks = 0.0
    ks_values = []
    for record in stream_iter:
        val = ks_det.update(record)
        ks_values.append(val)
        if val > max_ks:
            max_ks = val

    return max_ks, ks_values


def run_detection_with_threshold(classifier, reference, shifted, shift_onset, window_size, seed, threshold, score_offset=0.0):
    """Run detection using empirical threshold instead of CS bounds."""
    config = StreamConfig(
        shift_condition="paraphrase" if shifted else None,
        shift_onset_step=shift_onset, mixing_proportion=1.0, seed=seed,
    )
    sim = StreamSimulator(config=config, classifier=classifier, seed=seed, reference_examples=reference, shifted_examples=shifted or [])

    ref_window = ReferenceWindow(min_size=window_size, n_bootstrap=100)
    ref_records = []
    stream_iter = iter(sim)
    step = 0
    for record in stream_iter:
        ref_window.add(record)
        ref_records.append(record)
        step += 1
        if step >= window_size:
            break

    frozen = ref_window.freeze()
    ks_det = KSDetector(frozen_stats=frozen, window_size=window_size)
    for rec in ref_records:
        ks_det.update(rec)

    alarm_step = None
    pre_scores, post_scores = [], []
    for record in stream_iter:
        step += 1
        # Apply synthetic offset post-onset
        if score_offset and step > shift_onset and shifted:
            from dataclasses import replace
            record = replace(record, score=min(record.score + score_offset, 1.0))

        if step <= shift_onset:
            pre_scores.append(record.score)
        else:
            post_scores.append(record.score)

        val = ks_det.update(record)
        # Alarm: KS exceeds calibrated threshold after warmup
        if val > threshold and step > 2 * window_size and alarm_step is None:
            alarm_step = step

    return {
        "alarm_step": alarm_step,
        "detection_latency": (alarm_step - shift_onset) if alarm_step else None,
        "mean_pre": float(np.mean(pre_scores)) if pre_scores else None,
        "mean_post": float(np.mean(post_scores)) if post_scores else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full canary with empirical FAR calibration")
    parser.add_argument("--no-synthetic-shift", action="store_true",
                        help="Run without synthetic offset (real shift detection only)")
    args = parser.parse_args()
    score_offset = 0.0 if args.no_synthetic_shift else SYNTHETIC_SHIFT

    wall_start = time.time()

    checkpoint = os.environ.get("DEBERTA_CHECKPOINT_PATH")
    if checkpoint:
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        print(f"Classifier: DeBERTaAdapter ({checkpoint})")
        classifier = DeBERTaAdapter()
    else:
        from scripts.canary_run import CanaryClassifier
        print("DEBERTA_CHECKPOINT_PATH not set — using mock classifier")
        classifier = CanaryClassifier(dim=768)

    # Load reference data — unharmful only
    print("Loading WildGuardMix reference (unharmful only)...")
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=SEED)
    prompts = ds["prompt"]
    reference = [{"text": prompts[i], "source_dataset": "wildguardmix-unharmful"} for i in range(N_REFERENCE)]
    neg_pool = [{"text": prompts[N_REFERENCE + i], "source_dataset": "wildguardmix-unharmful"} for i in range(N_REFERENCE * 2)]
    print(f"  {len(ds)} unharmful examples available")

    # --- Empirical FAR calibration ---
    print(f"\nCalibrating alarm threshold ({N_CALIBRATION_RUNS} negative control runs)...")
    max_ks_values = []
    for run_i in range(N_CALIBRATION_RUNS):
        # Use different slices of the negative pool
        start = (run_i * 50) % len(neg_pool)
        run_examples = neg_pool[start:start + N_REFERENCE]
        if len(run_examples) < WINDOW_SIZE:
            run_examples = neg_pool[:N_REFERENCE]
        max_ks, _ = run_stream_ks(classifier, run_examples, WINDOW_SIZE, seed=SEED + run_i * 7)
        max_ks_values.append(max_ks)
        if (run_i + 1) % 10 == 0:
            print(f"  {run_i + 1}/{N_CALIBRATION_RUNS} done")

    threshold = float(np.percentile(max_ks_values, 97))
    print(f"  Calibrated threshold (97th pct of max KS): {threshold:.4f}")
    print(f"  Max KS range: [{min(max_ks_values):.4f}, {max(max_ks_values):.4f}]")
    print()

    # --- Run each shift condition ---
    results = []
    for ci, (condition, corpus_path) in enumerate(SHIFT_CORPORA.items()):
        print(f"--- {condition} ---")
        shifted = load_shifted(corpus_path)
        if not shifted:
            shifted = [{"text": prompts[N_REFERENCE + i], "source_dataset": "synthetic"} for i in range(300)]

        cond_seed = SEED + ci * 100

        # Positive run with synthetic offset
        pos = run_detection_with_threshold(
            classifier, reference, shifted, SHIFT_ONSET, WINDOW_SIZE, cond_seed, threshold, score_offset
        )

        # Negative control
        neg = run_detection_with_threshold(
            classifier, neg_pool[:N_REFERENCE], None, 0, WINDOW_SIZE, cond_seed + 1, threshold
        )

        # Debug negative control failure
        if neg["alarm_step"] is not None:
            print(f"  [NEG CTRL DEBUG] alarm at step {neg['alarm_step']}, seed={cond_seed+1}")
            # Re-run to get max KS
            max_ks, ks_vals = run_stream_ks(classifier, neg_pool[:N_REFERENCE], WINDOW_SIZE, seed=cond_seed + 1)
            print(f"  [NEG CTRL DEBUG] max_ks={max_ks:.4f} vs threshold={threshold:.4f}")

        neg_clean = neg["alarm_step"] is None
        results.append({
            "condition": condition,
            "neg_clean": neg_clean,
            "alarm": pos["alarm_step"] is not None,
            "latency": pos["detection_latency"],
            "mean_pre": pos["mean_pre"],
            "mean_post": pos["mean_post"],
        })
        pre_str = f"{pos['mean_pre']:.4f}" if pos['mean_pre'] else "-"
        post_str = f"{pos['mean_post']:.4f}" if pos['mean_post'] else "-"
        print(f"  alarm={pos['alarm_step']}, latency={pos['detection_latency']}, neg_clean={neg_clean}, pre={pre_str}, post={post_str}")
        print()

    # Summary table
    print("=" * 85)
    print(f"{'Condition':<28} {'Neg Clean':<10} {'Alarm':<7} {'Latency':<10} {'Pre':<8} {'Post':<8} {'Threshold'}")
    print("-" * 85)
    for r in results:
        pre = f"{r['mean_pre']:.4f}" if r['mean_pre'] is not None else "-"
        post = f"{r['mean_post']:.4f}" if r['mean_post'] is not None else "-"
        print(f"{r['condition']:<28} {'✓' if r['neg_clean'] else '✗':<10} "
              f"{'✓' if r['alarm'] else '✗':<7} "
              f"{str(r['latency'] or '-'):<10} "
              f"{pre:<8} {post:<8} {threshold:.4f}")
    print("=" * 85)
    print(f"Wall-clock time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
