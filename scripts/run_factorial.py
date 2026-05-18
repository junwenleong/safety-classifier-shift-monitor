"""Full factorial evaluation harness.

Iterates over classifiers × shift conditions × seeds × window sizes.
Writes results incrementally to results/factorial_results.jsonl.

Usage:
    export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
    export TEXT_MODERATION_CHECKPOINT_PATH=checkpoints/text-moderation-wildguardmix
    python scripts/run_factorial.py              # Regime A (default)
    python scripts/run_factorial.py --regime-c   # Regime C (adversarial success)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_full_canary import (
    SHIFT_CORPORA,
    load_shifted,
    run_detection_with_threshold,
    run_stream_ks,
)

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFT_CONDITIONS = ["paraphrase", "code-switch", "compositional-long-context", "temporal", "adversarial-suffix"]
SEEDS = list(range(20))
WINDOW_SIZES = [100, 200]
N_REFERENCE = 500
N_CALIBRATION = 50
CAL_PCT = 97
OUTPUT = Path("results/factorial_results.jsonl")
OUTPUT_REGIME_B = Path("results/regime_b_results.jsonl")
OUTPUT_REGIME_C = Path("results/regime_c_results.jsonl")


def get_classifier(name: str):
    if name == "deberta":
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        return DeBERTaAdapter()
    elif name == "text-moderation":
        from shift_detection_monitor.classifiers.gpt_oss_safeguard import TextModerationAdapter
        return TextModerationAdapter()
    elif name == "shieldgemma":
        from shift_detection_monitor.classifiers.shieldgemma import ShieldGemmaAdapter
        return ShieldGemmaAdapter()
    elif name == "llama-guard":
        from shift_detection_monitor.classifiers.llama_guard import LlamaGuard3Adapter
        return LlamaGuard3Adapter()


def load_reference():
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=42)
    prompts = ds["prompt"]
    ref = [{"text": prompts[i], "source_dataset": "wildguardmix-unharmful"} for i in range(N_REFERENCE)]
    neg = [{"text": prompts[N_REFERENCE + i], "source_dataset": "wildguardmix-unharmful"} for i in range(N_REFERENCE * 2)]
    return ref, neg


def load_done() -> set:
    if not OUTPUT.exists():
        return set()
    keys = set()
    with open(OUTPUT) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                keys.add((r["classifier"], r["shift_condition"], r["seed"], r["window_size"]))
    return keys


def load_done_regime_b() -> set:
    if not OUTPUT_REGIME_B.exists():
        return set()
    keys = set()
    with open(OUTPUT_REGIME_B) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                keys.add((r["classifier"], r["seed"], r["window_size"]))
    return keys


def run_regime_b():
    """Regime B: temporal split — detect shift using real temporal corpus."""
    wall_start = time.time()
    OUTPUT_REGIME_B.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_regime_b()

    regime_b_classifiers = ["deberta", "llama-guard"]
    regime_b_seeds = list(range(5))
    total = len(regime_b_classifiers) * len(regime_b_seeds) * len(WINDOW_SIZES)
    print(f"Regime B: {total} cells, {len(done)} already done, {total - len(done)} remaining")

    print("Loading reference data...")
    reference, neg_pool = load_reference()

    temporal_path = Path("data/shifted/temporal/output.jsonl")
    shifted = load_shifted(temporal_path, n=292)
    print(f"  Loaded {len(shifted)} temporal shift examples")

    for clf_name in regime_b_classifiers:
        cells = [(seed, ws) for seed, ws in product(regime_b_seeds, WINDOW_SIZES)
                 if (clf_name, seed, ws) not in done]
        if not cells:
            print(f"\n[{clf_name}] all done, skipping")
            continue

        print(f"\n[{clf_name}] {len(cells)} cells remaining, loading classifier...")
        classifier = get_classifier(clf_name)

        thresholds = {}
        for ws in WINDOW_SIZES:
            if ws not in thresholds:
                print(f"  Calibrating window_size={ws}...")
                max_ks = []
                for i in range(N_CALIBRATION):
                    start = (i * 50) % len(neg_pool)
                    examples = neg_pool[start:start + N_REFERENCE]
                    if len(examples) < ws:
                        examples = neg_pool[:N_REFERENCE]
                    mk, _ = run_stream_ks(classifier, examples, ws, seed=42 + i * 7)
                    max_ks.append(mk)
                thresholds[ws] = float(np.percentile(max_ks, CAL_PCT))
                print(f"    threshold={thresholds[ws]:.4f}")

        for seed, ws in cells:
            threshold = thresholds[ws]
            cond_seed = seed * 1000 + 500

            pos = run_detection_with_threshold(
                classifier, reference, shifted, N_REFERENCE, ws, cond_seed, threshold
            )
            neg = run_detection_with_threshold(
                classifier, neg_pool[:N_REFERENCE], None, 0, ws, cond_seed + 1, threshold
            )

            result = {
                "classifier": clf_name,
                "shift_condition": "temporal",
                "regime": "B",
                "seed": seed,
                "window_size": ws,
                "alarm_step": pos["alarm_step"],
                "detection_latency": pos["detection_latency"],
                "neg_clean": neg["alarm_step"] is None,
                "threshold": threshold,
                "is_valid_detection": (
                    pos["detection_latency"] is not None
                    and pos["detection_latency"] >= 0
                    and neg["alarm_step"] is None
                ),
            }

            with open(OUTPUT_REGIME_B, "a") as f:
                f.write(json.dumps(result) + "\n")

            lat = pos["detection_latency"] or "-"
            neg_ok = "✓" if result["neg_clean"] else "✗"
            print(f"  [{clf_name}/temporal/s{seed}/w{ws}] latency={lat} neg={neg_ok}")

    # Summary
    print("\n" + "=" * 60)
    print("REGIME B SUMMARY (temporal split, robustness check)")
    print("-" * 60)
    if OUTPUT_REGIME_B.exists():
        results = [json.loads(l) for l in open(OUTPUT_REGIME_B) if l.strip()]
        valid = [r for r in results if r.get("is_valid_detection")]
        print(f"Total cells: {len(results)}")
        print(f"Valid detections: {len(valid)}/{len(results)} = {len(valid)/len(results):.1%}")
        if valid:
            lats = [r["detection_latency"] for r in valid]
            print(f"Mean latency: {sum(lats)/len(lats):.1f} steps")
        for clf in regime_b_classifiers:
            v = [r for r in valid if r["classifier"] == clf]
            c = [r for r in results if r["classifier"] == clf]
            if c:
                lats = [r["detection_latency"] for r in v]
                mean = f"{sum(lats)/len(lats):.1f}" if lats else "-"
                print(f"  {clf}: {len(v)}/{len(c)} detected, mean={mean}")
    print(f"Wall-clock: {time.time() - wall_start:.0f}s")


def load_done_regime_c() -> set:
    if not OUTPUT_REGIME_C.exists():
        return set()
    keys = set()
    with open(OUTPUT_REGIME_C) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                keys.add((r["classifier"], r["seed"], r["window_size"]))
    return keys


def load_adversarial_success():
    """Load adversarial suffix examples where success=True."""
    path = Path("data/shifted/adversarial_suffix/deberta_suffixes.jsonl")
    if not path.exists():
        print(f"ERROR: {path} not found. Run GCG first or pull from Mac Studio.")
        sys.exit(1)
    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("success"):
                text = r.get("combined") or r.get("text", "")
                if text:
                    examples.append({"text": text[:1000], "source_dataset": "gcg-success"})
    print(f"  Loaded {len(examples)} successful adversarial suffix examples")
    return examples


def run_regime_c():
    """Regime C: adversarial success — detect shift from successful GCG attacks."""
    wall_start = time.time()
    OUTPUT_REGIME_C.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_regime_c()
    total = len(CLASSIFIERS) * len(SEEDS) * len(WINDOW_SIZES)
    print(f"Regime C: {total} cells, {len(done)} already done, {total - len(done)} remaining")

    print("Loading reference data...")
    reference, neg_pool = load_reference()

    print("Loading adversarial success examples...")
    shifted = load_adversarial_success()
    if len(shifted) < 50:
        print(f"WARNING: only {len(shifted)} successful examples — results may be noisy")

    for clf_name in CLASSIFIERS:
        cells = [(seed, ws) for seed, ws in product(SEEDS, WINDOW_SIZES)
                 if (clf_name, seed, ws) not in done]
        if not cells:
            print(f"\n[{clf_name}] all done, skipping")
            continue

        print(f"\n[{clf_name}] {len(cells)} cells remaining, loading classifier...")
        classifier = get_classifier(clf_name)

        thresholds = {}
        for ws in WINDOW_SIZES:
            if ws not in thresholds:
                print(f"  Calibrating window_size={ws}...")
                max_ks = []
                for i in range(N_CALIBRATION):
                    start = (i * 50) % len(neg_pool)
                    examples = neg_pool[start:start + N_REFERENCE]
                    if len(examples) < ws:
                        examples = neg_pool[:N_REFERENCE]
                    mk, _ = run_stream_ks(classifier, examples, ws, seed=42 + i * 7)
                    max_ks.append(mk)
                thresholds[ws] = float(np.percentile(max_ks, CAL_PCT))
                print(f"    threshold={thresholds[ws]:.4f}")

        for seed, ws in cells:
            threshold = thresholds[ws]
            cond_seed = seed * 1000 + 999

            pos = run_detection_with_threshold(
                classifier, reference, shifted, N_REFERENCE, ws, cond_seed, threshold
            )
            neg = run_detection_with_threshold(
                classifier, neg_pool[:N_REFERENCE], None, 0, ws, cond_seed + 1, threshold
            )

            result = {
                "classifier": clf_name,
                "shift_condition": "adversarial-success",
                "regime": "C",
                "seed": seed,
                "window_size": ws,
                "alarm_step": pos["alarm_step"],
                "detection_latency": pos["detection_latency"],
                "neg_clean": neg["alarm_step"] is None,
                "threshold": threshold,
                "is_valid_detection": (
                    pos["detection_latency"] is not None
                    and pos["detection_latency"] >= 0
                    and neg["alarm_step"] is None
                ),
            }

            with open(OUTPUT_REGIME_C, "a") as f:
                f.write(json.dumps(result) + "\n")

            lat = pos["detection_latency"] or "-"
            neg_ok = "✓" if result["neg_clean"] else "✗"
            print(f"  [{clf_name}/adv-success/s{seed}/w{ws}] latency={lat} neg={neg_ok}")

    # Summary
    print("\n" + "=" * 60)
    print("REGIME C SUMMARY")
    print("-" * 60)
    if OUTPUT_REGIME_C.exists():
        results = [json.loads(l) for l in open(OUTPUT_REGIME_C) if l.strip()]
        valid = [r for r in results if r.get("is_valid_detection")]
        print(f"Total cells: {len(results)}")
        print(f"Valid detections: {len(valid)}/{len(results)} = {len(valid)/len(results):.1%}")
        if valid:
            lats = [r["detection_latency"] for r in valid]
            print(f"Mean latency: {sum(lats)/len(lats):.1f} steps")
        for clf in CLASSIFIERS:
            v = [r for r in valid if r["classifier"] == clf]
            c = [r for r in results if r["classifier"] == clf]
            if c:
                lats = [r["detection_latency"] for r in v]
                mean = f"{sum(lats)/len(lats):.1f}" if lats else "-"
                print(f"  {clf}: {len(v)}/{len(c)} detected, mean={mean}")
    print(f"Wall-clock: {time.time() - wall_start:.0f}s")


def main():
    wall_start = time.time()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    total = len(CLASSIFIERS) * len(SHIFT_CONDITIONS) * len(SEEDS) * len(WINDOW_SIZES)
    print(f"Factorial: {total} cells, {len(done)} already done, {total - len(done)} remaining")

    print("Loading reference data...")
    reference, neg_pool = load_reference()

    # Group by classifier to avoid reloading models
    for clf_name in CLASSIFIERS:
        cells = [(s, seed, ws) for s, seed, ws in product(SHIFT_CONDITIONS, SEEDS, WINDOW_SIZES)
                 if (clf_name, s, seed, ws) not in done]
        if not cells:
            print(f"\n[{clf_name}] all done, skipping")
            continue

        print(f"\n[{clf_name}] {len(cells)} cells remaining, loading classifier...")
        classifier = get_classifier(clf_name)

        # Calibrate per window size
        thresholds = {}
        for ws in WINDOW_SIZES:
            if ws not in thresholds:
                print(f"  Calibrating window_size={ws}...")
                max_ks = []
                for i in range(N_CALIBRATION):
                    start = (i * 50) % len(neg_pool)
                    examples = neg_pool[start:start + N_REFERENCE]
                    if len(examples) < ws:
                        examples = neg_pool[:N_REFERENCE]
                    mk, _ = run_stream_ks(classifier, examples, ws, seed=42 + i * 7)
                    max_ks.append(mk)
                thresholds[ws] = float(np.percentile(max_ks, CAL_PCT))
                print(f"    threshold={thresholds[ws]:.4f}")

        # Run cells
        for shift_cond, seed, ws in cells:
            shifted = load_shifted(SHIFT_CORPORA.get(shift_cond), n=300)
            if not shifted:
                shifted = [{"text": reference[i]["text"], "source_dataset": "synthetic"} for i in range(300)]

            threshold = thresholds[ws]
            cond_seed = seed * 1000 + hash(shift_cond) % 1000

            pos = run_detection_with_threshold(
                classifier, reference, shifted, N_REFERENCE, ws, cond_seed, threshold
            )
            neg = run_detection_with_threshold(
                classifier, neg_pool[:N_REFERENCE], None, 0, ws, cond_seed + 1, threshold
            )

            result = {
                "classifier": clf_name,
                "shift_condition": shift_cond,
                "seed": seed,
                "window_size": ws,
                "alarm_step": pos["alarm_step"],
                "detection_latency": pos["detection_latency"],
                "neg_clean": neg["alarm_step"] is None,
                "pre_score": pos.get("mean_pre"),
                "post_score": pos.get("mean_post"),
                "threshold": threshold,
                "is_valid_detection": (
                    pos["detection_latency"] is not None
                    and pos["detection_latency"] >= 0
                    and neg["alarm_step"] is None
                ),
            }

            with open(OUTPUT, "a") as f:
                f.write(json.dumps(result) + "\n")

            lat = pos["detection_latency"] or "-"
            neg_ok = "✓" if result["neg_clean"] else "✗"
            print(f"  [{clf_name}/{shift_cond}/s{seed}/w{ws}] latency={lat} neg={neg_ok}")

    # Summary matrix
    print("\n" + "=" * 80)
    print("SUMMARY: Mean detection latency (classifiers × shift conditions)")
    print("-" * 80)

    results = []
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            results = [json.loads(line) for line in f if line.strip()]

    header = f"{'Classifier':<18}" + "".join(f"{s[:12]:<14}" for s in SHIFT_CONDITIONS)
    print(header)
    for clf in CLASSIFIERS:
        row = f"{clf:<18}"
        for shift in SHIFT_CONDITIONS:
            lats = [r["detection_latency"] for r in results
                    if r["classifier"] == clf and r["shift_condition"] == shift
                    and r.get("is_valid_detection", r.get("detection_latency") is not None and r.get("detection_latency", -1) >= 0)]
            row += f"{(sum(lats)/len(lats)):.0f}{'':>8}" if lats else f"{'-':<14}"
        print(row)

    print("=" * 80)
    print(f"Wall-clock time: {time.time() - wall_start:.0f}s")
    print(f"Results: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-b", action="store_true", help="Run Regime B (temporal split, robustness check)")
    parser.add_argument("--regime-c", action="store_true", help="Run Regime C (adversarial success, proof of concept)")
    args = parser.parse_args()

    if args.regime_b:
        run_regime_b()
    elif args.regime_c:
        run_regime_c()
    else:
        main()
