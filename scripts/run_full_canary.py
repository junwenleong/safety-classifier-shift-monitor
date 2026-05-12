"""Full canary: run monitor across all 5 shift conditions with DeBERTa.

Reports a summary table with detection results per condition.
Uses --synthetic-shift to validate detection machinery.

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.canary_run import CanaryClassifier, run_detection

SHIFT_CORPORA = {
    "paraphrase": Path("data/shifted/paraphrase/output.jsonl"),
    "code-switch": Path("data/shifted/code-switch/output.jsonl"),
    "compositional-long-context": Path("data/shifted/compositional/output.jsonl"),
    "temporal": Path("data/shifted/temporal/output.jsonl"),
    "adversarial-suffix": None,  # No corpus yet — use reference with synthetic offset only
}

WINDOW_SIZE = 100
SHIFT_ONSET = 500
N_REFERENCE = 500
SYNTHETIC_SHIFT = 0.5
SEED = 42


def load_shifted(path: Path | None, n: int = 300) -> list[dict]:
    """Load shifted examples from corpus, cycle if needed."""
    if path is None or not path.exists():
        return []
    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    if not raw:
        return []
    if len(raw) >= n:
        return raw[:n]
    return (raw * ((n // len(raw)) + 1))[:n]


def main():
    wall_start = time.time()

    # Load classifier
    checkpoint = os.environ.get("DEBERTA_CHECKPOINT_PATH")
    if checkpoint:
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        print(f"Classifier: DeBERTaAdapter ({checkpoint})")
        classifier = DeBERTaAdapter()
    else:
        print("DEBERTA_CHECKPOINT_PATH not set — using mock classifier")
        classifier = CanaryClassifier(dim=768)

    # Load reference data
    print("Loading WildGuardMix reference...")
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.shuffle(seed=SEED)
    prompts = ds["prompt"]
    reference = [{"text": prompts[i], "source_dataset": "wildguardmix"} for i in range(N_REFERENCE)]
    # Separate slice for negative controls
    neg_reference = [{"text": prompts[N_REFERENCE + i], "source_dataset": "wildguardmix"} for i in range(N_REFERENCE + 300)]

    print(f"Reference: {N_REFERENCE}, Window: {WINDOW_SIZE}, Onset: {SHIFT_ONSET}, Offset: +{SYNTHETIC_SHIFT}")
    print()

    # Run each condition
    results = []
    for ci, (condition, corpus_path) in enumerate(SHIFT_CORPORA.items()):
        print(f"--- {condition} ---")
        shifted = load_shifted(corpus_path)
        if not shifted:
            shifted = [{"text": prompts[N_REFERENCE + i], "source_dataset": "synthetic"} for i in range(300)]

        # Use per-condition seed for independent reference windows
        cond_seed = SEED + ci * 100

        # Diagnostic: iterate stream and print pre/post onset info
        from shift_detection_monitor.stream.simulator import StreamSimulator
        from shift_detection_monitor.config import StreamConfig
        diag_config = StreamConfig(
            shift_condition="paraphrase" if shifted else None,
            shift_onset_step=SHIFT_ONSET, mixing_proportion=1.0, seed=cond_seed,
        )
        diag_sim = StreamSimulator(
            config=diag_config, classifier=classifier, seed=cond_seed,
            reference_examples=reference, shifted_examples=shifted,
        )
        pre_records, post_records = [], []
        for rec in diag_sim:
            if rec.time_step < SHIFT_ONSET:
                pre_records.append(rec)
            else:
                post_records.append(rec)
        print(f"  Pre-onset ({len(pre_records)} records):")
        for rec in pre_records[:3]:
            print(f"    step={rec.time_step} shifted={rec.is_shifted} score={rec.score:.4f} text={rec.text[:50]}")
        print(f"    ...")
        for rec in pre_records[-3:]:
            print(f"    step={rec.time_step} shifted={rec.is_shifted} score={rec.score:.4f} text={rec.text[:50]}")
        pre_scores = [r.score for r in pre_records]
        post_scores = [r.score for r in post_records]
        print(f"  Mean pre-onset score: {sum(pre_scores)/len(pre_scores):.4f}, Mean post-onset score: {sum(post_scores)/len(post_scores):.4f}" if post_scores else "  No post-onset records")

        # Positive run
        pos = run_detection(
            classifier=classifier,
            reference_examples=reference,
            shifted_examples=shifted,
            shift_onset=SHIFT_ONSET,
            window_size=WINDOW_SIZE,
            seed=cond_seed,
            score_offset=SYNTHETIC_SHIFT,
        )

        # Negative control (different reference slice, different seed)
        neg = run_detection(
            classifier=classifier,
            reference_examples=neg_reference,
            shifted_examples=None,
            shift_onset=0,
            window_size=WINDOW_SIZE,
            seed=cond_seed + 1,
        )

        neg_clean = neg["alarm_step"] is None
        mean_pre = pos.get("mean_score_pre")
        mean_post = pos.get("mean_score_post")
        results.append({
            "condition": condition,
            "neg_clean": neg_clean,
            "alarm": pos["alarm_step"] is not None,
            "latency": pos["detection_latency"],
            "mean_pre": mean_pre,
            "mean_post": mean_post,
        })
        status = f"alarm={pos['alarm_step']}, latency={pos['detection_latency']}, neg_clean={neg_clean}"
        print(f"  {status}")
        if mean_pre is not None:
            print(f"  scores: pre={mean_pre:.4f} post={mean_post:.4f}")
        print()

    # Summary table
    print("=" * 80)
    print(f"{'Condition':<28} {'Neg Clean':<10} {'Alarm':<7} {'Latency':<10} {'Pre Score':<10} {'Post Score'}")
    print("-" * 80)
    for r in results:
        pre = f"{r['mean_pre']:.4f}" if r['mean_pre'] is not None else "-"
        post = f"{r['mean_post']:.4f}" if r['mean_post'] is not None else "-"
        print(f"{r['condition']:<28} {'✓' if r['neg_clean'] else '✗':<10} "
              f"{'✓' if r['alarm'] else '✗':<7} "
              f"{str(r['latency'] or '-'):<10} "
              f"{pre:<10} {post}")
    print("=" * 80)
    print(f"Wall-clock time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
