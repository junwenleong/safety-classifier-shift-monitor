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
    for condition, corpus_path in SHIFT_CORPORA.items():
        print(f"--- {condition} ---")
        shifted = load_shifted(corpus_path)
        if not shifted:
            # For adversarial-suffix: use reference examples as shifted (offset does the work)
            shifted = [{"text": prompts[N_REFERENCE + i], "source_dataset": "synthetic"} for i in range(300)]

        # Positive run
        pos = run_detection(
            classifier=classifier,
            reference_examples=reference,
            shifted_examples=shifted,
            shift_onset=SHIFT_ONSET,
            window_size=WINDOW_SIZE,
            seed=SEED,
            score_offset=SYNTHETIC_SHIFT,
        )

        # Negative control
        neg = run_detection(
            classifier=classifier,
            reference_examples=neg_reference,
            shifted_examples=None,
            shift_onset=0,
            window_size=WINDOW_SIZE,
            seed=SEED + 1,
        )

        neg_clean = neg["alarm_step"] is None
        results.append({
            "condition": condition,
            "neg_clean": neg_clean,
            "alarm": pos["alarm_step"] is not None,
            "latency": pos["detection_latency"],
            "mean_pre": pos["mean_score_pre"],
            "mean_post": pos["mean_score_post"],
        })
        status = f"alarm={pos['alarm_step']}, latency={pos['detection_latency']}, neg_clean={neg_clean}"
        print(f"  {status}")
        print()

    # Summary table
    print("=" * 80)
    print(f"{'Condition':<28} {'Neg Clean':<10} {'Alarm':<7} {'Latency':<10} {'Pre Score':<10} {'Post Score'}")
    print("-" * 80)
    for r in results:
        print(f"{r['condition']:<28} {'✓' if r['neg_clean'] else '✗':<10} "
              f"{'✓' if r['alarm'] else '✗':<7} "
              f"{str(r['latency'] or '-'):<10} "
              f"{r['mean_pre']:.4f}     {r['mean_post']:.4f}" if r['mean_pre'] else "")
    print("=" * 80)
    print(f"Wall-clock time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
