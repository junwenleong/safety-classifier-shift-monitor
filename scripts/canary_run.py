"""Canary run: single classifier, single shift, full pipeline test.

- Classifier: DeBERTa (base model, mocked for canary)
- Shift condition: paraphrase
- Regime A: synthetic injected shift
- Seed: 42, window size: 200
- Stream: 500 reference + 300 shifted
- Negative control: 800 reference, no shift
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from shift_detection_monitor.config import StreamConfig
from shift_detection_monitor.detection.alarm_controller import AlarmController
from shift_detection_monitor.detection.ks_detector import KSDetector
from shift_detection_monitor.detection.mmd_detector import MMDDetector
from shift_detection_monitor.detection.reference_window import ReferenceWindow
from shift_detection_monitor.stream.simulator import StreamSimulator
from shift_detection_monitor.types import ClassifierOutput, StreamRecord


class CanaryClassifier:
    """Mock classifier that produces distinct distributions for reference vs shifted text."""

    def __init__(self, dim: int = 1024):
        self._dim = dim

    @property
    def name(self) -> str:
        return "deberta-v3-large"

    @property
    def embedding_dim(self) -> int | None:
        return self._dim

    def predict(self, text: str) -> ClassifierOutput:
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        is_shifted = "shifted" in text or "altered" in text
        base_score = 0.7 if is_shifted else 0.2
        score = float(np.clip(base_score + rng.normal(0, 0.05), 0.0, 1.0))
        mean_offset = 2.0 if is_shifted else 0.0
        representation = (rng.standard_normal(self._dim) + mean_offset).astype(np.float64)
        return ClassifierOutput(score=score, representation=representation, metadata={})


def run_detection(
    classifier,
    reference_examples: list[dict],
    shifted_examples: list[dict] | None,
    shift_onset: int,
    window_size: int,
    seed: int,
    score_offset: float = 0.0,
) -> dict:
    """Run the full detection pipeline on a single stream."""
    config = StreamConfig(
        shift_condition="paraphrase" if shifted_examples else None,
        shift_onset_step=shift_onset,
        mixing_proportion=1.0,
        seed=seed,
    )

    simulator = StreamSimulator(
        config=config,
        classifier=classifier,
        seed=seed,
        reference_examples=reference_examples,
        shifted_examples=shifted_examples or [],
    )

    # Collect all records in one pass
    ref_window = ReferenceWindow(min_size=window_size, n_bootstrap=200)
    ref_records = []
    alarm_step = None
    cs_bounds = []
    step = 0

    # Phase 1: Fill reference window
    stream_iter = iter(simulator)
    for record in stream_iter:
        ref_window.add(record)
        ref_records.append(record)
        step += 1
        if step >= window_size:
            break

    frozen_stats = ref_window.freeze()

    # Phase 2: Compute KS null reference value by running KS detector over reference window
    # The KS statistic is always > 0 under the null; using 0.0 causes immediate alarming
    ks_ref_detector = KSDetector(frozen_stats=frozen_stats, window_size=window_size)
    ks_null_values = []
    for record in ref_records:
        ks_val = ks_ref_detector.update(record)
        ks_null_values.append(ks_val)
    ks_reference_value = float(np.mean(ks_null_values))

    # Phase 3: Set up detectors
    mmd_detector = MMDDetector(frozen_stats=frozen_stats, window_size=window_size)
    ks_detector = KSDetector(frozen_stats=frozen_stats, window_size=window_size)

    alarm_controller = AlarmController(
        alpha=0.05,
        correction_method="bonferroni",
        combined_window=50,
        window_mode="sliding",
        window_size=window_size,
        min_warmup_steps=2 * window_size,
        tail_bound="bounded",
        lower_bound=0.0,
        upper_bound=1.0,
    )

    mmd_engine = alarm_controller.register_detector("mmd", frozen_stats.mmd_reference_value)
    ks_engine = alarm_controller.register_detector("ks", ks_reference_value)

    # Phase 3: Continue streaming through detectors
    debug_steps = {300, 350, 400, 450}
    for record in stream_iter:
        step += 1

        # Synthetic shift: offset scores post-onset
        if score_offset and step > shift_onset and shifted_examples:
            from dataclasses import replace
            record = replace(record, score=min(record.score + score_offset, 1.0))

        mmd_val = mmd_detector.update(record)
        if mmd_val is not None:
            mmd_update = mmd_engine.update(mmd_val)
            alarm_controller.report_update("mmd", mmd_update)

        ks_val = ks_detector.update(record)
        ks_update = ks_engine.update(ks_val)
        alarm_controller.report_update("ks", ks_update)

        # Debug: print CS state at pre-shift steps
        if step in debug_steps and shifted_examples:
            mmd_info = f" | MMD=[{mmd_update.lower:.6f},{mmd_update.upper:.6f}] stat={mmd_update.statistic:.6f} alarm={mmd_update.alarm}" if mmd_val is not None else ""
            print(f"    [debug] step={step} KS=[{ks_update.lower:.4f},{ks_update.upper:.4f}] stat={ks_update.statistic:.4f} alarm={ks_update.alarm}{mmd_info}")

        # Track CS bounds post-shift
        if shifted_examples and step > shift_onset and len(cs_bounds) < 100:
            cs_bounds.append({
                "step": step,
                "ks_lower": ks_update.lower,
                "ks_upper": ks_update.upper,
                "ks_stat": ks_update.statistic,
            })

        alarms = alarm_controller.check_alarms()
        if alarms and alarm_step is None:
            alarm_step = alarms[0].time_step
            for a in alarms:
                print(f"    [ALARM] step={a.time_step} detector={a.detector} stat={a.statistic_value:.6f} ref={a.reference_value:.6f} CS=[{a.cs_lower:.6f},{a.cs_upper:.6f}]")

    latency = (alarm_step - shift_onset) if alarm_step and shifted_examples else None
    return {
        "alarm_step": alarm_step,
        "shift_onset": shift_onset,
        "detection_latency": latency,
        "fired_within_200": latency is not None and latency <= 200,
        "cs_bounds": cs_bounds,
        "total_steps": step,
        "ks_reference_value": ks_reference_value,
        "mmd_reference_value": frozen_stats.mmd_reference_value,
    }


def main(
    seed: int = 42,
    window_size: int = 200,
    shift_onset: int = 500,
    n_reference: int = 500,
    n_shifted: int = 300,
    dim: int = 1024,
    synthetic_shift: float = 0.0,
):
    wall_start = time.time()
    rng = np.random.default_rng(seed)

    # Use real DeBERTaAdapter if checkpoint is available, else mock
    checkpoint = os.environ.get("DEBERTA_CHECKPOINT_PATH")
    if checkpoint:
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        print(f"Using DeBERTaAdapter with checkpoint: {checkpoint}")
        classifier = DeBERTaAdapter()
    else:
        print("DEBERTA_CHECKPOINT_PATH not set — using mock classifier")
        classifier = CanaryClassifier(dim=dim)

    print("=" * 60)
    print("CANARY RUN: DeBERTa + Paraphrase Shift + Regime A")
    print("=" * 60)

    # Load real data from WildGuardMix
    print("\nLoading WildGuardMix prompts...")
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.shuffle(seed=seed)
    prompts = ds["prompt"]

    # Reference: first n_reference prompts
    reference = [{"text": prompts[i], "source_dataset": "wildguardmix"} for i in range(n_reference)]

    # Shifted: load from paraphrase corpus if available, else placeholder
    paraphrase_path = Path("data/shifted/paraphrase/output.jsonl")
    if paraphrase_path.exists():
        import json
        with open(paraphrase_path) as f:
            shifted = [json.loads(line) for line in f if line.strip()]
        if len(shifted) < n_shifted:
            n_shifted = len(shifted)
        else:
            shifted = shifted[:n_shifted]
        print(f"  Loaded {n_shifted} paraphrased examples from {paraphrase_path}")
    else:
        shifted = [
            {"text": "Rephrase: " + prompts[n_reference + i], "source_dataset": "placeholder"}
            for i in range(n_shifted)
        ]
        print(f"  No paraphrase corpus found — using placeholder ({n_shifted} examples)")

    print(f"  Reference: {n_reference}, Shifted: {n_shifted}")
    print(f"  First 3 reference texts:")
    for i in range(min(3, len(reference))):
        print(f"    [{i}] {reference[i]['text'][:80]}...")
    print(f"  First 3 shifted texts:")
    for i in range(min(3, len(shifted))):
        print(f"    [{i}] {shifted[i].get('shifted', shifted[i].get('text', ''))[:80]}...")

    # --- Positive run ---
    print("\n--- Positive Control (shift at step %d) ---" % shift_onset)
    if synthetic_shift:
        print(f"  Synthetic score offset: +{synthetic_shift}")
    pos = run_detection(
        classifier=classifier,
        reference_examples=reference,
        shifted_examples=shifted,
        shift_onset=shift_onset,
        window_size=window_size,
        seed=seed,
        score_offset=synthetic_shift,
    )

    if pos["alarm_step"]:
        print(f"  Alarm fired at step: {pos['alarm_step']}")
        print(f"  Detection latency: {pos['detection_latency']} steps")
        print(f"  Fired within 200 steps of onset: {pos['fired_within_200']}")
    else:
        print("  No alarm fired (detection missed)")
    print(f"  KS reference value: {pos['ks_reference_value']:.6f}")
    print(f"  MMD reference value: {pos['mmd_reference_value']:.6f}")

    # --- Negative control ---
    print("\n--- Negative Control (no shift) ---")
    # Use a different slice of WildGuardMix as pure reference (no shift)
    neg_start = n_reference + n_shifted
    neg_reference = [
        {"text": prompts[neg_start + i], "source_dataset": "wildguardmix"}
        for i in range(n_reference + n_shifted)
    ]
    neg = run_detection(
        classifier=classifier,
        reference_examples=neg_reference,
        shifted_examples=None,
        shift_onset=0,
        window_size=window_size,
        seed=seed + 1,
    )

    neg_clean = neg["alarm_step"] is None
    print(f"  Alarm fired: {not neg_clean}")
    print(f"  Negative control clean: {neg_clean}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Detection latency: {pos['detection_latency']}")
    print(f"  Negative control clean: {neg_clean}")

    if pos["cs_bounds"]:
        print(f"\n  CS bounds (first 10 of {len(pos['cs_bounds'])} post-shift steps):")
        for b in pos["cs_bounds"][:10]:
            print(f"    step={b['step']:4d}  KS=[{b['ks_lower']:.4f}, {b['ks_upper']:.4f}]  stat={b['ks_stat']:.4f}")

    wall_time = time.time() - wall_start
    print(f"\n  Wall-clock time: {wall_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Canary run: single-classifier shift detection pipeline test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--window-size", type=int, default=200, help="Reference window / detector window size")
    parser.add_argument("--shift-onset", type=int, default=500, help="Time step where shift begins")
    parser.add_argument("--n-reference", type=int, default=500, help="Number of reference examples")
    parser.add_argument("--n-shifted", type=int, default=300, help="Number of shifted examples")
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimensionality for mock classifier")
    parser.add_argument("--synthetic-shift", type=float, default=0.0,
                        help="Add fixed score offset post-onset to simulate adversarial shift (e.g. 0.3)")
    args = parser.parse_args()
    main(
        seed=args.seed,
        window_size=args.window_size,
        shift_onset=args.shift_onset,
        n_reference=args.n_reference,
        n_shifted=args.n_shifted,
        dim=args.dim,
        synthetic_shift=args.synthetic_shift,
    )
