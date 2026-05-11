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
import time

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


def make_examples(n: int, prefix: str, rng: np.random.Generator) -> list[dict]:
    """Generate synthetic examples with a given prefix."""
    return [
        {"text": f"{prefix} example {i}: {rng.random():.6f}", "source_dataset": prefix}
        for i in range(n)
    ]


def run_detection(
    classifier,
    reference_examples: list[dict],
    shifted_examples: list[dict] | None,
    shift_onset: int,
    window_size: int,
    seed: int,
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
    alarm_step = None
    cs_bounds = []
    step = 0

    # Phase 1: Fill reference window
    stream_iter = iter(simulator)
    for record in stream_iter:
        ref_window.add(record)
        step += 1
        if step >= window_size:
            break

    frozen_stats = ref_window.freeze()

    # Phase 2: Set up detectors
    mmd_detector = MMDDetector(frozen_stats=frozen_stats, window_size=window_size)
    ks_detector = KSDetector(frozen_stats=frozen_stats, window_size=window_size)

    alarm_controller = AlarmController(
        alpha=0.05,
        correction_method="bonferroni",
        combined_window=50,
        window_mode="sliding",
        window_size=window_size,
        min_warmup_steps=window_size,
        tail_bound="bounded",
        lower_bound=0.0,
        upper_bound=1.0,
    )

    mmd_engine = alarm_controller.register_detector("mmd", frozen_stats.mmd_reference_value)
    ks_engine = alarm_controller.register_detector("ks", 0.0)

    # Phase 3: Continue streaming through detectors
    for record in stream_iter:
        step += 1

        mmd_val = mmd_detector.update(record)
        if mmd_val is not None:
            mmd_update = mmd_engine.update(mmd_val)
            alarm_controller.report_update("mmd", mmd_update)

        ks_val = ks_detector.update(record)
        ks_update = ks_engine.update(ks_val)
        alarm_controller.report_update("ks", ks_update)

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

    latency = (alarm_step - shift_onset) if alarm_step and shifted_examples else None
    return {
        "alarm_step": alarm_step,
        "shift_onset": shift_onset,
        "detection_latency": latency,
        "fired_within_200": latency is not None and latency <= 200,
        "cs_bounds": cs_bounds,
        "total_steps": step,
    }


def main(
    seed: int = 42,
    window_size: int = 200,
    shift_onset: int = 500,
    n_reference: int = 500,
    n_shifted: int = 300,
    dim: int = 1024,
):
    wall_start = time.time()
    rng = np.random.default_rng(seed)

    classifier = CanaryClassifier(dim=dim)

    print("=" * 60)
    print("CANARY RUN: DeBERTa + Paraphrase Shift + Regime A")
    print("=" * 60)

    # Generate corpus
    reference = make_examples(n_reference, "reference", rng)
    shifted = make_examples(n_shifted, "shifted altered", rng)

    # --- Positive run ---
    print("\n--- Positive Control (shift at step %d) ---" % shift_onset)
    pos = run_detection(
        classifier=classifier,
        reference_examples=reference,
        shifted_examples=shifted,
        shift_onset=shift_onset,
        window_size=window_size,
        seed=seed,
    )

    if pos["alarm_step"]:
        print(f"  Alarm fired at step: {pos['alarm_step']}")
        print(f"  Detection latency: {pos['detection_latency']} steps")
        print(f"  Fired within 200 steps of onset: {pos['fired_within_200']}")
    else:
        print("  No alarm fired (detection missed)")

    # --- Negative control ---
    print("\n--- Negative Control (no shift) ---")
    neg_reference = make_examples(n_reference + n_shifted, "reference", np.random.default_rng(99))
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
    args = parser.parse_args()
    main(
        seed=args.seed,
        window_size=args.window_size,
        shift_onset=args.shift_onset,
        n_reference=args.n_reference,
        n_shifted=args.n_shifted,
        dim=args.dim,
    )
