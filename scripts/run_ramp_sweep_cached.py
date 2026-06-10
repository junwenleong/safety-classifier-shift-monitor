"""Ramp-rate and mixing-level sweep using CACHED scores (no live inference).

Uses the cached streams from cache_embeddings.py to simulate gradual drift
by subsampling shifted examples at various mixing proportions. This avoids
the checkpoint-loading issue in run_gradual_drift.py.

Usage:
    .venv/bin/python scripts/run_ramp_sweep_cached.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.detection.confidence_sequence import ConfidenceSequenceEngine
from shift_detection_monitor.detection.ks_detector import KSDetector
from shift_detection_monitor.detection.reference_window import ReferenceWindow
from shift_detection_monitor.types import StreamRecord

CACHE_DIR = Path("results/cached_streams")
NULL_SCORES = Path("results/null_scores.json")
OUTPUT = Path("results/ramp_rate_sweep.json")

CLASSIFIER = "deberta"
SHIFT = "paraphrase"
SHIFT_ONSET = 500
WINDOW_SIZE = 100
N_SEEDS = 10  # use all cached seeds
RAMP_DURATIONS = [50, 100, 150, 200]
MIXING_LEVELS = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
MIXING_30_EXTRA_SEEDS = 20  # extra seeds for 30% mixing (total n=30 with bootstrap)
FAST_RAMP = 50
CS_ALPHA = 0.05
N_CAL = 50  # null streams for KS threshold calibration
CAL_PCT = 97


def load_cached_stream(seed: int):
    """Load cached scores from the embedding cache."""
    path = CACHE_DIR / CLASSIFIER / SHIFT / f"seed_{seed}.npz"
    data = np.load(path)
    return data["scores"], data["is_shifted"]


def simulate_gradual_stream(ref_scores, shifted_scores, ramp_duration, max_mixing, seed):
    """Simulate a gradual-drift stream by mixing reference and shifted scores.
    
    Before SHIFT_ONSET: all reference scores.
    After SHIFT_ONSET: at each step, draw shifted with probability that
    ramps from 0 to max_mixing over ramp_duration steps.
    """
    rng = random.Random(seed + 9999)
    ref_pool = list(ref_scores)
    shift_pool = list(shifted_scores)
    rng.shuffle(shift_pool)

    stream = []
    shift_idx = 0
    ref_idx = 0

    # Pre-onset: use reference scores in order
    for ref_idx in range(min(SHIFT_ONSET, len(ref_pool))):
        stream.append(ref_pool[ref_idx])
    ref_idx += 1

    # Post-onset: mix according to ramp
    for t in range(300):  # up to 300 post-onset steps
        steps_since = t
        mix_prob = min(max_mixing, max_mixing * steps_since / ramp_duration)

        if rng.random() < mix_prob and shift_idx < len(shift_pool):
            stream.append(shift_pool[shift_idx])
            shift_idx += 1
        elif ref_idx < len(ref_pool):
            stream.append(ref_pool[ref_idx])
            ref_idx += 1
        else:
            break

    return np.array(stream)


def calibrate_ks_threshold(all_ref_scores):
    """Calibrate KS threshold from null streams."""
    rng = random.Random(42)
    max_ks_values = []

    for _ in range(N_CAL):
        pool = list(all_ref_scores)
        rng.shuffle(pool)
        pool = pool[:SHIFT_ONSET]

        # Build reference window from first WINDOW_SIZE
        ref_window = ReferenceWindow(min_size=WINDOW_SIZE, n_bootstrap=100)
        for i in range(WINDOW_SIZE):
            rec = StreamRecord(time_step=i, text="", score=pool[i], representation=None,
                             ground_truth_label=None, is_shifted=False,
                             source_dataset="ref", shift_condition=None)
            ref_window.add(rec)
        frozen = ref_window.freeze()
        ks_det = KSDetector(frozen_stats=frozen, window_size=WINDOW_SIZE)

        # Feed all scores and track max after warmup
        max_ks = 0.0
        for i, score in enumerate(pool):
            rec = StreamRecord(time_step=i, text="", score=score, representation=None,
                             ground_truth_label=None, is_shifted=False,
                             source_dataset="ref", shift_condition=None)
            val = ks_det.update(rec)
            if i >= 2 * WINDOW_SIZE and val > max_ks:
                max_ks = val
        max_ks_values.append(max_ks)

    return float(np.percentile(max_ks_values, CAL_PCT))


def run_ks_detection(stream_scores, threshold):
    """Run KS detection on a score stream."""
    ref_window = ReferenceWindow(min_size=WINDOW_SIZE, n_bootstrap=100)
    for i in range(min(WINDOW_SIZE, len(stream_scores))):
        rec = StreamRecord(time_step=i, text="", score=stream_scores[i], representation=None,
                         ground_truth_label=None, is_shifted=False,
                         source_dataset="ref", shift_condition=None)
        ref_window.add(rec)
    frozen = ref_window.freeze()
    ks_det = KSDetector(frozen_stats=frozen, window_size=WINDOW_SIZE)

    alarm_step = None
    for i, score in enumerate(stream_scores):
        rec = StreamRecord(time_step=i, text="", score=score, representation=None,
                         ground_truth_label=None, is_shifted=False,
                         source_dataset="ref", shift_condition=None)
        val = ks_det.update(rec)
        if val > threshold and i >= 2 * WINDOW_SIZE and alarm_step is None:
            alarm_step = i

    latency = (alarm_step - SHIFT_ONSET) if alarm_step is not None else None
    return latency


def run_cs_detection(stream_scores, ref_mean):
    """Run CS growing-window detection on a score stream."""
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
    for t, score in enumerate(stream_scores):
        result = engine.update(score)
        if result.alarm and alarm_step is None:
            alarm_step = t

    latency = (alarm_step - SHIFT_ONSET) if alarm_step is not None else None
    return latency


def main():
    print("=" * 60)
    print("RAMP-RATE & MIXING SWEEP (cached scores, no live inference)")
    print(f"  Classifier: {CLASSIFIER}, Shift: {SHIFT}")
    print(f"  Ramp durations: {RAMP_DURATIONS}")
    print(f"  Mixing levels: {MIXING_LEVELS}")
    print(f"  Seeds: {N_SEEDS}")
    print("=" * 60)

    # Load all cached scores
    all_scores = []
    all_ref_scores = []
    all_shifted_scores = []
    for seed in range(N_SEEDS):
        scores, is_shifted = load_cached_stream(seed)
        all_scores.append(scores)
        all_ref_scores.extend(scores[~is_shifted])
        all_shifted_scores.extend(scores[is_shifted])

    # Reference mean for CS
    null_data = json.load(open(NULL_SCORES))
    ref_mean = float(np.mean(null_data[CLASSIFIER]))
    print(f"\n  Reference mean: {ref_mean:.4f}")
    print(f"  Shifted mean: {np.mean(all_shifted_scores):.4f}")
    print(f"  Reference pool: {len(all_ref_scores)}, Shifted pool: {len(all_shifted_scores)}")

    # Calibrate KS threshold
    print("\n  Calibrating KS threshold...")
    threshold = calibrate_ks_threshold(all_ref_scores)
    print(f"  KS Threshold: {threshold:.4f}")

    results = {"threshold": threshold, "ref_mean": ref_mean, "classifier": CLASSIFIER, "shift": SHIFT}

    # Part 1: Ramp-rate sweep at max_mixing=0.5
    print(f"\n{'='*60}")
    print("PART 1: Ramp-rate sweep (max_mixing=0.5)")
    results["ramp_durations"] = {}
    for ramp_dur in RAMP_DURATIONS:
        ks_lats, cs_lats = [], []
        for seed in range(N_SEEDS):
            scores, is_shifted = load_cached_stream(seed)
            ref_scores = scores[~is_shifted]
            shifted_scores = scores[is_shifted]
            # Use pooled reference (all seeds) for post-onset filler
            stream = simulate_gradual_stream(all_ref_scores[:1000], shifted_scores, ramp_dur, 0.5, seed)
            ks_lats.append(run_ks_detection(stream, threshold))
            cs_lats.append(run_cs_detection(stream, ref_mean))

        ks_valid = [l for l in ks_lats if l is not None and l >= 0]
        cs_valid = [l for l in cs_lats if l is not None and l >= 0]
        results["ramp_durations"][str(ramp_dur)] = {
            "ks": {"n_detected": len(ks_valid), "detection_rate": len(ks_valid)/N_SEEDS,
                   "mean_latency": float(np.mean(ks_valid)) if ks_valid else None, "latencies": ks_lats},
            "cs": {"n_detected": len(cs_valid), "detection_rate": len(cs_valid)/N_SEEDS,
                   "mean_latency": float(np.mean(cs_valid)) if cs_valid else None, "latencies": cs_lats},
        }
        ks_str = f"{len(ks_valid)}/{N_SEEDS}" + (f" mean={np.mean(ks_valid):.1f}" if ks_valid else "")
        cs_str = f"{len(cs_valid)}/{N_SEEDS}" + (f" mean={np.mean(cs_valid):.1f}" if cs_valid else "")
        print(f"  Ramp {ramp_dur}: KS {ks_str}, CS {cs_str}")

    # Part 2: Mixing-level sweep at fast ramp
    print(f"\n{'='*60}")
    print(f"PART 2: Mixing-level sweep (ramp={FAST_RAMP} steps)")
    results["mixing_levels"] = {}
    for mix_level in MIXING_LEVELS:
        ks_lats, cs_lats = [], []
        for seed in range(N_SEEDS):
            scores, is_shifted = load_cached_stream(seed)
            ref_scores = scores[~is_shifted]
            shifted_scores = scores[is_shifted]
            # Use pooled reference for post-onset filler
            stream = simulate_gradual_stream(all_ref_scores[:1000], shifted_scores, FAST_RAMP, mix_level, seed)
            ks_lats.append(run_ks_detection(stream, threshold))
            cs_lats.append(run_cs_detection(stream, ref_mean))

        ks_valid = [l for l in ks_lats if l is not None and l >= 0]
        cs_valid = [l for l in cs_lats if l is not None and l >= 0]
        results["mixing_levels"][str(mix_level)] = {
            "ks": {"n_detected": len(ks_valid), "detection_rate": len(ks_valid)/N_SEEDS,
                   "mean_latency": float(np.mean(ks_valid)) if ks_valid else None, "latencies": ks_lats},
            "cs": {"n_detected": len(cs_valid), "detection_rate": len(cs_valid)/N_SEEDS,
                   "mean_latency": float(np.mean(cs_valid)) if cs_valid else None, "latencies": cs_lats},
        }
        ks_str = f"{len(ks_valid)}/{N_SEEDS}" + (f" mean={np.mean(ks_valid):.1f}" if ks_valid else "")
        cs_str = f"{len(cs_valid)}/{N_SEEDS}" + (f" mean={np.mean(cs_valid):.1f}" if cs_valid else "")
        print(f"  Mix {mix_level*100:.0f}%: KS {ks_str}, CS {cs_str}")

    # Part 3: Extended test at 30% mixing (n=30 for significance)
    print(f"\n{'='*60}")
    print(f"PART 3: 30% mixing extended (n=30, ramp={FAST_RAMP} steps)")
    print("  (Uses bootstrap resampling of reference pool for extra seeds)")

    ks_lats_30 = []
    cs_lats_30 = []
    # First 10 seeds: use cached shifted scores directly
    for seed in range(N_SEEDS):
        scores, is_shifted = load_cached_stream(seed)
        shifted_scores = scores[is_shifted]
        stream = simulate_gradual_stream(all_ref_scores[:1000], shifted_scores, FAST_RAMP, 0.3, seed)
        ks_lats_30.append(run_ks_detection(stream, threshold))
        cs_lats_30.append(run_cs_detection(stream, ref_mean))

    # Extra 20 seeds: resample shifted pool with different seeds
    for extra_seed in range(MIXING_30_EXTRA_SEEDS):
        # Use a different cached seed's shifted scores with offset random seed
        base_seed = extra_seed % N_SEEDS
        scores, is_shifted = load_cached_stream(base_seed)
        shifted_scores = scores[is_shifted]
        stream = simulate_gradual_stream(all_ref_scores[:1000], shifted_scores, FAST_RAMP, 0.3, seed=extra_seed + 100)
        ks_lats_30.append(run_ks_detection(stream, threshold))
        cs_lats_30.append(run_cs_detection(stream, ref_mean))

    ks_valid_30 = [l for l in ks_lats_30 if l is not None and l >= 0]
    cs_valid_30 = [l for l in cs_lats_30 if l is not None and l >= 0]

    results["mixing_30_extended"] = {
        "n": len(ks_lats_30),
        "ks": {"n_detected": len(ks_valid_30), "detection_rate": len(ks_valid_30)/len(ks_lats_30),
               "mean_latency": float(np.mean(ks_valid_30)) if ks_valid_30 else None, "latencies": ks_lats_30},
        "cs": {"n_detected": len(cs_valid_30), "detection_rate": len(cs_valid_30)/len(cs_lats_30),
               "mean_latency": float(np.mean(cs_valid_30)) if cs_valid_30 else None, "latencies": cs_lats_30},
    }
    print(f"  KS: {len(ks_valid_30)}/{len(ks_lats_30)}" + (f" mean={np.mean(ks_valid_30):.1f}" if ks_valid_30 else ""))
    print(f"  CS: {len(cs_valid_30)}/{len(cs_lats_30)}" + (f" mean={np.mean(cs_valid_30):.1f}" if cs_valid_30 else ""))

    # Fisher exact test
    from scipy.stats import fisher_exact
    table = [[len(cs_valid_30), len(cs_lats_30) - len(cs_valid_30)],
             [len(ks_valid_30), len(ks_lats_30) - len(ks_valid_30)]]
    _, p_fisher = fisher_exact(table, alternative='greater')
    print(f"  Fisher exact (CS > KS): p = {p_fisher:.4f}")
    results["mixing_30_extended"]["fisher_p"] = p_fisher

    # Save
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
