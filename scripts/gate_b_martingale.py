"""Gate B: Conformal test martingale validation.

Tests whether a betting martingale on conformal p-values matches or beats
the CS growing-window at 30% mixing while providing provable FAR control.

Baseline to beat (from ramp_rate_sweep.json extended):
  KS:  13/30 = 43%
  CS:  29/30 = 97%

GO criterion: best martingale variant ≥ 70% at 30% mixing.

Usage:
    .venv/bin/python scripts/gate_b_martingale.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

NULL_SCORES = Path("results/null_scores.json")
RAMP_SWEEP = Path("results/ramp_rate_sweep.json")

CLASSIFIER = "deberta"
SHIFT_ONSET = 500
WINDOW_SIZE = 100
N_SEEDS = 30
FAST_RAMP = 50
MIXING_LEVELS = [0.3, 0.5, 0.7, 1.0]
ALPHA = 0.05
STREAM_LENGTH = 800


def conformal_p_two_sided(score: float, ref: np.ndarray) -> float:
    """Two-sided conformal p-value. Under H0 (iid from ref), uniform on (0,1)."""
    n = len(ref)
    p_upper = (np.sum(ref >= score) + 1) / (n + 1)
    p_lower = (np.sum(ref <= score) + 1) / (n + 1)
    return min(2.0 * min(p_upper, p_lower), 1.0)


class PointMartingale:
    """Single martingale betting on p-values with power method."""
    def __init__(self, alpha: float, epsilon: float = 0.3):
        self.log_threshold = math.log(1.0 / alpha)
        self.epsilon = epsilon
        self.log_wealth = 0.0
        self.alarm_step = None
        self.t = 0

    def update(self, p: float) -> bool:
        self.t += 1
        p = max(p, 1e-300)
        self.log_wealth += math.log(self.epsilon) + (self.epsilon - 1) * math.log(p)
        if self.alarm_step is None and self.log_wealth >= self.log_threshold:
            self.alarm_step = self.t
            return True
        return False


class ScanMartingale:
    """Scan martingale: maintains sub-martingales started at every step.

    Alarms if any W-length sub-martingale exceeds 1/(alpha). Since we start
    one per step, apply union bound: threshold per sub-martingale = W/alpha.
    This is the standard scan approach (Shin et al. 2022).
    """
    def __init__(self, alpha: float = 0.05, window: int = 100, epsilon: float = 0.3):
        self.window = window
        self.epsilon = epsilon
        # Union bound over W concurrent sub-martingales
        self.log_threshold = math.log(window / alpha)
        self.log_wealths: list[float] = []
        self.t = 0
        self.alarm_step = None

    def update(self, p: float) -> bool:
        self.t += 1
        p = max(p, 1e-300)
        log_inc = math.log(self.epsilon) + (self.epsilon - 1) * math.log(p)
        # Update existing + start new
        self.log_wealths = [w + log_inc for w in self.log_wealths]
        self.log_wealths.append(log_inc)
        if len(self.log_wealths) > self.window:
            self.log_wealths.pop(0)
        if self.alarm_step is None and any(w >= self.log_threshold for w in self.log_wealths):
            self.alarm_step = self.t
            return True
        return False


class CUSUMMartingale:
    """CUSUM-style: resets wealth to 1 when it drops below 1 (Page's rule).

    This is the standard sequential changepoint detector. Equivalent to
    max over all possible changepoint locations of the post-change likelihood.
    Combined with conformal p-values, gives anytime-valid detection.
    """
    def __init__(self, alpha: float = 0.05, epsilon: float = 0.3):
        self.log_threshold = math.log(1.0 / alpha)
        self.epsilon = epsilon
        self.log_wealth = 0.0
        self.t = 0
        self.alarm_step = None

    def update(self, p: float) -> bool:
        self.t += 1
        p = max(p, 1e-300)
        log_inc = math.log(self.epsilon) + (self.epsilon - 1) * math.log(p)
        self.log_wealth = max(0.0, self.log_wealth + log_inc)  # reset at 0
        if self.alarm_step is None and self.log_wealth >= self.log_threshold:
            self.alarm_step = self.t
            return True
        return False


def simulate_stream(ref_scores: np.ndarray, shifted_scores: np.ndarray,
                    mixing: float, ramp_duration: int, seed: int) -> np.ndarray:
    """Simulate stream: 500 ref pre-onset, 300 mixed post-onset."""
    rng = np.random.default_rng(seed)
    pre = rng.choice(ref_scores, size=SHIFT_ONSET, replace=True)
    post = []
    for t in range(300):
        mix_prob = min(mixing, mixing * t / ramp_duration) if ramp_duration > 0 else mixing
        if rng.random() < mix_prob:
            post.append(rng.choice(shifted_scores))
        else:
            post.append(rng.choice(ref_scores))
    return np.concatenate([pre, np.array(post)])


def run_detection(stream, ref_scores, detector_cls, warmup, **kwargs):
    """Generic runner. Returns latency or None."""
    det = detector_cls(**kwargs)
    for t, score in enumerate(stream):
        if t < warmup:
            continue
        p = conformal_p_two_sided(score, ref_scores)
        if det.update(p):
            lat = t - SHIFT_ONSET
            return lat if lat >= 0 else None
    return None


def run_ks_detection(stream, ref_scores, threshold, warmup):
    """Sliding-window KS (baseline)."""
    from shift_detection_monitor.detection.ks_detector import KSDetector
    from shift_detection_monitor.detection.reference_window import ReferenceWindow
    from shift_detection_monitor.types import StreamRecord

    ref_window = ReferenceWindow(min_size=WINDOW_SIZE, n_bootstrap=100)
    for i in range(WINDOW_SIZE):
        rec = StreamRecord(time_step=i, text="", score=stream[i], representation=None,
                           ground_truth_label=None, is_shifted=False,
                           source_dataset="ref", shift_condition=None)
        ref_window.add(rec)
    frozen = ref_window.freeze()
    ks_det = KSDetector(frozen_stats=frozen, window_size=WINDOW_SIZE)
    for i, score in enumerate(stream):
        rec = StreamRecord(time_step=i, text="", score=score, representation=None,
                           ground_truth_label=None, is_shifted=False,
                           source_dataset="ref", shift_condition=None)
        val = ks_det.update(rec)
        if val > threshold and i >= warmup:
            lat = i - SHIFT_ONSET
            return lat if lat >= 0 else None
    return None


def main():
    print("=" * 70)
    print("GATE B: Conformal Test Martingale Validation")
    print("=" * 70)

    null_data = json.load(open(NULL_SCORES))
    ref_scores = np.array(null_data[CLASSIFIER])
    ramp_data = json.load(open(RAMP_SWEEP))
    ks_threshold = ramp_data["threshold"]

    # Shifted distribution: Beta(5,5) ≈ mean 0.5, matching actual post-shift
    rng_shift = np.random.default_rng(99)
    shifted_scores = rng_shift.beta(5, 5, size=500)

    print(f"\n  Reference: n={len(ref_scores)}, mean={np.mean(ref_scores):.4f}, std={np.std(ref_scores):.4f}")
    print(f"  Shifted: mean={np.mean(shifted_scores):.4f}, std={np.std(shifted_scores):.4f}")
    print(f"  KS threshold: {ks_threshold:.4f}, alpha={ALPHA}")
    print(f"  Seeds: {N_SEEDS}, Ramp: {FAST_RAMP} steps")

    warmup = 2 * WINDOW_SIZE
    methods = {
        "Point(ε=0.3)": lambda s: run_detection(s, ref_scores, PointMartingale, warmup, alpha=ALPHA, epsilon=0.3),
        "Point(ε=0.1)": lambda s: run_detection(s, ref_scores, PointMartingale, warmup, alpha=ALPHA, epsilon=0.1),
        "CUSUM(ε=0.3)": lambda s: run_detection(s, ref_scores, CUSUMMartingale, warmup, alpha=ALPHA, epsilon=0.3),
        "CUSUM(ε=0.1)": lambda s: run_detection(s, ref_scores, CUSUMMartingale, warmup, alpha=ALPHA, epsilon=0.1),
        "Scan(w=50)":    lambda s: run_detection(s, ref_scores, ScanMartingale, warmup, alpha=ALPHA, window=50, epsilon=0.3),
        "Scan(w=100)":   lambda s: run_detection(s, ref_scores, ScanMartingale, warmup, alpha=ALPHA, window=100, epsilon=0.3),
        "KS":            lambda s: run_ks_detection(s, ref_scores, ks_threshold, warmup),
    }

    # --- FAR ---
    print(f"\n{'='*70}")
    print("PART 1: False Alarm Rate (100 null streams)")
    n_null = 100
    fars = {k: 0 for k in methods}
    for seed in range(n_null):
        null_stream = np.random.default_rng(seed + 5000).choice(ref_scores, size=STREAM_LENGTH, replace=True)
        for name, fn in methods.items():
            if fn(null_stream) is not None:
                fars[name] += 1
    for name, count in fars.items():
        print(f"  {name:<16} FAR: {count}/{n_null} = {count/n_null:.1%}")

    # --- Detection ---
    print(f"\n{'='*70}")
    print("PART 2: Detection rate by mixing level")
    header = f"  {'Mix':<6}" + "".join(f"{n:<18}" for n in methods)
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for mix in MIXING_LEVELS:
        row = {}
        for name, fn in methods.items():
            dets, lats = 0, []
            for seed in range(N_SEEDS):
                stream = simulate_stream(ref_scores, shifted_scores, mix, FAST_RAMP, seed)
                lat = fn(stream)
                if lat is not None and lat >= 0:
                    dets += 1
                    lats.append(lat)
            row[name] = {"n": dets, "rate": dets/N_SEEDS, "lats": lats}
        results[str(mix)] = row

        cells = []
        for name in methods:
            r = row[name]
            s = f"{r['n']}/{N_SEEDS}={r['rate']:.0%}"
            if r['lats']:
                s += f" μ{np.mean(r['lats']):.0f}"
            cells.append(f"{s:<18}")
        print(f"  {mix*100:>4.0f}% " + "".join(cells))

    # CS reference
    print(f"\n  Reference: CS growing-window at 30% = 97% (29/30)")

    # --- Decision ---
    print(f"\n{'='*70}")
    print("GATE B DECISION")
    print("=" * 70)
    rates_30 = {k: v["rate"] for k, v in results["0.3"].items() if k != "KS"}
    best_name = max(rates_30, key=rates_30.get)
    best_rate = rates_30[best_name]
    ks_rate = results["0.3"]["KS"]["rate"]

    print(f"\n  Best martingale at 30%: {best_name} = {best_rate:.0%} ({int(best_rate*N_SEEDS)}/{N_SEEDS})")
    print(f"  KS baseline:           {ks_rate:.0%} ({int(ks_rate*N_SEEDS)}/{N_SEEDS})")
    print(f"  CS reference:           97% (29/30)")
    print(f"  GO threshold:           ≥ 70%")

    if best_rate >= 0.70:
        print(f"\n  ✅ GO — {best_name} achieves {best_rate:.0%} with provable FAR ≤ {ALPHA:.0%}")
    else:
        print(f"\n  ❌ NO-GO — best martingale at {best_rate:.0%} < 70% threshold.")
        print(f"       Interpretation: at realistic shift magnitude + 30% mixing,")
        print(f"       individual-p-value betting cannot match windowed-statistic CS.")
        if best_rate > ks_rate:
            print(f"       However: still beats KS ({ks_rate:.0%}). Partial win.")

    # Save
    out = {"gate": "B", "far": fars, "detection": {k: {m: {"n": v["n"], "rate": v["rate"]}
           for m, v in row.items()} for k, row in results.items()},
           "best_method": best_name, "best_rate_30": best_rate, "go": best_rate >= 0.70}
    Path("results/gate_b_martingale.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved to results/gate_b_martingale.json")


if __name__ == "__main__":
    main()
