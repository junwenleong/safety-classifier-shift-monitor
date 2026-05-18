"""Analyze factorial results with confidence intervals.

Wilson Score 95% CIs on rates, bootstrap 95% CIs on mean latencies.

Usage:
    python scripts/analyze_factorial.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("results/factorial_results.jsonl")
CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "code-switch", "compositional-long-context", "temporal", "adversarial-suffix"]
N_BOOT = 1000
RNG = np.random.default_rng(42)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson Score 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(values: list[float], n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Bootstrap 95% CI on the mean. Returns (mean, ci_lower, ci_upper)."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return (mean, mean, mean)
    boot_means = np.array([float(np.mean(RNG.choice(arr, size=len(arr), replace=True)))
                           for _ in range(n_boot)])
    return (mean, float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def load_and_backfill():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "is_valid_detection" not in r:
                r["is_valid_detection"] = (
                    r.get("detection_latency") is not None
                    and r["detection_latency"] >= 0
                    and r.get("neg_clean") is True
                )
            rows.append(r)
    return rows


def main():
    rows = load_and_backfill()
    n_valid = sum(1 for r in rows if r["is_valid_detection"])
    ci = wilson_ci(n_valid, len(rows))
    print(f"Total cells: {len(rows)}")
    print(f"Valid detections: {n_valid}/{len(rows)} = {n_valid/len(rows):.1%} [{ci[0]:.3f}, {ci[1]:.3f}]")
    print()

    # 1. Mean latency with bootstrap CIs
    print("=" * 100)
    print("MEAN DETECTION LATENCY with 95% bootstrap CI (valid detections only)")
    print("-" * 100)
    header = f"{'Classifier':<18}" + "".join(f"{s[:14]:<20}" for s in SHIFTS)
    print(header)

    for clf in CLASSIFIERS:
        row = f"{clf:<18}"
        for shift in SHIFTS:
            valid = [r["detection_latency"] for r in rows
                     if r["classifier"] == clf and r["shift_condition"] == shift
                     and r["is_valid_detection"]]
            if len(valid) >= 2:
                mean, lo, hi = bootstrap_ci(valid)
                row += f"{mean:>5.1f} [{lo:.0f},{hi:.0f}] n={len(valid)}  "
            elif len(valid) == 1:
                row += f"{valid[0]:>5.1f} (n=1)          "
            else:
                row += f"{'—':<20}"
        print(row)
    print()

    # 2. Detection rate with Wilson Score CIs
    print("=" * 100)
    print("DETECTION RATE with 95% Wilson Score CI")
    print("-" * 100)
    header = f"{'Classifier':<18}" + "".join(f"{s[:14]:<22}" for s in SHIFTS)
    print(header)

    for clf in CLASSIFIERS:
        row = f"{clf:<18}"
        for shift in SHIFTS:
            cells = [r for r in rows if r["classifier"] == clf and r["shift_condition"] == shift]
            k = sum(1 for r in cells if r["is_valid_detection"])
            n = len(cells)
            lo, hi = wilson_ci(k, n)
            row += f"{k}/{n} [{lo:.2f},{hi:.2f}]     "
        print(row)
    print()

    # 3. FAR with Wilson Score CIs
    print("=" * 100)
    print("FALSE ALARM RATE with 95% Wilson Score CI")
    print("-" * 100)
    for clf in CLASSIFIERS:
        cells = [r for r in rows if r["classifier"] == clf]
        k = sum(1 for r in cells if not r.get("neg_clean", True))
        n = len(cells)
        lo, hi = wilson_ci(k, n)
        print(f"  {clf:<18} {k}/{n} = {k/n:.3f} [{lo:.3f}, {hi:.3f}]")
    print()

    # 4. Window size effect with bootstrap CIs
    print("=" * 100)
    print("WINDOW SIZE EFFECT with 95% bootstrap CI")
    print("-" * 100)
    for ws in [100, 200]:
        valid = [r["detection_latency"] for r in rows
                 if r["window_size"] == ws and r["is_valid_detection"]]
        mean, lo, hi = bootstrap_ci(valid)
        print(f"  w={ws}: mean={mean:.1f} [{lo:.1f}, {hi:.1f}]  n={len(valid)}")
    print()

    # 5. Top/bottom 3
    print("=" * 100)
    print("TOP 3 (fastest) and BOTTOM 3 (slowest) with bootstrap CIs")
    print("-" * 100)
    combos = []
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            valid = [r["detection_latency"] for r in rows
                     if r["classifier"] == clf and r["shift_condition"] == shift
                     and r["is_valid_detection"]]
            if len(valid) >= 2:
                mean, lo, hi = bootstrap_ci(valid)
                combos.append((clf, shift, mean, lo, hi, len(valid)))

    combos.sort(key=lambda x: x[2])
    print("  FASTEST:")
    for clf, shift, mean, lo, hi, n in combos[:3]:
        print(f"    {clf} × {shift}: {mean:.1f} [{lo:.1f}, {hi:.1f}] (n={n})")
    print("  SLOWEST:")
    for clf, shift, mean, lo, hi, n in combos[-3:]:
        print(f"    {clf} × {shift}: {mean:.1f} [{lo:.1f}, {hi:.1f}] (n={n})")


if __name__ == "__main__":
    main()
