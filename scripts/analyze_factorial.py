"""Analyze factorial results for the paper.

Reads results/factorial_results.jsonl, backfills is_valid_detection,
and produces summary tables filtered to valid detections only.

Usage:
    python scripts/analyze_factorial.py
"""

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results/factorial_results.jsonl")

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "code-switch", "compositional-long-context", "temporal", "adversarial-suffix"]


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
    print(f"Total cells: {len(rows)}")
    print(f"Valid detections: {sum(1 for r in rows if r['is_valid_detection'])}")
    print()

    # 1. Summary table: mean latency (valid only), classifiers × shifts
    print("=" * 90)
    print("MEAN DETECTION LATENCY (valid detections only)")
    print("-" * 90)
    header = f"{'Classifier':<18}" + "".join(f"{s[:14]:<16}" for s in SHIFTS)
    print(header)

    for clf in CLASSIFIERS:
        row = f"{clf:<18}"
        for shift in SHIFTS:
            valid = [r["detection_latency"] for r in rows
                     if r["classifier"] == clf and r["shift_condition"] == shift
                     and r["is_valid_detection"]]
            if valid:
                row += f"{sum(valid)/len(valid):>6.1f} (n={len(valid)})  "
            else:
                row += f"{'- (n=0)':<16}"
        print(row)
    print()

    # 2. Detection rate per cell (fraction of seeds producing valid detections)
    print("=" * 90)
    print("DETECTION RATE (fraction of seeds with valid detection)")
    print("-" * 90)
    header = f"{'Classifier':<18}" + "".join(f"{s[:14]:<16}" for s in SHIFTS)
    print(header)

    for clf in CLASSIFIERS:
        row = f"{clf:<18}"
        for shift in SHIFTS:
            cells = [r for r in rows if r["classifier"] == clf and r["shift_condition"] == shift]
            n_valid = sum(1 for r in cells if r["is_valid_detection"])
            row += f"{n_valid}/{len(cells):<13}" if cells else f"{'0/0':<16}"
        print(row)
    print()

    # 3. False alarm rate per classifier
    print("=" * 90)
    print("FALSE ALARM RATE (fraction of cells with neg_clean=False)")
    print("-" * 90)
    for clf in CLASSIFIERS:
        cells = [r for r in rows if r["classifier"] == clf]
        n_false = sum(1 for r in cells if not r.get("neg_clean", True))
        print(f"  {clf:<18} {n_false}/{len(cells)} = {n_false/len(cells):.3f}")
    print()

    # 4. Window size effect
    print("=" * 90)
    print("WINDOW SIZE EFFECT (mean latency, valid detections only)")
    print("-" * 90)
    for ws in [100, 200]:
        valid = [r["detection_latency"] for r in rows
                 if r["window_size"] == ws and r["is_valid_detection"]]
        print(f"  w={ws}: mean={sum(valid)/len(valid):.1f}  n={len(valid)}")
    print()

    # 5. Top 3 and bottom 3 by mean latency
    print("=" * 90)
    print("TOP 3 (lowest latency) and BOTTOM 3 (highest latency)")
    print("-" * 90)
    combos = []
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            valid = [r["detection_latency"] for r in rows
                     if r["classifier"] == clf and r["shift_condition"] == shift
                     and r["is_valid_detection"]]
            if valid:
                combos.append((clf, shift, sum(valid) / len(valid), len(valid)))

    combos.sort(key=lambda x: x[2])
    print("  FASTEST:")
    for clf, shift, mean, n in combos[:3]:
        print(f"    {clf} × {shift}: {mean:.1f} (n={n})")
    print("  SLOWEST:")
    for clf, shift, mean, n in combos[-3:]:
        print(f"    {clf} × {shift}: {mean:.1f} (n={n})")


if __name__ == "__main__":
    main()
