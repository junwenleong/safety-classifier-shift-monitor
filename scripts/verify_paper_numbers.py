"""Verify all paper numbers against raw data.

Reads factorial_results.jsonl, computes every statistic in paper/draft.md,
asserts they match.

Usage:
    python scripts/verify_paper_numbers.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

RESULTS = Path("results/factorial_results.jsonl")
CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "code-switch", "compositional-long-context", "temporal", "adversarial-suffix"]

PASS = 0
FAIL = 0


def check(name: str, expected, actual, tol=0.1):
    global PASS, FAIL
    if abs(expected - actual) <= tol:
        print(f"  ✓ {name}: {actual:.4f} (expected {expected})")
        PASS += 1
    else:
        print(f"  ✗ {name}: {actual:.4f} != {expected} (diff={abs(expected-actual):.3f})")
        FAIL += 1


def main():
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    for r in rows:
        r["is_valid_detection"] = (
            r.get("detection_latency") is not None
            and r["detection_latency"] >= 0
            and r.get("neg_clean") is True
        )
    valid = [r for r in rows if r["is_valid_detection"]]

    print("PAPER NUMBER VERIFICATION (N=20, 800 cells)")
    print("=" * 60)

    # Abstract / overall
    print("\n[Abstract & Overall]")
    check("Total cells", 800, len(rows), tol=0)
    check("Valid detections", 693, len(valid), tol=0)
    check("Detection rate %", 86.6, len(valid) / len(rows) * 100, tol=0.1)

    w100 = [r["detection_latency"] for r in valid if r["window_size"] == 100]
    w200 = [r["detection_latency"] for r in valid if r["window_size"] == 200]
    check("w=100 mean latency", 39.5, sum(w100) / len(w100), tol=0.5)
    check("w=200 mean latency", 45.4, sum(w200) / len(w200), tol=0.5)

    # FAR per classifier
    print("\n[False Alarm Rates]")
    far_expected = {"deberta": 0.095, "text-moderation": 0.02, "llama-guard": 0.03, "shieldgemma": 0.085}
    for clf in CLASSIFIERS:
        cells = [r for r in rows if r["classifier"] == clf]
        far = sum(1 for r in cells if not r.get("neg_clean", True)) / len(cells)
        check(f"FAR {clf}", far_expected[clf], far, tol=0.005)

    # Variance decomposition (compute from raw data)
    print("\n[Variance Decomposition]")
    latencies = np.array([r["detection_latency"] for r in valid], dtype=float)
    clf_labels = [r["classifier"] for r in valid]
    shift_labels = [r["shift_condition"] for r in valid]
    grand_mean = np.mean(latencies)
    ss_total = np.sum((latencies - grand_mean) ** 2)

    ss_clf = sum(
        np.sum(np.array([c == clf for c in clf_labels])) *
        (np.mean(latencies[np.array([c == clf for c in clf_labels])]) - grand_mean) ** 2
        for clf in CLASSIFIERS
    )
    ss_shift = sum(
        np.sum(np.array([s == shift for s in shift_labels])) *
        (np.mean(latencies[np.array([s == shift for s in shift_labels])]) - grand_mean) ** 2
        for shift in SHIFTS
    )
    ss_int = 0
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            mask = np.array([c == clf and s == shift for c, s in zip(clf_labels, shift_labels)])
            if mask.sum() > 0:
                clf_mask = np.array([c == clf for c in clf_labels])
                shift_mask = np.array([s == shift for s in shift_labels])
                effect = np.mean(latencies[mask]) - np.mean(latencies[clf_mask]) - np.mean(latencies[shift_mask]) + grand_mean
                ss_int += mask.sum() * effect ** 2

    check("η² classifier", 0.243, ss_clf / ss_total, tol=0.002)
    check("η² shift_type", 0.237, ss_shift / ss_total, tol=0.002)
    check("η² interaction", 0.185, ss_int / ss_total, tol=0.002)
    check("η² residual", 0.335, (ss_total - ss_clf - ss_shift - ss_int) / ss_total, tol=0.002)

    # Key cell latencies from paper
    print("\n[Key Cell Latencies]")
    key_cells = [
        ("deberta", "adversarial-suffix", 36.6),
        ("text-moderation", "code-switch", 33.2),
        ("llama-guard", "code-switch", 93.4),
        ("llama-guard", "adversarial-suffix", 27.8),
        ("shieldgemma", "paraphrase", 85.0),
        ("shieldgemma", "adversarial-suffix", 26.8),
    ]
    for clf, shift, expected in key_cells:
        lats = [r["detection_latency"] for r in valid
                if r["classifier"] == clf and r["shift_condition"] == shift]
        actual = sum(lats) / len(lats) if lats else 0
        check(f"{clf} × {shift}", expected, actual, tol=0.5)

    # Detection rates for key cells
    print("\n[Key Detection Rates]")
    check("DeBERTa × adversarial invalid", 13,
          sum(1 for r in rows if r["classifier"] == "deberta" and r["shift_condition"] == "adversarial-suffix" and not r["is_valid_detection"]),
          tol=0)
    check("Llama Guard × code-switch detected", 40,
          sum(1 for r in rows if r["classifier"] == "llama-guard" and r["shift_condition"] == "code-switch" and r["is_valid_detection"]),
          tol=0)

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
    print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
