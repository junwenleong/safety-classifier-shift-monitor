"""Verify all paper numbers against raw data.

Reads factorial_results.jsonl and variance_decomposition.json,
computes every statistic in paper/draft.md, asserts they match.

Usage:
    python scripts/verify_paper_numbers.py
"""

import json
import math
import sys
from pathlib import Path

RESULTS = Path("results/factorial_results.jsonl")
VARIANCE = Path("results/variance_decomposition.json")

PASS = 0
FAIL = 0


def check(name: str, expected, actual, tol=0.1):
    global PASS, FAIL
    if abs(expected - actual) <= tol:
        print(f"  ✓ {name}: {actual} (expected {expected})")
        PASS += 1
    else:
        print(f"  ✗ {name}: {actual} != {expected} (diff={abs(expected-actual):.3f})")
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
    vd = json.loads(open(VARIANCE).read())

    print("PAPER NUMBER VERIFICATION")
    print("=" * 50)

    # Abstract / overall
    print("\n[Abstract & Overall]")
    check("Total cells", 200, len(rows), tol=0)
    check("Valid detections", 173, len(valid), tol=0)
    check("Detection rate %", 86.5, len(valid) / len(rows) * 100, tol=0.1)

    w100 = [r["detection_latency"] for r in valid if r["window_size"] == 100]
    check("w=100 mean latency", 36.3, sum(w100) / len(w100), tol=0.1)

    w200 = [r["detection_latency"] for r in valid if r["window_size"] == 200]
    check("w=200 mean latency", 45.3, sum(w200) / len(w200), tol=0.1)

    # FAR range
    fars = {}
    for clf in ["deberta", "text-moderation", "llama-guard", "shieldgemma"]:
        cells = [r for r in rows if r["classifier"] == clf]
        fars[clf] = sum(1 for r in cells if not r.get("neg_clean", True)) / len(cells)
    check("FAR min (text-moderation)", 0.04, min(fars.values()), tol=0.001)
    check("FAR max (shieldgemma)", 0.12, max(fars.values()), tol=0.001)

    # Variance decomposition
    print("\n[Variance Decomposition]")
    check("η² classifier", 0.196, vd["factor_variances"]["classifier"], tol=0.001)
    check("η² shift_type", 0.217, vd["factor_variances"]["shift_type"], tol=0.001)
    check("η² interaction", 0.265, vd["interaction_variances"]["classifier:shift_type"], tol=0.001)
    check("η² residual", 0.322, vd["residual_variance"], tol=0.001)

    # Key cell latencies
    print("\n[Key Cell Latencies]")
    for clf, shift, expected in [
        ("shieldgemma", "paraphrase", 97.4),
        ("deberta", "compositional-long-context", 26.2),
        ("llama-guard", "adversarial-suffix", 26.7),
        ("llama-guard", "code-switch", 69.0),
    ]:
        lats = [r["detection_latency"] for r in valid
                if r["classifier"] == clf and r["shift_condition"] == shift]
        actual = sum(lats) / len(lats) if lats else 0
        check(f"{clf} × {shift}", expected, actual, tol=0.1)

    # Summary
    print("\n" + "=" * 50)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
    print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
