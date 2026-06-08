"""Verify all paper numbers against raw data.

Covers: Regime A (800-cell factorial), Regime B (temporal jailbreaks),
Regime C (adversarial success), and conformal evaluation.

Usage:
    .venv/bin/python scripts/verify_paper_numbers.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

RESULTS = Path("results/factorial_results.jsonl")
REGIME_B = Path("results/regime_b_results.jsonl")
REGIME_C = Path("results/regime_c_results.jsonl")
CONFORMAL = Path("results/conformal_full.json")
PCA = Path("results/pca_experiment.json")

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
        print(f"  ✗ {name}: {actual:.4f} != {expected} (diff={abs(expected-actual):.4f})")
        FAIL += 1


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    # ========================================================================
    # REGIME A: Main factorial (800 cells)
    # ========================================================================
    rows = load_jsonl(RESULTS)
    for r in rows:
        r["is_valid_detection"] = (
            r.get("detection_latency") is not None
            and r["detection_latency"] >= 0
            and r.get("neg_clean") is True
        )
    valid = [r for r in rows if r["is_valid_detection"]]

    print("PAPER NUMBER VERIFICATION")
    print("=" * 70)

    print("\n[Regime A — Abstract & Overall]")
    check("Total cells", 800, len(rows), tol=0)
    check("Valid detections", 693, len(valid), tol=0)
    check("Detection rate %", 86.6, len(valid) / len(rows) * 100, tol=0.1)

    w100 = [r["detection_latency"] for r in valid if r["window_size"] == 100]
    w200 = [r["detection_latency"] for r in valid if r["window_size"] == 200]
    check("w=100 mean latency", 39.5, sum(w100) / len(w100), tol=0.5)
    check("w=200 mean latency", 45.4, sum(w200) / len(w200), tol=0.5)

    # FAR per classifier
    print("\n[Regime A — False Alarm Rates]")
    far_expected = {"deberta": 0.095, "text-moderation": 0.02, "llama-guard": 0.03, "shieldgemma": 0.085}
    for clf in CLASSIFIERS:
        cells = [r for r in rows if r["classifier"] == clf]
        far = sum(1 for r in cells if not r.get("neg_clean", True)) / len(cells)
        check(f"FAR {clf}", far_expected[clf], far, tol=0.005)

    # Variance decomposition
    print("\n[Regime A — Variance Decomposition]")
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

    # Key cell latencies
    print("\n[Regime A — Key Cell Latencies]")
    key_cells = [
        ("deberta", "paraphrase", 28.4),
        ("deberta", "adversarial-suffix", 36.6),
        ("text-moderation", "adversarial-suffix", 25.3),
        ("llama-guard", "code-switch", 93.4),
        ("llama-guard", "adversarial-suffix", 27.8),
        ("shieldgemma", "paraphrase", 85.0),
        ("shieldgemma", "adversarial-suffix", 26.8),
        ("shieldgemma", "temporal", 27.1),
    ]
    for clf, shift, expected in key_cells:
        lats = [r["detection_latency"] for r in valid
                if r["classifier"] == clf and r["shift_condition"] == shift]
        actual = sum(lats) / len(lats) if lats else 0
        check(f"{clf} × {shift}", expected, actual, tol=0.5)

    # Failure analysis
    print("\n[Regime A — Failure Analysis]")
    invalid = [r for r in rows if not r["is_valid_detection"]]
    check("Invalid cells total", 107, len(invalid), tol=0)

    deb_adv_invalid = sum(1 for r in rows
                          if r["classifier"] == "deberta"
                          and r["shift_condition"] == "adversarial-suffix"
                          and not r["is_valid_detection"])
    check("DeBERTa × adversarial invalid", 13, deb_adv_invalid, tol=0)

    lg_adv_invalid = sum(1 for r in rows
                         if r["classifier"] == "llama-guard"
                         and r["shift_condition"] == "adversarial-suffix"
                         and not r["is_valid_detection"])
    check("Llama Guard × adversarial invalid", 11, lg_adv_invalid, tol=0)

    # Window size paired difference
    print("\n[Regime A — Window Size Paired Difference]")
    w100_map = {}
    w200_map = {}
    for r in valid:
        key = (r["classifier"], r["shift_condition"], r["seed"])
        if r["window_size"] == 100:
            w100_map[key] = r["detection_latency"]
        elif r["window_size"] == 200:
            w200_map[key] = r["detection_latency"]
    paired_keys = sorted(set(w100_map.keys()) & set(w200_map.keys()))
    paired_diff = np.mean([w100_map[k] - w200_map[k] for k in paired_keys])
    check("Window paired diff (100-200)", -7.0, paired_diff, tol=0.5)
    check("N paired", 314, len(paired_keys), tol=0)

    # ========================================================================
    # REGIME B: Temporal jailbreaks
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Regime B — Temporal Jailbreaks]")
    regime_b = load_jsonl(REGIME_B)
    for r in regime_b:
        if "is_valid_detection" not in r:
            r["is_valid_detection"] = (
                r.get("detection_latency") is not None
                and r["detection_latency"] >= 0
                and r.get("neg_clean") is True
            )

    check("Regime B total cells", 20, len(regime_b), tol=0)
    b_valid = sum(1 for r in regime_b if r["is_valid_detection"])
    check("Regime B valid detections", 17, b_valid, tol=0)
    check("Regime B detection rate %", 85.0, b_valid / len(regime_b) * 100, tol=0.1)

    # ========================================================================
    # REGIME C: Adversarial success
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Regime C — Adversarial Success]")
    regime_c = load_jsonl(REGIME_C)
    for r in regime_c:
        if "is_valid_detection" not in r:
            r["is_valid_detection"] = (
                r.get("detection_latency") is not None
                and r["detection_latency"] >= 0
                and r.get("neg_clean") is True
            )

    # Paper says: DeBERTa fails to detect in 38/40 seed-window combinations
    deb_c = [r for r in regime_c if r["classifier"] == "deberta"]
    deb_c_invalid = sum(1 for r in deb_c if not r["is_valid_detection"])
    check("Regime C DeBERTa fails", 38, deb_c_invalid, tol=0)
    check("Regime C DeBERTa total", 40, len(deb_c), tol=0)

    # Paper says: Llama Guard detects in 14/40
    lg_c = [r for r in regime_c if r["classifier"] == "llama-guard"]
    lg_c_valid = sum(1 for r in lg_c if r["is_valid_detection"])
    check("Regime C Llama Guard detects", 14, lg_c_valid, tol=0)
    check("Regime C Llama Guard total", 40, len(lg_c), tol=0)

    # Total Regime C cells
    check("Regime C total cells", 160, len(regime_c), tol=0)

    # ========================================================================
    # CONFORMAL EVALUATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Conformal — Full Evaluation]")
    conformal = json.loads(open(CONFORMAL).read())

    # Paper Table: coverage gaps and recoveries
    conformal_expected = {
        ("deberta", "temporal"): {"gap": 0.085, "recovery": 0.160, "ess": 88},
        ("deberta", "paraphrase"): {"gap": 0.515, "recovery": 0.390, "ess": 46},
        ("deberta", "adversarial-suffix"): {"gap": 0.325, "recovery": 0.020, "ess": 206},
        ("text-moderation", "temporal"): {"gap": 0.090, "recovery": 0.020, "ess": 300},
        ("text-moderation", "paraphrase"): {"gap": 0.515, "recovery": 0.005, "ess": 300},
        ("llama-guard", "temporal"): {"gap": 0.350, "recovery": 0.020, "ess": 300},
        ("llama-guard", "paraphrase"): {"gap": 0.715, "recovery": 0.015, "ess": 300},
        ("shieldgemma", "temporal"): {"gap": 0.225, "recovery": 0.075, "ess": 300},
        ("shieldgemma", "paraphrase"): {"gap": 0.725, "recovery": 0.005, "ess": 300},
        ("shieldgemma", "adversarial-suffix"): {"gap": 0.445, "recovery": 0.060, "ess": 300},
    }

    for entry in conformal:
        clf = entry["classifier"]
        shift = entry["shift_type"]
        key = (clf, shift)
        if key in conformal_expected:
            exp = conformal_expected[key]
            check(f"Conformal {clf}×{shift} gap",
                  exp["gap"], entry["unweighted"]["coverage_gap"], tol=0.005)
            check(f"Conformal {clf}×{shift} recovery",
                  exp["recovery"], entry["weighted"]["coverage_recovery"], tol=0.005)
            check(f"Conformal {clf}×{shift} ESS",
                  exp["ess"], entry["weight_diagnostic"]["effective_sample_size"], tol=3.0)

    # Collapse counts
    n_collapse = sum(1 for e in conformal if e["density_ratio_collapse"])
    check("Conformal collapse count", 9, n_collapse, tol=0)
    n_non_collapse = sum(1 for e in conformal if not e["density_ratio_collapse"])
    check("Conformal non-collapse (DeBERTa)", 3, n_non_collapse, tol=0)

    # ========================================================================
    # PCA EXPERIMENT
    # ========================================================================
    print("\n" + "=" * 70)
    print("[PCA Experiment]")
    if PCA.exists():
        pca_data = json.loads(open(PCA).read())
        # Paper: PCA to 32d recovers 33pp for Llama Guard, 21pp for ShieldGemma on temporal
        for entry in pca_data:
            if entry.get("classifier") == "llama-guard" and entry.get("pca_dim") == 32:
                recovery = entry.get("weighted_coverage", 0) - entry.get("unweighted_coverage", 0)
                # Paper says coverage 0.555 → 0.885 = +33pp
                check("PCA-32 Llama Guard temporal recovery",
                      0.330, recovery, tol=0.02)
            elif entry.get("classifier") == "shieldgemma" and entry.get("pca_dim") == 32:
                recovery = entry.get("weighted_coverage", 0) - entry.get("unweighted_coverage", 0)
                check("PCA-32 ShieldGemma temporal recovery",
                      0.205, recovery, tol=0.02)
    else:
        print("  ⚠ PCA experiment results not found, skipping")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
    print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
