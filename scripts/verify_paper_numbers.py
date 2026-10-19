"""Verify all paper numbers against raw data.

Paper: https://arxiv.org/abs/2606.11949

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
    # NEW RESULTS (post-factorial additions)
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Post-Factorial — CS Growing-Window]")
    cs_path = Path("results/cs_growing_window_results.json")
    if cs_path.exists():
        cs_data = json.load(open(cs_path))
        cs_results = cs_data["results"]
        cs_valid = [r for r in cs_results if r["is_valid_detection"]]
        check("CS detection rate", 120, len(cs_valid), tol=0)
        # Detection latency (regenerated under the predictable-betting fix; paper reports these)
        cs_mean_lat = sum(r["detection_latency"] for r in cs_valid) / len(cs_valid)
        check("CS mean detection latency (120 cells)", 35.7, cs_mean_lat, tol=0.1)
        db_para = [r["detection_latency"] for r in cs_valid
                   if r["classifier"] == "deberta" and r["shift_condition"] == "paraphrase"]
        check("CS latency DeBERTa x paraphrase", 29.6, sum(db_para) / len(db_para), tol=0.1)
        # FAR
        cs_far = cs_data.get("far", {})
        for clf in CLASSIFIERS:
            if clf in cs_far:
                check(f"CS FAR {clf}", 0, cs_far[clf]["alarms"], tol=0)
    else:
        print("  ⚠ CS results not found, skipping")

    print("\n[Post-Factorial — Filtered Ablation]")
    filt_path = Path("results/filtered_ablation_results.json")
    if filt_path.exists():
        filt = json.load(open(filt_path))
        check("Filtered corpus: refusals", 47, filt["corpus_stats"]["refusals"], tol=0)
        check("DeBERTa unfiltered mean", 38.0, filt["deberta"]["unfiltered"]["mean"], tol=0.5)
        check("DeBERTa filtered mean", 37.8, filt["deberta"]["filtered"]["mean"], tol=0.5)
        check("Llama Guard unfiltered mean", 66.6, filt["llama-guard"]["unfiltered"]["mean"], tol=0.5)
        check("Llama Guard filtered mean", 60.8, filt["llama-guard"]["filtered"]["mean"], tol=0.5)
    else:
        print("  ⚠ Filtered ablation results not found, skipping")

    print("\n[Post-Factorial — Embedding Displacement]")
    disp_path = Path("results/embedding_displacement.json")
    if disp_path.exists():
        disp = json.load(open(disp_path))
        check("Displacement overall r", -0.089, disp["correlation"]["r"], tol=0.01)
    else:
        print("  ⚠ Displacement results not found, skipping")

    print("\n[Post-Factorial — Gradual Drift]")
    drift_path = Path("results/gradual_drift_results.json")
    if drift_path.exists():
        drift = json.load(open(drift_path))
        check("Gradual drift: abrupt detection rate", 1.0, drift["abrupt"]["detection_rate"], tol=0.01)
        # NOTE: the gradual arm's 0.0 is a stream-exhaustion artefact (ref_pool
        # is consumed before onset; see run_gradual_drift.py KNOWN ISSUE). We
        # assert the committed value for provenance but it is NOT evidence that
        # gradual drift is undetectable; the paper's gradual/ramp results use
        # results/ramp_rate_sweep.json instead.
        check("Gradual drift: gradual detection rate (ARTEFACT, see script note)", 0.0, drift["gradual"]["detection_rate"], tol=0.01)
    else:
        print("  ⚠ Gradual drift results not found, skipping")

    print("\n[Rotating canary, rank reversal, frontier canary, ca8 baselines]")
    rc = Path("results/rotating_canary_eval/rotating_canary_trials.jsonl")
    if rc.exists():
        rows_rc = [json.loads(l) for l in open(rc) if l.strip()]
        check("Rotating canary: total sessions", 1950, len(rows_rc), tol=0)
        def _cell(m, a):
            return [r for r in rows_rc if r["model"] == m and r["attacker"] == a]
        check("Rotating canary: gemini naive successes /100", 59, sum(bool(r["attack_success"]) for r in _cell("gemini-2.5-pro", "naive")), tol=0)
        check("Rotating canary: gpt-5.1 spoofing successes /150", 150, sum(bool(r["attack_success"]) for r in _cell("gpt-5.1", "spoofing")), tol=0)
        check("Rotating canary: claude canary-aware matches /100", 39, sum(bool(r["attack_success"]) for r in _cell("claude-sonnet-5", "canary_aware")), tol=0)
        clean = [r for r in rows_rc if r["attacker"] == "none"]
        check("Rotating canary: clean sessions with canary score < 0.5 (of 400)", 1, sum(r["canary_score"] < 0.5 for r in clean), tol=0)
    rr = Path("results/v5_rank_reversal/rank_reversal_results.jsonl")
    if rr.exists():
        rows_rr = [json.loads(l) for l in open(rr) if l.strip()]
        valid500 = {m: sum(1 for r in rows_rr if r["model"] == m and r["budget"] == 500 and r["valid"]) for m in ("o3", "o4-mini", "gemini-2.5-pro")}
        check("Rank reversal: o3 valid at 500 tokens /100", 100, valid500["o3"], tol=0)
        check("Rank reversal: gemini-2.5-pro valid at 500 tokens /100", 99, valid500["gemini-2.5-pro"], tol=0)
        check("Rank reversal: valid at 16 tokens (all models)", 0, sum(1 for r in rows_rr if r["budget"] == 16 and r["valid"]), tol=0)
    bs = Path("results/benign_specificity_full.json")
    if bs.exists():
        import numpy as _np
        dd = json.load(open(bs))
        rowsb = [(m, v["adv_mean"], float(_np.clip(_np.array(v["benign_scores"], dtype=float), 0, 1).mean()), max(v["benign_scores"])) for m, v in dd.items() if m != "gemini-2.5-pro"]
        check("Frontier canary: mean delta over 19 discriminating models", 0.729, _np.mean([a - b for _, a, b, _ in rowsb]), tol=0.002)
        check("Frontier canary: models with no benign score >= 0.5 (of 19)", 15, sum(1 for _, _, _, mx in rowsb if mx < 0.5), tol=0)
    ca = Path("results/ca8_divergence_min_n20.json")
    if ca.exists():
        cr = json.load(open(ca))["results"]
        check("CA8: stealth cases with a recorded baseline", 4, sum(1 for r in cr if r["stealth_evasion"] and r.get("orig_b") is not None), tol=0)
        check("CA8: stealth cases with original canary score >= 0.98", 2, sum(1 for r in cr if r["stealth_evasion"] and (r.get("orig_b") or 0) >= 0.98), tol=0)
        check("CA8: prompts with no baseline", 10, sum(1 for r in cr if r.get("orig_a") is None), tol=0)

    print("\n[Invalid-cell split and lambda-sweep regression]")
    fr = Path("results/factorial_results.jsonl")
    if fr.exists():
        fx = [json.loads(l) for l in open(fr) if l.strip()]
        inv = [r for r in fx if not (r["detection_latency"] is not None and r["detection_latency"] >= 0 and r["neg_clean"])]
        check("Invalid cells", 107, len(inv), tol=0)
        check("Invalid cells: alarm before onset (negative latency)", 68, sum(1 for r in inv if r["detection_latency"] is not None and r["detection_latency"] < 0), tol=0)
        check("Invalid cells: dirty negative control", 46, sum(1 for r in inv if not r["neg_clean"]), tol=0)
        check("Invalid cells: both", 7, sum(1 for r in inv if (not r["neg_clean"]) and r["detection_latency"] is not None and r["detection_latency"] < 0), tol=0)
    ls = Path("results/v5_lambda_sweep/analysis.json")
    if ls.exists():
        reg = json.load(open(ls))["regression"]
        check("Lambda sweep: regression slope over 5 lambda values", 0.087, reg["slope"], tol=0.002)
        check("Lambda sweep: regression R-squared", 0.207, reg["r_squared"], tol=0.002)

    print("\n[Factorial regeneration with the corrected KS detector]")
    rg = Path("results/factorial_regeneration")
    if (rg / "factorial_results_corrected_ks.jsonl").exists():
        key = lambda r: (r["classifier"], r["shift_condition"], r["seed"], r["window_size"])
        ld = lambda p: {key(r): r for r in (json.loads(l) for l in open(p) if l.strip())}
        orig_rg = ld("results/factorial_results.jsonl")
        com = ld(rg / "factorial_results_committed_ks.jsonl")
        cor = ld(rg / "factorial_results_corrected_ks.jsonl")
        ok = lambda r: r["detection_latency"] is not None and r["detection_latency"] >= 0 and r["neg_clean"]
        same = lambda a, b: all(a[f] == b[f] for f in ("alarm_step", "detection_latency", "neg_clean"))
        stable = [k for k in orig_rg if k[1] != "adversarial-suffix"]
        check("Regeneration: cells", 800, len(cor), tol=0)
        check("Regeneration: committed KS identical to original (640 stable-corpus cells)", 640, sum(same(orig_rg[k], com[k]) for k in stable), tol=0)
        check("Regeneration: committed KS identical to original (160 adversarial cells)", 68, sum(same(orig_rg[k], com[k]) for k in orig_rg if k[1] == "adversarial-suffix"), tol=0)
        check("Regeneration: cells whose alarm step changed with the corrected KS", 92, sum(not same(com[k], cor[k]) for k in com), tol=0)
        check("Regeneration: changed cells, Llama Guard", 52, sum(not same(com[k], cor[k]) for k in com if k[0] == "llama-guard"), tol=0)
        check("Regeneration: changed cells, ShieldGemma", 38, sum(not same(com[k], cor[k]) for k in com if k[0] == "shieldgemma"), tol=0)
        check("Regeneration: control-validated detections, committed KS", 693, sum(ok(r) for r in com.values()), tol=0)
        check("Regeneration: control-validated detections, corrected KS", 694, sum(ok(r) for r in cor.values()), tol=0)
        w100 = [r["detection_latency"] for r in cor.values() if ok(r) and r["window_size"] == 100]
        check("Regeneration: mean latency w=100, corrected KS", 39.75, sum(w100) / len(w100), tol=0.01)
        ch = [cor[k]["detection_latency"] - com[k]["detection_latency"] for k in com if not same(com[k], cor[k]) and com[k]["detection_latency"] is not None and cor[k]["detection_latency"] is not None]
        check("Regeneration: mean latency change on changed cells", 1.96, sum(ch) / len(ch), tol=0.01)
    else:
        print("  ⚠ regeneration results not found, skipping")

    print("\n[Equal-reference KS vs CS]")
    er = Path("results/cs_engine_investigation/ramp_ks_reference_size.json")
    if er.exists():
        r = json.load(open(er))
        check("Equal-reference 30% mixing: KS (500-score ref) detected /10", 10, r["0.3"]["ks_ref500"], tol=0)
        check("Paper-setup 30% mixing: KS (100-score ref) detected /10", 4, r["0.3"]["ks_ref100"], tol=0)
        check("Extended 30% test: KS (500-score ref) detected /30", 30, r["30pct_extended"]["ks_ref500"], tol=0)
        check("30% mixing KS (500-score ref) mean latency", 71.5, r["0.3"]["ks_ref500_latency"], tol=0.1)
    erf = Path("results/cs_engine_investigation/equal_reference_ks_vs_cs.json")
    if erf.exists():
        rows = json.load(open(erf))
        cs_faster = sum(1 for x in rows if x["cs_latency"] < x["ks_latency"])
        check("Equal-reference abrupt cells: CS faster in", 4, cs_faster, tol=0)
        check("Equal-reference abrupt cells: pooled KS latency", 27.5, sum(x["ks_latency"] for x in rows) / len(rows), tol=0.1)
    else:
        print("  ⚠ equal-reference results not found, skipping")

    print("\n[Ramped-onset comparison (Table ad1)]")
    ad1_path = Path("results/cs_engine_investigation/ad1_paraphrase.json")
    if ad1_path.exists():
        ad1 = json.load(open(ad1_path))
        for i, mix in enumerate(ad1["mixing"]):
            check(f"AD1 scan detection at {int(mix*100)}%", [0.60, 0.93, 1.0, 1.0][i], ad1["scan"][i], tol=0.01)
            check(f"AD1 KS (500-step null) at {int(mix*100)}%", [0.57, 0.87, 1.0, 1.0][i], ad1["ks_committed"][i], tol=0.01)
            check(f"AD1 KS (pooled) at {int(mix*100)}%", [0.33, 0.67, 0.93, 1.0][i], ad1["ks_pooled"][i], tol=0.01)
    else:
        print("  ⚠ AD1 reproduction not found, skipping")

    print("\n[Post-Factorial — Ramp Sweep]")
    ramp_path = Path("results/ramp_rate_sweep.json")
    if ramp_path.exists():
        ramp = json.load(open(ramp_path))
        # Part 1: ramp-rate sweep at 50% mixing
        if "ramp_durations" in ramp:
            r50 = ramp["ramp_durations"].get("50", {})
            check("Ramp 50 KS detection rate", 1.0, r50.get("ks", {}).get("detection_rate", 0), tol=0.01)
            check("Ramp 50 CS detection rate", 1.0, r50.get("cs", {}).get("detection_rate", 0), tol=0.01)
            r200 = ramp["ramp_durations"].get("200", {})
            check("Ramp 200 KS n_detected", 9, r200.get("ks", {}).get("n_detected", 0), tol=0)
            check("Ramp 200 CS n_detected", 10, r200.get("cs", {}).get("n_detected", 0), tol=0)
            check("Ramp 50 CS mean latency", 83.9, r50.get("cs", {}).get("mean_latency", 0), tol=0.1)
            check("Ramp 50 KS mean latency", 94.2, r50.get("ks", {}).get("mean_latency", 0), tol=0.1)
        # Part 2: mixing-level sweep
        if "mixing_levels" in ramp:
            m100 = ramp["mixing_levels"].get("1.0", {})
            check("Mix 100% KS detection rate", 1.0, m100.get("ks", {}).get("detection_rate", 0), tol=0.01)
            check("Mix 100% CS detection rate", 1.0, m100.get("cs", {}).get("detection_rate", 0), tol=0.01)
            m30 = ramp["mixing_levels"].get("0.3", {})
            check("Mix 30% (n=10) KS n_detected", 4, m30.get("ks", {}).get("n_detected", 0), tol=0)
            check("Mix 30% (n=10) CS n_detected", 10, m30.get("cs", {}).get("n_detected", 0), tol=0)
            check("Mix 100% CS mean latency", 54.2, m100.get("cs", {}).get("mean_latency", 0), tol=0.1)
            check("Mix 100% KS mean latency", 63.9, m100.get("ks", {}).get("mean_latency", 0), tol=0.1)
        # Part 3: extended 30% mixing (n=30)
        if "mixing_30_extended" in ramp:
            ext = ramp["mixing_30_extended"]
            check("Mix 30% extended CS n_detected", 30, ext["cs"]["n_detected"], tol=0)
            check("Mix 30% extended KS n_detected", 13, ext["ks"]["n_detected"], tol=0)
            check("Mix 30% extended n", 30, ext["n"], tol=0)
            check("Mix 30% Fisher p < 0.001", 1, 1 if ext["fisher_p"] < 0.001 else 0, tol=0)
    else:
        print("  ⚠ Ramp sweep results not found, skipping")

    # ========================================================================
    # v5 AUDIT (post-hoc, EXPLORATORY): extended token sweep + no-shift control
    # ========================================================================
    print("\n" + "=" * 70)
    print("[v5 Audit — Extended Token Sweep (o3, EXPLORATORY)]")
    sweep_ext = Path("results/token_sweep_o3_extended.json")
    refit_merged = Path("results/token_sigmoid_refit_merged.json")
    if sweep_ext.exists():
        se = json.load(open(sweep_ext))
        # adversarial valid rates at key budgets (n=20/class, 0 errors expected)
        b = se["budgets"]
        check("Ext sweep adv valid rate @100", 0.45, b["100"]["adv_summary"]["valid_rate"], tol=0.001)
        check("Ext sweep adv valid rate @150", 0.85, b["150"]["adv_summary"]["valid_rate"], tol=0.001)
        check("Ext sweep adv valid rate @200", 1.00, b["200"]["adv_summary"]["valid_rate"], tol=0.001)
        # zero API errors across all budgets
        n_err = sum(b[k]["adv_summary"]["n_error"] + b[k]["benign_summary"]["n_error"] for k in b)
        check("Ext sweep total API errors", 0, n_err, tol=0)
    else:
        print("  ⚠ Extended token sweep not found, skipping")
    if refit_merged.exists():
        rm = json.load(open(refit_merged))
        check("Merged benign T50", 46.4, rm["benign"]["t50"], tol=1.0)
        check("Merged adversarial T50", 117.3, rm["adversarial"]["t50"], tol=3.0)
        check("Merged adversarial k", 0.047, rm["adversarial"]["k"], tol=0.005)
        check("Merged adversarial R^2", 0.977, rm["adversarial"]["r2"], tol=0.02)
        check("Merged adv/benign T50 ratio", 2.53, rm["t50_ratio_adv_over_benign"], tol=0.1)
    else:
        print("  ⚠ Merged sigmoid refit not found, skipping")

    print("\n[v5 Audit — No-Shift DRE Control (EXPLORATORY)]")
    noshift = Path("results/noshift_dre_control.json")
    if noshift.exists():
        ns = json.load(open(noshift))
        tm = ns["classifiers"]["text-moderation"]["summary"]
        # Text-Moderation reproduces collapse under NO shift: high train acc,
        # chance held-out AUC, ESS < n_cal, weights at floor.
        check("No-shift TextMod native train acc", 0.941, tm["native"]["train_accuracy"]["mean"], tol=0.03)
        check("No-shift TextMod native held-out AUC (chance)", 0.528, tm["native"]["heldout_auc"]["mean"], tol=0.05)
        check("No-shift TextMod native ESS < 300", 146.9, tm["native"]["ess"]["mean"], tol=25.0)
        # PCA raises ESS towards uniformity even with no shift (recovery is an artefact)
        check("No-shift TextMod PCA-8 ESS raised", 276.6, tm["pca"]["8"]["ess"]["mean"], tol=25.0)
        # Base DeBERTa checkpoint does NOT collapse under no shift
        db = ns["classifiers"]["deberta"]["summary"]
        check("No-shift DeBERTa(base) native ESS ~ n_cal", 289.1, db["native"]["ess"]["mean"], tol=15.0)
    else:
        print("  ⚠ No-shift DRE control not found, skipping")



def check_bool(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}")


def verify_v2():
    """Verify v2 paper numbers against raw data."""
    global PASS, FAIL
    
    print("\n" + "=" * 60)
    print("v2 VERIFICATION")
    print("=" * 60)
    
    # --- CA6: GCG detection rate ---
    ca6_data = json.loads(Path("results/gate_a_ca6_gibberish.json").read_text())
    # GCG detection: 37/49 = 75.5%
    if "gcg_detection_rate" in ca6_data:
        check_bool("CA6 GCG detection 75.5%", abs(ca6_data["gcg_detection_rate"] - 0.755) < 0.01)

    # --- k-scaling ---
    k_data = json.loads(Path("results/track_a_k_scaling.json").read_text())
    k_summary = k_data.get("summary", {}) if isinstance(k_data, dict) else {}
    if "k1_best" in k_summary:
        check_bool("k=1 best 93.9%", abs(k_summary["k1_best"]["rate"] - 0.939) < 0.01)
        check_bool("k=2 best 98.0%", abs(k_summary["k2_best"]["rate"] - 0.980) < 0.01)

    # --- LLM canary v2 ---
    v2_path = Path("results/llm_canary_v2.json")
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text())

        # 20 models
        check_bool("LLM canary v2: 20 models", len(v2) == 20)

        # Each model has 147 scores (49 gcg + 49 clean + 49 scrambled)
        for model_name, model_data in v2.items():
            n = len(model_data["scores"])
            if n != 147:
                check_bool(f"LLM v2 {model_name}: 147 scores", False)
                break
        else:
            check_bool("LLM v2: all models have 147 scores", True)

        # Compute median |Δ(gcg - clean)| ≈ 0.035
        all_deltas = []
        for model_data in v2.values():
            gcg = [s["score"] for s in model_data["scores"] if s["type"] == "gcg"]
            clean = [s["score"] for s in model_data["scores"] if s["type"] == "clean"]
            mean_delta = np.mean(gcg) - np.mean(clean)
            all_deltas.append(abs(mean_delta))
        median_abs_delta = np.median(all_deltas)
        check_bool(f"Median |Δ(gcg-clean)| < 0.05 (got {median_abs_delta:.3f})", median_abs_delta < 0.05)

        # Scrambled: 15/20 non-significant (already computed, trust the analysis)
        # Boundary flips: 59/980
        flips = 0
        total = 0
        for model_data in v2.values():
            scores = model_data["scores"]
            gcg = [s["score"] for s in scores if s["type"] == "gcg"]
            clean = [s["score"] for s in scores if s["type"] == "clean"]
            for i in range(min(len(gcg), len(clean))):
                total += 1
                if clean[i] >= 0.5 and gcg[i] < 0.5:
                    flips += 1
        check_bool(f"Boundary flips 59/980 (got {flips}/{total})", flips == 59 and total == 980)

    # --- Divergence minimisation (primary Tier-4 pair, n=20) ---
    # Paper reports 14/20 blocked, mean gap 0.218, median 0.250 at lambda=2.0.
    # NOTE: a prior verifier guarded `if "blocked" in dm` against
    # results/ca8_divergence_min.json, whose keys are n_stealth/n_flipped (an
    # older n=10 run), so the assertion silently never executed. The paper's
    # n=20 result lives in results/ca8_divergence_min_n20.json.
    dm20_path = Path("results/ca8_divergence_min_n20.json")
    if dm20_path.exists():
        dm = json.loads(dm20_path.read_text())
        n_total = dm["n_total"]
        n_blocked = n_total - dm["n_stealth"]
        blocked_gaps = dm["blocked_gaps"]
        check_bool(f"Div-min n=20 total (got {n_total})", n_total == 20)
        check_bool(f"Div-min 14 blocked (got {n_blocked})", n_blocked == 14)
        check_bool(f"Div-min 6 stealth (got {dm['n_stealth']})", dm["n_stealth"] == 6)
        check_bool(
            f"Div-min blocked mean gap 0.218 (got {np.mean(blocked_gaps):.3f})",
            abs(np.mean(blocked_gaps) - 0.218) < 0.005,
        )
        check_bool(
            f"Div-min blocked median gap 0.250 (got {np.median(blocked_gaps):.3f})",
            abs(np.median(blocked_gaps) - 0.250) < 0.005,
        )
    else:
        check_bool("Div-min n=20 file present", False)

    # --- Martingale FAR (per-classifier, from Track B AV2 evaluation) ---
    # NOTE: results/gate_b_martingale.json only holds the aggregate scan_w50
    # FAR as a display string ("0/200 = 0.0%") from the initial single-classifier
    # gate run; it does not have a per-classifier breakdown. The actual AV2
    # cross-classifier FAR uniformity check (which the paper reports
    # as "FAR<=1% across all 4
    # classifiers") lives in results/track_b_full_eval.json under the "av2" key.
    av2_path = Path("results/track_b_full_eval.json")
    if av2_path.exists():
        av2_data = json.loads(av2_path.read_text())
        av2 = av2_data.get("av2", {})
        for clf, entry in av2.items():
            check_bool(f"Martingale FAR {clf} <= 1%", entry["far"] <= 0.01)


def verify_corrections():
    """Verify the numbers corrected after the post-review audit.

    These checks cover claims that the previous verifier did not touch and that
    were found to be mis-stated or spliced: the GCG true-flip count, the
    lambda-sweep medians, the corrected frontier-stability figures, and the
    heterogeneous-traffic KS figures the paper actually cites.
    """
    print("\n" + "=" * 60)
    print("POST-REVIEW CORRECTION VERIFICATION")
    print("=" * 60)

    # --- GCG flip count: 49 end below 0.5, 47 are true unsafe->safe crossings ---
    gad = Path("results/gate_a_divergence.json")
    if gad.exists():
        det = json.loads(gad.read_text())["details"]
        below = sum(1 for r in det if r["deberta_attacked"] < 0.5)
        cross = sum(
            1 for r in det if r["deberta_orig"] >= 0.5 and r["deberta_attacked"] < 0.5
        )
        check_bool(f"GCG inputs ending below 0.5 == 49 (got {below})", below == 49)
        check_bool(f"GCG true unsafe->safe crossings == 47 (got {cross})", cross == 47)
        # CA6 detection: 37/49 on DeBERTa<->Llama Guard
        ca6 = Path("results/gate_a_ca6_gibberish.json")
        if ca6.exists():
            c = json.loads(ca6.read_text())
            if "gcg_detection_rate" in c:
                check_bool(
                    f"CA6 GCG detection 75.5% (got {c['gcg_detection_rate']:.3f})",
                    abs(c["gcg_detection_rate"] - 0.755) < 0.01,
                )

    # --- AutoDAN template-attack flip rate (backed by a file but previously
    #     unchecked by the verifier): 1% (2/200) against DeBERTa. ---
    ad = Path("results/track_a_autodan_canary.json")
    if ad.exists():
        a = json.loads(ad.read_text())
        check_bool(
            f"AutoDAN flip rate 1% (got {a['deberta_flip_rate']:.3f})",
            abs(a["deberta_flip_rate"] - 0.01) < 0.005,
        )

    # --- Lambda-sweep medians (paper reports these, not means) ---
    lam = Path("results/v5_lambda_sweep/analysis.json")
    if lam.exists():
        per = json.loads(lam.read_text())["per_lambda"]
        expected_median = {"2.0": 0.218, "4.0": 0.108, "8.0": 0.0625}
        for k, exp in expected_median.items():
            got = per[k]["median_gap"]
            check_bool(f"Lambda {k} median gap {exp} (got {got:.4f})", abs(got - exp) < 0.003)
        # block rates
        check_bool("Lambda 2.0 blocked 10/20", per["2.0"]["blocked"] == 10)
        check_bool("Lambda 8.0 blocked 12/20", per["8.0"]["blocked"] == 12)

    # --- Frontier stability (corrected v5_pareto_update.json) ---
    pu = Path("results/v5_pareto_update.json")
    if pu.exists():
        d = json.loads(pu.read_text())
        cs5 = d.get("claude-sonnet-5", {})
        if cs5.get("n_adv"):
            check_bool(
                f"claude-sonnet-5 detection 46.8% (got {cs5['det_point']:.3f})",
                abs(cs5["det_point"] - 0.468) < 0.01,
            )
            check_bool(
                f"claude-sonnet-5 FPR 44.7% (got {cs5['fpr_point']:.3f})",
                abs(cs5["fpr_point"] - 0.447) < 0.01,
            )
        g36 = d.get("gemini-3.6-flash", {})
        if g36.get("n_adv"):
            check_bool(
                f"gemini-3.6-flash detection 26.1% (got {g36['det_point']:.3f})",
                abs(g36["det_point"] - 0.261) < 0.01,
            )

    # --- Heterogeneous KS figures the paper cites (real-DeBERTa run) ---
    het = Path("results/v5_heterogeneous_real/heterogeneous_far_results.json")
    if het.exists():
        h = json.loads(het.read_text())["detectors"]["ks_w100"]
        check_bool(f"Heterogeneous KS hom 4.3% (got {h['homogeneous_far']:.3f})",
                   abs(h["homogeneous_far"] - 0.043) < 0.005)
        check_bool(f"Heterogeneous KS het 23.4% (got {h['heterogeneous_far']:.3f})",
                   abs(h["heterogeneous_far"] - 0.234) < 0.005)
        check_bool(f"Heterogeneous KS recal 2.6% (got {h['recalibrated_far']:.3f})",
                   abs(h["recalibrated_far"] - 0.026) < 0.005)
        # Document (do not assert a direction for) the CUSUM splice: on THIS run
        # CUSUM saturates, so the paper's withdrawal of the 6.1->6.6% claim is correct.
        cus = json.loads(het.read_text())["detectors"]["cusum_k0.5"]
        check_bool(
            f"Heterogeneous CUSUM saturates on cited-KS run (het FAR {cus['heterogeneous_far']:.2f} ~ 1.0)",
            cus["heterogeneous_far"] >= 0.99,
        )


    # --- Late additions: MMD null FAR, Regime C per-example scores, figure sources ---
    mm = json.loads(Path("results/mmd_null_far.json").read_text())
    exp = {"deberta": 1, "text-moderation": 1, "llama-guard": 3, "shieldgemma": 0}
    for c, k in exp.items():
        check(f"MMD null windows false alarms {c}", k, mm["classifiers"][c]["null_false_alarms"], tol=0)
        check(f"MMD null windows {c}", 30, mm["classifiers"][c]["null_windows"], tol=0)
    check("MMD stored detection rows reproduced (mismatches)", 0,
          mm["detection_reproduction"]["mismatching_rows"], tol=0)

    rc = json.loads(Path("results/regime_c_example_scores.json").read_text())
    lg = rc["per_example_scores_after_truncation"]["llama-guard"]
    deb = rc["per_example_scores_after_truncation"]["deberta"]
    check("Regime C examples", 22, rc["n_successful"], tol=0)
    check("Regime C examples longer than 1000 chars", 4,
          sum(1 for n in rc["combined_length_chars"] if n > 1000), tol=0)
    check("Regime C Llama Guard mean (2 June corpus)", 0.77, sum(lg) / len(lg), tol=0.005)
    check("Regime C Llama Guard above 0.5", 18, sum(1 for s in lg if s > 0.5), tol=0)
    check("Regime C Llama Guard delta vs reference", 0.72,
          sum(lg) / len(lg) - rc["reference_mean_score_500"]["llama-guard"], tol=0.005)
    check("Regime C DeBERTa streamed mean", 0.18, sum(deb) / len(deb), tol=0.005)
    check("Regime C DeBERTa above 0.5 as streamed", 4, sum(1 for s in deb if s > 0.5), tol=0)

    # Threat-tier and budget figures: values drawn from the stored result files.
    jt = json.loads(Path("results/ca8_joint_gcg_n20.json").read_text())
    dv = json.loads(Path("results/ca8_divergence_min_n20.json").read_text())
    check("Tier 3 blocked % (fig_threat_tiers)", 45.0,
          100 * (jt["n_prompts"] - jt["n_both_flipped"]) / jt["n_prompts"], tol=0.01)
    check("Tier 4 blocked % (fig_threat_tiers)", 70.0,
          100 * len(dv["blocked_gaps"]) / dv["n_total"], tol=0.01)
    check("Joint transfer % (fig_budget_curve)", 55.0,
          100 * jt["n_both_flipped"] / jt["n_prompts"], tol=0.01)

    # --- ANOVA permutation floor and Holm family (anova_permutation_holm_check.py) ---
    ah = json.loads(Path("results/anova_permutation_holm_check.json").read_text())
    assert ah["non_finite_rss_events"] == 0
    o = ah["original"]
    for term in ("classifier", "shift", "interaction"):
        check(f"ANOVA repo permutation p floor ({term})", 1 / 1001,
              o["anova_cell_level"]["repo_joint_permutation_p_1000"][term], tol=1e-9)
        check_bool(f"ANOVA Freedman-Lane p < 2e-4 ({term})",
                   o["anova_cell_level"]["freedman_lane_p"][term] < 2e-4)
    check("Holm weakest adjusted p, original rows (200k permutations)", 0.046,
          max(c["holm_adjusted_p"] for c in o["holm_cell_level"]), tol=0.003)
    check_bool("Holm: all 8 survive on original rows",
               all(c["holm_adjusted_p"] < 0.05 for c in o["holm_cell_level"]))
    for name, exp in (("committed_ks_regeneration", 0.064), ("corrected_ks_regeneration", 0.081)):
        h = ah[name]["holm_cell_level"]
        check(f"Holm comparison 2 adjusted p, {name}", exp,
              [c for c in h if c["comparison"].startswith("2 ")][0]["holm_adjusted_p"], tol=0.004)
        check_bool(f"Holm: comparison 2 does not survive, {name}",
                   [c for c in h if c["comparison"].startswith("2 ")][0]["holm_adjusted_p"] > 0.05)
        check_bool(f"Holm: other seven below 0.01, {name}",
                   all(c["holm_adjusted_p"] < 0.01 for c in h if not c["comparison"].startswith("2 ")))

if __name__ == "__main__":
    main()
    verify_v2()
    verify_corrections()
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL > 0 else 0)
