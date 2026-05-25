"""Full analysis of 800-cell extended factorial results.

Computes: detection rates, latency tables, FAR, window effects,
variance decomposition (ANOVA), failure analysis, and N=5 vs N=20 comparison.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

RESULTS = Path("results/factorial_results.jsonl")
CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "code-switch", "compositional-long-context", "temporal", "adversarial-suffix"]
SEEDS = list(range(20))
WINDOW_SIZES = [100, 200]
N_BOOT = 1000
RNG = np.random.default_rng(42)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(values, n_boot=N_BOOT):
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return (mean, mean, mean)
    boots = np.array([float(np.mean(RNG.choice(arr, size=len(arr), replace=True)))
                      for _ in range(n_boot)])
    return (mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def load():
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


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def main():
    rows = load()

    # =========================================================================
    # 1. VALIDATION
    # =========================================================================
    section("1. DATA VALIDATION")
    print(f"Total rows: {len(rows)}")
    expected = len(CLASSIFIERS) * len(SHIFTS) * len(SEEDS) * len(WINDOW_SIZES)
    print(f"Expected:   {expected}")

    # Check for duplicates
    keys = [(r["classifier"], r["shift_condition"], r["seed"], r["window_size"]) for r in rows]
    dupes = len(keys) - len(set(keys))
    print(f"Duplicates: {dupes}")

    # Check completeness
    expected_keys = set()
    for c in CLASSIFIERS:
        for s in SHIFTS:
            for seed in SEEDS:
                for ws in WINDOW_SIZES:
                    expected_keys.add((c, s, seed, ws))
    actual_keys = set(keys)
    missing = expected_keys - actual_keys
    print(f"Missing:    {len(missing)}")
    if missing:
        for m in sorted(missing)[:10]:
            print(f"  {m}")

    # Data quality
    null_latency = sum(1 for r in rows if r.get("detection_latency") is None)
    neg_latency = sum(1 for r in rows if r.get("detection_latency") is not None and r["detection_latency"] < 0)
    valid = sum(1 for r in rows if r["is_valid_detection"])
    neg_dirty = sum(1 for r in rows if not r.get("neg_clean", True))
    print(f"\nNull latency (no alarm):  {null_latency}")
    print(f"Negative latency (early): {neg_latency}")
    print(f"Neg control dirty:        {neg_dirty}")
    print(f"Valid detections:          {valid}/{len(rows)}")

    # =========================================================================
    # 2. OVERALL DETECTION RATE
    # =========================================================================
    section("2. OVERALL DETECTION RATE")
    n_valid = sum(1 for r in rows if r["is_valid_detection"])
    ci = wilson_ci(n_valid, len(rows))
    print(f"Detection rate: {n_valid}/{len(rows)} = {n_valid/len(rows):.1%}")
    print(f"95% Wilson CI:  [{ci[0]:.3f}, {ci[1]:.3f}]")

    # =========================================================================
    # 3. DETECTION RATE PER CLASSIFIER AND PER SHIFT
    # =========================================================================
    section("3. DETECTION RATE BY FACTOR")

    print("\nBy classifier:")
    for clf in CLASSIFIERS:
        cells = [r for r in rows if r["classifier"] == clf]
        k = sum(1 for r in cells if r["is_valid_detection"])
        n = len(cells)
        ci = wilson_ci(k, n)
        print(f"  {clf:<18} {k:>3}/{n} = {k/n:.1%}  [{ci[0]:.3f}, {ci[1]:.3f}]")

    print("\nBy shift condition:")
    for shift in SHIFTS:
        cells = [r for r in rows if r["shift_condition"] == shift]
        k = sum(1 for r in cells if r["is_valid_detection"])
        n = len(cells)
        ci = wilson_ci(k, n)
        print(f"  {shift:<30} {k:>3}/{n} = {k/n:.1%}  [{ci[0]:.3f}, {ci[1]:.3f}]")

    print("\nDetection rate matrix (classifier × shift):")
    header = f"{'Classifier':<18}" + "".join(f"{s[:12]:<16}" for s in SHIFTS)
    print(header)
    for clf in CLASSIFIERS:
        row = f"{clf:<18}"
        for shift in SHIFTS:
            cells = [r for r in rows if r["classifier"] == clf and r["shift_condition"] == shift]
            k = sum(1 for r in cells if r["is_valid_detection"])
            n = len(cells)
            row += f"{k}/{n} ({k/n:.0%})      "
        print(row)

    # =========================================================================
    # 4. MEAN LATENCY TABLE
    # =========================================================================
    section("4. MEAN DETECTION LATENCY (valid detections only)")

    header = f"{'Classifier':<18}" + "".join(f"{s[:12]:<22}" for s in SHIFTS)
    print(header)
    print("-" * 128)
    for clf in CLASSIFIERS:
        row = f"{clf:<18}"
        for shift in SHIFTS:
            lats = [r["detection_latency"] for r in rows
                    if r["classifier"] == clf and r["shift_condition"] == shift
                    and r["is_valid_detection"]]
            if len(lats) >= 2:
                mean, lo, hi = bootstrap_ci(lats)
                row += f"{mean:>5.1f} [{lo:.0f},{hi:.0f}] n={len(lats):<3}"
            elif len(lats) == 1:
                row += f"{lats[0]:>5.1f} (n=1)           "
            else:
                row += f"{'—':<22}"
        print(row)

    # Grand mean
    all_lats = [r["detection_latency"] for r in rows if r["is_valid_detection"]]
    mean, lo, hi = bootstrap_ci(all_lats)
    print(f"\nGrand mean latency: {mean:.1f} [{lo:.1f}, {hi:.1f}] (n={len(all_lats)})")

    # =========================================================================
    # 5. FALSE ALARM RATE
    # =========================================================================
    section("5. FALSE ALARM RATE (neg control fired)")

    for clf in CLASSIFIERS:
        cells = [r for r in rows if r["classifier"] == clf]
        k = sum(1 for r in cells if not r.get("neg_clean", True))
        n = len(cells)
        ci = wilson_ci(k, n)
        print(f"  {clf:<18} {k:>2}/{n} = {k/n:.1%}  [{ci[0]:.3f}, {ci[1]:.3f}]")

    total_dirty = sum(1 for r in rows if not r.get("neg_clean", True))
    ci = wilson_ci(total_dirty, len(rows))
    print(f"\n  Overall FAR:       {total_dirty}/{len(rows)} = {total_dirty/len(rows):.1%}  [{ci[0]:.3f}, {ci[1]:.3f}]")

    # =========================================================================
    # 6. WINDOW SIZE EFFECT
    # =========================================================================
    section("6. WINDOW SIZE EFFECT")

    for ws in WINDOW_SIZES:
        valid_ws = [r["detection_latency"] for r in rows
                    if r["window_size"] == ws and r["is_valid_detection"]]
        mean, lo, hi = bootstrap_ci(valid_ws)
        det_rate = sum(1 for r in rows if r["window_size"] == ws and r["is_valid_detection"])
        total_ws = sum(1 for r in rows if r["window_size"] == ws)
        ci_rate = wilson_ci(det_rate, total_ws)
        print(f"  w={ws}: latency={mean:.1f} [{lo:.1f}, {hi:.1f}] n={len(valid_ws)}  |  rate={det_rate}/{total_ws}={det_rate/total_ws:.1%} [{ci_rate[0]:.3f}, {ci_rate[1]:.3f}]")

    # Paired difference
    paired_diffs = []
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            for seed in SEEDS:
                w100 = [r for r in rows if r["classifier"] == clf and r["shift_condition"] == shift
                        and r["seed"] == seed and r["window_size"] == 100 and r["is_valid_detection"]]
                w200 = [r for r in rows if r["classifier"] == clf and r["shift_condition"] == shift
                        and r["seed"] == seed and r["window_size"] == 200 and r["is_valid_detection"]]
                if w100 and w200:
                    paired_diffs.append(w100[0]["detection_latency"] - w200[0]["detection_latency"])
    if paired_diffs:
        mean_diff, lo, hi = bootstrap_ci(paired_diffs)
        print(f"\n  Paired difference (w100 - w200): {mean_diff:.1f} [{lo:.1f}, {hi:.1f}] (n={len(paired_diffs)} pairs)")

    # =========================================================================
    # 7. VARIANCE DECOMPOSITION (ANOVA)
    # =========================================================================
    section("7. VARIANCE DECOMPOSITION (two-way ANOVA on valid detections)")

    # Build data matrix for valid detections
    latencies = []
    clf_labels = []
    shift_labels = []
    for r in rows:
        if r["is_valid_detection"]:
            latencies.append(r["detection_latency"])
            clf_labels.append(r["classifier"])
            shift_labels.append(r["shift_condition"])

    latencies = np.array(latencies, dtype=float)
    n_total = len(latencies)
    grand_mean = np.mean(latencies)

    # SS Total
    ss_total = np.sum((latencies - grand_mean) ** 2)

    # SS Classifier
    ss_clf = 0
    for clf in CLASSIFIERS:
        mask = np.array([c == clf for c in clf_labels])
        if mask.sum() > 0:
            ss_clf += mask.sum() * (np.mean(latencies[mask]) - grand_mean) ** 2

    # SS Shift
    ss_shift = 0
    for shift in SHIFTS:
        mask = np.array([s == shift for s in shift_labels])
        if mask.sum() > 0:
            ss_shift += mask.sum() * (np.mean(latencies[mask]) - grand_mean) ** 2

    # SS Interaction
    ss_interaction = 0
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            mask = np.array([c == clf and s == shift for c, s in zip(clf_labels, shift_labels)])
            if mask.sum() > 0:
                clf_mask = np.array([c == clf for c in clf_labels])
                shift_mask = np.array([s == shift for s in shift_labels])
                cell_mean = np.mean(latencies[mask])
                clf_mean = np.mean(latencies[clf_mask])
                shift_mean = np.mean(latencies[shift_mask])
                interaction_effect = cell_mean - clf_mean - shift_mean + grand_mean
                ss_interaction += mask.sum() * interaction_effect ** 2

    ss_residual = ss_total - ss_clf - ss_shift - ss_interaction

    eta_clf = ss_clf / ss_total
    eta_shift = ss_shift / ss_total
    eta_interaction = ss_interaction / ss_total
    eta_residual = ss_residual / ss_total

    print(f"\n  N valid detections: {n_total}")
    print(f"  Grand mean latency: {grand_mean:.1f}")
    print(f"\n  {'Factor':<25} {'SS':>10} {'η²':>8} {'% variance':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Classifier':<25} {ss_clf:>10.1f} {eta_clf:>8.3f} {eta_clf*100:>10.1f}%")
    print(f"  {'Shift type':<25} {ss_shift:>10.1f} {eta_shift:>8.3f} {eta_shift*100:>10.1f}%")
    print(f"  {'Classifier × Shift':<25} {ss_interaction:>10.1f} {eta_interaction:>8.3f} {eta_interaction*100:>10.1f}%")
    print(f"  {'Residual':<25} {ss_residual:>10.1f} {eta_residual:>8.3f} {eta_residual*100:>10.1f}%")
    print(f"  {'Total':<25} {ss_total:>10.1f} {'1.000':>8} {'100.0%':>12}")

    # Permutation tests
    n_perm = 1000
    rng_perm = np.random.default_rng(42)

    def compute_ss_factor(lat, labels, levels):
        gm = np.mean(lat)
        ss = 0
        for lev in levels:
            mask = np.array([l == lev for l in labels])
            if mask.sum() > 0:
                ss += mask.sum() * (np.mean(lat[mask]) - gm) ** 2
        return ss

    # Permutation test for classifier
    perm_clf_count = 0
    for _ in range(n_perm):
        perm_labels = rng_perm.permutation(clf_labels).tolist()
        perm_ss = compute_ss_factor(latencies, perm_labels, CLASSIFIERS)
        if perm_ss >= ss_clf:
            perm_clf_count += 1
    p_clf = (perm_clf_count + 1) / (n_perm + 1)

    # Permutation test for shift
    perm_shift_count = 0
    for _ in range(n_perm):
        perm_labels = rng_perm.permutation(shift_labels).tolist()
        perm_ss = compute_ss_factor(latencies, perm_labels, SHIFTS)
        if perm_ss >= ss_shift:
            perm_shift_count += 1
    p_shift = (perm_shift_count + 1) / (n_perm + 1)

    print(f"\n  Permutation p-values (1000 permutations):")
    print(f"    Classifier:  p = {p_clf:.4f} {'***' if p_clf < 0.001 else '**' if p_clf < 0.01 else '*' if p_clf < 0.05 else 'ns'}")
    print(f"    Shift type:  p = {p_shift:.4f} {'***' if p_shift < 0.001 else '**' if p_shift < 0.01 else '*' if p_shift < 0.05 else 'ns'}")

    # Bootstrap CIs on eta-squared
    boot_eta_clf = []
    boot_eta_shift = []
    boot_eta_int = []
    for _ in range(N_BOOT):
        idx = RNG.choice(n_total, size=n_total, replace=True)
        b_lat = latencies[idx]
        b_clf = [clf_labels[i] for i in idx]
        b_shift = [shift_labels[i] for i in idx]
        b_gm = np.mean(b_lat)
        b_ss_total = np.sum((b_lat - b_gm) ** 2)
        if b_ss_total == 0:
            continue
        b_ss_clf = compute_ss_factor(b_lat, b_clf, CLASSIFIERS)
        b_ss_shift = compute_ss_factor(b_lat, b_shift, SHIFTS)
        boot_eta_clf.append(b_ss_clf / b_ss_total)
        boot_eta_shift.append(b_ss_shift / b_ss_total)
        boot_eta_int.append(1 - b_ss_clf/b_ss_total - b_ss_shift/b_ss_total - (b_ss_total - b_ss_clf - b_ss_shift)/b_ss_total)

    print(f"\n  Bootstrap 95% CIs on η²:")
    print(f"    Classifier:        [{np.percentile(boot_eta_clf, 2.5):.3f}, {np.percentile(boot_eta_clf, 97.5):.3f}]")
    print(f"    Shift type:        [{np.percentile(boot_eta_shift, 2.5):.3f}, {np.percentile(boot_eta_shift, 97.5):.3f}]")

    # =========================================================================
    # 8. FAILURE ANALYSIS
    # =========================================================================
    section("8. FAILURE ANALYSIS")

    invalid = [r for r in rows if not r["is_valid_detection"]]
    print(f"Total invalid cells: {len(invalid)}/{len(rows)}")

    # Categorize
    no_alarm = [r for r in invalid if r.get("detection_latency") is None]
    early_alarm = [r for r in invalid if r.get("detection_latency") is not None and r["detection_latency"] < 0]
    neg_dirty_inv = [r for r in invalid if not r.get("neg_clean", True)]

    print(f"\n  No alarm (null latency):     {len(no_alarm)}")
    print(f"  Early alarm (latency < 0):   {len(early_alarm)}")
    print(f"  Neg control dirty:           {len(neg_dirty_inv)}")
    # Some cells may have both early alarm AND dirty neg
    both = [r for r in invalid if (r.get("detection_latency") is not None and r["detection_latency"] < 0) and not r.get("neg_clean", True)]
    print(f"  Both early + dirty:          {len(both)}")

    print(f"\n  Failures by classifier:")
    for clf in CLASSIFIERS:
        inv_clf = [r for r in invalid if r["classifier"] == clf]
        print(f"    {clf:<18} {len(inv_clf):>3} invalid")

    print(f"\n  Failures by shift:")
    for shift in SHIFTS:
        inv_shift = [r for r in invalid if r["shift_condition"] == shift]
        print(f"    {shift:<30} {len(inv_shift):>3} invalid")

    print(f"\n  Top 5 failure cells (classifier × shift):")
    fail_counts = defaultdict(int)
    for r in invalid:
        fail_counts[(r["classifier"], r["shift_condition"])] += 1
    for (clf, shift), count in sorted(fail_counts.items(), key=lambda x: -x[1])[:5]:
        total_cell = sum(1 for r in rows if r["classifier"] == clf and r["shift_condition"] == shift)
        print(f"    {clf} × {shift}: {count}/{total_cell} invalid")

    # =========================================================================
    # 9. N=5 vs N=20 COMPARISON
    # =========================================================================
    section("9. N=5 vs N=20 COMPARISON")

    rows_n5 = [r for r in rows if r["seed"] < 5]
    rows_n20 = rows

    print(f"  N=5 subset: {len(rows_n5)} cells")
    print(f"  N=20 full:  {len(rows_n20)} cells")

    # Detection rates
    k5 = sum(1 for r in rows_n5 if r["is_valid_detection"])
    k20 = sum(1 for r in rows_n20 if r["is_valid_detection"])
    ci5 = wilson_ci(k5, len(rows_n5))
    ci20 = wilson_ci(k20, len(rows_n20))
    print(f"\n  Detection rate:")
    print(f"    N=5:  {k5}/{len(rows_n5)} = {k5/len(rows_n5):.1%}  [{ci5[0]:.3f}, {ci5[1]:.3f}]")
    print(f"    N=20: {k20}/{len(rows_n20)} = {k20/len(rows_n20):.1%}  [{ci20[0]:.3f}, {ci20[1]:.3f}]")

    # Mean latency comparison
    lats_n5 = [r["detection_latency"] for r in rows_n5 if r["is_valid_detection"]]
    lats_n20 = [r["detection_latency"] for r in rows_n20 if r["is_valid_detection"]]
    m5, lo5, hi5 = bootstrap_ci(lats_n5)
    m20, lo20, hi20 = bootstrap_ci(lats_n20)
    print(f"\n  Grand mean latency:")
    print(f"    N=5:  {m5:.1f} [{lo5:.1f}, {hi5:.1f}]")
    print(f"    N=20: {m20:.1f} [{lo20:.1f}, {hi20:.1f}]")

    # ANOVA comparison
    print(f"\n  η² comparison (N=5 vs N=20):")
    # N=5 ANOVA
    lats5 = np.array([r["detection_latency"] for r in rows_n5 if r["is_valid_detection"]], dtype=float)
    clf5 = [r["classifier"] for r in rows_n5 if r["is_valid_detection"]]
    shift5 = [r["shift_condition"] for r in rows_n5 if r["is_valid_detection"]]
    gm5 = np.mean(lats5)
    ss_total5 = np.sum((lats5 - gm5) ** 2)
    ss_clf5 = compute_ss_factor(lats5, clf5, CLASSIFIERS)
    ss_shift5 = compute_ss_factor(lats5, shift5, SHIFTS)
    ss_int5 = 0
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            mask = np.array([c == clf and s == shift for c, s in zip(clf5, shift5)])
            if mask.sum() > 0:
                clf_mask = np.array([c == clf for c in clf5])
                shift_mask = np.array([s == shift for s in shift5])
                cell_mean = np.mean(lats5[mask])
                clf_mean = np.mean(lats5[clf_mask])
                shift_mean = np.mean(lats5[shift_mask])
                ss_int5 += mask.sum() * (cell_mean - clf_mean - shift_mean + gm5) ** 2

    print(f"    {'Factor':<25} {'N=5 η²':>8} {'N=20 η²':>8} {'Δ':>8}")
    print(f"    {'-'*55}")
    print(f"    {'Classifier':<25} {ss_clf5/ss_total5:>8.3f} {eta_clf:>8.3f} {eta_clf - ss_clf5/ss_total5:>+8.3f}")
    print(f"    {'Shift type':<25} {ss_shift5/ss_total5:>8.3f} {eta_shift:>8.3f} {eta_shift - ss_shift5/ss_total5:>+8.3f}")
    print(f"    {'Interaction':<25} {ss_int5/ss_total5:>8.3f} {eta_interaction:>8.3f} {eta_interaction - ss_int5/ss_total5:>+8.3f}")
    print(f"    {'Residual':<25} {1-ss_clf5/ss_total5-ss_shift5/ss_total5-ss_int5/ss_total5:>8.3f} {eta_residual:>8.3f}")

    # Per-cell latency comparison for top interactions
    print(f"\n  Top interaction effects (N=20):")
    interactions = []
    for clf in CLASSIFIERS:
        clf_mask = np.array([c == clf for c in clf_labels])
        clf_mean_lat = np.mean(latencies[clf_mask]) if clf_mask.sum() > 0 else grand_mean
        for shift in SHIFTS:
            shift_mask = np.array([s == shift for s in shift_labels])
            shift_mean_lat = np.mean(latencies[shift_mask]) if shift_mask.sum() > 0 else grand_mean
            cell_mask = np.array([c == clf and s == shift for c, s in zip(clf_labels, shift_labels)])
            if cell_mask.sum() > 0:
                cell_mean = np.mean(latencies[cell_mask])
                effect = cell_mean - clf_mean_lat - shift_mean_lat + grand_mean
                interactions.append((clf, shift, effect, cell_mean, cell_mask.sum()))

    interactions.sort(key=lambda x: -abs(x[2]))
    print(f"    {'Combination':<45} {'Effect':>8} {'Cell mean':>10} {'n':>4}")
    for clf, shift, effect, cell_mean, n in interactions[:6]:
        print(f"    {clf} × {shift:<28} {effect:>+8.1f} {cell_mean:>10.1f} {n:>4}")

    # =========================================================================
    # 10. POWER ANALYSIS
    # =========================================================================
    section("10. STATISTICAL POWER (N=20)")

    # Within-cell SD
    cell_sds = []
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            lats = [r["detection_latency"] for r in rows
                    if r["classifier"] == clf and r["shift_condition"] == shift
                    and r["is_valid_detection"]]
            if len(lats) >= 3:
                cell_sds.append(np.std(lats, ddof=1))
    pooled_sd = np.sqrt(np.mean(np.array(cell_sds)**2))
    # MDE for two-sample t-test at n=20 per group (approx)
    # Using n_per_cell as average valid detections per cell
    avg_n_per_cell = n_total / (len(CLASSIFIERS) * len(SHIFTS))
    # Cohen's d for 80% power at alpha=0.05, two-sided
    # For n=20: d ≈ 0.91 (from power tables)
    mde_steps = 0.91 * pooled_sd
    print(f"  Pooled within-cell SD: {pooled_sd:.1f} steps")
    print(f"  Avg valid detections per cell: {avg_n_per_cell:.1f}")
    print(f"  MDE (Cohen's d=0.91, 80% power): {mde_steps:.1f} steps")
    print(f"  (vs N=5 MDE: {1.46 * pooled_sd:.1f} steps)")


if __name__ == "__main__":
    main()
