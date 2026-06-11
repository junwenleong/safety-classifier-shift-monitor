"""Sensitivity analysis: mixed-effects model + TPR/FAR decomposition.

1. Seed-clustered bootstrap for η² CIs (alternative to mixed-effects)
2. Raw TPR and control FAR separated from the combined valid-detection metric
"""
import json
from pathlib import Path
import numpy as np

RESULTS = Path("results/factorial_results.jsonl")
CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "code-switch", "compositional-long-context", "temporal", "adversarial-suffix"]


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def compute_eta_squared(rows):
    """Compute η² for classifier, shift, interaction from a list of rows."""
    latencies = np.array([r["detection_latency"] for r in rows], dtype=float)
    clf_labels = [r["classifier"] for r in rows]
    shift_labels = [r["shift_condition"] for r in rows]
    grand_mean = np.mean(latencies)
    ss_total = np.sum((latencies - grand_mean) ** 2)
    if ss_total < 1e-12:
        return {"classifier": 0, "shift_type": 0, "interaction": 0, "residual": 1}

    # SS classifier
    ss_clf = 0
    for clf in set(clf_labels):
        mask = [c == clf for c in clf_labels]
        n_i = sum(mask)
        mean_i = np.mean(latencies[mask])
        ss_clf += n_i * (mean_i - grand_mean) ** 2

    # SS shift
    ss_shift = 0
    for s in set(shift_labels):
        mask = [x == s for x in shift_labels]
        n_i = sum(mask)
        mean_i = np.mean(latencies[mask])
        ss_shift += n_i * (mean_i - grand_mean) ** 2

    # SS interaction
    ss_int = 0
    for clf in set(clf_labels):
        for s in set(shift_labels):
            mask = np.array([c == clf and x == s for c, x in zip(clf_labels, shift_labels)])
            if mask.sum() > 0:
                clf_mask = np.array([c == clf for c in clf_labels])
                shift_mask = np.array([x == s for x in shift_labels])
                effect = np.mean(latencies[mask]) - np.mean(latencies[clf_mask]) - np.mean(latencies[shift_mask]) + grand_mean
                ss_int += mask.sum() * effect ** 2

    ss_res = max(ss_total - ss_clf - ss_shift - ss_int, 0)
    return {
        "classifier": ss_clf / ss_total,
        "shift_type": ss_shift / ss_total,
        "interaction": ss_int / ss_total,
        "residual": ss_res / ss_total,
    }


def main():
    rows = load_jsonl(RESULTS)
    for r in rows:
        r["is_valid_detection"] = (
            r.get("detection_latency") is not None
            and r["detection_latency"] >= 0
            and r.get("neg_clean") is True
        )

    valid = [r for r in rows if r["is_valid_detection"]]

    # =====================================================================
    # PART 1: Original ANOVA (for comparison)
    # =====================================================================
    print("=" * 70)
    print("PART 1: Original η² (individual-observation resampling)")
    print("=" * 70)
    orig = compute_eta_squared(valid)
    print(f"  Classifier:  {orig['classifier']:.4f}")
    print(f"  Shift type:  {orig['shift_type']:.4f}")
    print(f"  Interaction: {orig['interaction']:.4f}")
    print(f"  Residual:    {orig['residual']:.4f}")
    print(f"  N = {len(valid)}")

    # =====================================================================
    # PART 2: Seed-clustered bootstrap
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 2: Seed-clustered bootstrap (resample SEEDS, not observations)")
    print("=" * 70)

    # Group by seed — each seed is a cluster
    seeds = sorted(set(r["seed"] for r in valid))
    seed_groups = {s: [r for r in valid if r["seed"] == s] for s in seeds}

    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_results = {"classifier": [], "shift_type": [], "interaction": [], "residual": []}

    for _ in range(n_boot):
        # Resample seeds with replacement
        boot_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        boot_rows = []
        for s in boot_seeds:
            boot_rows.extend(seed_groups[s])
        if len(boot_rows) < 10:
            continue
        eta = compute_eta_squared(boot_rows)
        for k in boot_results:
            boot_results[k].append(eta[k])

    print(f"  Bootstrap iterations: {n_boot}")
    print(f"  Cluster unit: seed (n_clusters = {len(seeds)})")
    print(f"\n  {'Factor':<14} {'Point':>7} {'2.5%':>7} {'97.5%':>7} {'Stable?':>8}")
    print(f"  {'-'*50}")
    for factor in ["classifier", "shift_type", "interaction", "residual"]:
        point = orig[factor]
        lo = np.percentile(boot_results[factor], 2.5)
        hi = np.percentile(boot_results[factor], 97.5)
        stable = "YES" if lo > 0.05 else "marginal"
        print(f"  {factor:<14} {point:>7.3f} [{lo:>6.3f}, {hi:>6.3f}]  {stable:>8}")

    # =====================================================================
    # PART 3: Mixed-effects via statsmodels (if available)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 3: Mixed-effects model (seed as random intercept)")
    print("=" * 70)
    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        df = pd.DataFrame(valid)
        df["latency"] = df["detection_latency"].astype(float)

        # Fit: latency ~ classifier * shift_condition + (1 | seed)
        model = smf.mixedlm(
            "latency ~ C(classifier) * C(shift_condition)",
            data=df,
            groups=df["seed"],
        )
        result = model.fit(reml=True, method="lbfgs")
        print(f"  Converged: {result.converged}")
        print(f"  Random effect variance (seed): {result.cov_re.iloc[0,0]:.2f}")
        print(f"  Residual variance: {result.scale:.2f}")
        print(f"  Group var / (Group var + Residual var) = {result.cov_re.iloc[0,0] / (result.cov_re.iloc[0,0] + result.scale):.4f}")
        print(f"  → Seed explains {result.cov_re.iloc[0,0] / (result.cov_re.iloc[0,0] + result.scale)*100:.1f}% of residual variance")
        print(f"\n  Key insight: If seed variance is small relative to residual,")
        print(f"  the fixed-effects η² estimates are robust to clustering.")

        # Compare fixed-effects F-stats to check if conclusions change
        # Use Type III anova
        from statsmodels.stats.anova import anova_lm
        import statsmodels.api as sm

        ols_model = smf.ols("latency ~ C(classifier) * C(shift_condition)", data=df).fit()
        anova_table = sm.stats.anova_lm(ols_model, typ=2)
        print(f"\n  OLS ANOVA (Type II) for comparison:")
        print(f"  {'Source':<35} {'F':>8} {'p':>10}")
        for idx in anova_table.index:
            if idx != "Residual":
                f_val = anova_table.loc[idx, "F"]
                p_val = anova_table.loc[idx, "PR(>F)"]
                print(f"  {idx:<35} {f_val:>8.1f} {p_val:>10.2e}")

    except ImportError as e:
        print(f"  statsmodels not available: {e}")
        print("  Using seed-clustered bootstrap results above instead.")
    except Exception as e:
        print(f"  Model fitting failed: {e}")
        print("  Using seed-clustered bootstrap results above instead.")

    # =====================================================================
    # PART 4: TPR / FAR decomposition
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 4: TPR and FAR decomposition (appendix table)")
    print("=" * 70)

    print(f"\n  {'Classifier':<16} {'Raw TPR':>10} {'Control FAR':>12} {'Valid Det.':>11}")
    print(f"  {'-'*52}")

    for clf in CLASSIFIERS:
        clf_rows = [r for r in rows if r["classifier"] == clf]

        # Raw TPR: cells where detection_latency >= 0 (regardless of neg control)
        shifted_cells = [r for r in clf_rows
                         if r.get("detection_latency") is not None and r["detection_latency"] >= 0]
        total_shifted = len(clf_rows)  # all 200 cells per classifier
        raw_tpr = len(shifted_cells) / total_shifted if total_shifted > 0 else 0

        # Control FAR: proportion of cells where neg_clean is False
        neg_dirty = sum(1 for r in clf_rows if not r.get("neg_clean", True))
        control_far = neg_dirty / total_shifted if total_shifted > 0 else 0

        # Valid detection (combined): latency >= 0 AND neg_clean
        valid_det = sum(1 for r in clf_rows if r["is_valid_detection"])
        valid_rate = valid_det / total_shifted if total_shifted > 0 else 0

        print(f"  {clf:<16} {raw_tpr:>9.3f} {control_far:>11.3f} {valid_rate:>10.3f}")

    print(f"\n  Per classifier × shift:")
    print(f"  {'Classifier':<16} {'Shift':<24} {'Raw TPR':>8} {'FAR':>6} {'Valid':>6} {'n':>4}")
    print(f"  {'-'*68}")
    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            cells = [r for r in rows if r["classifier"] == clf and r["shift_condition"] == shift]
            n_cells = len(cells)
            if n_cells == 0:
                continue
            detected = sum(1 for r in cells
                           if r.get("detection_latency") is not None and r["detection_latency"] >= 0)
            neg_dirty = sum(1 for r in cells if not r.get("neg_clean", True))
            valid_d = sum(1 for r in cells if r["is_valid_detection"])
            tpr = detected / n_cells
            far = neg_dirty / n_cells
            vr = valid_d / n_cells
            print(f"  {clf:<16} {shift:<24} {tpr:>7.2f} {far:>5.2f} {vr:>5.2f} {n_cells:>4}")

    # Summary
    print(f"\n  OVERALL:")
    all_detected = sum(1 for r in rows
                       if r.get("detection_latency") is not None and r["detection_latency"] >= 0)
    all_neg_dirty = sum(1 for r in rows if not r.get("neg_clean", True))
    all_valid = sum(1 for r in rows if r["is_valid_detection"])
    n_total = len(rows)
    print(f"    Raw TPR (latency≥0):     {all_detected}/{n_total} = {all_detected/n_total:.3f}")
    print(f"    Control FAR (neg dirty): {all_neg_dirty}/{n_total} = {all_neg_dirty/n_total:.3f}")
    print(f"    Valid detection (both):  {all_valid}/{n_total} = {all_valid/n_total:.3f}")


if __name__ == "__main__":
    main()
