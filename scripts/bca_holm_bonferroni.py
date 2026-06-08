"""BCa bootstrap (10K resamples) + Holm-Bonferroni on highlighted comparisons.

Replaces percentile bootstrap with BCa throughout. Applies multiplicity
correction to the 8-10 comparisons explicitly discussed in the paper.

Usage:
    .venv/bin/python scripts/bca_holm_bonferroni.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

RESULTS = Path("results/factorial_results.jsonl")
N_BOOT = 10_000
ALPHA = 0.05
RNG = np.random.default_rng(42)


# --- BCa Bootstrap ---

def bca_ci(data: np.ndarray, stat_fn=np.mean, n_boot: int = N_BOOT,
           alpha: float = ALPHA) -> tuple[float, float, float]:
    """BCa bootstrap CI. Returns (estimate, lower, upper)."""
    n = len(data)
    if n < 2:
        val = float(stat_fn(data))
        return (val, val, val)

    theta_hat = float(stat_fn(data))

    # Bootstrap distribution
    boot_thetas = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(data, size=n, replace=True)
        boot_thetas[i] = stat_fn(sample)

    # Bias correction: z0
    prop_below = np.mean(boot_thetas < theta_hat)
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = sp_stats.norm.ppf(prop_below)

    # Acceleration: jackknife
    jackknife_vals = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_vals[i] = stat_fn(jack_sample)
    jack_mean = np.mean(jackknife_vals)
    diffs = jack_mean - jackknife_vals
    a_hat = np.sum(diffs**3) / (6.0 * (np.sum(diffs**2))**1.5 + 1e-300)

    # Adjusted percentiles
    z_alpha_lo = sp_stats.norm.ppf(alpha / 2)
    z_alpha_hi = sp_stats.norm.ppf(1 - alpha / 2)

    def adjusted_percentile(z_alpha):
        numerator = z0 + z_alpha
        denom = 1 - a_hat * numerator
        if abs(denom) < 1e-10:
            return 0.5
        return sp_stats.norm.cdf(z0 + numerator / denom)

    alpha1 = adjusted_percentile(z_alpha_lo)
    alpha2 = adjusted_percentile(z_alpha_hi)

    # Clamp to valid percentile range
    alpha1 = np.clip(alpha1, 0.5 / n_boot, 1 - 0.5 / n_boot)
    alpha2 = np.clip(alpha2, 0.5 / n_boot, 1 - 0.5 / n_boot)

    lower = float(np.percentile(boot_thetas, 100 * alpha1))
    upper = float(np.percentile(boot_thetas, 100 * alpha2))

    return (theta_hat, lower, upper)


def bca_diff_ci(data1: np.ndarray, data2: np.ndarray,
                n_boot: int = N_BOOT, alpha: float = ALPHA) -> tuple[float, float, float]:
    """BCa CI on the difference of means (data1 - data2), unpaired."""
    n1, n2 = len(data1), len(data2)
    theta_hat = float(np.mean(data1) - np.mean(data2))

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        s1 = RNG.choice(data1, size=n1, replace=True)
        s2 = RNG.choice(data2, size=n2, replace=True)
        boot_diffs[i] = np.mean(s1) - np.mean(s2)

    # Bias correction
    prop_below = np.clip(np.mean(boot_diffs < theta_hat), 1e-10, 1 - 1e-10)
    z0 = sp_stats.norm.ppf(prop_below)

    # Acceleration via jackknife on combined sample
    combined = np.concatenate([data1, data2])
    n = len(combined)
    labels = np.array([0] * n1 + [1] * n2)
    jackknife_vals = np.empty(n)
    for i in range(n):
        jack_combined = np.delete(combined, i)
        jack_labels = np.delete(labels, i)
        m1 = np.mean(jack_combined[jack_labels == 0]) if np.sum(jack_labels == 0) > 0 else 0
        m2 = np.mean(jack_combined[jack_labels == 1]) if np.sum(jack_labels == 1) > 0 else 0
        jackknife_vals[i] = m1 - m2
    jack_mean = np.mean(jackknife_vals)
    diffs = jack_mean - jackknife_vals
    a_hat = np.sum(diffs**3) / (6.0 * (np.sum(diffs**2))**1.5 + 1e-300)

    z_lo = sp_stats.norm.ppf(alpha / 2)
    z_hi = sp_stats.norm.ppf(1 - alpha / 2)

    def adj_pct(z_a):
        num = z0 + z_a
        den = 1 - a_hat * num
        if abs(den) < 1e-10:
            return 0.5
        return sp_stats.norm.cdf(z0 + num / den)

    a1 = np.clip(adj_pct(z_lo), 0.5 / n_boot, 1 - 0.5 / n_boot)
    a2 = np.clip(adj_pct(z_hi), 0.5 / n_boot, 1 - 0.5 / n_boot)

    lower = float(np.percentile(boot_diffs, 100 * a1))
    upper = float(np.percentile(boot_diffs, 100 * a2))

    return (theta_hat, lower, upper)


def bca_paired_diff_ci(data1: np.ndarray, data2: np.ndarray,
                       n_boot: int = N_BOOT, alpha: float = ALPHA) -> tuple[float, float, float]:
    """BCa CI on paired difference (data1 - data2)."""
    diffs = data1 - data2
    return bca_ci(diffs, stat_fn=np.mean, n_boot=n_boot, alpha=alpha)


# --- Holm-Bonferroni ---

def holm_bonferroni(p_values: list[tuple[str, float]]) -> list[tuple[str, float, float, bool]]:
    """Apply Holm-Bonferroni correction.

    Args:
        p_values: list of (comparison_name, p_value)

    Returns:
        list of (name, raw_p, adjusted_p, significant) sorted by raw p
    """
    m = len(p_values)
    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    results = []
    for i, (name, p) in enumerate(sorted_pvals):
        adjusted_p = min(p * (m - i), 1.0)
        # Enforce monotonicity
        if i > 0 and adjusted_p < results[-1][2]:
            adjusted_p = results[-1][2]
        significant = adjusted_p < ALPHA
        results.append((name, p, adjusted_p, significant))
    return results


def bootstrap_p_value(data1: np.ndarray, data2: np.ndarray,
                      n_boot: int = N_BOOT, paired: bool = False) -> float:
    """Bootstrap permutation p-value for difference in means."""
    if paired:
        diffs = data1 - data2
        observed = np.mean(diffs)
        count = 0
        for _ in range(n_boot):
            signs = RNG.choice([-1, 1], size=len(diffs))
            boot_diff = np.mean(diffs * signs)
            if abs(boot_diff) >= abs(observed):
                count += 1
        return count / n_boot
    else:
        observed = abs(np.mean(data1) - np.mean(data2))
        combined = np.concatenate([data1, data2])
        n1 = len(data1)
        count = 0
        for _ in range(n_boot):
            perm = RNG.permutation(combined)
            perm_diff = abs(np.mean(perm[:n1]) - np.mean(perm[n1:]))
            if perm_diff >= observed:
                count += 1
        return count / n_boot


# --- Main ---

def main():
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    for r in rows:
        r["is_valid_detection"] = (
            r.get("detection_latency") is not None
            and r["detection_latency"] >= 0
            and r.get("neg_clean") is True
        )
    valid = [r for r in rows if r["is_valid_detection"]]

    def get_latencies(clf=None, shift=None, ws=None):
        return np.array([
            r["detection_latency"] for r in valid
            if (clf is None or r["classifier"] == clf)
            and (shift is None or r["shift_condition"] == shift)
            and (ws is None or r["window_size"] == ws)
        ], dtype=float)

    # ========================================================================
    # PART 1: BCa CIs on all key statistics reported in the paper
    # ========================================================================
    print("=" * 80)
    print("BCa BOOTSTRAP CIs (10,000 resamples, seed=42)")
    print("=" * 80)

    # Overall mean latencies by window size
    print("\n--- Window Size Effect ---")
    for ws in [100, 200]:
        lats = get_latencies(ws=ws)
        mean, lo, hi = bca_ci(lats)
        print(f"  w={ws}: {mean:.1f} [{lo:.1f}, {hi:.1f}] (n={len(lats)})")

    # Per-cell latencies
    print("\n--- Per-Cell Mean Latencies (BCa 95% CI) ---")
    classifiers = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
    shifts = ["paraphrase", "code-switch", "compositional-long-context",
              "temporal", "adversarial-suffix"]

    header = f"{'Classifier':<16}" + "".join(f"{s[:12]:<22}" for s in shifts)
    print(header)
    for clf in classifiers:
        row = f"{clf:<16}"
        for shift in shifts:
            lats = get_latencies(clf=clf, shift=shift)
            if len(lats) >= 2:
                mean, lo, hi = bca_ci(lats)
                row += f"{mean:>5.1f} [{lo:.0f},{hi:.0f}]       "
            else:
                row += f"{'—':<22}"
        print(row)

    # ========================================================================
    # PART 2: Highlighted comparisons with p-values
    # ========================================================================
    print("\n" + "=" * 80)
    print("HIGHLIGHTED COMPARISONS (bootstrap permutation p-values)")
    print("=" * 80)

    comparisons = []

    # 1. Encoder vs decoder on paraphrase
    enc_para = np.concatenate([get_latencies("deberta", "paraphrase"),
                               get_latencies("text-moderation", "paraphrase")])
    dec_para = np.concatenate([get_latencies("llama-guard", "paraphrase"),
                               get_latencies("shieldgemma", "paraphrase")])
    diff, lo, hi = bca_diff_ci(dec_para, enc_para)
    p = bootstrap_p_value(enc_para, dec_para)
    comparisons.append(("Decoder vs Encoder on paraphrase", p))
    print(f"\n  1. Decoder vs Encoder on paraphrase:")
    print(f"     Diff = {diff:.1f} [{lo:.1f}, {hi:.1f}], p = {p:.4f}")

    # 2. Encoder vs decoder on adversarial suffix
    enc_adv = np.concatenate([get_latencies("deberta", "adversarial-suffix"),
                              get_latencies("text-moderation", "adversarial-suffix")])
    dec_adv = np.concatenate([get_latencies("llama-guard", "adversarial-suffix"),
                              get_latencies("shieldgemma", "adversarial-suffix")])
    diff, lo, hi = bca_diff_ci(enc_adv, dec_adv)
    p = bootstrap_p_value(enc_adv, dec_adv)
    comparisons.append(("Encoder vs Decoder on adversarial suffix", p))
    print(f"\n  2. Encoder vs Decoder on adversarial suffix:")
    print(f"     Diff = {diff:.1f} [{lo:.1f}, {hi:.1f}], p = {p:.4f}")

    # 3. DeBERTa: adversarial vs paraphrase (crossover within-classifier)
    deb_para = get_latencies("deberta", "paraphrase")
    deb_adv = get_latencies("deberta", "adversarial-suffix")
    diff, lo, hi = bca_diff_ci(deb_adv, deb_para)
    p = bootstrap_p_value(deb_para, deb_adv)
    comparisons.append(("DeBERTa adversarial vs paraphrase", p))
    print(f"\n  3. DeBERTa: adversarial vs paraphrase:")
    print(f"     Diff = {diff:.1f} [{lo:.1f}, {hi:.1f}], p = {p:.4f}")

    # 4. Llama Guard: paraphrase vs adversarial (opposite direction)
    lg_para = get_latencies("llama-guard", "paraphrase")
    lg_adv = get_latencies("llama-guard", "adversarial-suffix")
    diff, lo, hi = bca_diff_ci(lg_para, lg_adv)
    p = bootstrap_p_value(lg_adv, lg_para)
    comparisons.append(("Llama Guard paraphrase vs adversarial", p))
    print(f"\n  4. Llama Guard: paraphrase vs adversarial:")
    print(f"     Diff = {diff:.1f} [{lo:.1f}, {hi:.1f}], p = {p:.4f}")

    # 5. Window size effect (paired by cell)
    # Match cells by (classifier, shift_condition, seed)
    w100_map = {}
    w200_map = {}
    for r in valid:
        key = (r["classifier"], r["shift_condition"], r["seed"])
        if r["window_size"] == 100:
            w100_map[key] = r["detection_latency"]
        elif r["window_size"] == 200:
            w200_map[key] = r["detection_latency"]
    paired_keys = sorted(set(w100_map.keys()) & set(w200_map.keys()))
    w100_paired = np.array([w100_map[k] for k in paired_keys], dtype=float)
    w200_paired = np.array([w200_map[k] for k in paired_keys], dtype=float)
    diff, lo, hi = bca_paired_diff_ci(w100_paired, w200_paired)
    p = bootstrap_p_value(w100_paired, w200_paired, paired=True)
    comparisons.append(("Window 100 vs 200 (paired)", p))
    print(f"\n  5. Window 100 vs 200 (paired, n={len(paired_keys)}):")
    print(f"     Diff = {diff:.1f} [{lo:.1f}, {hi:.1f}], p = {p:.4f}")

    # 6. Slowest cell: Llama Guard × code-switch vs grand mean
    lg_cs = get_latencies("llama-guard", "code-switch")
    all_lats = get_latencies()
    diff, lo, hi = bca_diff_ci(lg_cs, all_lats)
    p = bootstrap_p_value(lg_cs, all_lats)
    comparisons.append(("Llama Guard×code-switch vs grand mean", p))
    print(f"\n  6. Llama Guard × code-switch vs grand mean:")
    print(f"     Diff = {diff:.1f} [{lo:.1f}, {hi:.1f}], p = {p:.4f}")

    # 7. ShieldGemma paraphrase vs ShieldGemma adversarial
    sg_para = get_latencies("shieldgemma", "paraphrase")
    sg_adv = get_latencies("shieldgemma", "adversarial-suffix")
    diff, lo, hi = bca_diff_ci(sg_para, sg_adv)
    p = bootstrap_p_value(sg_adv, sg_para)
    comparisons.append(("ShieldGemma paraphrase vs adversarial", p))
    print(f"\n  7. ShieldGemma: paraphrase vs adversarial:")
    print(f"     Diff = {diff:.1f} [{lo:.1f}, {hi:.1f}], p = {p:.4f}")

    # 8. FAR spread: DeBERTa vs Text-Moderation
    deb_far_cells = [r for r in rows if r["classifier"] == "deberta"]
    tm_far_cells = [r for r in rows if r["classifier"] == "text-moderation"]
    deb_dirty = np.array([0 if r.get("neg_clean", True) else 1 for r in deb_far_cells], dtype=float)
    tm_dirty = np.array([0 if r.get("neg_clean", True) else 1 for r in tm_far_cells], dtype=float)
    diff, lo, hi = bca_diff_ci(deb_dirty, tm_dirty)
    p = bootstrap_p_value(deb_dirty, tm_dirty)
    comparisons.append(("FAR: DeBERTa vs Text-Moderation", p))
    print(f"\n  8. FAR: DeBERTa vs Text-Moderation:")
    print(f"     Diff = {diff:.3f} [{lo:.3f}, {hi:.3f}], p = {p:.4f}")

    # ========================================================================
    # PART 3: Holm-Bonferroni correction
    # ========================================================================
    print("\n" + "=" * 80)
    print("HOLM-BONFERRONI CORRECTION")
    print(f"  Family size: {len(comparisons)} comparisons, α = {ALPHA}")
    print("=" * 80)

    results = holm_bonferroni(comparisons)
    print(f"\n  {'Comparison':<48} {'Raw p':<10} {'Adj p':<10} {'Sig?'}")
    print("  " + "-" * 80)
    for name, raw_p, adj_p, sig in results:
        marker = "✓" if sig else "✗"
        print(f"  {name:<48} {raw_p:<10.4f} {adj_p:<10.4f} {marker}")

    n_sig = sum(1 for _, _, _, s in results if s)
    n_nonsig = len(results) - n_sig
    print(f"\n  {n_sig}/{len(results)} comparisons survive Holm-Bonferroni correction at α={ALPHA}")
    if n_nonsig > 0:
        print(f"  ⚠ {n_nonsig} comparison(s) do NOT survive correction:")
        for name, raw_p, adj_p, sig in results:
            if not sig:
                print(f"    - {name} (raw p={raw_p:.4f}, adjusted p={adj_p:.4f})")

    # ========================================================================
    # PART 4: Updated paper-ready statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("PAPER-READY STATISTICS (BCa, 10K resamples)")
    print("=" * 80)

    # Overall
    all_w100 = get_latencies(ws=100)
    all_w200 = get_latencies(ws=200)
    m1, l1, h1 = bca_ci(all_w100)
    m2, l2, h2 = bca_ci(all_w200)
    print(f"\n  Overall w=100: {m1:.1f} [{l1:.1f}, {h1:.1f}]")
    print(f"  Overall w=200: {m2:.1f} [{l2:.1f}, {h2:.1f}]")
    print(f"  Paired diff:   {diff:.1f} [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()
