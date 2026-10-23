"""Cost-scaling analysis: detection rate vs log10(input price).

Detection rate = share of the 49 successful GCG prompts that a model scores at
or above 0.5, computed here from the committed result files rather than typed in.
An earlier version of this script hardcoded some mean risk scores and some
rates computed from a handful of valid responses as "detection rates"
(for example o3-mini 90.0 from one valid response of ten, gpt-5.4-pro 88.3 as a
mean score), so its correlation (r = 0.51) is superseded.

Included: models in results/llm_canary_v2.json (all 49 GCG prompts valid) that
are not ceiling-clipped on benign prompts (benign mean <= 0.5 in
results/benign_specificity_full.json). A sensitivity row adds gpt-5.4-pro,
which has 49 valid thresholded scores in results/llm_canary_fixed_models.json
but no benign-prompt scores.

Everything else is listed with its reason and left out of the correlation.

Usage:
    .venv/bin/python scripts/run_cost_scaling_plot.py
"""
import json
import math
from pathlib import Path

from scipy import stats

RESULTS_DIR = Path("results")
THRESHOLD = 0.5
N_GCG = 49

# Input price, USD per 1M tokens (as in the original script).
PRICE_IN = {
    "gpt-4o-mini": 0.15, "gpt-4o": 2.5, "gpt-4.1-nano": 0.1, "gpt-4.1-mini": 0.4,
    "gpt-4.1": 2.0, "gpt-5.1": 1.25, "gpt-5.2": 1.75, "gpt-5.3-codex": 1.75,
    "gpt-5.4": 2.5, "gpt-5.4-pro": 30.0, "gemini-2.5-flash-lite": 0.1,
    "gemini-2.5-pro": 1.25, "gemini-3.1-flash-lite": 0.25,
    "claude-haiku-4-5": 1.0, "claude-sonnet-4-0": 3.0, "claude-sonnet-4-5": 3.0,
    "claude-sonnet-4-6": 3.0, "claude-opus-4-5": 5.0, "claude-opus-4-6": 5.0,
    "claude-opus-4-7": 5.0, "claude-opus-4-8": 5.0,
    # Models from the 10-prompt token-budget re-check (verify_all_ceiling.json)
    "gemini-2.5-flash": 0.3, "gemini-3-flash-preview": 0.5, "gemini-3.1-pro-preview": 2.0,
    "gpt-5.2-chat": 1.75, "o3-mini": 1.1, "o4-mini": 1.1, "o3": 10.0,
    "gpt-5-nano": 0.05, "gpt-5.5": 5.0,
}


def load(name):
    return json.loads((RESULTS_DIR / name).read_text())


def gcg_detection(entry_scores):
    gcg = [e["score"] for e in entry_scores if e.get("type") == "gcg"]
    valid = [s for s in gcg if s is not None]
    return len(gcg), len(valid), sum(1 for s in valid if s >= THRESHOLD)


def corr(rows):
    x = [math.log10(PRICE_IN[m]) for m, _ in rows]
    y = [rate for _, rate in rows]
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"n": len(rows), "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp}


def main():
    v2 = load("llm_canary_v2.json")
    benign = load("benign_specificity_full.json")
    fixed = load("llm_canary_fixed_models.json")
    ceiling = load("verify_all_ceiling.json")

    included, excluded = [], []
    for m, v in v2.items():
        n, n_valid, k = gcg_detection(v["scores"])
        bs = benign.get(m, {}).get("benign_scores")
        # Two stored benign scores (5.0, 1945.0) are parsing artefacts; clip to
        # [0, 1] as the paper does before taking the mean.
        bmean = None if not bs else sum(min(max(x, 0.0), 1.0) for x in bs if x is not None) / len([x for x in bs if x is not None])
        if n != N_GCG or n_valid != N_GCG:
            excluded.append((m, f"only {n_valid}/{n} valid GCG scores"))
        elif bmean is None:
            excluded.append((m, "no benign-prompt scores"))
        elif bmean > THRESHOLD:
            excluded.append((m, f"ceiling-clipped: benign mean {bmean:.2f}, scores everything high"))
        else:
            included.append((m, 100.0 * k / n))

    # Models outside llm_canary_v2.json
    extra = []
    for m, v in fixed.items():
        n, n_valid, k = gcg_detection(v["scores"])
        if m == "gpt-5.4-pro" and n_valid == N_GCG:
            extra.append((m, 100.0 * k / n))
        else:
            excluded.append((m, "scores near 1.0 on every input (no benign-prompt scores)"))
    for m, v in ceiling.items():
        valid = 10 - v["empty_adv"]
        excluded.append((m, f"re-check on 10 prompts: {valid} valid adversarial responses, "
                            f"mean score {v['adv_mean']}, not a detection rate"))
    excluded.append(("gpt-5-nano", "no score for 19/20 benign prompts"))

    print("COST-SCALING ANALYSIS: detection rate (score >= 0.5 on 49 GCG prompts) vs log10(input price)")
    print("=" * 80)
    print(f"\nIncluded ({len(included)} models):")
    for m, d in sorted(included, key=lambda t: PRICE_IN[t[0]]):
        print(f"  {m:<26} det={d:5.1f}%  price=${PRICE_IN[m]:<6.2f}/1M")

    primary = corr(included)
    sensitivity = corr(included + extra)
    print(f"\nPrimary    (n={primary['n']}): Pearson r={primary['pearson_r']:.3f} (p={primary['pearson_p']:.3f}), "
          f"Spearman rho={primary['spearman_rho']:.3f} (p={primary['spearman_p']:.3f})")
    print(f"+gpt-5.4-pro (n={sensitivity['n']}): Pearson r={sensitivity['pearson_r']:.3f} "
          f"(p={sensitivity['pearson_p']:.3f}), Spearman rho={sensitivity['spearman_rho']:.3f} "
          f"(p={sensitivity['spearman_p']:.3f})")
    print("Superseded earlier value: Pearson r = 0.507 over 28 mixed entries")

    print(f"\nExcluded ({len(excluded)}):")
    for m, why in excluded:
        print(f"  {m:<30} {why}")

    output = {
        "definition": "detection = share of 49 GCG prompts scored >= 0.5; x = log10(input price per 1M tokens)",
        "primary": primary,
        "sensitivity_with_gpt_5_4_pro": sensitivity,
        "superseded_pearson_r": 0.506768161046633,
        "included": [{"model": m, "detection_pct": d, "price_in": PRICE_IN[m]} for m, d in included],
        "excluded": [{"model": m, "reason": why} for m, why in excluded],
    }
    (RESULTS_DIR / "cost_scaling_analysis.json").write_text(json.dumps(output, indent=2))
    print("\nSaved to results/cost_scaling_analysis.json")


if __name__ == "__main__":
    main()
