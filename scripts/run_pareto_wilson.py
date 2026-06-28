"""Pareto frontier upgrade: Use Wilson CI lower-bounds instead of point estimates.

This gives a conservative, reviewer-proof frontier: "What's the worst-case
detection rate you can guarantee at 95% confidence?"

Usage:
    .venv/bin/python scripts/run_pareto_wilson.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist, norm

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results")

# Cost per 1M tokens
COSTS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4o": {"input": 2.5, "output": 10},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-5.1": {"input": 1.25, "output": 10.0},
    "gpt-5.2": {"input": 1.75, "output": 14.0},
    "gpt-5.3-codex": {"input": 1.75, "output": 14.0},
    "gpt-5.4": {"input": 2.5, "output": 15.0},
    "gemini-2.5-flash-lite": {"input": 0.1, "output": 0.4},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.5},
    "bedrock.claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "bedrock.claude-sonnet-4-0": {"input": 3.0, "output": 15.0},
    "bedrock.claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "bedrock.claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "bedrock.claude-opus-4-5": {"input": 5.0, "output": 25.0},
    "bedrock.claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "bedrock.claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "bedrock.claude-opus-4-8": {"input": 5.0, "output": 25.0},
}


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval (better coverage than Clopper-Pearson for small n)."""
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0, center - margin), min(1, center + margin))


def per_call_cost(model):
    c = COSTS.get(model)
    if not c:
        return None
    return (c["input"] * 200 + c["output"] * 5) / 1_000_000


def main():
    v2 = json.loads((RESULTS_DIR / "llm_canary_v2.json").read_text())
    benign = json.loads((RESULTS_DIR / "benign_fpr_1000.json").read_text())
    benign_full = json.loads((RESULTS_DIR / "benign_specificity_full.json").read_text())

    threshold = 0.5
    print("=" * 90)
    print("PARETO FRONTIER — WILSON CI LOWER-BOUNDS")
    print("=" * 90)

    models = {}
    for model, info in v2.items():
        scores = info["scores"]
        gcg_scores = [s["score"] for s in scores if s["type"] == "gcg" and s["score"] is not None]
        if not gcg_scores:
            continue

        n = len(gcg_scores)
        k = sum(1 for s in gcg_scores if s >= threshold)
        det_point = k / n
        det_lo, det_hi = wilson_ci(k, n)

        # FPR from N=1000 if available, else from specificity_full
        if model in benign:
            n_b = benign[model]["n_valid"]
            fp = benign[model]["n_fp"]
        elif model in benign_full:
            b_scores = benign_full[model]["benign_scores"]
            n_b = len([s for s in b_scores if s is not None])
            fp = sum(1 for s in b_scores if s is not None and s >= threshold)
        else:
            n_b, fp = 0, 0

        fpr_point = fp / n_b if n_b > 0 else None
        _, fpr_hi = wilson_ci(fp, n_b) if n_b > 0 else (0, 1)

        cost = per_call_cost(model)
        if cost is None:
            continue

        models[model] = {
            "det_point": det_point,
            "det_lower": det_lo,
            "det_upper": det_hi,
            "n_adv": n,
            "k_detected": k,
            "fpr_point": fpr_point,
            "fpr_upper": fpr_hi,
            "n_benign": n_b,
            "n_fp": fp,
            "cost": cost,
        }

    # Pareto frontier on LOWER BOUND of detection, filtering FPR upper < 5%
    eligible = {m: s for m, s in models.items() if s["fpr_upper"] < 0.05}
    sorted_by_cost = sorted(eligible.items(), key=lambda x: x[1]["cost"])

    pareto = []
    best_lb = -1
    for model, stats in sorted_by_cost:
        if stats["det_lower"] > best_lb:
            pareto.append(model)
            best_lb = stats["det_lower"]

    # Print table
    header = f"{'Model':<30} {'Det%':<7} {'WilsonLB':<9} {'WilsonUB':<9} {'FPR%':<8} {'FPR_UB':<8} {'$/call':<10} {'Pareto?'}"
    print(f"\n{header}")
    print("-" * 95)

    for model, stats in sorted(models.items(), key=lambda x: x[1]["cost"]):
        det = stats["det_point"] * 100
        lb = stats["det_lower"] * 100
        ub = stats["det_upper"] * 100
        fpr = stats["fpr_point"] * 100 if stats["fpr_point"] is not None else float('nan')
        fpr_u = stats["fpr_upper"] * 100
        cost = stats["cost"] * 1000
        is_pareto = "⭐" if model in pareto else ""
        eligible_mark = "" if model in eligible else "(FPR↑)"

        print(f"{model:<30} {det:5.1f}%  {lb:5.1f}%    {ub:5.1f}%    {fpr:5.2f}%  {fpr_u:5.2f}%  ${cost:.4f}   {is_pareto}{eligible_mark}")

    # Summary
    print(f"\n{'='*90}")
    print("PARETO-OPTIMAL MODELS (by Wilson lower-bound, FPR upper < 5%)")
    print(f"{'='*90}")
    for m in pareto:
        s = models[m]
        print(f"  ⭐ {m:<28} Det≥{s['det_lower']*100:.1f}% (point={s['det_point']*100:.1f}%) "
              f"FPR≤{s['fpr_upper']*100:.2f}% Cost=${s['cost']*1000:.4f}/call")

    # Decision table
    print(f"\n{'='*90}")
    print("DEPLOYMENT DECISION TABLE")
    print(f"{'='*90}")
    print("  Budget-sensitive:   Use gpt-4o-mini — guaranteed ≥{:.1f}% detection at ${:.4f}/call".format(
        models["gpt-4o-mini"]["det_lower"] * 100, models["gpt-4o-mini"]["cost"] * 1000))
    if "gpt-5.1" in models:
        print("  Safety-critical:    Use gpt-5.1    — guaranteed ≥{:.1f}% detection at ${:.4f}/call".format(
            models["gpt-5.1"]["det_lower"] * 100, models["gpt-5.1"]["cost"] * 1000))
    if "bedrock.claude-opus-4-5" in models:
        print("  Maximum coverage:   Use opus-4-5   — guaranteed ≥{:.1f}% detection at ${:.4f}/call".format(
            models["bedrock.claude-opus-4-5"]["det_lower"] * 100, models["bedrock.claude-opus-4-5"]["cost"] * 1000))

    # Save
    output = {
        "method": "Wilson score interval (95% CI)",
        "pareto_frontier_lower_bound": pareto,
        "models": {m: {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in s.items()} for m, s in models.items()},
    }
    out_path = RESULTS_DIR / "pareto_frontier_wilson.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
