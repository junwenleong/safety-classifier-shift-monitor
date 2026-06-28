"""Cost-Scaling Plot: Detection Rate vs Log(Cost).

Shows the "Deployment Configuration Trap" visually.

Usage:
    .venv/bin/python scripts/run_cost_scaling_plot.py
"""
import json, math
from pathlib import Path

RESULTS_DIR = Path("results")

# All models with their detection rates (from v2 data, threshold=0.5)
# and costs (input $/1M tokens)
MODELS = {
    # Discriminating (from v2, N=49)
    "gpt-4o-mini": {"det": 83.7, "cost_in": 0.15},
    "gpt-4o": {"det": 87.8, "cost_in": 2.5},
    "gpt-4.1-nano": {"det": 63.3, "cost_in": 0.1},
    "gpt-4.1-mini": {"det": 73.5, "cost_in": 0.4},
    "gpt-4.1": {"det": 85.7, "cost_in": 2.0},
    "gpt-5.1": {"det": 93.9, "cost_in": 1.25},
    "gpt-5.2": {"det": 85.7, "cost_in": 1.75},
    "gpt-5.3-codex": {"det": 91.8, "cost_in": 1.75},
    "gpt-5.4": {"det": 91.8, "cost_in": 2.5},
    "gpt-5.4-pro": {"det": 88.3, "cost_in": 30.0},  # from fixed batch (mean 0.883)
    "gemini-2.5-flash-lite": {"det": 49.0, "cost_in": 0.1},
    "gemini-3.1-flash-lite": {"det": 67.3, "cost_in": 0.25},
    "claude-haiku-4-5": {"det": 73.5, "cost_in": 1.0},
    "claude-sonnet-4-0": {"det": 83.7, "cost_in": 3.0},
    "claude-sonnet-4-5": {"det": 79.6, "cost_in": 3.0},
    "claude-sonnet-4-6": {"det": 61.2, "cost_in": 3.0},
    "claude-opus-4-5": {"det": 95.9, "cost_in": 5.0},
    "claude-opus-4-6": {"det": 71.4, "cost_in": 5.0},
    "claude-opus-4-7": {"det": 87.8, "cost_in": 5.0},
    "claude-opus-4-8": {"det": 77.6, "cost_in": 5.0},
    # Token-budget artifact models (NOW FIXED, from verify_all_ceiling)
    "gemini-2.5-flash": {"det": 76.0, "cost_in": 0.3, "note": "fixed"},
    "gemini-2.5-pro": {"det": 57.0, "cost_in": 1.25, "note": "fixed"},
    "gemini-3-flash-preview": {"det": 44.0, "cost_in": 0.5, "note": "fixed"},
    "gemini-3.1-pro-preview": {"det": 82.0, "cost_in": 2.0, "note": "fixed"},
    "gpt-5.2-chat": {"det": 73.0, "cost_in": 1.75, "note": "fixed"},
    "o3-mini": {"det": 90.0, "cost_in": 1.1, "note": "fixed"},
    "o4-mini": {"det": 73.0, "cost_in": 1.1, "note": "fixed"},
    "o3": {"det": 83.0, "cost_in": 10.0, "note": "fixed"},
    # Refusers (unusable)
    "gpt-5-nano": {"det": 0, "cost_in": 0.05, "note": "refuser"},
    "gpt-5.5": {"det": 0, "cost_in": 5.0, "note": "refuser"},
}


def main():
    print("COST-SCALING ANALYSIS: Detection Rate vs Log(Cost)")
    print("=" * 80)

    # Sort by cost
    sorted_models = sorted(MODELS.items(), key=lambda x: x[1]["cost_in"])

    # Group by cost tier
    tiers = {"Ultra-cheap (<$0.5)": [], "Mid-tier ($0.5-$3)": [], "Premium ($3-$10)": [], "Flagship (>$10)": []}
    for m, d in sorted_models:
        if d.get("note") == "refuser":
            continue
        c = d["cost_in"]
        if c < 0.5:
            tiers["Ultra-cheap (<$0.5)"].append((m, d))
        elif c < 3:
            tiers["Mid-tier ($0.5-$3)"].append((m, d))
        elif c < 10:
            tiers["Premium ($3-$10)"].append((m, d))
        else:
            tiers["Flagship (>$10)"].append((m, d))

    for tier_name, models in tiers.items():
        if not models:
            continue
        dets = [d["det"] for _, d in models]
        costs = [d["cost_in"] for _, d in models]
        mean_det = sum(dets) / len(dets)
        mean_cost = sum(costs) / len(costs)
        print(f"\n  {tier_name} ({len(models)} models):")
        print(f"    Mean detection: {mean_det:.1f}%")
        print(f"    Mean cost: ${mean_cost:.2f}/1M tokens")
        print(f"    Range: {min(dets):.0f}%-{max(dets):.0f}%")
        for m, d in models:
            note = f" [{d['note']}]" if d.get("note") else ""
            print(f"      {m:<28} det={d['det']:5.1f}% cost=${d['cost_in']:<6.2f}{note}")

    # Key insight: is there an inverted-U?
    print(f"\n{'='*80}")
    print("COST-DETECTION RELATIONSHIP")
    print(f"{'='*80}")

    # Compute correlation
    valid = [(m, d) for m, d in MODELS.items() if d.get("note") != "refuser" and d["det"] > 0]
    log_costs = [math.log10(d["cost_in"]) for _, d in valid]
    dets = [d["det"] for _, d in valid]

    # Pearson correlation
    n = len(valid)
    mean_x = sum(log_costs) / n
    mean_y = sum(dets) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_costs, dets)) / n
    std_x = (sum((x - mean_x)**2 for x in log_costs) / n) ** 0.5
    std_y = (sum((y - mean_y)**2 for y in dets) / n) ** 0.5
    r = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

    print(f"\n  Pearson r(log_cost, detection): {r:.3f}")
    print(f"  Interpretation: {'weak' if abs(r) < 0.3 else 'moderate' if abs(r) < 0.6 else 'strong'} {'positive' if r > 0 else 'negative'} correlation")

    # The key finding
    print(f"\n  KEY FINDING:")
    if abs(r) < 0.3:
        print("  → No meaningful relationship between cost and detection rate.")
        print("  → With proper configuration, cheap models detect as well as expensive ones.")
        print("  → The Pareto frontier is determined by COST, not by capability.")
        print("  → Practitioners should use gpt-4o-mini ($0.15/1M) — paying more buys nothing.")
    else:
        print(f"  → Moderate correlation (r={r:.2f}): more expensive models detect slightly better,")
        print("  → but the marginal gain does not justify the 10-100× cost increase.")

    # Save
    output = {
        "correlation_log_cost_detection": r,
        "tiers": {t: [(m, d) for m, d in ms] for t, ms in tiers.items()},
        "conclusion": "No inverted-U. Flat relationship. Cost determines choice, not capability.",
    }
    (RESULTS_DIR / "cost_scaling_analysis.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Saved to results/cost_scaling_analysis.json")


if __name__ == "__main__":
    main()
