"""Variance decomposition of factorial results.

Feeds factorial_results.jsonl into VarianceDecomposer and prints
factor importance (classifier, shift_type, interaction, residual).

Usage:
    python scripts/run_variance_decomposition.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.evaluation.results import CellResult
from shift_detection_monitor.evaluation.variance_decomposer import VarianceDecomposer

RESULTS = Path("results/factorial_results.jsonl")
OUTPUT = Path("results/variance_decomposition.json")


def load_cell_results() -> list[CellResult]:
    rows = []
    with open(RESULTS) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            # Backfill is_valid_detection
            is_valid = (
                r.get("detection_latency") is not None
                and r["detection_latency"] >= 0
                and r.get("neg_clean") is True
            )
            if not is_valid:
                continue  # filter to valid detections only

            rows.append(CellResult(
                classifier=r["classifier"],
                shift_condition=r["shift_condition"],
                ground_truth_regime="B",  # temporal split regime for all factorial cells
                window_size=r["window_size"],
                seed=r["seed"],
                detection_latency=r["detection_latency"],
                false_alarm_rate=0.0 if r.get("neg_clean", True) else 1.0,
                n_abstentions=0,
                n_predictions=1,
                is_negative_control=False,
                is_positive_control=False,
            ))
    return rows


def main():
    results = load_cell_results()
    print(f"Valid cells loaded: {len(results)}")

    decomposer = VarianceDecomposer(min_observations_per_cell=3)
    decomp = decomposer.fit(results)

    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION (η² proportions)")
    print("-" * 60)
    print(f"  Classifier:          {decomp.factor_variances['classifier']:.4f}")
    print(f"  Shift type:          {decomp.factor_variances['shift_type']:.4f}")
    print(f"  Classifier×Shift:    {decomp.interaction_variances['classifier:shift_type']:.4f}")
    print(f"  Residual:            {decomp.residual_variance:.4f}")
    total = (decomp.factor_variances['classifier'] + decomp.factor_variances['shift_type']
             + decomp.interaction_variances['classifier:shift_type'] + decomp.residual_variance)
    print(f"  Sum:                 {total:.4f}")

    print("\nEFFECT SIZES (η² with 95% bootstrap CI)")
    print("-" * 60)
    for factor, es in decomp.effect_sizes.items():
        print(f"  {factor:<20} η²={es.estimate:.4f}  [{es.ci_lower:.4f}, {es.ci_upper:.4f}]")

    # Top 3 interactions by magnitude
    print("\nTOP 3 INTERACTIONS (classifier × shift_type, by |effect|)")
    print("-" * 60)
    interaction_effects = {}
    from collections import defaultdict
    import numpy as np

    clf_groups: dict[str, list[float]] = defaultdict(list)
    shift_groups: dict[str, list[float]] = defaultdict(list)
    cell_groups: dict[str, list[float]] = defaultdict(list)

    for r in results:
        lat = r.detection_latency
        assert lat is not None
        clf_groups[r.classifier].append(lat)
        shift_groups[r.shift_condition].append(lat)
        cell_groups[f"{r.classifier}:{r.shift_condition}"].append(lat)

    grand_mean = np.mean([r.detection_latency for r in results])
    for key, vals in cell_groups.items():
        clf, shift = key.split(":", 1)
        effect = np.mean(vals) - np.mean(clf_groups[clf]) - np.mean(shift_groups[shift]) + grand_mean
        interaction_effects[key] = float(effect)

    top3 = sorted(interaction_effects.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for key, effect in top3:
        print(f"  {key:<45} {effect:+.2f}")

    if decomp.flagged_cells:
        print(f"\nFLAGGED CELLS (< min observations): {len(decomp.flagged_cells)}")
        for cell in decomp.flagged_cells:
            print(f"  {cell}")
    else:
        print("\nNo cells flagged for insufficient observations.")

    print("=" * 60)

    # Save to JSON
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "factor_variances": decomp.factor_variances,
        "interaction_variances": decomp.interaction_variances,
        "effect_sizes": {
            k: {"estimate": v.estimate, "ci_lower": v.ci_lower, "ci_upper": v.ci_upper}
            for k, v in decomp.effect_sizes.items()
        },
        "flagged_cells": decomp.flagged_cells,
        "residual_variance": decomp.residual_variance,
        "top3_interactions": [{"cell": k, "effect": v} for k, v in top3],
        "n_valid_cells": len(results),
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
