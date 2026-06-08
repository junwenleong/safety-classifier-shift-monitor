"""Step 3: Embedding displacement scatter — L2 distance vs detection latency.

Tests whether embedding-space displacement mirrors the score-std vs latency
correlation (r=0.968). If yes: confirmed mechanistic finding. If not: hypothesis.

Requires: results/cached_streams/ from cache_embeddings.py

Usage:
    .venv/bin/python scripts/embedding_displacement.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

CACHE_DIR = Path("results/cached_streams")
FACTORIAL = Path("results/factorial_results.jsonl")
OUTPUT = Path("results/embedding_displacement.json")

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "temporal", "adversarial-suffix"]
SHIFT_ONSET = 500


def main():
    print("=" * 60)
    print("EMBEDDING DISPLACEMENT ANALYSIS")
    print("=" * 60)

    # Load factorial latencies for correlation
    rows = [json.loads(l) for l in open(FACTORIAL) if l.strip()]
    for r in rows:
        r["is_valid_detection"] = (
            r.get("detection_latency") is not None
            and r["detection_latency"] >= 0
            and r.get("neg_clean") is True
        )

    results = []

    for clf in CLASSIFIERS:
        for shift in SHIFTS:
            displacements = []
            for seed in range(10):
                path = CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                if "embeddings" not in data:
                    break
                embeddings = data["embeddings"]
                is_shifted = data["is_shifted"]

                ref_embs = embeddings[~is_shifted]
                shift_embs = embeddings[is_shifted]

                if len(ref_embs) == 0 or len(shift_embs) == 0:
                    continue

                # Mean L2 displacement between centroids
                ref_centroid = ref_embs.mean(axis=0)
                shift_centroid = shift_embs.mean(axis=0)
                l2_dist = float(np.linalg.norm(shift_centroid - ref_centroid))
                displacements.append(l2_dist)

            if not displacements:
                print(f"  {clf} × {shift}: no embeddings available")
                continue

            mean_disp = float(np.mean(displacements))

            # Get mean detection latency from factorial
            lats = [r["detection_latency"] for r in rows
                    if r["classifier"] == clf and r["shift_condition"] == shift
                    and r["is_valid_detection"]]
            mean_lat = float(np.mean(lats)) if lats else None

            results.append({
                "classifier": clf,
                "shift": shift,
                "mean_displacement": mean_disp,
                "mean_latency": mean_lat,
                "n_seeds": len(displacements),
            })
            print(f"  {clf} × {shift}: displacement={mean_disp:.3f}, latency={mean_lat:.1f}")

    # Correlation analysis
    print("\n" + "=" * 60)
    print("CORRELATION: displacement vs latency")
    print("=" * 60)

    disps = [r["mean_displacement"] for r in results if r["mean_latency"] is not None]
    lats = [r["mean_latency"] for r in results if r["mean_latency"] is not None]

    if len(disps) >= 3:
        r_val, p_val = sp_stats.pearsonr(disps, lats)
        print(f"\n  Pearson r = {r_val:.3f} (p = {p_val:.4f})")
        print(f"  Direction: {'larger displacement → slower detection' if r_val > 0 else 'larger displacement → faster detection'}")

        # Per-shift correlations
        print("\n  Per-shift:")
        for shift in SHIFTS:
            s_disps = [r["mean_displacement"] for r in results if r["shift"] == shift and r["mean_latency"] is not None]
            s_lats = [r["mean_latency"] for r in results if r["shift"] == shift and r["mean_latency"] is not None]
            if len(s_disps) >= 3:
                r_s, _ = sp_stats.pearsonr(s_disps, s_lats)
                print(f"    {shift}: r = {r_s:.3f}")

        # Compare to score-std correlation (r=0.968)
        print(f"\n  Score-std vs latency (from mechanistic_analysis.py): r=0.968")
        print(f"  Embedding displacement vs latency: r={r_val:.3f}")
        if abs(r_val) > 0.7:
            print("  → Displacement MIRRORS score pattern. Mechanistic finding confirmed.")
        else:
            print("  → Displacement does NOT clearly mirror score pattern. Report as hypothesis.")
    else:
        print("  Insufficient data for correlation")

    # Save
    output_data = {"results": results, "correlation": {"r": r_val, "p": p_val} if len(disps) >= 3 else None}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
