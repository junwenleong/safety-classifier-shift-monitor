"""Step 5: PCA-reduced conformal sweep on Llama Guard and ShieldGemma.

Tests whether PCA fix for density-ratio collapse generalizes across shift types.
Sweep: 4, 8, 16, 32 dimensions × temporal + paraphrase.

Requires: results/cached_streams/ from cache_embeddings.py

Usage:
    .venv/bin/python scripts/run_pca_conformal_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

CACHE_DIR = Path("results/cached_streams")
OUTPUT = Path("results/pca_conformal_sweep.json")

CLASSIFIERS = ["llama-guard", "shieldgemma"]
SHIFTS = ["temporal", "paraphrase"]
PCA_DIMS = [4, 8, 16, 32]
SHIFT_ONSET = 500
N_CAL = 300  # calibration set size
TARGET_COVERAGE = 0.90
WEIGHT_CLIP = (0.1, 10.0)


def compute_conformal(cal_scores, cal_weights, test_scores, target_coverage):
    """Compute weighted conformal coverage."""
    n = len(cal_scores)
    # Nonconformity scores (1 - score for unsafe class)
    nc_scores = 1.0 - cal_scores

    # Weighted quantile
    if cal_weights is not None:
        w = cal_weights / cal_weights.sum()
        sorted_idx = np.argsort(nc_scores)
        cumw = np.cumsum(w[sorted_idx])
        quantile_idx = np.searchsorted(cumw, target_coverage)
        quantile_idx = min(quantile_idx, len(sorted_idx) - 1)
        threshold = nc_scores[sorted_idx[quantile_idx]]
    else:
        threshold = np.quantile(nc_scores, target_coverage)

    # Test coverage
    test_nc = 1.0 - test_scores
    covered = (test_nc <= threshold).mean()
    return float(covered)


def estimate_density_ratios(cal_embeddings, test_embeddings, pca_dim):
    """Estimate density ratios via logistic regression on PCA-reduced embeddings."""
    combined = np.vstack([cal_embeddings, test_embeddings])

    # PCA reduction
    pca = PCA(n_components=pca_dim)
    combined_pca = pca.fit_transform(combined)
    variance_retained = pca.explained_variance_ratio_.sum()

    cal_pca = combined_pca[:len(cal_embeddings)]
    test_pca = combined_pca[len(cal_embeddings):]

    # Logistic regression: source=0, target=1
    X = np.vstack([cal_pca, test_pca])
    y = np.array([0] * len(cal_pca) + [1] * len(test_pca))

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, y)

    # Density ratios for calibration points
    probs = lr.predict_proba(cal_pca)[:, 1]
    ratios = probs / (1 - probs + 1e-10)

    # Clip
    ratios = np.clip(ratios, WEIGHT_CLIP[0], WEIGHT_CLIP[1])

    ess = float((ratios.sum()) ** 2 / (ratios ** 2).sum())

    return ratios, ess, variance_retained


def main():
    print("=" * 60)
    print("PCA CONFORMAL SWEEP: 4/8/16/32 dims × temporal + paraphrase")
    print("=" * 60)

    results = []

    for clf in CLASSIFIERS:
        print(f"\n  Classifier: {clf}")
        for shift in SHIFTS:
            # Aggregate embeddings across seeds for larger calibration set
            all_ref_embs = []
            all_ref_scores = []
            all_shift_embs = []
            all_shift_scores = []

            for seed in range(10):
                path = CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                if "embeddings" not in data:
                    print(f"    {shift}: no embeddings, skipping")
                    break
                embs = data["embeddings"]
                scores = data["scores"]
                is_shifted = data["is_shifted"]

                all_ref_embs.append(embs[~is_shifted])
                all_ref_scores.append(scores[~is_shifted])
                all_shift_embs.append(embs[is_shifted])
                all_shift_scores.append(scores[is_shifted])

            if not all_ref_embs:
                continue

            ref_embs = np.concatenate(all_ref_embs)
            ref_scores = np.concatenate(all_ref_scores)
            shift_embs = np.concatenate(all_shift_embs)
            shift_scores = np.concatenate(all_shift_scores)

            # Use first N_CAL reference as calibration, rest as... reference
            cal_embs = ref_embs[:N_CAL]
            cal_scores = ref_scores[:N_CAL]
            test_scores = shift_scores[:200]
            test_embs = shift_embs[:200]

            # Unweighted baseline
            unweighted_cov = compute_conformal(cal_scores, None, test_scores, TARGET_COVERAGE)

            print(f"    {shift} (unweighted coverage={unweighted_cov:.3f}):")

            for dim in PCA_DIMS:
                if dim > cal_embs.shape[1]:
                    continue

                ratios, ess, var_retained = estimate_density_ratios(cal_embs, test_embs, dim)
                weighted_cov = compute_conformal(cal_scores, ratios, test_scores, TARGET_COVERAGE)
                recovery = weighted_cov - unweighted_cov

                result = {
                    "classifier": clf,
                    "shift_type": shift,
                    "pca_dim": dim,
                    "unweighted_coverage": float(unweighted_cov),
                    "weighted_coverage": float(weighted_cov),
                    "recovery": float(recovery),
                    "ess": float(ess),
                    "variance_retained": float(var_retained),
                }
                results.append(result)
                print(f"      dim={dim:>2}: coverage {unweighted_cov:.3f}→{weighted_cov:.3f} (+{recovery:.3f}), ESS={ess:.1f}, var_retained={var_retained:.3f}")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
