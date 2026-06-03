"""P5: PCA experiment on generative classifier embeddings.

Tests whether dimensionality reduction before density ratio estimation
fixes the collapse problem for generative classifiers (Llama Guard, ShieldGemma).

Approach:
1. Extract embeddings from calibration + shifted data
2. Apply PCA to 32-d and 64-d
3. Re-run density ratio estimation on reduced embeddings
4. Re-run conformal evaluation with corrected weights
5. Compare coverage recovery vs full-dimensional baseline

Usage (Mac Studio):
    export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
    python scripts/run_pca_experiment.py

Outputs:
    results/pca_experiment.json — results for each (classifier, PCA dim) pair
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.adaptation.conformal import ConformalAbstentionLayer
from shift_detection_monitor.adaptation.density_ratio import DensityRatioEstimator
from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import StreamRecord

# Focus on generative classifiers where collapse occurs, plus deberta as control
CLASSIFIERS = ["deberta", "llama-guard", "shieldgemma"]
PCA_DIMS = [32, 64, 128]
N_CALIBRATION = 300
N_POST_SHIFT = 200
TARGET_ERROR_RATE = 0.1
OUTPUT = Path("results/pca_experiment.json")


def get_classifier(name: str):
    if name == "deberta":
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        return DeBERTaAdapter()
    elif name == "shieldgemma":
        from shift_detection_monitor.classifiers.shieldgemma import ShieldGemmaAdapter
        return ShieldGemmaAdapter()
    elif name == "llama-guard":
        from shift_detection_monitor.classifiers.llama_guard import LlamaGuard3Adapter
        return LlamaGuard3Adapter()
    else:
        raise ValueError(f"Unknown classifier: {name}")


def load_reference(n: int):
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=42)
    return [{"text": ds["prompt"][i], "label": 0} for i in range(n)]


def load_temporal_shift(n: int):
    path = Path("data/shifted/temporal/output.jsonl")
    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    return [{"text": r["text"], "label": 1} for r in raw[:n]]


def run_pca_conformal(cal_embeddings, cal_scores, cal_labels, post_embeddings, post_scores, post_labels, pca_dim):
    """Run conformal with PCA-reduced embeddings for density ratio."""
    # Fit PCA on calibration embeddings
    actual_dim = min(pca_dim, cal_embeddings.shape[1], cal_embeddings.shape[0])
    pca = PCA(n_components=actual_dim)
    cal_reduced = pca.fit_transform(cal_embeddings)
    post_reduced = pca.transform(post_embeddings)
    explained_var = float(np.sum(pca.explained_variance_ratio_))

    # Density ratio on reduced embeddings
    dr = DensityRatioEstimator(method="logistic", max_weight=10.0)
    dr.fit(cal_reduced, post_reduced)
    cal_weights = dr.weights(cal_reduced)

    # Weight diagnostic
    frac_at_floor = float(np.sum(cal_weights <= 0.11) / len(cal_weights))
    frac_at_ceil = float(np.sum(cal_weights >= 9.9) / len(cal_weights))
    ess = float((np.sum(cal_weights) ** 2) / np.sum(cal_weights ** 2))
    collapse = (frac_at_floor + frac_at_ceil) > 0.95

    # Weighted conformal coverage
    # Normalize weights
    w_norm = cal_weights / np.sum(cal_weights)

    # Compute weighted quantile for threshold
    nonconformity_scores = 1.0 - cal_scores  # simple nonconformity
    sorted_idx = np.argsort(nonconformity_scores)
    sorted_nc = nonconformity_scores[sorted_idx]
    sorted_w = w_norm[sorted_idx]
    cum_w = np.cumsum(sorted_w)
    threshold_idx = np.searchsorted(cum_w, 1 - TARGET_ERROR_RATE)
    threshold_idx = min(threshold_idx, len(sorted_nc) - 1)
    weighted_threshold = sorted_nc[threshold_idx]

    # Unweighted threshold for comparison
    uw_threshold = np.quantile(nonconformity_scores, 1 - TARGET_ERROR_RATE)

    # Coverage on post-shift
    post_nc = 1.0 - post_scores
    uw_coverage = float(np.mean(post_nc <= uw_threshold))
    wt_coverage = float(np.mean(post_nc <= weighted_threshold))

    return {
        "pca_dim": actual_dim,
        "explained_variance": explained_var,
        "frac_at_floor": frac_at_floor,
        "frac_at_ceil": frac_at_ceil,
        "collapse": collapse,
        "effective_sample_size": ess,
        "max_weight": float(np.max(cal_weights)),
        "min_weight": float(np.min(cal_weights)),
        "median_weight": float(np.median(cal_weights)),
        "unweighted_coverage": uw_coverage,
        "weighted_coverage": wt_coverage,
        "coverage_recovery": wt_coverage - uw_coverage,
        "weighted_threshold": float(weighted_threshold),
        "unweighted_threshold": float(uw_threshold),
    }


def main():
    print("P5: PCA Experiment — Fixing Density-Ratio Collapse")
    print("=" * 70)

    # Load data
    print("Loading reference data...")
    cal_examples = load_reference(N_CALIBRATION)
    print("Loading temporal shift data...")
    post_examples = load_temporal_shift(N_POST_SHIFT)

    all_results = []

    for clf_name in CLASSIFIERS:
        print(f"\n{'='*60}")
        print(f"[{clf_name}] Loading classifier...")
        classifier = get_classifier(clf_name)

        # Extract embeddings
        print(f"  Running inference on calibration ({N_CALIBRATION})...")
        cal_scores = []
        cal_embeddings = []
        cal_labels = []
        for ex in cal_examples:
            output = classifier.predict(ex["text"])
            cal_scores.append(output.score)
            cal_embeddings.append(output.representation)
            cal_labels.append(ex["label"])

        print(f"  Running inference on post-shift ({N_POST_SHIFT})...")
        post_scores = []
        post_embeddings = []
        post_labels = []
        for ex in post_examples:
            output = classifier.predict(ex["text"])
            post_scores.append(output.score)
            post_embeddings.append(output.representation)
            post_labels.append(ex["label"])

        cal_emb = np.array(cal_embeddings)
        post_emb = np.array(post_embeddings)
        cal_sc = np.array(cal_scores)
        post_sc = np.array(post_scores)
        cal_lb = np.array(cal_labels)
        post_lb = np.array(post_labels)

        print(f"  Embedding dim: {cal_emb.shape[1]}")

        # Baseline: full-dimensional density ratio
        print(f"  [full-dim] Running baseline...")
        dr_full = DensityRatioEstimator(method="logistic", max_weight=10.0)
        dr_full.fit(cal_emb, post_emb)
        full_weights = dr_full.weights(cal_emb)
        full_collapse = float(np.sum((full_weights <= 0.11) | (full_weights >= 9.9)) / len(full_weights)) > 0.95

        baseline = {
            "classifier": clf_name,
            "pca_dim": "full",
            "embedding_dim": int(cal_emb.shape[1]),
            "collapse": full_collapse,
            "frac_at_floor": float(np.sum(full_weights <= 0.11) / len(full_weights)),
            "effective_sample_size": float((np.sum(full_weights) ** 2) / np.sum(full_weights ** 2)),
            "median_weight": float(np.median(full_weights)),
        }
        all_results.append(baseline)
        collapse_str = "COLLAPSE" if full_collapse else "ok"
        print(f"    {collapse_str} | ESS={baseline['effective_sample_size']:.1f}/{N_CALIBRATION}")

        # PCA experiments
        for pca_dim in PCA_DIMS:
            if pca_dim >= cal_emb.shape[1]:
                continue
            print(f"  [PCA-{pca_dim}] Running...")
            result = run_pca_conformal(cal_emb, cal_sc, cal_lb, post_emb, post_sc, post_lb, pca_dim)
            result["classifier"] = clf_name
            result["embedding_dim"] = int(cal_emb.shape[1])
            all_results.append(result)

            collapse_str = "COLLAPSE" if result["collapse"] else "FIXED"
            print(f"    {collapse_str} | ESS={result['effective_sample_size']:.1f}/{N_CALIBRATION} | "
                  f"recovery={result['coverage_recovery']:+.3f} | "
                  f"explained_var={result['explained_variance']:.3f}")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nResults saved to {OUTPUT}")

    # Summary
    print("\n" + "=" * 80)
    print("PCA EXPERIMENT SUMMARY")
    print("-" * 80)
    print(f"{'Classifier':<15} {'Dim':<8} {'Collapse':<10} {'ESS':<10} {'Recovery':<10} {'Expl.Var'}")
    print("-" * 80)
    for r in all_results:
        dim = str(r.get("pca_dim", "?"))
        collapse = "YES" if r.get("collapse") else "no"
        ess = f"{r.get('effective_sample_size', 0):.1f}"
        recovery = f"{r.get('coverage_recovery', 0):+.3f}" if "coverage_recovery" in r else "—"
        ev = f"{r.get('explained_variance', 0):.3f}" if "explained_variance" in r else "—"
        print(f"{r['classifier']:<15} {dim:<8} {collapse:<10} {ess:<10} {recovery:<10} {ev}")
    print("=" * 80)


if __name__ == "__main__":
    main()
