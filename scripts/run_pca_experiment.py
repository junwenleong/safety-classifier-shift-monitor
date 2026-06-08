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


def run_pca_conformal(cal_data, post_data, cal_embeddings, post_embeddings, pca_dim):
    """Run conformal with PCA-reduced embeddings for density ratio.

    Uses ConformalAbstentionLayer (same as P1+P4) to ensure correct
    nonconformity score convention.
    """
    from shift_detection_monitor.types import ClassifierOutput

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

    # Unweighted coverage via ConformalAbstentionLayer
    layer_uw = ConformalAbstentionLayer(
        target_error_rate=TARGET_ERROR_RATE,
        conformal_mode="unweighted",
        calibration_set=cal_data,
    )
    n_covered_uw = sum(1 for o, l in post_data if l in layer_uw.predict_set(o))
    uw_coverage = n_covered_uw / len(post_data)

    # Weighted coverage: build FrozenReferenceStats with PCA-reduced embeddings,
    # then use ConformalAbstentionLayer with on_alarm
    # We need to create synthetic ClassifierOutputs with PCA-reduced representations
    cal_data_pca = [
        (ClassifierOutput(score=o.score, representation=cal_reduced[i]), label)
        for i, (o, label) in enumerate(cal_data)
    ]
    layer_wt = ConformalAbstentionLayer(
        target_error_rate=TARGET_ERROR_RATE,
        conformal_mode="weighted-on-alarm",
        calibration_set=cal_data_pca,
    )
    frozen = FrozenReferenceStats(
        kernel_bandwidth=1.0,
        reference_cdf=np.array([o.score for o, _ in cal_data]),
        reference_embeddings=cal_reduced,
        mmd_null_distribution=np.zeros(100),
        mmd_reference_value=0.0,
        pca_components=None,
        pca_mean=None,
        n_reference=len(cal_data),
    )
    post_records = [
        StreamRecord(
            time_step=i, text="", score=o.score, representation=post_reduced[i],
            ground_truth_label=label, is_shifted=True,
            source_dataset="temporal", shift_condition="temporal",
        )
        for i, (o, label) in enumerate(post_data)
    ]
    layer_wt.on_alarm(post_records, frozen)
    n_covered_wt = sum(1 for o, l in post_data if l in layer_wt.predict_set(o))
    wt_coverage = n_covered_wt / len(post_data)

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
        "unweighted_coverage": float(uw_coverage),
        "weighted_coverage": float(wt_coverage),
        "coverage_recovery": float(wt_coverage - uw_coverage),
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

        # Extract embeddings and build (ClassifierOutput, label) pairs
        print(f"  Running inference on calibration ({N_CALIBRATION})...")
        cal_data = []
        cal_embeddings = []
        for ex in cal_examples:
            output = classifier.predict(ex["text"])
            cal_data.append((output, ex["label"]))
            cal_embeddings.append(output.representation)

        print(f"  Running inference on post-shift ({N_POST_SHIFT})...")
        post_data = []
        post_embeddings = []
        for ex in post_examples:
            output = classifier.predict(ex["text"])
            post_data.append((output, ex["label"]))
            post_embeddings.append(output.representation)

        cal_emb = np.array(cal_embeddings)
        post_emb = np.array(post_embeddings)

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
            result = run_pca_conformal(cal_data, post_data, cal_emb, post_emb, pca_dim)
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
