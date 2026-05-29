"""Conformal abstention layer evaluation under temporal shift.

Evaluates unweighted vs weighted conformal prediction sets on DeBERTa.
Shows coverage degradation under shift and partial recovery with weighting.

Usage:
    export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
    python scripts/run_conformal_eval.py [--classifier deberta]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.adaptation.conformal import ConformalAbstentionLayer
from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import ClassifierOutput, StreamRecord

N_CALIBRATION = 300
N_PRE_SHIFT = 200
N_POST_SHIFT = 200
TARGET_ERROR_RATE = 0.1


def get_classifier(name: str):
    if name == "deberta":
        from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
        return DeBERTaAdapter()
    elif name == "text-moderation":
        from shift_detection_monitor.classifiers.gpt_oss_safeguard import TextModerationAdapter
        return TextModerationAdapter()
    elif name == "shieldgemma":
        from shift_detection_monitor.classifiers.shieldgemma import ShieldGemmaAdapter
        return ShieldGemmaAdapter()
    elif name == "llama-guard":
        from shift_detection_monitor.classifiers.llama_guard import LlamaGuard3Adapter
        return LlamaGuard3Adapter()
    else:
        raise ValueError(f"Unknown classifier: {name}")


def load_reference(n: int):
    """Load unharmful WildGuardMix examples as reference (label=0, safe)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=42)
    return [{"text": ds["prompt"][i], "label": 0} for i in range(n)]


def load_temporal_shift(n: int):
    """Load temporal shift examples (label=1, unsafe — shifted distribution)."""
    path = Path("data/shifted/temporal/output.jsonl")
    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    # Temporal shift corpus: these are shifted inputs, treat as unsafe for conformal eval
    return [{"text": r["text"], "label": 1} for r in raw[:n]]


def run_classifier(classifier, examples: list[dict]) -> list[tuple[ClassifierOutput, int]]:
    """Run classifier on examples, return (output, label) pairs."""
    results = []
    for ex in examples:
        output = classifier.predict(ex["text"])
        results.append((output, ex["label"]))
    return results


def evaluate_coverage(layer: ConformalAbstentionLayer, test_data: list[tuple[ClassifierOutput, int]]):
    """Compute empirical coverage and abstention count on test data."""
    n_covered = 0
    n_abstain = 0
    for output, label in test_data:
        pred_set = layer.predict_set(output)
        if label in pred_set:
            n_covered += 1
        if len(pred_set) != 1:
            n_abstain += 1
    coverage = n_covered / len(test_data) if test_data else 0.0
    return coverage, n_abstain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="deberta")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"results/conformal_{args.classifier.replace('-', '_')}.json"

    print(f"Classifier: {args.classifier}")
    print(f"Target coverage: {1 - TARGET_ERROR_RATE:.0%}")
    print()

    # Load data
    print("Loading reference data (500 unharmful)...")
    ref_examples = load_reference(N_CALIBRATION + N_PRE_SHIFT)
    cal_examples = ref_examples[:N_CALIBRATION]
    pre_shift_examples = ref_examples[N_CALIBRATION:]

    print("Loading temporal shift data (200 examples)...")
    post_shift_examples = load_temporal_shift(N_POST_SHIFT)

    # Run classifier
    print(f"Running {args.classifier} on calibration set ({N_CALIBRATION})...")
    classifier = get_classifier(args.classifier)
    cal_data = run_classifier(classifier, cal_examples)

    print(f"Running {args.classifier} on pre-shift test ({N_PRE_SHIFT})...")
    pre_data = run_classifier(classifier, pre_shift_examples)

    print(f"Running {args.classifier} on post-shift test ({N_POST_SHIFT})...")
    post_data = run_classifier(classifier, post_shift_examples)

    # --- Unweighted conformal ---
    print("\nFitting unweighted conformal layer...")
    layer_uw = ConformalAbstentionLayer(
        target_error_rate=TARGET_ERROR_RATE,
        conformal_mode="unweighted",
        calibration_set=cal_data,
    )
    cov_pre_uw, abs_pre_uw = evaluate_coverage(layer_uw, pre_data)
    cov_post_uw, abs_post_uw = evaluate_coverage(layer_uw, post_data)

    # --- Weighted conformal ---
    print("Fitting weighted conformal layer...")
    layer_wt = ConformalAbstentionLayer(
        target_error_rate=TARGET_ERROR_RATE,
        conformal_mode="weighted-on-alarm",
        calibration_set=cal_data,
    )

    # Build FrozenReferenceStats with calibration embeddings
    cal_embeddings = np.array([o.representation for o, _ in cal_data if o.representation is not None])
    frozen = FrozenReferenceStats(
        kernel_bandwidth=1.0,
        reference_cdf=np.array([o.score for o, _ in cal_data]),
        reference_embeddings=cal_embeddings,
        mmd_null_distribution=np.zeros(100),
        mmd_reference_value=0.0,
        pca_components=None,
        pca_mean=None,
        n_reference=len(cal_data),
    )

    # Build post-alarm StreamRecords for on_alarm
    post_records = [
        StreamRecord(
            time_step=i,
            text="",
            score=o.score,
            representation=o.representation,
            ground_truth_label=label,
            is_shifted=True,
            source_dataset="temporal",
            shift_condition="temporal",
        )
        for i, (o, label) in enumerate(post_data)
    ]

    layer_wt.on_alarm(post_records, frozen)
    cov_post_wt, abs_post_wt = evaluate_coverage(layer_wt, post_data)
    # Pre-shift coverage for weighted (before alarm, same as unweighted)
    cov_pre_wt, abs_pre_wt = evaluate_coverage(layer_wt, pre_data)

    # --- Weight clipping diagnostic ---
    from shift_detection_monitor.adaptation.density_ratio import DensityRatioEstimator
    target_emb = np.array([r.representation for r in post_records if r.representation is not None])
    dr_estimator = DensityRatioEstimator(method="logistic", max_weight=10.0)
    dr_estimator.fit(cal_embeddings, target_emb)
    cal_weights = dr_estimator.weights(cal_embeddings)
    n_at_upper = np.sum(cal_weights >= 10.0 - 1e-6)
    n_at_lower = np.sum(cal_weights <= 0.1 + 1e-6)
    frac_clipped = (n_at_upper + n_at_lower) / len(cal_weights)
    effective_n = (np.sum(cal_weights) ** 2) / np.sum(cal_weights ** 2)
    weight_diagnostic = {
        "frac_at_upper_clip": float(n_at_upper / len(cal_weights)),
        "frac_at_lower_clip": float(n_at_lower / len(cal_weights)),
        "frac_clipped_total": float(frac_clipped),
        "effective_sample_size": float(effective_n),
        "max_weight": float(np.max(cal_weights)),
        "min_weight": float(np.min(cal_weights)),
        "median_weight": float(np.median(cal_weights)),
    }

    # --- Results table ---
    print("\n" + "=" * 75)
    print(f"CONFORMAL EVALUATION — {args.classifier} — temporal shift")
    print(f"Target coverage: {1 - TARGET_ERROR_RATE:.0%} | Cal: {N_CALIBRATION} | Pre: {N_PRE_SHIFT} | Post: {N_POST_SHIFT}")
    print("-" * 75)
    print(f"{'Mode':<22} {'Pre-shift':<12} {'Post-shift':<12} {'Gap':<10} {'Abstain (post)'}")
    print(f"{'Unweighted':<22} {cov_pre_uw:<12.3f} {cov_post_uw:<12.3f} {cov_pre_uw - cov_post_uw:<10.3f} {abs_post_uw}")
    print(f"{'Weighted-on-alarm':<22} {cov_pre_wt:<12.3f} {cov_post_wt:<12.3f} {cov_pre_wt - cov_post_wt:<10.3f} {abs_post_wt}")
    print("=" * 75)

    if cov_pre_uw - cov_post_uw > 0.05:
        print("\n→ Unweighted conformal loses coverage under shift (gap > 5%)")
    else:
        print("\n→ Coverage gap is small — shift may not strongly affect this classifier")

    if cov_post_wt > cov_post_uw:
        print(f"→ Weighted conformal recovers {cov_post_wt - cov_post_uw:.3f} coverage")
    else:
        print("→ Weighted conformal did not improve post-shift coverage")

    print(f"\nWeight clipping diagnostic:")
    print(f"  Frac at upper clip (≥10): {weight_diagnostic['frac_at_upper_clip']:.3f}")
    print(f"  Frac at lower clip (≤0.1): {weight_diagnostic['frac_at_lower_clip']:.3f}")
    print(f"  Total clipped: {weight_diagnostic['frac_clipped_total']:.3f}")
    print(f"  Effective sample size: {weight_diagnostic['effective_sample_size']:.1f} / {N_CALIBRATION}")
    print(f"  Weight range: [{weight_diagnostic['min_weight']:.3f}, {weight_diagnostic['max_weight']:.3f}]")
    print(f"  Median weight: {weight_diagnostic['median_weight']:.3f}")

    # Save results JSON
    results = {
        "classifier": args.classifier,
        "target_error_rate": TARGET_ERROR_RATE,
        "n_calibration": N_CALIBRATION,
        "n_pre_shift": N_PRE_SHIFT,
        "n_post_shift": N_POST_SHIFT,
        "unweighted": {
            "pre_shift_coverage": float(cov_pre_uw),
            "post_shift_coverage": float(cov_post_uw),
            "coverage_gap": float(cov_pre_uw - cov_post_uw),
            "post_shift_abstentions": abs_post_uw,
        },
        "weighted_on_alarm": {
            "pre_shift_coverage": float(cov_pre_wt),
            "post_shift_coverage": float(cov_post_wt),
            "coverage_gap": float(cov_pre_wt - cov_post_wt),
            "post_shift_abstentions": abs_post_wt,
        },
        "weight_diagnostic": weight_diagnostic,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
