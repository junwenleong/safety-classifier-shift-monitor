"""P1+P4: Unified conformal evaluation across all classifiers and shift types.

Extends the existing temporal-only conformal evaluation to cover:
- 4 classifiers: deberta, text-moderation, llama-guard, shieldgemma
- 3 shift types: temporal, paraphrase, adversarial-suffix

Produces Table 3 as a 3-shift × 4-classifier matrix showing:
- Coverage gap (unweighted vs weighted)
- Weight diagnostic (density-ratio collapse detection)

Usage (Mac Studio):
    export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
    export TEXT_MODERATION_CHECKPOINT_PATH=checkpoints/text-moderation-wildguardmix
    python scripts/run_conformal_eval_full.py

Outputs:
    results/conformal_full.json — all results in structured format
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shift_detection_monitor.adaptation.conformal import ConformalAbstentionLayer
from shift_detection_monitor.adaptation.density_ratio import DensityRatioEstimator
from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import StreamRecord

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFT_TYPES = ["temporal", "paraphrase", "adversarial-suffix"]
N_CALIBRATION = 300
N_PRE_SHIFT = 200
N_POST_SHIFT = 200
TARGET_ERROR_RATE = 0.1
OUTPUT = Path("results/conformal_full.json")


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
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=42)
    return [{"text": ds["prompt"][i], "label": 0} for i in range(n)]


def load_shift(shift_type: str, n: int):
    paths = {
        "temporal": Path("data/shifted/temporal/output.jsonl"),
        "paraphrase": Path("data/shifted/paraphrase/output.jsonl"),
        "adversarial-suffix": Path("data/shifted/adversarial_suffix/deberta_suffixes.jsonl"),
    }
    path = paths[shift_type]
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return None

    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    examples = []
    for r in raw:
        text = r.get("text") or r.get("combined") or r.get("shifted", "")
        if text:
            examples.append({"text": text[:1000], "label": 1})

    if len(examples) < n:
        examples = (examples * ((n // len(examples)) + 1))[:n]
    return examples[:n]


def run_classifier_batch(classifier, examples):
    """Run classifier, return (ClassifierOutput, label) pairs."""
    from shift_detection_monitor.types import ClassifierOutput
    results = []
    for ex in examples:
        output = classifier.predict(ex["text"])
        results.append((output, ex["label"]))
    return results


def evaluate_coverage(layer, test_data):
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


def run_single_eval(classifier, clf_name, shift_type, cal_examples, pre_shift_examples, post_shift_examples):
    """Run conformal evaluation for one classifier × one shift type."""
    print(f"  [{clf_name} × {shift_type}] Running inference...")

    # Run classifier
    cal_data = run_classifier_batch(classifier, cal_examples)
    pre_data = run_classifier_batch(classifier, pre_shift_examples)
    post_data = run_classifier_batch(classifier, post_shift_examples)

    # Unweighted conformal
    layer_uw = ConformalAbstentionLayer(
        target_error_rate=TARGET_ERROR_RATE,
        conformal_mode="unweighted",
        calibration_set=cal_data,
    )
    cov_pre_uw, _ = evaluate_coverage(layer_uw, pre_data)
    cov_post_uw, abs_post_uw = evaluate_coverage(layer_uw, post_data)

    # Weighted conformal
    layer_wt = ConformalAbstentionLayer(
        target_error_rate=TARGET_ERROR_RATE,
        conformal_mode="weighted-on-alarm",
        calibration_set=cal_data,
    )

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

    post_records = [
        StreamRecord(
            time_step=i, text="", score=o.score, representation=o.representation,
            ground_truth_label=label, is_shifted=True,
            source_dataset=shift_type, shift_condition=shift_type,
        )
        for i, (o, label) in enumerate(post_data)
    ]

    layer_wt.on_alarm(post_records, frozen)
    cov_post_wt, abs_post_wt = evaluate_coverage(layer_wt, post_data)

    # Weight diagnostic
    target_emb = np.array([r.representation for r in post_records if r.representation is not None])
    dr = DensityRatioEstimator(method="logistic", max_weight=10.0)
    dr.fit(cal_embeddings, target_emb)
    cal_weights = dr.weights(cal_embeddings)

    weight_diagnostic = {
        "frac_at_upper_clip": float(np.sum(cal_weights >= 9.9) / len(cal_weights)),
        "frac_at_lower_clip": float(np.sum(cal_weights <= 0.11) / len(cal_weights)),
        "frac_clipped_total": float(np.sum((cal_weights >= 9.9) | (cal_weights <= 0.11)) / len(cal_weights)),
        "effective_sample_size": float((np.sum(cal_weights) ** 2) / np.sum(cal_weights ** 2)),
        "max_weight": float(np.max(cal_weights)),
        "min_weight": float(np.min(cal_weights)),
        "median_weight": float(np.median(cal_weights)),
    }

    result = {
        "classifier": clf_name,
        "shift_type": shift_type,
        "unweighted": {
            "pre_shift_coverage": float(cov_pre_uw),
            "post_shift_coverage": float(cov_post_uw),
            "coverage_gap": float(cov_pre_uw - cov_post_uw),
            "post_shift_abstentions": abs_post_uw,
        },
        "weighted": {
            "post_shift_coverage": float(cov_post_wt),
            "coverage_recovery": float(cov_post_wt - cov_post_uw),
            "post_shift_abstentions": abs_post_wt,
        },
        "weight_diagnostic": weight_diagnostic,
        "density_ratio_collapse": weight_diagnostic["frac_clipped_total"] > 0.95,
    }

    gap = result["unweighted"]["coverage_gap"]
    recovery = result["weighted"]["coverage_recovery"]
    collapse = "COLLAPSE" if result["density_ratio_collapse"] else "ok"
    print(f"    gap={gap:.3f} recovery={recovery:+.3f} weights={collapse}")

    return result


def main():
    print("P1+P4: Unified Conformal Evaluation")
    print(f"Classifiers: {CLASSIFIERS}")
    print(f"Shift types: {SHIFT_TYPES}")
    print(f"Target coverage: {1 - TARGET_ERROR_RATE:.0%}")
    print("=" * 70)

    # Load reference data (shared across all evaluations)
    print("\nLoading reference data...")
    ref_examples = load_reference(N_CALIBRATION + N_PRE_SHIFT)
    cal_examples = ref_examples[:N_CALIBRATION]
    pre_shift_examples = ref_examples[N_CALIBRATION:]

    # Load all shift data
    shift_data = {}
    for shift_type in SHIFT_TYPES:
        print(f"Loading {shift_type} shift data...")
        data = load_shift(shift_type, N_POST_SHIFT)
        if data is None:
            continue
        shift_data[shift_type] = data

    all_results = []

    # Group by classifier to avoid reloading models
    for clf_name in CLASSIFIERS:
        print(f"\n{'='*60}")
        print(f"[{clf_name}] Loading classifier...")
        classifier = get_classifier(clf_name)

        for shift_type in SHIFT_TYPES:
            if shift_type not in shift_data:
                continue
            result = run_single_eval(
                classifier, clf_name, shift_type,
                cal_examples, pre_shift_examples, shift_data[shift_type],
            )
            all_results.append(result)

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nResults saved to {OUTPUT}")

    # Print summary table
    print("\n" + "=" * 80)
    print("TABLE 3: Coverage Gap (unweighted) | Recovery (weighted) | Collapse?")
    print("-" * 80)
    header = f"{'Classifier':<18}" + "".join(f"{s:<22}" for s in SHIFT_TYPES)
    print(header)
    print("-" * 80)
    for clf_name in CLASSIFIERS:
        row = f"{clf_name:<18}"
        for shift_type in SHIFT_TYPES:
            match = [r for r in all_results if r["classifier"] == clf_name and r["shift_type"] == shift_type]
            if match:
                r = match[0]
                gap = r["unweighted"]["coverage_gap"]
                rec = r["weighted"]["coverage_recovery"]
                col = "†" if r["density_ratio_collapse"] else " "
                row += f"{gap:.3f} / {rec:+.3f} {col}  "
            else:
                row += f"{'—':<22}"
        print(row)
    print("-" * 80)
    print("† = density-ratio collapse (weights clipped to floor)")
    print("=" * 80)


if __name__ == "__main__":
    main()
