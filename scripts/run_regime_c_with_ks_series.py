"""P2: Regime C rerun with stored KS statistic time series.

Reruns Regime C (adversarial success detection) and stores the full KS
statistic time series per classifier, enabling:
1. Reproducibility without model inference
2. A figure showing DeBERTa (flat) vs Llama Guard (spike) trajectories

Usage (Mac Studio):
    export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
    export TEXT_MODERATION_CHECKPOINT_PATH=checkpoints/text-moderation-wildguardmix
    python scripts/run_regime_c_with_ks_series.py

Outputs:
    results/regime_c_ks_series.json  — full KS time series per classifier
    paper/figures/fig_regime_c_ks_trajectories.pdf — trajectory comparison figure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_full_canary import load_shifted, run_stream_ks
from shift_detection_monitor.config import StreamConfig
from shift_detection_monitor.detection.ks_detector import KSDetector
from shift_detection_monitor.detection.reference_window import ReferenceWindow
from shift_detection_monitor.stream.simulator import StreamSimulator

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
N_REFERENCE = 500
N_CALIBRATION = 50
CAL_PCT = 97
WINDOW_SIZE = 200
SEED = 0
OUTPUT_SERIES = Path("results/regime_c_ks_series.json")
OUTPUT_FIGURE = Path("paper/figures/fig_regime_c_ks_trajectories.pdf")


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


def load_reference():
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=42)
    prompts = ds["prompt"]
    ref = [{"text": prompts[i], "source_dataset": "wildguardmix-unharmful"} for i in range(N_REFERENCE)]
    neg = [{"text": prompts[N_REFERENCE + i], "source_dataset": "wildguardmix-unharmful"} for i in range(N_REFERENCE * 2)]
    return ref, neg


def load_adversarial_success():
    path = Path("data/shifted/adversarial_suffix/deberta_suffixes.jsonl")
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("success"):
                text = r.get("combined") or r.get("text", "")
                if text:
                    examples.append({"text": text[:1000], "source_dataset": "gcg-success"})
    print(f"  Loaded {len(examples)} successful adversarial examples")
    return examples


def run_with_ks_series(classifier, reference, shifted, shift_onset, window_size, seed, threshold):
    """Run detection and return full KS time series."""
    config = StreamConfig(
        shift_condition="adversarial-suffix" if shifted else None,
        shift_onset_step=shift_onset, mixing_proportion=1.0, seed=seed,
    )
    sim = StreamSimulator(
        config=config, classifier=classifier, seed=seed,
        reference_examples=reference, shifted_examples=shifted or [],
    )

    ref_window = ReferenceWindow(min_size=window_size, n_bootstrap=100)
    ref_records = []
    stream_iter = iter(sim)
    step = 0
    for record in stream_iter:
        ref_window.add(record)
        ref_records.append(record)
        step += 1
        if step >= window_size:
            break

    frozen = ref_window.freeze()
    ks_det = KSDetector(frozen_stats=frozen, window_size=window_size)
    for rec in ref_records:
        ks_det.update(rec)

    ks_series = []
    alarm_step = None
    for record in stream_iter:
        step += 1
        val = ks_det.update(record)
        ks_series.append({"step": step, "ks": float(val), "is_shifted": record.is_shifted})
        if val > threshold and step > 2 * window_size and alarm_step is None:
            alarm_step = step

    return {
        "ks_series": ks_series,
        "alarm_step": alarm_step,
        "detection_latency": (alarm_step - shift_onset) if alarm_step else None,
        "threshold": threshold,
        "shift_onset": shift_onset,
    }


def plot_trajectories(all_results: dict) -> None:
    """Plot KS trajectories for all classifiers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    colors = {"deberta": "#7f8c8d", "text-moderation": "#2980b9",
              "llama-guard": "#e74c3c", "shieldgemma": "#f39c12"}
    labels = {"deberta": "DeBERTa-v3", "text-moderation": "Text-Moderation",
              "llama-guard": "Llama Guard 3", "shieldgemma": "ShieldGemma"}

    for clf_name, result in all_results.items():
        series = result["ks_series"]
        steps = [s["step"] for s in series]
        ks_vals = [s["ks"] for s in series]
        ax.plot(steps, ks_vals, label=labels[clf_name], color=colors[clf_name], lw=1.5)

    # Mark shift onset
    onset = list(all_results.values())[0]["shift_onset"]
    ax.axvline(onset, color="black", ls="--", lw=0.8, alpha=0.5, label="Shift onset")

    # Mark threshold (use first available)
    threshold = list(all_results.values())[0]["threshold"]
    ax.axhline(threshold, color="gray", ls=":", lw=0.8, alpha=0.5, label=f"Threshold ({threshold:.3f})")

    ax.set_xlabel("Stream Step")
    ax.set_ylabel("KS Statistic")
    ax.set_title("Regime C: KS Statistic Trajectories (Adversarial Success)")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight", dpi=150)
    fig.savefig(OUTPUT_FIGURE.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Figure saved to {OUTPUT_FIGURE}")
    plt.close()


def main():
    print("P2: Regime C with KS time series storage")
    print("=" * 60)

    print("Loading reference data...")
    reference, neg_pool = load_reference()

    print("Loading adversarial success examples...")
    shifted = load_adversarial_success()

    all_results = {}
    for clf_name in CLASSIFIERS:
        print(f"\n[{clf_name}] Loading classifier...")
        classifier = get_classifier(clf_name)

        # Calibrate threshold
        print(f"  Calibrating (window_size={WINDOW_SIZE})...")
        max_ks = []
        for i in range(N_CALIBRATION):
            start = (i * 50) % len(neg_pool)
            examples = neg_pool[start:start + N_REFERENCE]
            if len(examples) < WINDOW_SIZE:
                examples = neg_pool[:N_REFERENCE]
            mk, _ = run_stream_ks(classifier, examples, WINDOW_SIZE, seed=42 + i * 7)
            max_ks.append(mk)
        threshold = float(np.percentile(max_ks, CAL_PCT))
        print(f"  Threshold: {threshold:.4f}")

        # Run with KS series stored
        cond_seed = SEED * 1000 + 999
        result = run_with_ks_series(
            classifier, reference, shifted, N_REFERENCE, WINDOW_SIZE, cond_seed, threshold
        )
        result["classifier"] = clf_name
        all_results[clf_name] = result

        lat = result["detection_latency"] or "none"
        print(f"  Detection latency: {lat}")
        print(f"  KS series length: {len(result['ks_series'])}")

    # Save full results
    OUTPUT_SERIES.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SERIES.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nKS series saved to {OUTPUT_SERIES}")

    # Plot
    plot_trajectories(all_results)

    # Summary
    print("\n" + "=" * 60)
    print("REGIME C KS SERIES SUMMARY")
    print("-" * 60)
    for clf_name, result in all_results.items():
        lat = result["detection_latency"] or "none"
        max_ks = max(s["ks"] for s in result["ks_series"]) if result["ks_series"] else 0
        print(f"  {clf_name:<18} latency={lat:<6} max_ks={max_ks:.4f} threshold={result['threshold']:.4f}")


if __name__ == "__main__":
    main()
