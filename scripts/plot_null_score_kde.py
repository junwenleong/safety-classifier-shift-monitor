"""P3: Generate null score distribution KDE figure.

Extracts safety scores from all 4 classifiers on in-distribution data,
then produces a 4-panel KDE plot showing score distributions under the null.

This explains the 5x FAR spread across classifiers — different score
distribution shapes produce different calibration thresholds.

Usage (Mac Studio):
    export DEBERTA_CHECKPOINT_PATH=checkpoints/deberta-wildguardmix
    export TEXT_MODERATION_CHECKPOINT_PATH=checkpoints/text-moderation-wildguardmix
    python scripts/plot_null_score_kde.py

If null scores already exist (from a prior run), regenerates the figure only:
    python scripts/plot_null_score_kde.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CLASSIFIERS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
CLASSIFIER_LABELS = {
    "deberta": "DeBERTa-v3 (86M)",
    "text-moderation": "Text-Moderation (304M)",
    "llama-guard": "Llama Guard 3 (8B)",
    "shieldgemma": "ShieldGemma (9B)",
}
N_SAMPLES = 500
OUTPUT_SCORES = Path("results/null_scores.json")
OUTPUT_FIGURE = Path("paper/figures/fig_null_score_kde.pdf")


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


def load_reference(n: int) -> list[dict]:
    """Load n unharmful WildGuardMix examples."""
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    ds = ds.filter(lambda x: x["prompt_harm_label"] == "unharmful")
    ds = ds.shuffle(seed=42)
    return [{"text": ds["prompt"][i]} for i in range(n)]


def extract_null_scores() -> dict[str, list[float]]:
    """Run all classifiers on reference data and return scores."""
    print(f"Loading {N_SAMPLES} reference examples...")
    examples = load_reference(N_SAMPLES)

    all_scores = {}
    for clf_name in CLASSIFIERS:
        print(f"\n[{clf_name}] Loading classifier...")
        classifier = get_classifier(clf_name)
        scores = []
        for i, ex in enumerate(examples):
            output = classifier.predict(ex["text"])
            scores.append(float(output.score))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{N_SAMPLES}")
        all_scores[clf_name] = scores
        print(f"  Done. Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")

    # Save
    OUTPUT_SCORES.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SCORES.write_text(json.dumps(all_scores, indent=2) + "\n")
    print(f"\nScores saved to {OUTPUT_SCORES}")
    return all_scores


def plot_kde(all_scores: dict[str, list[float]]) -> None:
    """Generate 4-panel KDE figure."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axes = axes.flatten()

    for ax, clf_name in zip(axes, CLASSIFIERS):
        scores = np.array(all_scores[clf_name])
        kde = gaussian_kde(scores, bw_method="scott")
        x = np.linspace(0, 1, 500)
        density = kde(x)

        ax.fill_between(x, density, alpha=0.3, color="steelblue")
        ax.plot(x, density, color="steelblue", lw=1.5)
        ax.set_title(CLASSIFIER_LABELS[clf_name], fontsize=10)
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate stats
        ax.text(0.95, 0.95, f"μ={np.mean(scores):.3f}\nσ={np.std(scores):.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[2].set_xlabel("Safety Score")
    axes[3].set_xlabel("Safety Score")
    fig.suptitle("Null Score Distributions (In-Distribution)", fontsize=11, y=0.98)
    plt.tight_layout()

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight", dpi=150)
    fig.savefig(OUTPUT_FIGURE.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Figure saved to {OUTPUT_FIGURE}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference, plot from existing scores")
    args = parser.parse_args()

    if args.plot_only:
        if not OUTPUT_SCORES.exists():
            print(f"ERROR: {OUTPUT_SCORES} not found. Run without --plot-only first.")
            sys.exit(1)
        all_scores = json.loads(OUTPUT_SCORES.read_text())
    else:
        all_scores = extract_null_scores()

    plot_kde(all_scores)


if __name__ == "__main__":
    main()
