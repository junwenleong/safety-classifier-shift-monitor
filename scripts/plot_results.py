"""Generate publication-quality figures for the paper.

Figure 1: Detection latency heatmap (classifiers × shift conditions)
Figure 2: Variance decomposition horizontal bar chart

Usage:
    python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/factorial_results.jsonl")
VARIANCE = Path("results/variance_decomposition.json")
OUTDIR = Path("paper/figures")

CLASSIFIERS = ["DeBERTa", "Text-Moderation", "Llama Guard", "ShieldGemma"]
CLF_KEYS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["Paraphrase", "Code-switch", "Compositional", "Temporal", "Adversarial"]
SHIFT_KEYS = ["paraphrase", "code-switch", "compositional-long-context", "temporal", "adversarial-suffix"]

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 150,
})


def load_factorial():
    with open(RESULTS) as f:
        return [json.loads(l) for l in f if l.strip()]


def fig1_heatmap(rows):
    """Detection latency heatmap, classifiers × shift conditions."""
    matrix = np.full((4, 5), np.nan)
    counts = np.zeros((4, 5), dtype=int)

    for r in rows:
        valid = (r.get("detection_latency") is not None
                 and r["detection_latency"] >= 0
                 and r.get("neg_clean") is True)
        if not valid:
            continue
        i = CLF_KEYS.index(r["classifier"])
        j = SHIFT_KEYS.index(r["shift_condition"])
        if np.isnan(matrix[i, j]):
            matrix[i, j] = 0.0
        matrix[i, j] += r["detection_latency"]
        counts[i, j] += 1

    # Compute means
    with np.errstate(invalid="ignore"):
        matrix = np.where(counts > 0, matrix / counts, np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Use reversed Blues so dark = fast, light = slow
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=20, vmax=100)

    # Hatching for cells with < 3 valid detections
    for i in range(4):
        for j in range(5):
            if counts[i, j] < 3:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, hatch="///", edgecolor="gray", lw=0.5))

    # Annotate cells
    for i in range(4):
        for j in range(5):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "white" if val > 65 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
                ax.text(j, i + 0.3, f"n={counts[i,j]}", ha="center", va="center",
                        fontsize=6.5, color=color, alpha=0.8)

    ax.set_xticks(range(5))
    ax.set_xticklabels(SHIFTS, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASSIFIERS, fontsize=9)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Mean detection latency (steps)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Detection Latency: Classifiers × Shift Conditions", fontsize=10, pad=8)

    plt.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "fig1_latency_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig1_latency_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {OUTDIR / 'fig1_latency_heatmap.pdf'}")


def fig2_variance(vd):
    """Variance decomposition horizontal bar chart."""
    labels = ["Residual", "Classifier", "Shift type", "Classifier × Shift"]
    values = [
        vd["residual_variance"],
        vd["factor_variances"]["classifier"],
        vd["factor_variances"]["shift_type"],
        vd["interaction_variances"]["classifier:shift_type"],
    ]

    # CIs (absolute bounds) — only for main effects
    ci_clf = (vd["effect_sizes"]["classifier"]["ci_lower"],
              vd["effect_sizes"]["classifier"]["ci_upper"])
    ci_shift = (vd["effect_sizes"]["shift_type"]["ci_lower"],
                vd["effect_sizes"]["shift_type"]["ci_upper"])

    # Convert to asymmetric errors relative to bar value
    errors_low = [0, values[1] - ci_clf[0], values[2] - ci_shift[0], 0]
    errors_high = [0, ci_clf[1] - values[1], ci_shift[1] - values[2], 0]

    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    colors = ["#bdbdbd", "#31a354", "#3182bd", "#e6550d"]
    y_pos = range(len(labels))

    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    # Add error bars only for items with CIs
    for i in range(len(values)):
        if errors_low[i] > 0 or errors_high[i] > 0:
            ax.errorbar(values[i], i, xerr=[[errors_low[i]], [errors_high[i]]],
                        fmt="none", color="black", lw=1.0, capsize=3, capthick=0.8)

    # Annotate values
    for i, v in enumerate(values):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Proportion of variance (η²)", fontsize=9)
    ax.set_xlim(0, 0.42)
    ax.axvline(x=0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    ax.set_title("Variance Decomposition of Detection Latency", fontsize=10, pad=8)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig2_variance_decomposition.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig2_variance_decomposition.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {OUTDIR / 'fig2_variance_decomposition.pdf'}")


def main():
    rows = load_factorial()
    fig1_heatmap(rows)

    with open(VARIANCE) as f:
        vd = json.loads(f.read())
    fig2_variance(vd)


if __name__ == "__main__":
    main()
