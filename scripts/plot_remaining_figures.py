"""Regenerate the six committed paper figures that shipped without a drawing script.

Draws, from STORED RESULT FILES ONLY (no model inference, no API calls, no network):

  fig_threat_tiers      bar chart by threat tier 1-4 (detection for tiers 1-2, attack blocked for 3-4)
  fig_k_scaling         detection rate by number of canaries k=1,2,3 (labels 93.9/98.0/98.0)
  fig_confidence_gating scatter: canary baseline confidence B_orig vs transfer outcome
  fig_budget_curve      transfer rate (%), single-target vs joint GCG, 50 steps
  fig_llm_canary_split  mean score per model, benign vs adversarial
  fig_scrambled_violin  violin of delta (adversarial - clean) over all models x prompts

Matplotlib style follows scripts/plot_null_score_kde.py (steelblue fills, top/right
spines hidden, savefig both .pdf and .png, dpi=150, bbox_inches="tight").

Output goes to /tmp/smsr_final/figs_regen/ by default (override with --out).
The script writes to paper/figures/ only when --out is given that path explicitly.

Usage:
    REPRO_MANIFEST_DISABLE=1 .venv/bin/python scripts/plot_remaining_figures.py
    REPRO_MANIFEST_DISABLE=1 .venv/bin/python scripts/plot_remaining_figures.py --out /some/dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
STEELBLUE = "steelblue"


def _load(rel: str):
    return json.loads((RESULTS / rel).read_text())


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"{name}{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_dir / (name + '.pdf')}")


def _hide_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# fig_k_scaling  <- results/track_a_k_scaling.json
# ---------------------------------------------------------------------------
def plot_k_scaling(out_dir: Path) -> None:
    d = _load("track_a_k_scaling.json")
    # Best-of-k (the paper reports best single, best pair, and k=3 all-three).
    k1 = d["summary"]["k1_best"]["rate"]
    k2 = d["summary"]["k2_best"]["rate"]
    k3 = d["summary"]["k3"]["rate"]
    rates = np.array([k1, k2, k3]) * 100.0

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([1, 2, 3], rates, color=STEELBLUE, width=0.6)
    for x, r in zip([1, 2, 3], rates):
        ax.text(x, r + 1.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Number of Canaries ($k$)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, 105)
    _hide_spines(ax)
    _save(fig, out_dir, "fig_k_scaling")


# ---------------------------------------------------------------------------
# fig_threat_tiers
#   Tier 1: 98.0% detection (k=2)        results/track_a_k_scaling.json
#   Tier 2: 93.9% detection (k=1 best)   results/track_a_k_scaling.json
#   Tier 3: attack blocked in 9/20       results/ca8_joint_gcg_n20.json
#           (20 prompts minus n_both_flipped = 11)
#   Tier 4: attack blocked in 14/20      results/ca8_divergence_min_n20.json
#           (len(blocked_gaps) = 14)
#   Tiers 1-2 are detection rates; tiers 3-4 are the share of attacks the
#   adversary could not complete. The two measures are drawn with different
#   fills and labelled, not presented as one quantity.
#   The version of this figure committed with paper v4 plotted 70% and 60% for
#   tiers 3 and 4; those values are in no result file and disagreed with the
#   paper's own table (45% and 70%), so they were replaced.
# ---------------------------------------------------------------------------
def plot_threat_tiers(out_dir: Path) -> None:
    d = _load("track_a_k_scaling.json")
    joint = _load("ca8_joint_gcg_n20.json")
    div = _load("ca8_divergence_min_n20.json")
    tier1 = d["summary"]["k2_best"]["rate"] * 100.0
    tier2 = d["summary"]["k1_best"]["rate"] * 100.0
    tier3 = 100.0 * (joint["n_prompts"] - joint["n_both_flipped"]) / joint["n_prompts"]
    tier4 = 100.0 * len(div["blocked_gaps"]) / div["n_total"]
    rates = [tier1, tier2, tier3, tier4]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([1, 2], rates[:2], color=STEELBLUE, width=0.6, label="Detected")
    ax.bar([3, 4], rates[2:], color="darkorange", width=0.6, label="Attack blocked")
    for x, r in zip([1, 2, 3, 4], rates):
        ax.text(x, r + 1.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Threat Tier")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 118)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _hide_spines(ax)
    _save(fig, out_dir, "fig_threat_tiers")


# ---------------------------------------------------------------------------
# fig_confidence_gating  <- results/ca8_cheap_canary.json
#   B_orig = canary_orig; transferred iff the canary was flipped (canary_attacked < 0.5).
#   Reproduces results/confidence_gating_validation.json summary exactly
#   (n=40, 14 transferred, rate>=0.99 = 4.3%, rate<0.99 = 76.5%).
# ---------------------------------------------------------------------------
def plot_confidence_gating(out_dir: Path) -> None:
    rows = _load("ca8_cheap_canary.json")["canary_results"]
    borig = np.array([r["canary_orig"] for r in rows])
    transferred = np.array([r["canary_attacked"] < 0.5 for r in rows])

    fig, ax = plt.subplots(figsize=(4, 3))
    y = np.where(transferred, 1.0, 0.0)
    ax.scatter(borig, y, color=STEELBLUE, alpha=0.6, s=35)
    ax.axvline(0.99, ls="--", color="gray", lw=1.0,
               label="$B_{orig} = 0.99$ threshold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Resisted", "Transferred"])
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("Canary Baseline Confidence ($B_{orig}$)")
    ax.set_ylabel("Transfer Outcome")
    ax.legend(loc="center left", fontsize=8)
    _hide_spines(ax)
    _save(fig, out_dir, "fig_confidence_gating")


# ---------------------------------------------------------------------------
# fig_budget_curve
#   single-target: results/ca8_transfer_minimal.json  (3/10 = 30%, 50-step passive)
#   joint:         results/ca8_joint_gcg_n20.json     (11/20 = 55%, lambda=0.5, 50 steps)
#   The version committed with paper v4 plotted 70% (7/10, ca8_joint_gcg_proper.json),
#   a different and smaller run than the 55% (11/20) the paper's text reports.
# ---------------------------------------------------------------------------
def plot_budget_curve(out_dir: Path) -> None:
    single = _load("ca8_transfer_minimal.json")
    joint = _load("ca8_joint_gcg_n20.json")
    single_rate = 100.0 * single["n_transfer"] / single["n_prompts"]
    joint_rate = 100.0 * joint["n_both_flipped"] / joint["n_prompts"]
    joint_steps = joint.get("steps", 50)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([0, 1], [single_rate, joint_rate],
           color=["steelblue", "darkorange"], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Single-target\n(50 steps, n={single['n_prompts']})",
                        f"Joint\n({joint_steps} steps, n={joint['n_prompts']})"])
    for x, r in zip([0, 1], [single_rate, joint_rate]):
        ax.text(x, r + 1.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("GCG Attack Strategy")
    ax.set_ylabel("Transfer Rate (%)")
    ax.set_ylim(0, max(single_rate, joint_rate) * 1.25)
    _hide_spines(ax)
    _save(fig, out_dir, "fig_budget_curve")


# ---------------------------------------------------------------------------
# fig_llm_canary_split  <- results/benign_specificity_full.json
#   benign_mean and adv_mean per model. The committed figure shows these 7 models
#   with a dotted divider between "ceiling-clipped" (benign high) and discriminating.
# ---------------------------------------------------------------------------
LLM_SPLIT_MODELS = [
    ("gpt-4o", "gpt-4o"),
    ("gpt-4.1", "gpt-4.1"),
    ("gpt-5.1", "gpt-5.1"),
    ("claude-sonnet-4-0", "claude-s4"),
    ("gemini-2.5-pro", "gem-2.5p"),
    ("claude-sonnet-4-5", "claude-s4.5"),
    ("claude-sonnet-4-6", "claude-s4.6"),
]


def plot_llm_canary_split(out_dir: Path) -> None:
    d = _load("benign_specificity_full.json")
    labels, benign, adv = [], [], []
    for key, label in LLM_SPLIT_MODELS:
        labels.append(label)
        benign.append(d[key]["benign_mean"])
        adv.append(d[key]["adv_mean"])
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.2, 3))
    ax.bar(x - w / 2, benign, w, label="Benign", color="steelblue")
    ax.bar(x + w / 2, adv, w, label="Adversarial", color="darkorange")
    # dotted divider between the 4 ceiling-clipped and the 3 discriminating models
    ax.axvline(3.5, ls=":", color="black", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    _hide_spines(ax)
    _save(fig, out_dir, "fig_llm_canary_split")


# ---------------------------------------------------------------------------
# fig_scrambled_violin  <- results/llm_canary_v2.json
#   Delta(adv - clean) per prompt per model, across all 20 models x 49 prompts
#   = 980 prompt-model pairs.
# ---------------------------------------------------------------------------
def plot_scrambled_violin(out_dir: Path) -> None:
    d = _load("llm_canary_v2.json")
    deltas = []
    for _model, v in d.items():
        scores = v["scores"]
        gcg = [s["score"] for s in scores if s["type"] == "gcg"]
        clean = [s["score"] for s in scores if s["type"] == "clean"]
        n = min(len(gcg), len(clean))
        for i in range(n):
            if gcg[i] is not None and clean[i] is not None:
                deltas.append(gcg[i] - clean[i])
    deltas = np.array(deltas)

    fig, ax = plt.subplots(figsize=(4, 3))
    parts = ax.violinplot([deltas], positions=[0], showmeans=False,
                          showextrema=False, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(STEELBLUE)
        pc.set_alpha(0.6)
    ax.hlines(float(np.median(deltas)), -0.25, 0.25, color="black", lw=1.0, ls="--")
    ax.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
    ax.set_xticks([0])
    ax.set_xticklabels(["All models $\\times$ prompts"])
    ax.set_ylabel(r"$\Delta$ (adversarial $-$ clean)")
    ax.set_ylim(-1.05, 1.05)
    _hide_spines(ax)
    _save(fig, out_dir, "fig_scrambled_violin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/smsr_final/figs_regen",
                        help="Output directory (default: /tmp/smsr_final/figs_regen). "
                             "paper/figures/ is never written.")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    if out_dir == (REPO / "paper" / "figures").resolve():
        raise SystemExit("Refusing to write to paper/figures/.")

    print(f"Writing regenerated figures to {out_dir}")
    plot_threat_tiers(out_dir)
    plot_k_scaling(out_dir)
    plot_confidence_gating(out_dir)
    plot_budget_curve(out_dir)
    plot_llm_canary_split(out_dir)
    plot_scrambled_violin(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
