"""Regenerate paper/latex/figures/fig_ad1_killer_chart.pdf from the reproduced AD1 detection rates.

Run scripts/investigations/ad1_ks_threshold_sensitivity.py with AD1_SHIFT=paraphrase first; it writes
results/cs_engine_investigation/ad1_paraphrase.json, which this script reads. Run from the repo root.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "results/cs_engine_investigation/ad1_paraphrase.json").read_text())
mix = [int(round(m * 100)) for m in d["mixing"]]
fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.plot(mix, [100 * v for v in d["scan"]], marker="s", color="tab:orange", label="Scan martingale")
styles = [("ks_committed", "KS, 500-step null threshold", "tab:blue", "-"),
          ("ks_longnull", "KS, 800-step null threshold", "tab:blue", "--"),
          ("ks_pooled", "KS, pooled 800-step null", "tab:blue", ":")]
for key, label, colour, ls in styles:
    ax.plot(mix, [100 * v for v in d[key]], marker="o", color=colour, linestyle=ls, label=label)
ax.set_xlabel("Mixing Rate (%)")
ax.set_ylabel("Detection Rate (%)")
ax.set_xticks(mix)
ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
out = ROOT / "paper/latex/figures/fig_ad1_killer_chart.pdf"
fig.savefig(out)
print("wrote", out)
