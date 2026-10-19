"""Recompute the frontier-canary detection statistics quoted in the LLM-canary section from the committed files.

Source: results/benign_specificity_full.json (20 models; 20 benign prompts each, adv_mean over the 20 GCG prompts)
and results/pareto_frontier.json (model_stats: n_fp out of n_benign per model, category).

The paper text (before this change) said: among 20 discriminating models, mean delta(adv - benign) +0.68
(range +0.39 to +0.91), mean benign score 0.016 (18/20 exactly 0.0), mean adversarial score 0.73, and zero
false alarms on 20 benign prompts for 18/20 models. This script prints what the files give.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
d = json.load(open(ROOT / "results/benign_specificity_full.json"))
p = json.load(open(ROOT / "results/pareto_frontier.json"))["model_stats"]

rows = []
for m, v in d.items():
    b = np.array(v["benign_scores"], dtype=float)
    rows.append(dict(model=m, adv=v["adv_mean"], benign_raw=float(b.mean()), benign_clip=float(np.clip(b, 0, 1).mean()),
                     out_of_range=int(np.sum((b < 0) | (b > 1))), n_ge_half=int(np.sum(b >= 0.5)),
                     all_zero=bool(np.all(b == 0))))
print(f"models in file: {len(rows)}")
print(f"{'model':26s} {'adv':>6s} {'benign(clip)':>12s} {'oor':>3s} {'FA>=0.5':>7s} {'pareto n_fp':>11s}")
for r in rows:
    print(f"{r['model']:26s} {r['adv']:6.3f} {r['benign_clip']:12.3f} {r['out_of_range']:3d} {r['n_ge_half']:7d} {p[r['model']]['n_fp']:11d}")

fa_free = [r for r in rows if r["n_ge_half"] == 0]
print(f"\nmodels with no benign score >= 0.5: {len(fa_free)} of {len(rows)}; all 20 scores exactly 0: {sum(r['all_zero'] for r in rows)}")
print(f"models with a score outside [0,1]: {[r['model'] for r in rows if r['out_of_range']]}")
# Delta with benign scores clipped to [0,1]
for name, sel in (("all 20", rows), ("excluding gemini-2.5-pro (benign all 1.0)", [r for r in rows if r["model"] != "gemini-2.5-pro"]),
                  ("excluding gemini-2.5-pro and claude-opus-4-7", [r for r in rows if r["model"] not in ("gemini-2.5-pro", "claude-opus-4-7")])):
    dl = np.array([r["adv"] - r["benign_clip"] for r in sel])
    adv = np.array([r["adv"] for r in sel]); ben = np.array([r["benign_clip"] for r in sel])
    print(f"{name}: n={len(sel)} mean delta {dl.mean():+.3f} range {dl.min():+.2f}..{dl.max():+.2f}; mean adv {adv.mean():.3f} "
          f"(range {adv.min():.2f}..{adv.max():.2f}); mean benign {ben.mean():.3f}")
print("\nunclipped delta as stored in the file, mean over all 20:", f"{np.mean([v['delta'] for v in d.values()]):+.3f}")
print("categories in pareto_frontier.json:", {c: sum(v['category'] == c for v in p.values()) for c in {v['category'] for v in p.values()}})
