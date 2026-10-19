"""Recompute tab:rank-reversal (V5-1) from results/v5_rank_reversal/rank_reversal_results.jsonl.

Per model x budget: N, valid-response share (Wilson 95%), mean score on adversarial and benign prompts among valid
responses, and AUC (adversarial vs benign) among valid responses. Prints the record keys first so the filter is visible.
"""
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(line) for line in open(ROOT / "results/v5_rank_reversal/rank_reversal_results.jsonl") if line.strip()]
print("records:", len(rows), "keys:", sorted(rows[0]))
print(json.dumps(rows[0])[:400])


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


kmodel, kbud, klab, kscore = "model", "budget", "prompt_class", "score"
g = defaultdict(list)
for r in rows:
    g[(r[kmodel], r[kbud])].append(r)
print(f"\n{'model':16s} {'budget':>6s} {'N':>4s} {'valid':>6s} {'Wilson95':>16s} {'adv mean':>9s} {'ben mean':>9s} {'AUC':>6s}")
for (m, b), v in sorted(g.items()):
    valid = [r for r in v if r["valid"] and r[kscore] is not None and 0.0 <= float(r[kscore]) <= 1.0]
    adv = [float(r[kscore]) for r in valid if str(r[klab]).lower() in ("adversarial", "adv", "true", "1", "attack")]
    ben = [float(r[kscore]) for r in valid if str(r[klab]).lower() in ("benign", "ben", "false", "0", "clean")]
    lo, hi = wilson(len(valid), len(v))
    print(f"{m:16s} {b:6} {len(v):4d} {len(valid)/len(v):6.0%} [{lo:.3f}, {hi:.3f}] "
          f"{(np.mean(adv) if adv else float('nan')):9.3f} {(np.mean(ben) if ben else float('nan')):9.3f} {auc(adv, ben):6.3f}")
