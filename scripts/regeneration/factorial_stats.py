"""Summary statistics of a factorial results file (paper-style), for ORIGINAL vs regenerated comparison.

Usage: python factorial_stats.py FILE.jsonl [--shifts a,b,c]   (restrict to some shift conditions)
Valid (control-validated) detection: detection_latency not None and >= 0 and neg_clean.
Reports: cells, valid rate, raw TPR (latency >= 0), per-classifier FAR (neg_clean False), mean latency by window,
cell mean latencies, and sequential-SS eta-squared (classifier -> shift -> interaction) on valid detections.
"""
import json
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1]
shifts = None
if "--shifts" in sys.argv:
    shifts = sys.argv[sys.argv.index("--shifts") + 1].split(",")
rows = [json.loads(line) for line in open(path) if line.strip()]
if shifts:
    rows = [r for r in rows if r["shift_condition"] in shifts]
valid = [r for r in rows if r["detection_latency"] is not None and r["detection_latency"] >= 0 and r["neg_clean"]]
raw = [r for r in rows if r["detection_latency"] is not None and r["detection_latency"] >= 0]
print(f"{path}: cells {len(rows)}, valid {len(valid)} ({len(valid)/len(rows):.1%}), raw TPR {len(raw)/len(rows):.1%}")
for clf in sorted({r["classifier"] for r in rows}):
    c = [r for r in rows if r["classifier"] == clf]
    far = sum(not r["neg_clean"] for r in c) / len(c)
    v = [r for r in valid if r["classifier"] == clf]
    print(f"  {clf:16s} FAR {far:.1%}  valid {len(v)}/{len(c)} ({len(v)/len(c):.1%})  mean latency {np.mean([r['detection_latency'] for r in v]):.1f}")
for w in (100, 200):
    v = [r["detection_latency"] for r in valid if r["window_size"] == w]
    print(f"  window {w}: valid {len(v)}, mean latency {np.mean(v):.2f}")
print("  grand mean latency (all valid): %.2f" % np.mean([r["detection_latency"] for r in valid]))

cells = defaultdict(list)
for r in valid:
    cells[(r["classifier"], r["shift_condition"])].append(r["detection_latency"])
print("  cell mean latency (n):")
for k in sorted(cells):
    print(f"    {k[0]:16s} {k[1]:28s} {np.mean(cells[k]):6.1f} ({len(cells[k])})")

# sequential SS (type I), order classifier, shift, interaction
y = np.array([r["detection_latency"] for r in valid], dtype=float)
C = np.array([r["classifier"] for r in valid])
S = np.array([r["shift_condition"] for r in valid])
grand = y.mean()
ss_tot = ((y - grand) ** 2).sum()


def fit_ss(design_cols):
    X = np.column_stack([np.ones(len(y))] + design_cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return ((y - X @ beta) ** 2).sum()


def dummies(a):
    levels = sorted(set(a))
    return [(a == l).astype(float) for l in levels[1:]]


dc, ds = dummies(C), dummies(S)
inter = [a * b for a in dc for b in ds]
rss0 = ss_tot
rss_c = fit_ss(dc)
rss_cs = fit_ss(dc + ds)
rss_full = fit_ss(dc + ds + inter)
print("  eta^2 (sequential): classifier %.3f, shift %.3f, interaction %.3f, residual %.3f" % (
    (rss0 - rss_c) / ss_tot, (rss_c - rss_cs) / ss_tot, (rss_cs - rss_full) / ss_tot, rss_full / ss_tot))
