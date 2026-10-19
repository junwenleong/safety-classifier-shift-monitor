"""Compare regenerated factorial rows with the original rows, cell by cell.

Usage: python compare_runs.py ORIGINAL.jsonl NEW.jsonl [OLD.jsonl]
Reports, per (classifier, shift): number of cells, cells with identical (alarm_step, detection_latency, neg_clean),
max |pre_score| / |post_score| differences, and threshold equality.  With OLD (re-run with the committed KS on the same cache),
reports the detector-only difference (NEW vs OLD) separately from the environment difference (OLD vs ORIGINAL).
"""
import json
import sys
from collections import defaultdict

key = lambda r: (r["classifier"], r["shift_condition"], r["seed"], r["window_size"])


def load(p):
    return {key(r): r for r in map(json.loads, open(p)) if r}


def cmp(a, b, label):
    print(f"\n=== {label} ===")
    by = defaultdict(list)
    for k in a:
        if k in b:
            by[(k[0], k[1])].append(k)
    tot = same = 0
    for g in sorted(by):
        ks = by[g]
        ident = [k for k in ks if all(a[k][f] == b[k][f] for f in ("alarm_step", "detection_latency", "neg_clean"))]
        pre = max(abs(a[k]["pre_score"] - b[k]["pre_score"]) for k in ks)
        post = max(abs(a[k]["post_score"] - b[k]["post_score"]) for k in ks)
        thr = all(abs(a[k]["threshold"] - b[k]["threshold"]) < 1e-9 for k in ks)
        tot += len(ks)
        same += len(ident)
        print(f"{g[0]:16s} {g[1]:28s} cells {len(ks):3d}  identical outcome {len(ident):3d}  max|pre| {pre:.1e}  max|post| {post:.1e}  thresholds equal {thr}")
    print(f"TOTAL cells compared {tot}, identical outcome {same}")


o = load(sys.argv[1])
n = load(sys.argv[2])
cmp(o, n, "ORIGINAL vs NEW (corrected KS)")
if len(sys.argv) > 3:
    old = load(sys.argv[3])
    cmp(o, old, "ORIGINAL vs OLD (committed KS, same cache)")
    cmp(old, n, "OLD vs NEW (detector change only)")
