"""Recompute the rotating-canary table (tab:rotating-canary) from the committed trial log.

Prints, per model x attacker/condition: N, attack-success count and rate, Wilson 95% CI, and the fraction of
sessions with canary_suppressed, plus the grand total, so the table in paper.tex can be compared cell by cell.
"""
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(line) for line in open(ROOT / "results/rotating_canary_eval/rotating_canary_trials.jsonl") if line.strip()]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


g = defaultdict(list)
for r in rows:
    g[(r["model"], r["attacker"], r["condition"])].append(r)
print(f"records: {len(rows)}")
print(f"{'model':18s} {'attacker':18s} {'condition':14s} {'N':>4s} {'succ':>5s} {'ASR':>6s} {'Wilson95':>16s} {'suppressed':>10s}")
for k in sorted(g):
    v = g[k]
    n = len(v)
    s = sum(bool(x["attack_success"]) for x in v)
    sup = sum(bool(x["canary_suppressed"]) for x in v)
    lo, hi = wilson(s, n)
    print(f"{k[0]:18s} {k[1]:18s} {k[2]:14s} {n:4d} {s:5d} {s/n:6.3f} [{lo:.3f}, {hi:.3f}] {sup/n:9.1%}")
print("sessions per model:", dict(Counter(r["model"] for r in rows)))
