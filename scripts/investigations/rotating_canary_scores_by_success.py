"""Canary score distribution among successful attacks, by model x attacker (rotating-canary trial log).

The paper says spoofing attacks succeed "without triggering canary alarms". Alarm rule in the paper's design is a
canary score below a threshold; this prints the canary_score mean and share of scores below 0.5 for successful and
unsuccessful attacked sessions, so that claim can be compared with the record.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(line) for line in open(ROOT / "results/rotating_canary_eval/rotating_canary_trials.jsonl") if line.strip()]
g = defaultdict(list)
for r in rows:
    g[(r["model"], r["attacker"], bool(r["attack_success"]))].append(r["canary_score"])
print(f"{'model':16s} {'attacker':20s} {'success':>7s} {'n':>4s} {'mean canary':>11s} {'share<0.5':>10s} {'min':>5s} {'max':>5s}")
for k in sorted(g):
    v = np.array(g[k], dtype=float)
    print(f"{k[0]:16s} {k[1]:20s} {str(k[2]):>7s} {len(v):4d} {v.mean():11.3f} {np.mean(v < 0.5):10.2f} {v.min():5.2f} {v.max():5.2f}")
