"""Investigation script (2026-10-18). Run from the repo root.
Requires the cached score streams (rebuild with scripts/cache_embeddings.py or copy them);
set CS_STREAMS=<dir containing deberta/paraphrase/seed_0.npz ...>. Engine variants under
scripts/investigations/engine_*.py are copies of earlier versions of
shift_detection_monitor/detection/confidence_sequence.py (git: 0380167 and 00d01b1).
"""
import os
import importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
nul = json.load(open(ROOT / "results/null_scores.json"))
CL = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
def load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m.ConfidenceSequenceEngine
NEW = load("shift_detection_monitor/detection/confidence_sequence.py", "n")
def alarm(stream, mu0):
    e = NEW(alpha=0.05, reference_value=mu0, window_mode="growing", tail_bound="bounded", lower_bound=0.0, upper_bound=1.0, min_warmup_steps=100)
    for x in stream:
        if e.update(float(x)).alarm: return True
    return False
rng = np.random.default_rng(11)
N = 600
print("Population = the 500 null scores of each classifier (true mean known exactly = pool mean).")
print("Monitored streams: 500 fresh draws from the population. mu0 setting varies.\n")
print(f"{'classifier':16s} {'known mu0':>10s} {'n_ref=250':>10s} {'n_ref=500':>10s} {'n_ref=2000':>11s}")
for c in CL:
    pool = np.array(nul[c]); mu = pool.mean(); row = []
    for ref in [None, 250, 500, 2000]:
        al = 0
        for _ in range(N):
            m0 = mu if ref is None else rng.choice(pool, size=ref, replace=True).mean()
            al += alarm(rng.choice(pool, size=500, replace=True), m0)
        row.append(al / N)
    print(f"{c:16s} {row[0]:10.3f} {row[1]:10.3f} {row[2]:10.3f} {row[3]:11.3f}")
