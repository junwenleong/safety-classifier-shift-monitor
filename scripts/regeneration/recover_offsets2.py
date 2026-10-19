"""Recover cond_seed offsets from the stream order alone (independent of the KS statistic).

In the positive stream (onset 500, mixing 1.0) the first 500 records are the shuffled reference pool and the next 300 the shuffled shifted pool.
pre_score = mean of reference scores at shuffled positions [w, 500); post_score = mean of all shifted scores (order-independent).
For each original row, find offsets o in 0..999 with cond_seed = seed*1000 + o such that pre_score matches to 1e-9.
Usage: python recover_offsets2.py <classifier>
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scripts.run_factorial as rf  # noqa: E402
from scripts.run_full_canary import load_shifted  # noqa: E402
from memo import MemoClassifier  # noqa: E402

name = sys.argv[1]
orig = [json.loads(line) for line in open(os.environ["ORIGINAL"]) if line.strip()]
memo = MemoClassifier(lambda: (_ for _ in ()).throw(RuntimeError("cache miss")), os.path.join(os.environ["SCORE_CACHE"], f"{name}.pkl"))
reference, _ = rf.load_reference()
ref_scores = np.array([memo.predict(e["text"]).score for e in reference])
res = {}
for shift in rf.SHIFT_CONDITIONS:
    shifted = load_shifted(rf.SHIFT_CORPORA[shift], n=300)
    sh_scores = np.array([memo.predict(e["text"]).score for e in shifted])
    rows = [r for r in orig if r["classifier"] == name and r["shift_condition"] == shift]
    pdiff = max(abs(r["post_score"] - sh_scores.mean()) for r in rows)
    post_ok = np.mean([abs(r["post_score"] - sh_scores.mean()) < 1e-9 for r in rows])
    per_seed = {}
    for r in rows:
        if r["window_size"] != 100:
            continue
        hits = []
        for o in range(1000):
            idx = list(range(len(ref_scores)))
            random.Random(r["seed"] * 1000 + o).shuffle(idx)
            pre = ref_scores[idx][100:500].mean()
            if abs(pre - r["pre_score"]) < 1e-9:
                hits.append(o)
        per_seed[r["seed"]] = hits
    common = set(range(1000))
    for h in per_seed.values():
        common &= set(h)
    print(f"{name}/{shift}: post_score matches shifted-pool mean (1e-5) in {post_ok:.0%} of rows, max |diff| {pdiff:.2e} | offsets per seed (w=100): "
          f"{ {s: h[:4] for s, h in per_seed.items()} } | offsets common to all seeds: {sorted(common)}", flush=True)
    res[shift] = {"post_match": float(post_ok), "per_seed": per_seed, "common": sorted(common)}
json.dump(res, open(os.path.join(os.environ["SCORE_CACHE"], f"offsets2_{name}.json"), "w"), indent=1)
