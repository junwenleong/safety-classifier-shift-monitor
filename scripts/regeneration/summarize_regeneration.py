"""Summarise the regenerated factorial: detector-only effect (committed KS vs corrected KS on identical cached scores and streams)
and the comparison with the original May-2026 rows.  Writes a JSON summary next to the inputs.

Usage: python summarize_regeneration.py ORIGINAL.jsonl OLD.jsonl NEW.jsonl OUT.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

orig_p, old_p, new_p, out_p = sys.argv[1:5]
k = lambda r: (r["classifier"], r["shift_condition"], r["seed"], r["window_size"])
load = lambda p: {k(r): r for r in map(json.loads, open(p)) if r}
O, A, B = load(orig_p), load(old_p), load(new_p)  # original, committed-KS regeneration, corrected-KS regeneration
assert set(O) == set(A) == set(B) and len(O) == 800
valid = lambda r: r["detection_latency"] is not None and r["detection_latency"] >= 0 and r["neg_clean"]
STABLE = ["paraphrase", "code-switch", "compositional-long-context", "temporal"]
out = {}


def stats(rows):
    rs = list(rows)
    v = [r for r in rs if valid(r)]
    return {"cells": len(rs), "valid": len(v), "valid_rate": len(v) / len(rs),
            "raw_tpr": float(np.mean([r["detection_latency"] is not None and r["detection_latency"] >= 0 for r in rs])),
            "far": float(np.mean([not r["neg_clean"] for r in rs])),
            "mean_latency": float(np.mean([r["detection_latency"] for r in v])),
            "w100_latency": float(np.mean([r["detection_latency"] for r in v if r["window_size"] == 100])),
            "w200_latency": float(np.mean([r["detection_latency"] for r in v if r["window_size"] == 200]))}


for name, rows in (("original", O), ("regenerated_committed_ks", A), ("regenerated_corrected_ks", B)):
    out[name] = {"all": stats(rows.values()), "stable_corpora": stats(r for r in rows.values() if r["shift_condition"] in STABLE),
                 "adversarial_suffix": stats(r for r in rows.values() if r["shift_condition"] == "adversarial-suffix"),
                 "by_classifier": {c: stats(r for r in rows.values() if r["classifier"] == c) for c in sorted({k_[0] for k_ in rows})}}

same = lambda a, b: all(a[f] == b[f] for f in ("alarm_step", "detection_latency", "neg_clean"))
chg = [x for x in A if not same(A[x], B[x])]
by = defaultdict(int)
for x in chg:
    by[x[0]] += 1
dl = [B[x]["detection_latency"] - A[x]["detection_latency"] for x in chg if A[x]["detection_latency"] is not None and B[x]["detection_latency"] is not None]
out["detector_only"] = {"cells_changed": len(chg), "by_classifier": dict(by),
                        "latency_delta_mean_on_changed": float(np.mean(dl)), "latency_delta_min": int(min(dl)), "latency_delta_max": int(max(dl)),
                        "detection_status_changed": sum((A[x]["detection_latency"] is None) != (B[x]["detection_latency"] is None) for x in chg),
                        "neg_clean_changed": sum(A[x]["neg_clean"] != B[x]["neg_clean"] for x in chg),
                        "valid_changed": sum(valid(A[x]) != valid(B[x]) for x in chg),
                        "thresholds": {f"{c}|{w}": {"committed": sorted({A[x]["threshold"] for x in A if x[0] == c and x[3] == w})[0],
                                                    "corrected": sorted({B[x]["threshold"] for x in B if x[0] == c and x[3] == w})[0]}
                                       for c in sorted({x[0] for x in A}) for w in (100, 200)}}
st = [x for x in O if x[1] in STABLE]
out["reproduction"] = {
    "stable_corpora_cells": len(st),
    "committed_ks_identical_to_original": sum(same(O[x], A[x]) for x in st),
    "corrected_ks_identical_to_original": sum(same(O[x], B[x]) for x in st),
    "adversarial_cells": 160,
    "adversarial_committed_ks_identical_to_original": sum(same(O[x], A[x]) for x in O if x[1] == "adversarial-suffix"),
    "max_abs_pre_score_diff_committed_ks": float(max(abs(O[x]["pre_score"] - A[x]["pre_score"]) for x in O)),
    "max_abs_post_score_diff_stable": float(max(abs(O[x]["post_score"] - A[x]["post_score"]) for x in st)),
}
cell = lambda rows, c, s: float(np.mean([r["detection_latency"] for r in rows.values() if r["classifier"] == c and r["shift_condition"] == s and valid(r)]))
out["cell_latency"] = {f"{c}|{s}": {"original": cell(O, c, s), "committed": cell(A, c, s), "corrected": cell(B, c, s)}
                       for c in sorted({x[0] for x in O}) for s in sorted({x[1] for x in O})}
json.dump(out, open(out_p, "w"), indent=1)
print(json.dumps({k_: out[k_] for k_ in ("detector_only", "reproduction")}, indent=1)[:2600])
print("max |cell mean latency change| corrected vs committed:", max(abs(v["corrected"] - v["committed"]) for v in out["cell_latency"].values()))
print("max |cell mean latency change| corrected vs original :", max(abs(v["corrected"] - v["original"]) for v in out["cell_latency"].values()))
