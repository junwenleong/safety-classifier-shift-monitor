"""AD1 sensitivity to the KS threshold rule, using the committed run_all_pending.py harness settings.

Setup (copied from scripts/run_all_pending.py run_track_b_evaluation): reference = the 500 null scores
of the classifier, shifted pool = Beta(5,5) x 500 (seed 99), 30 seeds, 500 pre-onset steps then 300
post-onset steps with a 50-step ramp to the target mixing rate, KS window 100, warm-up 200,
scan martingale eps=0.3 window=50 alpha=0.05 (package ScanMartingale).

KS threshold rules compared (only this changes):
  committed : per-run, 97th percentile of the max KS over 30 null streams of length 500 (what the script does)
  long-null : 97th percentile of the max KS over 30 null streams of length 800 (the full monitored length),
              i.e. a threshold that controls false alarms over the whole stream
  pooled    : one threshold for all runs, 97th percentile of the max KS over 200 null streams of length 800
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from shift_detection_monitor.detection.conformal_martingale import ScanMartingale  # noqa: E402
from shift_detection_monitor.detection.ks_detector import KSDetector  # noqa: E402
from shift_detection_monitor.detection.reference_window import FrozenReferenceStats  # noqa: E402
from shift_detection_monitor.types import StreamRecord  # noqa: E402

CLF = os.environ.get("AD1_CLF", "deberta")
null_scores = json.load(open("results/null_scores.json"))
N_SEEDS, ONSET, WARM, ALPHA, RAMP = 30, 500, 200, 0.05, 50
ref = np.array(null_scores[CLF])


def frozen_stats(r):
    s = np.sort(np.array(r, dtype=np.float64)); n = len(s)
    return FrozenReferenceStats(kernel_bandwidth=1.0, reference_cdf=s, reference_embeddings=np.zeros((n, 2)),
                                mmd_null_distribution=np.zeros(10), mmd_reference_value=0.0,
                                pca_components=None, pca_mean=None, n_reference=n)


def rec(score, t):
    return StreamRecord(t, "", score, None, None, False, "eval", None)


frozen = frozen_stats(ref)
SHIFT_SRC = os.environ.get("AD1_SHIFT", "beta55")
if SHIFT_SRC == "beta55":
    shifted = np.random.default_rng(99).beta(5, 5, size=500)   # what scripts/run_all_pending.py uses
else:                                                           # real cached shifted scores for CLF x paraphrase
    _d = Path(os.environ.get("CS_STREAMS", "results/cached_streams")) / CLF / SHIFT_SRC
    shifted = np.concatenate([np.load(_d / f"seed_{k}.npz")["scores"][np.load(_d / f"seed_{k}.npz")["is_shifted"]] for k in range(10)])
print("shifted pool:", SHIFT_SRC, "n =", len(shifted), "mean =", round(float(shifted.mean()), 3), "| reference mean =", round(float(ref.mean()), 3))


def stream(mixing, seed):
    rng = np.random.default_rng(seed)
    pre = rng.choice(ref, size=ONSET, replace=True)
    post = []
    for t in range(300):
        p = min(mixing, mixing * t / RAMP)
        post.append(rng.choice(shifted) if rng.random() < p else rng.choice(ref))
    return np.concatenate([pre, np.array(post)])


def run_scan(s):
    d = ScanMartingale(frozen, alpha=ALPHA, window=50, epsilon=0.3)
    for t, x in enumerate(s):
        d.update(rec(x, t))
        if d.alarm_step is not None:
            return (d.alarm_step - ONSET) >= 0
    return False


def ks_max_over_null(length, trial):
    k = KSDetector(frozen_stats=frozen, window_size=100)
    r = np.random.default_rng(trial + 7777)
    mx = 0.0
    for t, x in enumerate(r.choice(frozen.reference_cdf, size=length, replace=True)):
        v = k.update(rec(x, t))
        if t >= WARM and v > mx:
            mx = v
    return mx


thr = {
    "committed (30 null x 500 steps)": float(np.percentile([ks_max_over_null(500, i) for i in range(30)], 97)),
    "long-null (30 null x 800 steps)": float(np.percentile([ks_max_over_null(800, i) for i in range(30)], 97)),
    "pooled (200 null x 800 steps)": float(np.percentile([ks_max_over_null(800, 1000 + i) for i in range(200)], 97)),
}


def run_ks(s, th):
    k = KSDetector(frozen_stats=frozen, window_size=100)
    for t, x in enumerate(s):
        v = k.update(rec(x, t))
        if t >= ONSET and v > th:
            return True
    return False


print(f"classifier {CLF}; KS thresholds:", {k: round(v, 4) for k, v in thr.items()})
print(f"{'mixing':>7s} {'scan':>6s} " + " ".join(f"{k.split(' ')[0]:>10s}" for k in thr))
rows_out = {"mixing": [], "scan": [], "ks_committed": [], "ks_longnull": [], "ks_pooled": [],
            "classifier": CLF, "shifted_pool": SHIFT_SRC, "n_streams": N_SEEDS}
for mix in (0.15, 0.20, 0.25, 0.30):
    streams = [stream(mix, s) for s in range(N_SEEDS)]
    scan = np.mean([run_scan(s) for s in streams])
    ks = [np.mean([run_ks(s, th) for s in streams]) for th in thr.values()]
    print(f"{mix:7.2f} {scan:6.0%} " + " ".join(f"{v:10.0%}" for v in ks))
    rows_out["mixing"].append(mix); rows_out["scan"].append(float(scan))
    for key, v in zip(("ks_committed", "ks_longnull", "ks_pooled"), ks):
        rows_out[key].append(float(v))
_out = ROOT / f"results/cs_engine_investigation/ad1_{SHIFT_SRC}.json"
_out.write_text(json.dumps(rows_out, indent=1))
print("wrote", _out.relative_to(ROOT))
# false alarms of each KS rule on null streams (800 steps) for context
for name, th in thr.items():
    fa = np.mean([run_ks(np.random.default_rng(55000 + i).choice(ref, 800), th) for i in range(100)])
    print(f"KS null false-alarm rate ({name}): {fa:.0%}")
sa = np.mean([run_scan(np.random.default_rng(66000 + i).choice(ref, 800)) for i in range(100)])
print(f"Scan null false-alarm rate: {sa:.0%}")
