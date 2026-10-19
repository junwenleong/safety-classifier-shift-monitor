"""Equal-reference KS vs CS on the 120 cached abrupt-shift streams (4 classifiers x 3 shifts x 10 seeds).

The factorial's KS detector freezes its reference from the first window_size scores of each stream; the
CS engine receives mu0 from the 500 null scores. Here KS is given the same 500 null scores as reference
(per classifier), calibrated with the same rule (97th percentile of the maximum KS over null streams,
50 streams of 500 steps, warm-up 2w as in the factorial path), and CS is the working-tree engine.
Latency = first alarm at or after the shift onset (step 500) minus 500; pre-onset alarms are not counted as detections.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from shift_detection_monitor.detection.confidence_sequence import ConfidenceSequenceEngine  # noqa: E402
from shift_detection_monitor.detection.ks_detector import KSDetector  # noqa: E402
from shift_detection_monitor.detection.reference_window import FrozenReferenceStats  # noqa: E402
from shift_detection_monitor.types import StreamRecord  # noqa: E402

STREAMS = Path(os.environ.get("CS_STREAMS", "results/cached_streams"))
null_scores = json.load(open("results/null_scores.json"))
CLFS = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]
SHIFTS = ["paraphrase", "temporal", "adversarial-suffix"]
W, ONSET = 100, 500


def frozen_from(scores):
    s = np.sort(np.asarray(scores, dtype=np.float64)); n = len(s)
    return FrozenReferenceStats(kernel_bandwidth=1.0, reference_cdf=s, reference_embeddings=np.zeros((n, 2)),
                                mmd_null_distribution=np.zeros(10), mmd_reference_value=0.0,
                                pca_components=None, pca_mean=None, n_reference=n)


def rec(x, t):
    return StreamRecord(t, "", float(x), None, None, False, "eval", None)


def ks_alarm_step(stream, frozen, thr):
    k = KSDetector(frozen_stats=frozen, window_size=W)
    for t, x in enumerate(stream):
        v = k.update(rec(x, t))
        if v > thr and t > 2 * W:
            return t
    return None


def cs_alarm_step(stream, mu0):
    e = ConfidenceSequenceEngine(alpha=0.05, reference_value=mu0, window_mode="growing", tail_bound="bounded",
                                 lower_bound=0.0, upper_bound=1.0, min_warmup_steps=W)
    for t, x in enumerate(stream):
        if e.update(float(x)).alarm:
            return t
    return None


rows = []
for clf in CLFS:
    null = np.array(null_scores[clf]); frozen = frozen_from(null); mu0 = float(null.mean())
    maxes = []
    for i in range(50):
        k = KSDetector(frozen_stats=frozen, window_size=W); mx = 0.0
        for t, x in enumerate(np.random.default_rng(100 + i).choice(null, size=ONSET, replace=True)):
            v = k.update(rec(x, t))
            if t >= 2 * W and v > mx:
                mx = v
        maxes.append(mx)
    thr = float(np.percentile(maxes, 97))
    for shift in SHIFTS:
        ks_l, cs_l, ks_pre = [], [], 0
        for seed in range(10):
            sc = np.load(STREAMS / clf / shift / f"seed_{seed}.npz")["scores"]
            a = ks_alarm_step(sc, frozen, thr); b = cs_alarm_step(sc, mu0)
            if a is not None and a < ONSET: ks_pre += 1
            ks_l.append(None if a is None or a < ONSET else a - ONSET)
            cs_l.append(None if b is None or b < ONSET else b - ONSET)
        kv = [x for x in ks_l if x is not None]; cv = [x for x in cs_l if x is not None]
        rows.append((clf, shift, thr, len(kv), np.mean(kv) if kv else float("nan"), len(cv), np.mean(cv) if cv else float("nan"), ks_pre))

print(f"{'classifier':16s} {'shift':20s} {'KS thr':>7s} {'KS n':>5s} {'KS lat':>7s} {'CS n':>5s} {'CS lat':>7s} {'CS/KS':>6s} {'KS pre-onset alarms':>20s}")
ks_all, cs_all, cs_faster = [], [], 0
for r in rows:
    print(f"{r[0]:16s} {r[1]:20s} {r[2]:7.3f} {r[3]:5d} {r[4]:7.1f} {r[5]:5d} {r[6]:7.1f} {r[6]/r[4]:6.2f} {r[7]:20d}")
    cs_faster += r[6] < r[4]
print(f"\nCS faster in {cs_faster} of 12 cells; pooled mean latency KS {np.nanmean([r[4] for r in rows]):.1f} vs CS {np.nanmean([r[6] for r in rows]):.1f}")
json.dump([dict(zip(("classifier", "shift", "ks_threshold", "ks_n", "ks_latency", "cs_n", "cs_latency", "ks_pre_onset_alarms"), [float(x) if isinstance(x, (np.floating,)) else x for x in r])) for r in rows],
          open(ROOT / "results/cs_engine_investigation/equal_reference_ks_vs_cs.json", "w"), indent=1)
