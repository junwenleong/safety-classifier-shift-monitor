"""Does the ramp sweep's CS-vs-KS contrast depend on the KS reference size?

scripts/run_ramp_sweep_cached.py gives the KS detector a reference frozen from the FIRST 100 SCORES of
each stream (WINDOW_SIZE) and calibrates one threshold from windows of the pooled reference scores. The CS
detector is given mu0 = mean of the 500 null scores. This script re-runs the same streams, the same
simulator and the same CS engine, and gives KS a reference frozen from the 500 null scores, calibrated the
same way (97th percentile of the maximum KS over N_CAL null streams; here with 500-score reference), so only
the KS reference size differs.

Writes results/cs_engine_investigation/ramp_ks_reference_size.json and prints a table.
"""
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

spec = importlib.util.spec_from_file_location("rs", ROOT / "scripts/run_ramp_sweep_cached.py")
rs = importlib.util.module_from_spec(spec)
sys.modules["rs"] = rs
spec.loader.exec_module(rs)
rs.CACHE_DIR = Path(os.environ.get("CS_STREAMS", "results/cached_streams"))

from shift_detection_monitor.detection.ks_detector import KSDetector  # noqa: E402
from shift_detection_monitor.detection.reference_window import FrozenReferenceStats  # noqa: E402
from shift_detection_monitor.types import StreamRecord  # noqa: E402

all_ref, all_sh = [], []
for s in range(rs.N_SEEDS):
    sc, sh = rs.load_cached_stream(s)
    all_ref.extend(sc[~sh]); all_sh.extend(sc[sh])
null = np.array(json.load(open(rs.NULL_SCORES))[rs.CLASSIFIER])
ref_mean = float(null.mean())


def frozen_from(scores):
    s = np.sort(np.asarray(scores, dtype=np.float64)); n = len(s)
    return FrozenReferenceStats(kernel_bandwidth=1.0, reference_cdf=s, reference_embeddings=np.zeros((n, 2)),
                                mmd_null_distribution=np.zeros(10), mmd_reference_value=0.0,
                                pca_components=None, pca_mean=None, n_reference=n)


def rec(x, t):
    return StreamRecord(t, "", x, None, None, False, "eval", None)


frozen500 = frozen_from(null)


def ks_max_null(trial, length=500):
    k = KSDetector(frozen_stats=frozen500, window_size=rs.WINDOW_SIZE)
    r = np.random.default_rng(900 + trial)
    mx = 0.0
    for t, x in enumerate(r.choice(null, size=length, replace=True)):
        v = k.update(rec(x, t))
        if t >= 2 * rs.WINDOW_SIZE and v > mx:
            mx = v
    return mx


thr500 = float(np.percentile([ks_max_null(i) for i in range(50)], rs.CAL_PCT))
thr100 = float(rs.calibrate_ks_threshold(all_ref))


def ks500(stream):
    """Latency (steps after onset) of the first KS alarm at or after onset, else None."""
    k = KSDetector(frozen_stats=frozen500, window_size=rs.WINDOW_SIZE)
    for t, x in enumerate(stream):
        v = k.update(rec(x, t))
        if t >= rs.SHIFT_ONSET and v > thr500:
            return t - rs.SHIFT_ONSET
    return None


def detected(l):
    return l is not None and l >= 0


rows = {}
print(f"KS threshold with 100-score reference (as in the paper's sweep): {thr100:.4f}")
print(f"KS threshold with 500-score reference (same calibration rule):  {thr500:.4f}\n")
print(f"{'mixing':>7s}  CS (mean latency)  KS100 paper (lat)  KS500 (lat)   detected of 10 streams")
for mix in rs.MIXING_LEVELS:
    L = {"cs": [], "k100": [], "k500": []}
    for seed in range(rs.N_SEEDS):
        sc, sh = rs.load_cached_stream(seed)
        st = rs.simulate_gradual_stream(all_ref[:1000], sc[sh], rs.FAST_RAMP, mix, seed)
        L["cs"].append(rs.run_cs_detection(st, ref_mean)); L["k100"].append(rs.run_ks_detection(st, thr100)); L["k500"].append(ks500(st))
    def summ(v):
        d = [x for x in v if detected(x)]
        return len(d), (round(float(np.mean(d)), 1) if d else None)
    (c, cl), (a, al), (b, bl) = summ(L["cs"]), summ(L["k100"]), summ(L["k500"])
    rows[str(mix)] = {"cs": c, "cs_latency": cl, "ks_ref100": a, "ks_ref100_latency": al, "ks_ref500": b, "ks_ref500_latency": bl, "n": rs.N_SEEDS}
    print(f"{mix:7.1f} {c:3d} ({cl}) {a:6d} ({al}) {b:6d} ({bl})")

# extended 30% test (30 runs: 10 cached + 20 resamples), as in the paper
c30 = a30 = b30 = 0
sims = []
for seed in range(rs.N_SEEDS):
    sc, sh = rs.load_cached_stream(seed); sims.append(rs.simulate_gradual_stream(all_ref[:1000], sc[sh], rs.FAST_RAMP, 0.3, seed))
for extra in range(rs.MIXING_30_EXTRA_SEEDS):
    sc, sh = rs.load_cached_stream(extra % rs.N_SEEDS)
    sims.append(rs.simulate_gradual_stream(all_ref[:1000], sc[sh], rs.FAST_RAMP, 0.3, seed=extra + 100))
lc, la, lb = [], [], []
for st in sims:
    lc.append(rs.run_cs_detection(st, ref_mean)); la.append(rs.run_ks_detection(st, thr100)); lb.append(ks500(st))
c30, a30, b30 = (sum(detected(x) for x in v) for v in (lc, la, lb))
m = lambda v: round(float(np.mean([x for x in v if detected(x)])), 1)
rows["30pct_extended"] = {"cs": c30, "cs_latency": m(lc), "ks_ref100": a30, "ks_ref100_latency": m(la), "ks_ref500": b30, "ks_ref500_latency": m(lb), "n": len(sims)}
print(f"\n30% mixing extended (n={len(sims)}): CS {c30} (lat {m(lc)}), KS(100-score ref, paper) {a30} (lat {m(la)}), KS(500-score ref) {b30} (lat {m(lb)})")
rows["thresholds"] = {"ks_ref100": thr100, "ks_ref500": thr500}
(ROOT / "results/cs_engine_investigation/ramp_ks_reference_size.json").write_text(json.dumps(rows, indent=1))
