"""Investigation script (2026-10-18). Run from the repo root.
Requires the cached score streams (rebuild with scripts/cache_embeddings.py or copy them);
set CS_STREAMS=<dir containing deberta/paraphrase/seed_0.npz ...>. Engine variants under
scripts/investigations/engine_*.py are copies of earlier versions of
shift_detection_monitor/detection/confidence_sequence.py (git: 0380167 and 00d01b1).
"""
import importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

def load_cls(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m.ConfidenceSequenceEngine
ENG = {"may": load_cls("scripts/investigations/engine_cs_may.py", "e1"), "sep27": load_cls("scripts/investigations/engine_cs_sep27.py", "e2"), "new": load_cls("shift_detection_monitor/detection/confidence_sequence.py", "e3")}

spec = importlib.util.spec_from_file_location("rs", ROOT / "scripts/run_ramp_sweep_cached.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
rs.CACHE_DIR = Path(os.environ.get("CS_STREAMS","results/cached_streams"))

all_scores, all_ref, all_sh = [], [], []
for s in range(rs.N_SEEDS):
    sc, sh = rs.load_cached_stream(s); all_ref.extend(sc[~sh]); all_sh.extend(sc[sh])
ref_mean = float(np.mean(json.load(open(rs.NULL_SCORES))[rs.CLASSIFIER]))

def cs_lat(stream):
    return rs.run_cs_detection(stream, ref_mean)

committed = json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", "662f359:results/ramp_rate_sweep.json"]))

def summarize(lats):
    v = [l for l in lats if l is not None and l >= 0]
    pre = [l for l in lats if l is not None and l < 0]
    return len(v), (float(np.mean(v)) if v else None), len(pre)

out = {}
for name, Eng in ENG.items():
    rs.ConfidenceSequenceEngine = Eng
    r = {}
    # Part 2: mixing levels at ramp 50, 10 seeds
    for mix in rs.MIXING_LEVELS:
        lats = []
        for seed in range(rs.N_SEEDS):
            sc, sh = rs.load_cached_stream(seed)
            lats.append(cs_lat(rs.simulate_gradual_stream(all_ref[:1000], sc[sh], rs.FAST_RAMP, mix, seed)))
        r[f"mix{mix}"] = summarize(lats)
    # Part 1: ramp durations at mixing 0.5
    for ramp in rs.RAMP_DURATIONS:
        lats = []
        for seed in range(rs.N_SEEDS):
            sc, sh = rs.load_cached_stream(seed)
            lats.append(cs_lat(rs.simulate_gradual_stream(all_ref[:1000], sc[sh], ramp, 0.5, seed)))
        r[f"ramp{ramp}"] = summarize(lats)
    # Part 3: 30% extended n=30
    lats = []
    for seed in range(rs.N_SEEDS):
        sc, sh = rs.load_cached_stream(seed)
        lats.append(cs_lat(rs.simulate_gradual_stream(all_ref[:1000], sc[sh], rs.FAST_RAMP, 0.3, seed)))
    for extra in range(rs.MIXING_30_EXTRA_SEEDS):
        sc, sh = rs.load_cached_stream(extra % rs.N_SEEDS)
        lats.append(cs_lat(rs.simulate_gradual_stream(all_ref[:1000], sc[sh], rs.FAST_RAMP, 0.3, seed=extra + 100)))
    r["mix30_ext_n30"] = summarize(lats)
    out[name] = r

print("committed (June) CS side:")
for k, v in committed["mixing_levels"].items(): print(f"  mix {k}: n_detected {v['cs']['n_detected']}/10 mean_latency {v['cs']['mean_latency']}")
for k, v in committed["ramp_durations"].items(): print(f"  ramp {k}: n_detected {v['cs']['n_detected']}/10 mean_latency {v['cs']['mean_latency']}")
e = committed["mixing_30_extended"]; print(f"  mix30 extended: {e['cs']['n_detected']}/{e['n']} mean_latency {e['cs']['mean_latency']}")
print()
for name, r in out.items():
    print(f"== {name} engine   (n_detected, mean_latency, n_alarms_before_onset)")
    for k, v in r.items(): print(f"   {k:16s} {v}")
