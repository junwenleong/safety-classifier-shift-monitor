"""Investigation script (2026-10-18). Run from the repo root.
Requires the cached score streams (rebuild with scripts/cache_embeddings.py or copy them);
set CS_STREAMS=<dir containing deberta/paraphrase/seed_0.npz ...>. Engine variants under
scripts/investigations/engine_*.py are copies of earlier versions of
shift_detection_monitor/detection/confidence_sequence.py (git: 0380167 and 00d01b1).
"""
import os
import importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; S = Path(os.environ.get("CS_STREAMS","results/cached_streams"))
nul = json.load(open(ROOT / "results/null_scores.json"))
CL = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]; SH = ["paraphrase", "temporal", "adversarial-suffix"]
def load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m.ConfidenceSequenceEngine
E = {"may": load("scripts/investigations/engine_cs_may.py", "m1"), "sep27": load("scripts/investigations/engine_cs_sep27.py", "m2"), "new": load("shift_detection_monitor/detection/confidence_sequence.py", "m3")}
def run(Eng, sc, mu0, **kw):
    e = Eng(alpha=0.05, reference_value=mu0, window_mode="growing", tail_bound="bounded", lower_bound=0.0, upper_bound=1.0, min_warmup_steps=100)
    for t, s in enumerate(sc):
        if e.update(float(s)).alarm: return t
    return None
old_json = {(r["classifier"], r["shift_condition"], r["seed"]): r["alarm_step"] for r in json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", "a42401e:results/cs_growing_window_results.json"]))["results"]}
res = {k: {} for k in E}
for c in CL:
    mu0 = float(np.mean(nul[c]))
    for s in SH:
        for k in range(10):
            sc = np.load(S / c / s / f"seed_{k}.npz")["scores"]
            for name, Eng in E.items(): res[name][(c, s, k)] = run(Eng, sc, mu0)
for name in E:
    match = sum(res[name][key] == old_json[key] for key in old_json)
    pre = sum(1 for v in res[name].values() if v is not None and v < 500)
    lat = np.mean([v - 500 for v in res[name].values() if v is not None and v >= 500])
    print(f"{name:6s}: reproduces committed June JSON in {match:3d}/120 cells | alarms before onset {pre:3d} | mean latency over post-onset alarms {lat:6.1f}")
jun_pre = sum(1 for v in old_json.values() if v < 500)
print(f"committed June JSON: alarms before onset {jun_pre}; mean latency {np.mean([v-500 for v in old_json.values()]):.1f}")
diffs = [(k, old_json[k], res['sep27'][k]) for k in old_json if old_json[k] != res['sep27'][k]][:6]
print("sample (cell, June alarm step, sep27 alarm step):", diffs)
