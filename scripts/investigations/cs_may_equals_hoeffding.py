"""Investigation script (2026-10-18). Run from the repo root.
Requires the cached score streams (rebuild with scripts/cache_embeddings.py or copy them);
set CS_STREAMS=<dir containing deberta/paraphrase/seed_0.npz ...>. Engine variants under
scripts/investigations/engine_*.py are copies of earlier versions of
shift_detection_monitor/detection/confidence_sequence.py (git: 0380167 and 00d01b1).
"""
import os
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; S = Path(os.environ.get("CS_STREAMS","results/cached_streams"))
nul = json.load(open(ROOT / "results/null_scores.json"))
CL = ["deberta", "text-moderation", "llama-guard", "shieldgemma"]; SH = ["paraphrase", "temporal", "adversarial-suffix"]
def load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m.ConfidenceSequenceEngine
New = load("shift_detection_monitor/detection/confidence_sequence.py", "n1"); May = load("scripts/investigations/engine_cs_may.py", "m1")
ALPHA, WARM = 0.05, 100

def hoeffding_rule(sc, mu0):
    """Alarm when mu0 is outside mean_t +/- sqrt(log(1/alpha)/(2t)) (range 1), after warmup. No wealth."""
    s = 0.0
    for t, x in enumerate(sc, start=1):
        s += x
        if t >= WARM and abs(s / t - mu0) > math.sqrt(math.log(1 / ALPHA) / (2 * t)):
            return t - 1
    return None

def run(Eng, sc, mu0):
    e = Eng(alpha=ALPHA, reference_value=mu0, window_mode="growing", tail_bound="bounded", lower_bound=0.0, upper_bound=1.0, min_warmup_steps=WARM)
    for t, x in enumerate(sc):
        if e.update(float(x)).alarm: return t
    return None

cells = [(c, s, k) for c in CL for s in SH for k in range(10)]
may_alarm, hoef_alarm, new_alarm = {}, {}, {}
for c, s, k in cells:
    mu0 = float(np.mean(nul[c])); sc = np.load(S / c / s / f"seed_{k}.npz")["scores"]
    may_alarm[(c, s, k)] = run(May, sc, mu0); hoef_alarm[(c, s, k)] = hoeffding_rule(sc, mu0); new_alarm[(c, s, k)] = run(New, sc, mu0)
same = sum(may_alarm[k] == hoef_alarm[k] for k in cells)
print(f"May engine alarm == plain Hoeffding-on-running-mean rule in {same}/120 cells")
# FAR on reference-only (first 500 steps), per rule
for name, fn in [("May (Hoeffding)", lambda sc, m: run(May, sc, m)), ("New (wealth, predictable)", lambda sc, m: run(New, sc, m))]:
    al = tot = 0
    for c in CL:
        mu0 = float(np.mean(nul[c]))
        for k in range(10):
            sc = np.load(S / c / "paraphrase" / f"seed_{k}.npz")["scores"][:500]
            tot += 1; al += fn(sc, mu0) is not None
    print(f"reference-only FAR {name}: {al}/{tot}")
# Why wealth is slower than Hoeffding here: effect size per cell
mu0 = float(np.mean(nul["deberta"])); sc = np.load(S / "deberta" / "paraphrase" / "seed_0.npz")["scores"]
print(f"deberta/paraphrase seed0: mu0={mu0:.3f}, pre-onset mean={sc[:500].mean():.3f}, post-onset mean={sc[500:600].mean():.3f}")
