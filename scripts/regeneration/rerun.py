"""Run scripts/run_factorial.py unmodified, with a memoising classifier layer and the original per-(shift, seed) stream seeds.

The harness derives cond_seed = seed*1000 + hash(shift_cond) % 1000.  String hashing is salted per process, and the original run was split
across processes, so the original offsets differ by seed.  Here hash() returns 0 inside the harness module and run_detection_with_threshold
is wrapped so that cond_seed is replaced by seed*1000 + OFFSET[shift][seed] (recovered by tools/recover_offsets2.py); the negative control keeps
its +1 relation to the positive seed exactly as in the harness.

Environment: SCORE_CACHE (dir with <classifier>.pkl), CLASSIFIERS (comma list), OFFSETS (json {classifier: {shift: {seed: offset}}}).
cwd must be a run directory (new/ or old/).  The harness resumes from results/factorial_results.jsonl in that directory.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shift_detection_monitor  # noqa: E402
import scripts.run_factorial as rf  # noqa: E402
from memo import MemoClassifier  # noqa: E402

print("package:", shift_detection_monitor.__file__, flush=True)
cache_dir = os.environ["SCORE_CACHE"]
OFF = json.load(open(os.environ["OFFSETS"]))
_orig_get = rf.get_classifier
_orig_load_shifted = rf.load_shifted
_orig_run = rf.run_detection_with_threshold
_memos = {}
_path_to_cond = {str(p): c for c, p in rf.SHIFT_CORPORA.items()}
state = {"clf": None, "delta": 0}


class Tagged(list):
    cond = None


def get_classifier(name):
    state["clf"] = name
    m = MemoClassifier(lambda: _orig_get(name), os.path.join(cache_dir, f"{name}.pkl"))
    _memos[name] = m
    return m


def load_shifted(path, n=300):
    out = Tagged(_orig_load_shifted(path, n=n))
    out.cond = _path_to_cond.get(str(path))
    return out


def run_detection(classifier, reference, shifted, shift_onset, window_size, cond_seed, threshold, *a, **k):
    if shifted is not None and getattr(shifted, "cond", None):
        seed = cond_seed // 1000
        real = seed * 1000 + int(OFF[state["clf"]][shifted.cond][str(seed)])
        state["delta"] = real - cond_seed
        cond_seed = real
    else:
        cond_seed = cond_seed + state["delta"]
    return _orig_run(classifier, reference, shifted, shift_onset, window_size, cond_seed, threshold, *a, **k)


rf.get_classifier = get_classifier
rf.load_shifted = load_shifted
rf.run_detection_with_threshold = run_detection
rf.hash = lambda x: 0
rf.CLASSIFIERS = os.environ["CLASSIFIERS"].split(",")
rf.main()
for n, m in _memos.items():
    m.flush()
    print(f"[{n}] cache hits {m.hits}, misses {m.misses}", flush=True)
