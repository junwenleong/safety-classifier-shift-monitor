"""Score every distinct text the factorial can touch for one classifier and store it in the cache.

Texts: the reference examples (first 500 unharmful WildGuardMix prompts after the harness's shuffle),
the negative pool (next 1000), and each shifted corpus (300 examples, loaded as the harness does).
Usage (cwd = a run directory, e.g. ~/factorial_rerun/new):  python ../tools/warm_cache.py <classifier>
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import scripts.run_factorial as rf  # noqa: E402
from scripts.run_full_canary import load_shifted  # noqa: E402
from memo import MemoClassifier, key_of  # noqa: E402
import shift_detection_monitor  # noqa: E402

name = sys.argv[1]
cache_dir = os.environ["SCORE_CACHE"]
print("package:", shift_detection_monitor.__file__)

reference, neg_pool = rf.load_reference()
texts = [e["text"] for e in reference] + [e["text"] for e in neg_pool]
for cond, path in rf.SHIFT_CORPORA.items() if hasattr(rf, "SHIFT_CORPORA") else []:
    shifted = load_shifted(path, n=300)
    print(f"  corpus {cond}: {len(shifted)} examples ({'MISSING -> would fall back to synthetic' if not shifted else 'ok'})")
    texts += [e["text"] for e in shifted]
distinct = list(dict.fromkeys(texts))
print(f"{name}: {len(texts)} texts, {len(distinct)} distinct")

memo = MemoClassifier(lambda: rf.get_classifier(name), os.path.join(cache_dir, f"{name}.pkl"))
t0 = time.time()
todo = [t for t in distinct if key_of(t) not in memo._cache]
print(f"{len(todo)} to score ({len(distinct) - len(todo)} already cached)")
for i, t in enumerate(todo):
    memo.predict(t)
    if (i + 1) % 250 == 0:
        print(f"  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
memo.flush()
print(f"done: {len(memo)} cached, {memo.misses} scored in {time.time() - t0:.0f}s ({memo.infer_seconds / max(1, memo.misses):.3f}s/prompt)")
