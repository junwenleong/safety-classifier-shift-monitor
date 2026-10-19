"""Recompute the MMD calibration and the null-window false-alarm check.

scripts/run_mmd_evaluation.py prints the null-window FAR (its "Step 3") but does
not save it, so the paper's MMD FAR (DeBERTa 3.3%, Text-Moderation 3.3%,
ShieldGemma 0.0%, Llama Guard 10%) had no stored source. This script imports the
evaluation script's own functions (no re-implementation), reproduces its
calibration, its null-window check and its detection loop on the cached
embeddings, writes the thresholds and per-classifier FAR, and checks that the
detection results match the committed results/mmd_evaluation.json exactly.

No model inference: numpy only. Needs results/cached_streams/ (git-ignored,
about 536 MB) in the working directory.

Usage (from the repository root):
    .venv/bin/python scripts/investigations/mmd_null_far.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mmd_evaluation as ev  # noqa: E402
from shift_detection_monitor.detection.mmd_detector import compute_mmd_squared  # noqa: E402

OUT = ROOT / "results" / "mmd_null_far.json"
SAVED = ROOT / "results" / "mmd_evaluation.json"


def main() -> None:
    pooled = {}
    for clf in ev.CLASSIFIERS:
        parts = []
        for shift in ev.SHIFTS:
            for seed in ev.SEEDS:
                p = ev.CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if not p.exists():
                    continue
                d = np.load(p)
                if "embeddings" not in d:
                    break
                parts.append(d["embeddings"][: ev.SHIFT_ONSET])
        pooled[clf] = np.concatenate(parts)

    out = {"config": {
        "window_size": ev.WINDOW_SIZE, "n_calibration_permutations": ev.N_CALIBRATION_PERMUTATIONS,
        "alpha": ev.TARGET_FAR_ALPHA, "seeds": ev.SEEDS, "shifts": ev.SHIFTS,
    }, "classifiers": {}}

    new_results = []
    for clf in ev.CLASSIFIERS:
        refs = pooled[clf]
        bw = ev.median_bandwidth(refs)
        thr = ev.calibrate_mmd_threshold(refs, bw)

        ref_sample = refs[np.random.default_rng(99).choice(len(refs), 200, replace=False)]
        n_false = n_windows = 0
        per_seed = []
        for seed in ev.SEEDS:
            p = ev.CACHE_DIR / clf / ev.SHIFTS[0] / f"seed_{seed}.npz"
            if not p.exists():
                continue
            ref_portion = np.load(p)["embeddings"][: ev.SHIFT_ONSET]
            nf = nw = 0
            for start in range(ev.WINDOW_SIZE, len(ref_portion) - ev.WINDOW_SIZE, ev.WINDOW_SIZE):
                m = compute_mmd_squared(ref_sample, ref_portion[start: start + ev.WINDOW_SIZE], bw)
                nw += 1
                nf += int(m > thr)
            per_seed.append({"seed": seed, "false": nf, "windows": nw})
            n_false += nf
            n_windows += nw
        out["classifiers"][clf] = {
            "n_pooled_reference": int(len(refs)), "bandwidth": bw, "threshold": thr,
            "null_false_alarms": n_false, "null_windows": n_windows,
            "null_far": n_false / max(n_windows, 1), "per_seed": per_seed,
        }

        for shift in ev.SHIFTS:
            for seed in ev.SEEDS:
                p = ev.CACHE_DIR / clf / shift / f"seed_{seed}.npz"
                if not p.exists():
                    continue
                d = np.load(p)
                r = ev.run_mmd_detection(d["embeddings"], d["is_shifted"], refs, bw, thr)
                new_results.append({"classifier": clf, "shift_condition": shift, "seed": seed, **r})

    saved = json.load(open(SAVED))["results"]
    key = lambda r: (r["classifier"], r["shift_condition"], r["seed"])  # noqa: E731
    a = {key(r): r for r in saved}
    b = {key(r): r for r in new_results}
    mism = [k for k in a if k not in b or a[k] != b[k]]
    out["detection_reproduction"] = {
        "saved_rows": len(a), "recomputed_rows": len(b),
        "mismatching_rows": len(mism) + len([k for k in b if k not in a]),
        "all_latency_equal_window_size": all(
            r["detection_latency"] == ev.WINDOW_SIZE for r in new_results),
    }
    OUT.write_text(json.dumps(out, indent=1))
    for c, v in out["classifiers"].items():
        print(f"{c}: thr={v['threshold']:.6f} bw={v['bandwidth']:.3f} "
              f"FAR={v['null_false_alarms']}/{v['null_windows']}")
    print(out["detection_reproduction"])


if __name__ == "__main__":
    main()
