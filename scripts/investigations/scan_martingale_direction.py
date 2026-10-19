"""Does the sub-threshold experiment's ScanMartingaleDetector respond to an upward shift?

Uses the experiment's own class (imported from scripts/exp_subthreshold_detection.py) on synthetic
streams: a reference of 500 scores, then a post-onset segment drawn from the reference
(null), shifted UP (scores increase) or shifted DOWN. Reports the final log-martingale and whether
the detector fires within 1,000 post-onset steps.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("sub", ROOT / "scripts/exp_subthreshold_detection.py")
sub = importlib.util.module_from_spec(spec)
sys.modules["sub"] = sub
spec.loader.exec_module(sub)

rng = np.random.default_rng(3)
ref = np.clip(rng.normal(0.2, 0.15, 500), 0, 1)
det = sub.ScanMartingaleDetector(epsilon=0.3, threshold=20.0)

def final_log_martingale(post):
    ref_ = ref
    log_m = 0.0
    for x in post:
        p = max(float(np.mean(ref_ <= x)), 1e-10)
        log_m += np.log(max(0.3 * p ** (0.3 - 1), 1e-300))
    return log_m

def fires(post):
    return det.run(np.concatenate([ref, post]), onset=500).detected

N = 1000
cases = {
    "null (resample reference)": rng.choice(ref, N),
    "UP shift (+0.3 on every score)": np.clip(rng.choice(ref, N) + 0.3, 0, 1),
    "UP shift, 30% mixing": np.where(rng.random(N) < 0.3, np.clip(rng.choice(ref, N) + 0.4, 0, 1), rng.choice(ref, N)),
    "DOWN shift (-0.15 on every score)": np.clip(rng.choice(ref, N) - 0.15, 0, 1),
}
print(f"{'post-onset segment':36s} {'final log M':>12s} {'fires (thr 20)':>15s}")
for name, post in cases.items():
    print(f"{name:36s} {final_log_martingale(post):12.1f} {str(fires(post)):>15s}")
