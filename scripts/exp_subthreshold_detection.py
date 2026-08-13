"""Experiment 2: Sub-threshold persistent contamination detection.

Tests whether CUSUM/EWMA detectors can detect sustained low-rate mixing
that KS and scan martingale miss (the ≤12% blind spot from v4).

Uses cached null scores (results/null_scores.json) and models shifted
distributions from the factorial's observed score statistics.

Pre-registered in docs/pre_registration_v5.md, Experiment 2.

Usage:
    .venv/bin/python scripts/exp_subthreshold_detection.py
    .venv/bin/python scripts/exp_subthreshold_detection.py --quick  # 50 null streams for testing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Configuration (pre-registered)
# ============================================================================

RESULTS_DIR = Path("results/v5_subthreshold")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NULL_SCORES_FILE = Path("results/null_scores.json")

# Pre-registered parameters
CLASSIFIERS = ["deberta"]  # Llama Guard deferred until Mac Studio available
MIXING_RATES = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
STREAM_LENGTH = 5000
PRE_SHIFT_LENGTH = 500
N_NULL_STREAMS = 500
N_SHIFTED_STREAMS = 100
WINDOW_SIZE = 100  # for KS/scan baseline

# CUSUM parameters (grid search for 5% FAR calibration)
CUSUM_K_GRID = [0.25, 0.5, 0.75, 1.0, 1.5]  # allowance (in std units)

# EWMA parameters
EWMA_LAMBDA_GRID = [0.05, 0.1, 0.2]

# Shifted distribution model: DeBERTa scores on paraphrase shift
# From factorial: post_score mean ~0.53, pre_score mean ~0.016
# We model shifted scores as Beta(a, b) fitted to known statistics
SHIFTED_SCORE_PARAMS = {
    "deberta": {
        "paraphrase": {"mean": 0.53, "std": 0.25},  # from factorial post-shift
    },
}

SEED_BASE = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Detectors
# ============================================================================

@dataclass
class DetectionResult:
    detected: bool
    alarm_step: int | None  # step where alarm fired (None if not detected)
    latency: int | None     # steps from onset to alarm


class KSDetector:
    """Sliding-window KS detector (baseline from v4)."""

    def __init__(self, window_size: int, threshold: float):
        self.window_size = window_size
        self.threshold = threshold

    def run(self, stream: np.ndarray, onset: int) -> DetectionResult:
        """Run KS detection on a score stream."""
        ref = stream[:onset]  # pre-shift reference
        for t in range(onset + self.window_size, len(stream)):
            window = stream[t - self.window_size : t]
            ks_stat, _ = stats.ks_2samp(ref[-self.window_size:], window)
            if ks_stat > self.threshold:
                return DetectionResult(True, t, t - onset)
        return DetectionResult(False, None, None)


class ScanMartingaleDetector:
    """Conformal scan martingale (baseline from v4)."""

    def __init__(self, epsilon: float = 0.3, threshold: float = 20.0):
        self.epsilon = epsilon
        self.threshold = threshold

    def run(self, stream: np.ndarray, onset: int) -> DetectionResult:
        """Run scan martingale on a score stream using conformal p-values."""
        # Compute conformal p-values against pre-shift reference
        ref = stream[:onset]
        log_martingale = 0.0

        for t in range(onset, len(stream)):
            # Conformal p-value: rank of current score in reference
            p = np.mean(ref <= stream[t])
            p = max(p, 1e-10)  # avoid log(0)

            # Betting function: epsilon-mixture
            bet = self.epsilon * p ** (self.epsilon - 1)
            log_martingale += np.log(max(bet, 1e-300))

            if log_martingale > np.log(self.threshold):
                return DetectionResult(True, t, t - onset)

        return DetectionResult(False, None, None)


class CUSUMDetector:
    """Page's CUSUM for detecting upward mean shift in scores."""

    def __init__(self, k: float, h: float, mu0: float, sigma0: float):
        """
        k: allowance parameter (slack in std units: k * sigma)
        h: decision threshold
        mu0: in-control mean
        sigma0: in-control std
        """
        self.k = k * sigma0  # convert from std units to raw
        self.h = h
        self.mu0 = mu0

    def run(self, stream: np.ndarray, onset: int) -> DetectionResult:
        """Run CUSUM starting from onset."""
        S_pos = 0.0  # positive CUSUM (detects upward shift)
        S_neg = 0.0  # negative CUSUM (detects downward shift)

        for t in range(onset, len(stream)):
            x = stream[t]
            S_pos = max(0, S_pos + (x - self.mu0) - self.k)
            S_neg = max(0, S_neg - (x - self.mu0) - self.k)

            if S_pos > self.h or S_neg > self.h:
                return DetectionResult(True, t, t - onset)

        return DetectionResult(False, None, None)


class EWMADetector:
    """Exponentially Weighted Moving Average detector."""

    def __init__(self, lam: float, L: float, mu0: float, sigma0: float):
        """
        lam: smoothing parameter (0 < lam <= 1)
        L: control limit width (in sigma_z units)
        mu0: in-control mean
        sigma0: in-control std
        """
        self.lam = lam
        self.L = L
        self.mu0 = mu0
        self.sigma0 = sigma0
        # Asymptotic std of EWMA statistic
        self.sigma_z = sigma0 * np.sqrt(lam / (2 - lam))

    def run(self, stream: np.ndarray, onset: int) -> DetectionResult:
        """Run EWMA starting from onset."""
        z = self.mu0  # initialize at in-control mean

        for t in range(onset, len(stream)):
            z = self.lam * stream[t] + (1 - self.lam) * z
            # Time-varying control limits (exact for finite samples)
            n = t - onset + 1
            sigma_z_t = self.sigma0 * np.sqrt(
                (self.lam / (2 - self.lam)) * (1 - (1 - self.lam) ** (2 * n))
            )
            ucl = self.mu0 + self.L * sigma_z_t
            lcl = self.mu0 - self.L * sigma_z_t

            if z > ucl or z < lcl:
                return DetectionResult(True, t, t - onset)

        return DetectionResult(False, None, None)


class GrowingWindowCSDetector:
    """Growing-window confidence sequence (from v4 paper)."""

    def __init__(self, alpha: float = 0.05, mu0: float = 0.0):
        self.alpha = alpha
        self.mu0 = mu0

    def run(self, stream: np.ndarray, onset: int) -> DetectionResult:
        """Run growing-window CS from onset."""
        cumsum = 0.0
        for t in range(onset, len(stream)):
            n = t - onset + 1
            cumsum += stream[t] - self.mu0
            # Wald-type CS boundary
            boundary = np.sqrt(n * np.log(np.log(max(n, 2)) / self.alpha))
            if abs(cumsum) > boundary:
                return DetectionResult(True, t, t - onset)

        return DetectionResult(False, None, None)


# ============================================================================
# Stream generation
# ============================================================================

def generate_null_stream(
    null_scores: np.ndarray, length: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a null (no-shift) stream by resampling cached scores."""
    return rng.choice(null_scores, size=length, replace=True)


def generate_shifted_stream(
    null_scores: np.ndarray,
    shifted_mean: float,
    shifted_std: float,
    mixing_rate: float,
    length: int,
    pre_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a stream with constant post-onset mixing."""
    stream = np.empty(length)

    # Pre-shift: pure null
    stream[:pre_shift] = rng.choice(null_scores, size=pre_shift, replace=True)

    # Post-shift: mixing
    for t in range(pre_shift, length):
        if rng.random() < mixing_rate:
            # Draw from shifted distribution (truncated normal [0,1])
            s = rng.normal(shifted_mean, shifted_std)
            stream[t] = np.clip(s, 0.0, 1.0)
        else:
            stream[t] = rng.choice(null_scores)

    return stream


# ============================================================================
# Calibration: find thresholds for 5% FAR on null streams
# ============================================================================

def calibrate_detectors(
    null_scores: np.ndarray,
    n_null_streams: int,
    stream_length: int,
    target_far: float = 0.05,
) -> dict:
    """Calibrate detector thresholds to achieve target FAR on null streams."""
    log(f"Calibrating detectors on {n_null_streams} null streams (target FAR={target_far})...")

    mu0 = float(np.mean(null_scores))
    sigma0 = float(np.std(null_scores))
    log(f"  Null distribution: mean={mu0:.6f}, std={sigma0:.6f}")

    # Generate null streams and collect max statistics per detector type
    rng = np.random.default_rng(SEED_BASE)

    # KS: collect max KS stat per null stream
    ks_max_stats = []
    for i in range(min(n_null_streams, 200)):  # KS calibration is expensive
        stream = generate_null_stream(null_scores, stream_length, rng)
        max_ks = 0.0
        ref = stream[:PRE_SHIFT_LENGTH]
        for t in range(PRE_SHIFT_LENGTH + WINDOW_SIZE, len(stream)):
            window = stream[t - WINDOW_SIZE : t]
            ks, _ = stats.ks_2samp(ref[-WINDOW_SIZE:], window)
            max_ks = max(max_ks, ks)
        ks_max_stats.append(max_ks)

    ks_threshold = float(np.percentile(ks_max_stats, 100 * (1 - target_far)))
    log(f"  KS threshold (97.5th pct): {ks_threshold:.4f}")

    # CUSUM: find h that gives target FAR for each k
    cusum_params = {}
    for k in CUSUM_K_GRID:
        alarms = 0
        for i in range(n_null_streams):
            stream = generate_null_stream(null_scores, stream_length, rng)
            # Run with generous h, count how many alarm
            det = CUSUMDetector(k=k, h=50.0, mu0=mu0, sigma0=sigma0)
            # Actually: calibrate h by finding the threshold
            # Collect max CUSUM values under null
            pass
        # Simpler approach: binary search for h
        cusum_params[k] = _calibrate_cusum_h(
            null_scores, k, mu0, sigma0, n_null_streams, stream_length, target_far, rng
        )

    # EWMA: find L that gives target FAR for each lambda
    ewma_params = {}
    for lam in EWMA_LAMBDA_GRID:
        ewma_params[lam] = _calibrate_ewma_L(
            null_scores, lam, mu0, sigma0, n_null_streams, stream_length, target_far, rng
        )

    # Scan martingale: fixed threshold (theoretical)
    scan_threshold = 1.0 / target_far  # = 20 for 5% FAR (Ville's inequality)

    return {
        "ks_threshold": ks_threshold,
        "cusum_params": cusum_params,
        "ewma_params": ewma_params,
        "scan_threshold": scan_threshold,
        "mu0": mu0,
        "sigma0": sigma0,
    }


def _calibrate_cusum_h(
    null_scores, k, mu0, sigma0, n_streams, stream_length, target_far, rng
) -> float:
    """Binary search for CUSUM h that achieves target FAR."""
    h_low, h_high = 0.1, 100.0

    for _ in range(20):  # binary search iterations
        h_mid = (h_low + h_high) / 2
        n_alarms = 0
        for i in range(min(n_streams, 200)):
            stream = generate_null_stream(null_scores, stream_length, rng)
            det = CUSUMDetector(k=k, h=h_mid, mu0=mu0, sigma0=sigma0)
            result = det.run(stream, PRE_SHIFT_LENGTH)
            if result.detected:
                n_alarms += 1
        far = n_alarms / min(n_streams, 200)
        if far > target_far:
            h_low = h_mid
        else:
            h_high = h_mid

    return (h_low + h_high) / 2


def _calibrate_ewma_L(
    null_scores, lam, mu0, sigma0, n_streams, stream_length, target_far, rng
) -> float:
    """Binary search for EWMA L that achieves target FAR."""
    L_low, L_high = 1.0, 10.0

    for _ in range(20):
        L_mid = (L_low + L_high) / 2
        n_alarms = 0
        for i in range(min(n_streams, 200)):
            stream = generate_null_stream(null_scores, stream_length, rng)
            det = EWMADetector(lam=lam, L=L_mid, mu0=mu0, sigma0=sigma0)
            result = det.run(stream, PRE_SHIFT_LENGTH)
            if result.detected:
                n_alarms += 1
        far = n_alarms / min(n_streams, 200)
        if far > target_far:
            L_low = L_mid
        else:
            L_high = L_mid

    return (L_low + L_high) / 2


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(quick: bool = False):
    """Run the full sub-threshold detection experiment."""
    log("=" * 70)
    log("EXPERIMENT 2: Sub-Threshold Persistent Contamination Detection")
    log("Pre-registered: docs/pre_registration_v5.md")
    log("=" * 70)

    # Load null scores
    with open(NULL_SCORES_FILE) as f:
        all_null_scores = json.load(f)

    n_null = N_NULL_STREAMS if not quick else 50
    n_shifted = N_SHIFTED_STREAMS if not quick else 20

    results = {}

    for classifier in CLASSIFIERS:
        log(f"\n{'='*50}")
        log(f"Classifier: {classifier}")
        log(f"{'='*50}")

        null_scores = np.array(all_null_scores[classifier])
        log(f"Null scores: n={len(null_scores)}, mean={null_scores.mean():.6f}, std={null_scores.std():.6f}")

        # Get shifted distribution parameters
        shift_params = SHIFTED_SCORE_PARAMS[classifier]["paraphrase"]
        shifted_mean = shift_params["mean"]
        shifted_std = shift_params["std"]
        log(f"Shifted dist model: mean={shifted_mean}, std={shifted_std}")

        # Calibrate detectors
        cal = calibrate_detectors(null_scores, n_null, STREAM_LENGTH)
        log(f"Calibration complete:")
        log(f"  KS threshold: {cal['ks_threshold']:.4f}")
        log(f"  CUSUM h params: { {k: f'{v:.2f}' for k,v in cal['cusum_params'].items()} }")
        log(f"  EWMA L params: { {k: f'{v:.2f}' for k,v in cal['ewma_params'].items()} }")

        mu0 = cal["mu0"]
        sigma0 = cal["sigma0"]

        # Build detector instances
        detectors = {
            "KS_w100": KSDetector(WINDOW_SIZE, cal["ks_threshold"]),
            "Scan_eps0.3": ScanMartingaleDetector(epsilon=0.3, threshold=cal["scan_threshold"]),
            "GrowingCS": GrowingWindowCSDetector(alpha=0.05, mu0=mu0),
        }

        # Add best CUSUM (pick k with tightest calibration)
        best_k = min(cal["cusum_params"].keys(), key=lambda k: cal["cusum_params"][k])
        detectors[f"CUSUM_k{best_k}"] = CUSUMDetector(
            k=best_k, h=cal["cusum_params"][best_k], mu0=mu0, sigma0=sigma0
        )
        # Also add a conservative CUSUM
        for k in [0.5, 1.0]:
            if k in cal["cusum_params"]:
                detectors[f"CUSUM_k{k}"] = CUSUMDetector(
                    k=k, h=cal["cusum_params"][k], mu0=mu0, sigma0=sigma0
                )

        # Add best EWMA
        for lam in [0.05, 0.1]:
            if lam in cal["ewma_params"]:
                detectors[f"EWMA_lam{lam}"] = EWMADetector(
                    lam=lam, L=cal["ewma_params"][lam], mu0=mu0, sigma0=sigma0
                )

        log(f"\nDetectors: {list(detectors.keys())}")

        # Run on null streams (verify FAR)
        log(f"\nVerifying FAR on {n_null} null streams...")
        far_results = {name: 0 for name in detectors}
        rng_null = np.random.default_rng(SEED_BASE + 1000)

        for i in range(n_null):
            stream = generate_null_stream(null_scores, STREAM_LENGTH, rng_null)
            for name, det in detectors.items():
                result = det.run(stream, PRE_SHIFT_LENGTH)
                if result.detected:
                    far_results[name] += 1

        far_rates = {name: count / n_null for name, count in far_results.items()}
        log(f"FAR rates: {  {k: f'{v:.3f}' for k,v in far_rates.items()} }")

        # Run on shifted streams at each mixing rate
        classifier_results = {"calibration": cal, "far": far_rates, "mixing_rates": {}}

        for mixing_rate in MIXING_RATES:
            log(f"\n  Mixing rate: {mixing_rate*100:.0f}%")
            det_results = {name: {"detected": 0, "latencies": []} for name in detectors}

            rng_shift = np.random.default_rng(SEED_BASE + int(mixing_rate * 10000))

            for i in range(n_shifted):
                stream = generate_shifted_stream(
                    null_scores, shifted_mean, shifted_std,
                    mixing_rate, STREAM_LENGTH, PRE_SHIFT_LENGTH, rng_shift
                )

                for name, det in detectors.items():
                    result = det.run(stream, PRE_SHIFT_LENGTH)
                    if result.detected:
                        det_results[name]["detected"] += 1
                        det_results[name]["latencies"].append(result.latency)

            # Compute summary statistics
            mixing_summary = {}
            for name, dr in det_results.items():
                n_det = dr["detected"]
                rate = n_det / n_shifted
                latencies = dr["latencies"]
                mixing_summary[name] = {
                    "detection_rate": rate,
                    "n_detected": n_det,
                    "n_total": n_shifted,
                    "wilson_ci_low": _wilson_ci(n_det, n_shifted, 0.05)[0],
                    "wilson_ci_high": _wilson_ci(n_det, n_shifted, 0.05)[1],
                    "median_latency": float(np.median(latencies)) if latencies else None,
                    "p90_latency": float(np.percentile(latencies, 90)) if len(latencies) > 1 else None,
                }

            classifier_results["mixing_rates"][str(mixing_rate)] = mixing_summary

            # Log summary
            for name, s in mixing_summary.items():
                det_str = f"{s['detection_rate']*100:.0f}%"
                lat_str = f"lat={s['median_latency']:.0f}" if s['median_latency'] else "n/a"
                log(f"    {name:20s}: {det_str:5s} [{s['wilson_ci_low']:.2f}, {s['wilson_ci_high']:.2f}] {lat_str}")

        results[classifier] = classifier_results

    # Compute MDM (Minimum Detectable Mixing) per detector
    log(f"\n{'='*70}")
    log("MINIMUM DETECTABLE MIXING (80% power, 5% FAR)")
    log("=" * 70)

    for clf, clf_results in results.items():
        log(f"\n{clf}:")
        for det_name in detectors.keys():
            mdm = None
            for rate in sorted(MIXING_RATES):
                rate_str = str(rate)
                if rate_str in clf_results["mixing_rates"]:
                    dr = clf_results["mixing_rates"][rate_str].get(det_name, {})
                    if dr.get("detection_rate", 0) >= 0.80:
                        mdm = rate
                        break
            mdm_str = f"{mdm*100:.0f}%" if mdm else ">30%"
            log(f"  {det_name:20s}: MDM = {mdm_str}")

    # Save results
    output_file = RESULTS_DIR / "subthreshold_detection.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {output_file}")

    return results


def _wilson_ci(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode: 50 null, 20 shifted")
    args = parser.parse_args()
    run_experiment(quick=args.quick)
