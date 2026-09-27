"""
Long-memory sequential detectors: CUSUM and EWMA.

These close the sub-threshold blind spot identified in the sliding-window
KS detector and the scan martingale: persistent, low-rate score
contamination (roughly <=12% mixing) that neither channel reliably
catches. Ported from scripts/exp_subthreshold_detection.py (the v5
pre-registered experiment, docs/pre_registration_v5.md Experiment 2) into
first-class, streaming-interface detectors matching KSDetector/
MMDDetector, so a reader trying to reproduce README findings #8/#9 from
the shipped package (rather than the standalone script) can find them.

Unlike KSDetector/MMDDetector, these are GROWING statistics: they
accumulate evidence since construction (or the last reset()) rather than
comparing a fixed-size sliding window. This is intentional -- it is
exactly the mechanism that lets them detect weaker, more persistent shifts
that a bounded window under-powers, at the cost of higher detection
latency (see paper.tex's "Minimum detectable mixing" table: CUSUM(k=0.5)
and EWMA(lambda=0.05) both reach 4% MDM at roughly 900-step latency, vs.
KS's 20% MDM at ~365 steps).

Like KSDetector/MMDDetector, update() returns the raw detector statistic,
not an alarm decision -- calibrate an alarm threshold empirically via
null-stream simulation (matching how the rest of this package treats
FAR control) rather than relying on a closed-form guarantee. Both
detectors' own docstrings note this explicitly: Page's CUSUM reset gives
the detector multiple "chances" per horizon, so a naive log(1/alpha)-style
threshold under-corrects for FAR; see conformal_martingale.py's
CUSUMMartingale for a related but distinct betting-based construction
with a different (also heuristic) FAR-correction argument.
"""

from __future__ import annotations

import numpy as np

from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import StreamRecord


class CUSUMDetector:
    """Page's two-sided CUSUM for detecting a shift in the mean score.

    Ported from scripts/exp_subthreshold_detection.py::CUSUMDetector. The
    v5 evaluation used k=0.5 (in std-of-reference units), which achieved a
    minimum detectable mixing (MDM) of 4% at ~930-step latency.

    Parameters
    ----------
    frozen_stats : FrozenReferenceStats
        Used only for its reference_cdf, from which the in-control mean
        (mu0) and standard deviation (sigma0) are computed.
    k : float
        Allowance parameter in units of sigma0 (slack subtracted from each
        increment before accumulating). Larger k requires a larger
        sustained shift before the statistic grows; smaller k detects
        weaker shifts at the cost of higher false-alarm sensitivity to
        noise.
    """

    def __init__(
        self,
        frozen_stats: FrozenReferenceStats,
        k: float = 0.5,
    ) -> None:
        ref = np.asarray(frozen_stats.reference_cdf, dtype=np.float64)
        if ref.size == 0:
            raise ValueError("frozen_stats.reference_cdf must be non-empty")
        self._mu0 = float(np.mean(ref))
        # No epsilon floor here: sigma0 only ever multiplies (k * sigma0),
        # never divides, in the CUSUM recursion, so a zero-variance
        # reference just means k_raw=0 -- safe, and keeps this an exact
        # match to scripts/exp_subthreshold_detection.py's recursion
        # (see test_cusum_matches_original_script_recursion).
        self._sigma0 = float(np.std(ref))
        self.k = k
        self._k_raw = k * self._sigma0
        self._s_pos = 0.0
        self._s_neg = 0.0

    def update(self, record: StreamRecord) -> float:
        """Add one score to the running CUSUM and return max(S+, S-).

        S+ accumulates evidence of a sustained upward shift, S- of a
        sustained downward shift. Both reset to 0 whenever they would go
        negative (Page's rule), so a single outlier cannot accumulate
        indefinitely -- this is what gives CUSUM long memory for
        *persistent* shifts without false-alarming on isolated noise.
        """
        x = record.score
        self._s_pos = max(0.0, self._s_pos + (x - self._mu0) - self._k_raw)
        self._s_neg = max(0.0, self._s_neg - (x - self._mu0) - self._k_raw)
        return max(self._s_pos, self._s_neg)

    def reset(self) -> None:
        """Reset both cumulative sums to zero (e.g. after an alarm)."""
        self._s_pos = 0.0
        self._s_neg = 0.0


class EWMADetector:
    """Exponentially Weighted Moving Average detector.

    Ported from scripts/exp_subthreshold_detection.py::EWMADetector. The
    v5 evaluation used lambda=0.05, which achieved the same 4% minimum
    detectable mixing as CUSUM(k=0.5).

    Parameters
    ----------
    frozen_stats : FrozenReferenceStats
        Used only for its reference_cdf, from which the in-control mean
        (mu0) and standard deviation (sigma0) are computed.
    lam : float
        Smoothing parameter, 0 < lam <= 1. Smaller lam weighs history more
        heavily (longer memory, more sensitive to small sustained shifts,
        slower to react to abrupt ones).
    """

    def __init__(
        self,
        frozen_stats: FrozenReferenceStats,
        lam: float = 0.05,
    ) -> None:
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"lam must be in (0, 1], got {lam}")
        ref = np.asarray(frozen_stats.reference_cdf, dtype=np.float64)
        if ref.size == 0:
            raise ValueError("frozen_stats.reference_cdf must be non-empty")
        self._mu0 = float(np.mean(ref))
        self._sigma0 = float(np.std(ref)) + 1e-12
        self.lam = lam
        self._z = self._mu0
        self._n = 0

    def update(self, record: StreamRecord) -> float:
        """Add one score to the EWMA and return the normalized deviation
        |z - mu0| / sigma_z(n).

        sigma_z(n) is the finite-sample-exact (not merely asymptotic)
        standard deviation of the EWMA statistic after n observations:
        sigma_z(n) = sigma0 * sqrt( lam/(2-lam) * (1 - (1-lam)^(2n)) ),
        which converges to the standard asymptotic EWMA control-limit
        formula sigma0 * sqrt(lam/(2-lam)) as n grows. Returning the
        normalized statistic (rather than the raw z) means a single
        constant threshold L is comparable across the whole stream,
        matching how KSDetector/MMDDetector expose a single comparable
        statistic for external, empirically-calibrated alarm thresholds.
        """
        self._n += 1
        self._z = self.lam * record.score + (1 - self.lam) * self._z
        sigma_z_n = self._sigma0 * np.sqrt(
            (self.lam / (2 - self.lam)) * (1 - (1 - self.lam) ** (2 * self._n))
        )
        sigma_z_n = max(sigma_z_n, 1e-12)
        return abs(self._z - self._mu0) / sigma_z_n

    def reset(self) -> None:
        """Reset the EWMA statistic to the in-control mean (e.g. after an alarm)."""
        self._z = self._mu0
        self._n = 0
