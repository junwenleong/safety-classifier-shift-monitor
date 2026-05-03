"""
KS-based shift detector.

Computes the one-sample Kolmogorov-Smirnov statistic comparing a sliding
window of stream scores against the frozen reference CDF.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import StreamRecord


class KSDetector:
    """Computes one-sample KS statistic comparing sliding window scores
    against the frozen reference CDF.

    Parameters
    ----------
    frozen_stats : FrozenReferenceStats
        Frozen reference statistics containing the sorted reference scores
        (reference_cdf).
    window_size : int
        Size of the sliding window for stream scores.
    """

    def __init__(
        self,
        frozen_stats: FrozenReferenceStats,
        window_size: int,
    ) -> None:
        self._reference_cdf = frozen_stats.reference_cdf  # sorted reference scores
        self._n_ref = len(self._reference_cdf)
        self._window_size = window_size
        self._window: deque[float] = deque()

    def update(self, record: StreamRecord) -> float:
        """Add a record's score to the sliding window and compute KS statistic.

        Returns the KS statistic D_n = sup_x |F_n(x) - F_ref(x)|.
        """
        self._window.append(record.score)
        if len(self._window) > self._window_size:
            self._window.popleft()

        return self._compute_ks_statistic()

    def _compute_ks_statistic(self) -> float:
        """Compute the one-sample KS statistic.

        D_n = sup_x |F_n(x) - F_ref(x)|

        Uses the standard algorithm matching scipy.stats.ks_1samp:
        For sorted sample values x_1 <= ... <= x_n:
            D+ = max_i (i/n - F_ref(x_i))
            D- = max_i (F_ref(x_i) - (i-1)/n)
            D  = max(D+, D-)

        where F_ref is the empirical CDF of the reference scores.
        """
        window_sorted = np.sort(np.array(list(self._window), dtype=np.float64))
        n = len(window_sorted)

        if n == 0:
            return 0.0

        # Compute F_ref(x_i) for each window value using the reference CDF
        # F_ref(x) = (number of reference scores <= x) / n_ref
        cdf_vals = np.searchsorted(self._reference_cdf, window_sorted, side="right") / self._n_ref

        # D+ = max(i/n - F_ref(x_i)) for i = 1..n
        d_plus = np.max(np.arange(1, n + 1) / n - cdf_vals)

        # D- = max(F_ref(x_i) - (i-1)/n) for i = 1..n
        d_minus = np.max(cdf_vals - np.arange(0, n) / n)

        return float(max(d_plus, d_minus))
