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
    """Computes the two-sample KS statistic D = sup_x |F_window(x) - F_ref(x)|
    comparing the sliding window's empirical CDF against the frozen reference
    empirical CDF.

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
        """Compute the two-sample KS statistic between the window and reference
        empirical CDFs.

            D = sup_x |F_window(x) - F_ref(x)|

        Both F_window and F_ref are *empirical* CDFs. The supremum of their
        absolute difference is attained at one of the distinct observed values
        across the pooled samples, so we evaluate both right-continuous
        empirical CDFs at every pooled value and take the max absolute gap.
        This handles ties and discrete/coincident jumps correctly (e.g. for
        identical window and reference the statistic is exactly 0).

        Reproducibility note: a prior implementation evaluated D- against the
        right-continuous reference CDF while using (i-1)/n left-limit window
        counts. That mishandles coincident jumps and inflates the statistic
        under ties (it returned 0.5 for an identical window and reference of
        [0, 1]). Safety-classifier scores are highly tied near 0, so that bug
        inflated the measured KS statistic; results produced before this fix
        (committed factorial artefacts) must be regenerated with model
        inference before being compared against the two-empirical-CDF estimand
        defined in the paper. See git history for the superseded version.
        """
        n = len(self._window)
        if n == 0:
            return 0.0

        window_sorted = np.sort(np.array(list(self._window), dtype=np.float64))

        # Evaluate both empirical CDFs at every distinct pooled value and take
        # the sup of their absolute difference (standard two-sample KS).
        pooled = np.unique(
            np.concatenate([window_sorted, self._reference_cdf])
        )
        f_window = np.searchsorted(window_sorted, pooled, side="right") / n
        f_ref = np.searchsorted(self._reference_cdf, pooled, side="right") / self._n_ref

        return float(np.max(np.abs(f_window - f_ref)))
