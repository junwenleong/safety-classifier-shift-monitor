"""
MMD-based shift detector using Gaussian kernel.

Computes the Maximum Mean Discrepancy between frozen reference embeddings
and a sliding window of stream embeddings. Uses the median-heuristic
bandwidth frozen at calibration time.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.distance import pdist

from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import StreamRecord


def compute_mmd_squared(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """Compute unbiased MMD² between two sets of embeddings.

    Uses the Gaussian kernel k(x, y) = exp(-||x - y||² / (2σ²))
    and the unbiased estimator:
        MMD²_u = (1/(m(m-1))) Σ_{i≠j} k(x_i, x_j)
               + (1/(n(n-1))) Σ_{i≠j} k(y_i, y_j)
               - (2/(mn)) Σ_{i,j} k(x_i, y_j)

    Parameters
    ----------
    X : np.ndarray
        Reference embeddings, shape (m, d).
    Y : np.ndarray
        Stream window embeddings, shape (n, d).
    bandwidth : float
        Gaussian kernel bandwidth σ.

    Returns
    -------
    float
        The unbiased MMD² estimate.
    """
    m = X.shape[0]
    n = Y.shape[0]

    if m < 2 or n < 2:
        return 0.0

    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    # XX kernel: sum of off-diagonal entries
    xx_sq_dists = pdist(X, "sqeuclidean")
    kxx_sum = 2.0 * np.sum(np.exp(-gamma * xx_sq_dists))
    kxx_term = kxx_sum / (m * (m - 1))

    # YY kernel: sum of off-diagonal entries
    yy_sq_dists = pdist(Y, "sqeuclidean")
    kyy_sum = 2.0 * np.sum(np.exp(-gamma * yy_sq_dists))
    kyy_term = kyy_sum / (n * (n - 1))

    # XY kernel: all cross-pairs
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    sq_dists_xy = np.sum(diff**2, axis=2)
    kxy_sum = np.sum(np.exp(-gamma * sq_dists_xy))
    kxy_term = 2.0 * kxy_sum / (m * n)

    return float(kxx_term + kyy_term - kxy_term)


class MMDDetector:
    """Computes MMD between reference embeddings and a sliding window of stream embeddings.

    Uses Gaussian kernel with frozen median-heuristic bandwidth.

    Parameters
    ----------
    frozen_stats : FrozenReferenceStats
        Frozen reference statistics containing reference embeddings,
        kernel bandwidth, and optional PCA projection.
    window_size : int
        Size of the sliding window for stream embeddings.
    """

    def __init__(
        self,
        frozen_stats: FrozenReferenceStats,
        window_size: int,
    ) -> None:
        self._frozen_stats = frozen_stats
        self._window_size = window_size
        self._window: deque[np.ndarray] = deque()

    def update(self, record: StreamRecord) -> float | None:
        """Add a record to the sliding window and compute MMD.

        Returns None if the record has no representation vector.
        Returns the MMD² statistic value otherwise.
        """
        if record.representation is None:
            return None

        embedding = record.representation.copy()

        # Apply PCA projection if available
        if (
            self._frozen_stats.pca_components is not None
            and self._frozen_stats.pca_mean is not None
        ):
            embedding = (embedding - self._frozen_stats.pca_mean) @ self._frozen_stats.pca_components.T

        # Add to sliding window
        self._window.append(embedding)
        if len(self._window) > self._window_size:
            self._window.popleft()

        # Need at least 2 samples in the window for unbiased estimator
        if len(self._window) < 2:
            return 0.0

        window_embeddings = np.array(list(self._window), dtype=np.float64)
        return compute_mmd_squared(
            self._frozen_stats.reference_embeddings,
            window_embeddings,
            self._frozen_stats.kernel_bandwidth,
        )
