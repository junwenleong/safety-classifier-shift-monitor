"""
Reference window calibration for shift detection.

Collects initial stream records under the known in-distribution regime,
then freezes reference statistics (kernel bandwidth, empirical CDF,
bootstrap MMD null distribution) for use by the MMD and KS detectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from shift_detection_monitor.types import CalibrationError, StreamRecord


@dataclass(frozen=True)
class FrozenReferenceStats:
    """Immutable reference statistics, serializable for reproducibility."""

    kernel_bandwidth: float
    reference_cdf: np.ndarray  # sorted scores for empirical CDF
    reference_embeddings: np.ndarray  # (n_ref, d) after dim reduction
    mmd_null_distribution: np.ndarray  # bootstrap MMD values under H0
    mmd_reference_value: float  # mean of mmd_null_distribution
    pca_components: np.ndarray | None  # PCA projection matrix, None if not applied
    pca_mean: np.ndarray | None  # PCA centering vector
    n_reference: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrozenReferenceStats):
            return NotImplemented
        if self.kernel_bandwidth != other.kernel_bandwidth:
            return False
        if self.mmd_reference_value != other.mmd_reference_value:
            return False
        if self.n_reference != other.n_reference:
            return False
        if not np.array_equal(self.reference_cdf, other.reference_cdf):
            return False
        if not np.array_equal(self.reference_embeddings, other.reference_embeddings):
            return False
        if not np.array_equal(self.mmd_null_distribution, other.mmd_null_distribution):
            return False
        # PCA components
        if (self.pca_components is None) != (other.pca_components is None):
            return False
        if self.pca_components is not None and other.pca_components is not None:
            if not np.array_equal(self.pca_components, other.pca_components):
                return False
        # PCA mean
        if (self.pca_mean is None) != (other.pca_mean is None):
            return False
        if self.pca_mean is not None and other.pca_mean is not None:
            if not np.array_equal(self.pca_mean, other.pca_mean):
                return False
        return True

    def __hash__(self) -> int:
        return hash(
            (
                self.kernel_bandwidth,
                self.mmd_reference_value,
                self.n_reference,
                self.reference_cdf.tobytes(),
                self.reference_embeddings.tobytes(),
            )
        )


def _compute_mmd_squared_unbiased(
    X: np.ndarray, Y: np.ndarray, bandwidth: float
) -> float:
    """Compute unbiased MMD² between two sets of embeddings using Gaussian kernel."""
    m = X.shape[0]
    n = Y.shape[0]

    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    # k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    # XX kernel
    xx_dists = pdist(X, "sqeuclidean")
    xx_kernel = np.exp(-gamma * xx_dists)
    kxx_sum = 2.0 * np.sum(xx_kernel)  # sum of off-diagonal
    kxx_term = kxx_sum / (m * (m - 1)) if m > 1 else 0.0

    # YY kernel
    yy_dists = pdist(Y, "sqeuclidean")
    yy_kernel = np.exp(-gamma * yy_dists)
    kyy_sum = 2.0 * np.sum(yy_kernel)
    kyy_term = kyy_sum / (n * (n - 1)) if n > 1 else 0.0

    # XY kernel (all pairs)
    # ||x_i - y_j||^2 for all i, j
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    sq_dists_xy = np.sum(diff**2, axis=2)
    xy_kernel = np.exp(-gamma * sq_dists_xy)
    kxy_term = 2.0 * np.sum(xy_kernel) / (m * n)

    return kxx_term + kyy_term - kxy_term


class ReferenceWindow:
    """Collects and freezes reference statistics for shift detection."""

    def __init__(
        self,
        min_size: int,
        dim_reduction_threshold: int | None = None,
        n_bootstrap: int = 1000,
    ) -> None:
        self._min_size = min_size
        self._dim_reduction_threshold = dim_reduction_threshold
        self._n_bootstrap = n_bootstrap
        self._records: list[StreamRecord] = []
        self._frozen = False

    def add(self, record: StreamRecord) -> None:
        """Add a record to the reference window.

        Raises CalibrationError if already frozen.
        """
        if self._frozen:
            raise CalibrationError("Cannot add records to a frozen reference window.")
        self._records.append(record)

    def freeze(self) -> FrozenReferenceStats:
        """Freeze reference statistics.

        1. Collect all scores and embeddings from added records
        2. If embeddings exist and dim > dim_reduction_threshold, fit PCA and project
        3. Compute median-heuristic bandwidth on (projected) embeddings
        4. Compute empirical CDF (sorted scores)
        5. Compute bootstrap null distribution for MMD (permutation test, n_bootstrap times)
        6. mmd_reference_value = mean of null distribution
        7. Return FrozenReferenceStats

        Raises CalibrationError if window has fewer records than min_size.
        """
        if len(self._records) < self._min_size:
            raise CalibrationError(
                f"Reference window has {len(self._records)} records, "
                f"but minimum is {self._min_size}."
            )

        self._frozen = True

        # 1. Collect scores and embeddings
        scores = np.array([r.score for r in self._records], dtype=np.float64)
        embeddings_list = [
            r.representation for r in self._records if r.representation is not None
        ]

        has_embeddings = len(embeddings_list) > 0
        pca_components: np.ndarray | None = None
        pca_mean: np.ndarray | None = None

        if has_embeddings:
            embeddings = np.array(embeddings_list, dtype=np.float64)

            # 2. Dimensionality reduction if needed
            if (
                self._dim_reduction_threshold is not None
                and embeddings.shape[1] > self._dim_reduction_threshold
            ):
                n_components = min(
                    self._dim_reduction_threshold, embeddings.shape[0] - 1
                )
                pca = PCA(n_components=n_components)
                embeddings = pca.fit_transform(embeddings)
                pca_components = pca.components_
                pca_mean = pca.mean_

            # 3. Median-heuristic bandwidth
            pairwise_dists = pdist(embeddings, "euclidean")
            bandwidth = float(np.median(pairwise_dists))
            # Avoid zero bandwidth
            if bandwidth == 0.0:
                bandwidth = 1.0

            # 5. Bootstrap null distribution for MMD
            rng = np.random.default_rng(42)
            n = embeddings.shape[0]
            null_mmd_values = np.zeros(self._n_bootstrap, dtype=np.float64)

            for b in range(self._n_bootstrap):
                # Permutation test: shuffle combined pool and split
                perm = rng.permutation(n)
                half = n // 2
                X_perm = embeddings[perm[:half]]
                Y_perm = embeddings[perm[half : half * 2]]
                if X_perm.shape[0] >= 2 and Y_perm.shape[0] >= 2:
                    null_mmd_values[b] = _compute_mmd_squared_unbiased(
                        X_perm, Y_perm, bandwidth
                    )

            # 6. mmd_reference_value = mean of null distribution
            # If near zero, use 75th percentile for a stable non-degenerate reference
            # Clamp to non-negative: MMD² under the null is zero-centered
            mmd_reference_value = float(np.mean(null_mmd_values))
            if abs(mmd_reference_value) < 1e-4:
                mmd_reference_value = float(np.percentile(null_mmd_values, 75))
            mmd_reference_value = max(mmd_reference_value, 0.0)
        else:
            # No embeddings available — use defaults
            embeddings = np.empty((0, 0), dtype=np.float64)
            bandwidth = 1.0
            null_mmd_values = np.zeros(self._n_bootstrap, dtype=np.float64)
            mmd_reference_value = 0.0

        # 4. Empirical CDF (sorted scores)
        reference_cdf = np.sort(scores)

        return FrozenReferenceStats(
            kernel_bandwidth=bandwidth,
            reference_cdf=reference_cdf,
            reference_embeddings=embeddings,
            mmd_null_distribution=null_mmd_values,
            mmd_reference_value=mmd_reference_value,
            pca_components=pca_components,
            pca_mean=pca_mean,
            n_reference=len(self._records),
        )


def save_frozen_stats(stats: FrozenReferenceStats, directory: Path) -> None:
    """Serialize FrozenReferenceStats to a directory with .npy files and metadata JSON."""
    directory.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.save(directory / "reference_cdf.npy", stats.reference_cdf)
    np.save(directory / "reference_embeddings.npy", stats.reference_embeddings)
    np.save(directory / "mmd_null_distribution.npy", stats.mmd_null_distribution)

    if stats.pca_components is not None:
        np.save(directory / "pca_components.npy", stats.pca_components)
    if stats.pca_mean is not None:
        np.save(directory / "pca_mean.npy", stats.pca_mean)

    # Save scalar metadata as JSON
    metadata = {
        "kernel_bandwidth": stats.kernel_bandwidth,
        "mmd_reference_value": stats.mmd_reference_value,
        "n_reference": stats.n_reference,
        "has_pca_components": stats.pca_components is not None,
        "has_pca_mean": stats.pca_mean is not None,
    }
    with open(directory / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_frozen_stats(directory: Path) -> FrozenReferenceStats:
    """Deserialize FrozenReferenceStats from a directory."""
    # Load metadata
    with open(directory / "metadata.json") as f:
        metadata = json.load(f)

    # Load numpy arrays
    reference_cdf = np.load(directory / "reference_cdf.npy")
    reference_embeddings = np.load(directory / "reference_embeddings.npy")
    mmd_null_distribution = np.load(directory / "mmd_null_distribution.npy")

    pca_components = None
    if metadata["has_pca_components"]:
        pca_components = np.load(directory / "pca_components.npy")

    pca_mean = None
    if metadata["has_pca_mean"]:
        pca_mean = np.load(directory / "pca_mean.npy")

    return FrozenReferenceStats(
        kernel_bandwidth=metadata["kernel_bandwidth"],
        reference_cdf=reference_cdf,
        reference_embeddings=reference_embeddings,
        mmd_null_distribution=mmd_null_distribution,
        mmd_reference_value=metadata["mmd_reference_value"],
        pca_components=pca_components,
        pca_mean=pca_mean,
        n_reference=metadata["n_reference"],
    )
