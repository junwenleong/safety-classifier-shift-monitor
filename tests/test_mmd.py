"""Property tests for MMDDetector and compute_mmd_squared.

Tests Properties 10, 11, and 12 from the design document.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.distance import cdist, pdist, squareform

from shift_detection_monitor.detection.mmd_detector import (
    MMDDetector,
    compute_mmd_squared,
)
from shift_detection_monitor.detection.reference_window import (
    FrozenReferenceStats,
    ReferenceWindow,
)
from shift_detection_monitor.types import StreamRecord


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def st_two_embedding_sets(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate two embedding sets X, Y and a positive bandwidth σ."""
    d = draw(st.integers(min_value=2, max_value=16))
    m = draw(st.integers(min_value=2, max_value=20))
    n = draw(st.integers(min_value=2, max_value=20))
    sigma = draw(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    X = draw(
        arrays(
            dtype=np.float64,
            shape=(m, d),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    Y = draw(
        arrays(
            dtype=np.float64,
            shape=(n, d),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return X, Y, sigma


def _analytical_mmd_squared(
    X: np.ndarray, Y: np.ndarray, bandwidth: float
) -> float:
    """Reference implementation of unbiased MMD² using the analytical formula.

    MMD²_u = (1/(m(m-1))) Σ_{i≠j} k(x_i, x_j)
           + (1/(n(n-1))) Σ_{i≠j} k(y_i, y_j)
           - (2/(mn)) Σ_{i,j} k(x_i, y_j)
    """
    m = X.shape[0]
    n = Y.shape[0]
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    # XX kernel matrix
    K_XX = np.exp(-gamma * squareform(pdist(X, "sqeuclidean")))
    # Zero diagonal for unbiased estimator
    np.fill_diagonal(K_XX, 0.0)
    kxx_term = np.sum(K_XX) / (m * (m - 1))

    # YY kernel matrix
    K_YY = np.exp(-gamma * squareform(pdist(Y, "sqeuclidean")))
    np.fill_diagonal(K_YY, 0.0)
    kyy_term = np.sum(K_YY) / (n * (n - 1))

    # XY kernel matrix
    K_XY = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    kxy_term = 2.0 * np.sum(K_XY) / (m * n)

    return float(kxx_term + kyy_term - kxy_term)


# ---------------------------------------------------------------------------
# Property 10: MMD Gaussian Kernel Correctness
# ---------------------------------------------------------------------------


@given(data=st_two_embedding_sets())
@settings(max_examples=100)
def test_p10_mmd_matches_analytical_formula(
    data: tuple[np.ndarray, np.ndarray, float],
) -> None:
    """P10: For any two embedding sets X, Y and bandwidth σ, computed MMD²
    matches the analytical formula within floating-point tolerance.

    **Validates: Requirements 4.1**
    """
    # Feature: shift-detection-monitor, Property 10: MMD Gaussian Kernel Correctness
    X, Y, sigma = data
    computed = compute_mmd_squared(X, Y, sigma)
    expected = _analytical_mmd_squared(X, Y, sigma)
    np.testing.assert_allclose(computed, expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Property 11: MMD Dimensionality Reduction Consistency
# ---------------------------------------------------------------------------


@given(
    n_records=st.integers(min_value=10, max_value=25),
    high_dim=st.integers(min_value=20, max_value=40),
    threshold=st.integers(min_value=4, max_value=10),
    data=st.data(),
)
@settings(max_examples=100)
def test_p11_pca_applied_before_bandwidth(
    n_records: int,
    high_dim: int,
    threshold: int,
    data: st.DataObject,
) -> None:
    """P11: For embeddings exceeding dim threshold, PCA is applied before
    bandwidth computation; same projection applied to stream embeddings;
    projected dim equals target.

    **Validates: Requirements 4.4, 4.5**
    """
    # Feature: shift-detection-monitor, Property 11: MMD Dimensionality Reduction Consistency
    # Ensure high_dim > threshold so PCA is triggered
    actual_dim = max(high_dim, threshold + 1)

    rw = ReferenceWindow(
        min_size=2,
        dim_reduction_threshold=threshold,
        n_bootstrap=10,
    )

    records = []
    for i in range(n_records):
        emb = data.draw(
            arrays(
                dtype=np.float64,
                shape=(actual_dim,),
                elements=st.floats(
                    min_value=-5.0,
                    max_value=5.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        record = StreamRecord(
            time_step=i,
            text=f"text_{i}",
            score=data.draw(
                st.floats(
                    min_value=0.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            ),
            representation=emb,
            ground_truth_label=None,
            is_shifted=False,
            source_dataset="wildguardmix",
            shift_condition=None,
        )
        records.append(record)
        rw.add(record)

    stats = rw.freeze()

    # PCA should have been applied
    assert stats.pca_components is not None
    assert stats.pca_mean is not None

    # Projected dimensionality should equal min(threshold, n_records - 1)
    expected_dim = min(threshold, n_records - 1)
    assert stats.reference_embeddings.shape[1] == expected_dim

    # The MMDDetector should apply the same projection to stream embeddings
    detector = MMDDetector(frozen_stats=stats, window_size=5)

    # Feed a new stream record with the original high dimensionality
    new_emb = data.draw(
        arrays(
            dtype=np.float64,
            shape=(actual_dim,),
            elements=st.floats(
                min_value=-5.0,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    new_record = StreamRecord(
        time_step=n_records,
        text="stream_text",
        score=0.5,
        representation=new_emb,
        ground_truth_label=None,
        is_shifted=True,
        source_dataset="wildguardmix",
        shift_condition="paraphrase",
    )

    result = detector.update(new_record)
    # Should return a float (not None), meaning the projection was applied
    assert result is not None
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Property 12: MMD Handles Missing Representations
# ---------------------------------------------------------------------------


@given(
    time_step=st.integers(min_value=0, max_value=10000),
    score=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_p12_none_representation_returns_none(
    time_step: int,
    score: float,
) -> None:
    """P12: For StreamRecords with representation=None, MMDDetector returns
    None without error.

    **Validates: Requirements 14.3**
    """
    # Feature: shift-detection-monitor, Property 12: MMD Handles Missing Representations
    # Create minimal frozen stats
    rng = np.random.default_rng(42)
    ref_embeddings = rng.standard_normal((10, 4))
    stats = FrozenReferenceStats(
        kernel_bandwidth=1.0,
        reference_cdf=np.sort(rng.random(10)),
        reference_embeddings=ref_embeddings,
        mmd_null_distribution=rng.random(100),
        mmd_reference_value=0.01,
        pca_components=None,
        pca_mean=None,
        n_reference=10,
    )

    detector = MMDDetector(frozen_stats=stats, window_size=5)

    record = StreamRecord(
        time_step=time_step,
        text="test text",
        score=score,
        representation=None,
        ground_truth_label=None,
        is_shifted=False,
        source_dataset="wildguardmix",
        shift_condition=None,
    )

    result = detector.update(record)
    assert result is None
