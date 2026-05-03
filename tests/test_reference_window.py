"""Property tests for ReferenceWindow and FrozenReferenceStats.

Tests Properties 3, 4, 5, and 6 from the design document.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.distance import pdist

from shift_detection_monitor.detection.reference_window import (
    FrozenReferenceStats,
    ReferenceWindow,
    load_frozen_stats,
    save_frozen_stats,
)
from shift_detection_monitor.types import CalibrationError, StreamRecord


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_SCORE_FLOAT = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
_SMALL_TEXT = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(codec="utf-8", categories=("L", "N")),
)


@st.composite
def st_records_with_embeddings(
    draw: st.DrawFn,
    min_count: int = 5,
    max_count: int = 30,
    dim: int | None = None,
) -> list[StreamRecord]:
    """Generate a list of StreamRecords all with embeddings of the same dimension."""
    n = draw(st.integers(min_value=min_count, max_value=max_count))
    d = dim if dim is not None else draw(st.integers(min_value=2, max_value=32))
    records = []
    for i in range(n):
        emb = draw(
            arrays(
                dtype=np.float64,
                shape=(d,),
                elements=st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        records.append(
            StreamRecord(
                time_step=i,
                text=f"text_{i}",
                score=draw(_SCORE_FLOAT),
                representation=emb,
                ground_truth_label=draw(st.one_of(st.none(), st.sampled_from([0, 1]))),
                is_shifted=False,
                source_dataset="wildguardmix",
                shift_condition=None,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Property 3: Reference Window Size Invariant
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=5, max_value=30),
    min_size=st.integers(min_value=2, max_value=30),
    dim=st.integers(min_value=2, max_value=16),
    data=st.data(),
)
@settings(max_examples=100)
def test_p3_window_size_invariant(
    n: int, min_size: int, dim: int, data: st.DataObject
) -> None:
    """P3: For any configured size N >= min M, frozen window has exactly N records;
    N < M raises CalibrationError.

    **Validates: Requirements 2.1, 2.4**
    """
    # Feature: shift-detection-monitor, Property 3: Reference Window Size Invariant
    rw = ReferenceWindow(min_size=min_size, n_bootstrap=10)

    for i in range(n):
        emb = data.draw(
            arrays(
                dtype=np.float64,
                shape=(dim,),
                elements=st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        record = StreamRecord(
            time_step=i,
            text=f"text_{i}",
            score=data.draw(_SCORE_FLOAT),
            representation=emb,
            ground_truth_label=None,
            is_shifted=False,
            source_dataset="wildguardmix",
            shift_condition=None,
        )
        rw.add(record)

    if n < min_size:
        try:
            rw.freeze()
            raise AssertionError("Expected CalibrationError was not raised")
        except CalibrationError:
            pass  # Expected
    else:
        stats = rw.freeze()
        assert stats.n_reference == n


# ---------------------------------------------------------------------------
# Property 4: Median Heuristic Bandwidth Correctness
# ---------------------------------------------------------------------------


@given(records=st_records_with_embeddings(min_count=5, max_count=20))
@settings(max_examples=100)
def test_p4_bandwidth_is_median_pairwise_distance(
    records: list[StreamRecord],
) -> None:
    """P4: Frozen bandwidth equals median of pairwise Euclidean distances
    among reference embeddings after dim reduction.

    **Validates: Requirements 2.2**
    """
    # Feature: shift-detection-monitor, Property 4: Median Heuristic Bandwidth Correctness
    rw = ReferenceWindow(min_size=2, n_bootstrap=10)
    for r in records:
        rw.add(r)

    stats = rw.freeze()

    # Recompute expected bandwidth from the (possibly projected) embeddings
    embeddings = stats.reference_embeddings
    if embeddings.shape[0] >= 2 and embeddings.shape[1] > 0:
        expected_bandwidth = float(np.median(pdist(embeddings, "euclidean")))
        if expected_bandwidth == 0.0:
            expected_bandwidth = 1.0
        assert abs(stats.kernel_bandwidth - expected_bandwidth) < 1e-10


# ---------------------------------------------------------------------------
# Property 5: Reference CDF Correctness
# ---------------------------------------------------------------------------


@given(records=st_records_with_embeddings(min_count=5, max_count=20))
@settings(max_examples=100)
def test_p5_cdf_is_empirical_cdf_of_scores(
    records: list[StreamRecord],
) -> None:
    """P5: Frozen CDF is the empirical CDF of reference scores.

    **Validates: Requirements 2.3**
    """
    # Feature: shift-detection-monitor, Property 5: Reference CDF Correctness
    rw = ReferenceWindow(min_size=2, n_bootstrap=10)
    for r in records:
        rw.add(r)

    stats = rw.freeze()

    # The reference_cdf should be the sorted scores
    expected_cdf = np.sort(np.array([r.score for r in records], dtype=np.float64))
    np.testing.assert_array_equal(stats.reference_cdf, expected_cdf)


# ---------------------------------------------------------------------------
# Property 6: Serialization Round-Trip
# ---------------------------------------------------------------------------


@given(records=st_records_with_embeddings(min_count=5, max_count=15))
@settings(max_examples=100)
def test_p6_serialization_round_trip(
    records: list[StreamRecord],
) -> None:
    """P6: Serialization round-trip of FrozenReferenceStats preserves
    numerical fields within 1e-12 relative error.

    **Validates: Requirements 2.5**
    """
    # Feature: shift-detection-monitor, Property 6: Reference Statistics Serialization Round-Trip
    rw = ReferenceWindow(min_size=2, n_bootstrap=10)
    for r in records:
        rw.add(r)

    original = rw.freeze()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "frozen_stats"
        save_frozen_stats(original, path)
        loaded = load_frozen_stats(path)

    # Scalar fields
    assert loaded.n_reference == original.n_reference
    _assert_close_scalar(loaded.kernel_bandwidth, original.kernel_bandwidth)
    _assert_close_scalar(loaded.mmd_reference_value, original.mmd_reference_value)

    # Array fields
    _assert_close_array(loaded.reference_cdf, original.reference_cdf)
    _assert_close_array(loaded.reference_embeddings, original.reference_embeddings)
    _assert_close_array(loaded.mmd_null_distribution, original.mmd_null_distribution)

    # PCA fields
    if original.pca_components is not None:
        assert loaded.pca_components is not None
        _assert_close_array(loaded.pca_components, original.pca_components)
    else:
        assert loaded.pca_components is None

    if original.pca_mean is not None:
        assert loaded.pca_mean is not None
        _assert_close_array(loaded.pca_mean, original.pca_mean)
    else:
        assert loaded.pca_mean is None


def _assert_close_scalar(actual: float, expected: float, rtol: float = 1e-12) -> None:
    """Assert two scalars are close within relative tolerance."""
    if expected == 0.0:
        assert abs(actual) < 1e-12
    else:
        assert abs(actual - expected) / abs(expected) < rtol


def _assert_close_array(
    actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-12
) -> None:
    """Assert two arrays are close within relative tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-15)
