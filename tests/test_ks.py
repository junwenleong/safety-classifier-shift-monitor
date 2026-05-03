"""Property tests for KSDetector.

Tests Property 13 from the design document.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats as scipy_stats

from shift_detection_monitor.detection.ks_detector import KSDetector
from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import StreamRecord


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_SCORE_FLOAT = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def st_ks_scenario(
    draw: st.DrawFn,
) -> tuple[np.ndarray, list[float], int]:
    """Generate a reference CDF (sorted scores), a list of stream scores,
    and a window size."""
    n_ref = draw(st.integers(min_value=5, max_value=50))
    n_stream = draw(st.integers(min_value=2, max_value=30))
    window_size = draw(st.integers(min_value=n_stream, max_value=max(n_stream, 50)))

    ref_scores = draw(
        arrays(
            dtype=np.float64,
            shape=(n_ref,),
            elements=_SCORE_FLOAT,
        )
    )
    ref_cdf = np.sort(ref_scores)

    stream_scores = [
        draw(_SCORE_FLOAT) for _ in range(n_stream)
    ]

    return ref_cdf, stream_scores, window_size


# ---------------------------------------------------------------------------
# Property 13: KS Statistic Correctness
# ---------------------------------------------------------------------------


@given(scenario=st_ks_scenario())
@settings(max_examples=100)
def test_p13_ks_matches_scipy(
    scenario: tuple[np.ndarray, list[float], int],
) -> None:
    """P13: For any sliding window of scores and reference CDF, computed KS
    statistic matches scipy.stats.ks_1samp within floating-point tolerance.

    **Validates: Requirements 5.1**
    """
    # Feature: shift-detection-monitor, Property 13: KS Statistic Correctness
    ref_cdf, stream_scores, window_size = scenario

    # Build minimal frozen stats with the reference CDF
    rng = np.random.default_rng(0)
    ref_embeddings = rng.standard_normal((5, 2))
    stats = FrozenReferenceStats(
        kernel_bandwidth=1.0,
        reference_cdf=ref_cdf,
        reference_embeddings=ref_embeddings,
        mmd_null_distribution=np.zeros(10),
        mmd_reference_value=0.0,
        pca_components=None,
        pca_mean=None,
        n_reference=len(ref_cdf),
    )

    detector = KSDetector(frozen_stats=stats, window_size=window_size)

    # Feed all stream scores
    last_ks = 0.0
    for i, score in enumerate(stream_scores):
        record = StreamRecord(
            time_step=i,
            text=f"text_{i}",
            score=score,
            representation=None,
            ground_truth_label=None,
            is_shifted=False,
            source_dataset="wildguardmix",
            shift_condition=None,
        )
        last_ks = detector.update(record)

    # Compute expected KS using scipy
    # The window contains all stream_scores (since window_size >= n_stream)
    window_scores = np.array(stream_scores, dtype=np.float64)

    # scipy.stats.ks_1samp compares a sample against a CDF function.
    # We need to define the reference CDF as a function.
    def ref_cdf_func(x: float | np.ndarray) -> float | np.ndarray:
        """Empirical CDF of the reference scores."""
        return np.searchsorted(ref_cdf, x, side="right") / len(ref_cdf)

    scipy_result = scipy_stats.ks_1samp(window_scores, ref_cdf_func)
    expected_ks = scipy_result.statistic

    np.testing.assert_allclose(last_ks, expected_ks, rtol=1e-10, atol=1e-12)
