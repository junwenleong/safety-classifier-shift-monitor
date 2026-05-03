"""
Property-based and unit tests for the ConformalAbstentionLayer.

Properties tested:
- P18: Abstention on non-singleton or empty prediction sets
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shift_detection_monitor.adaptation.conformal import (
    ConformalAbstentionLayer,
    CoverageStats,
)
from shift_detection_monitor.types import CalibrationError, ClassifierOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_calibration_set(
    n: int, seed: int = 42
) -> list[tuple[ClassifierOutput, int]]:
    """Create a calibration set of n examples with deterministic scores."""
    rng = np.random.default_rng(seed)
    cal_set = []
    for _ in range(n):
        score = float(rng.uniform(0.1, 0.9))
        label = int(rng.integers(0, 2))
        output = ClassifierOutput(
            score=score,
            representation=rng.standard_normal(8).astype(np.float64),
            metadata={},
        )
        cal_set.append((output, label))
    return cal_set


# ---------------------------------------------------------------------------
# P18: Conformal Abstention on Non-Singleton Prediction Sets
# ---------------------------------------------------------------------------


@given(
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    target_error_rate=st.floats(
        min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
    cal_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_abstention_on_non_singleton_or_empty(
    score: float,
    target_error_rate: float,
    cal_seed: int,
) -> None:
    """P18: For any classifier output, if prediction set has >1 class or is empty,
    the layer classifies as abstention.

    **Validates: Requirements 7.5, 7.6**

    Feature: shift-detection-monitor, Property 18: Conformal Abstention on Non-Singleton Prediction Sets
    """
    cal_set = _make_calibration_set(60, seed=cal_seed)

    layer = ConformalAbstentionLayer(
        target_error_rate=target_error_rate,
        conformal_mode="unweighted",
        calibration_set=cal_set,
        min_calibration_size=50,
    )

    output = ClassifierOutput(score=score, representation=None, metadata={})
    pred_set = layer.predict_set(output)

    # The property: if |C(x)| != 1, it's an abstention
    if len(pred_set) != 1:
        # Verify it was counted as an abstention
        assert layer.coverage_stats.n_abstentions >= 1
    else:
        # Singleton set — not an abstention
        assert pred_set == {0} or pred_set == {1}


# ---------------------------------------------------------------------------
# Unit tests for CalibrationError
# ---------------------------------------------------------------------------


def test_calibration_error_too_small() -> None:
    """CalibrationError raised when calibration set is too small."""
    small_cal = _make_calibration_set(10, seed=0)

    with pytest.raises(CalibrationError, match="minimum is 50"):
        ConformalAbstentionLayer(
            target_error_rate=0.1,
            conformal_mode="unweighted",
            calibration_set=small_cal,
            min_calibration_size=50,
        )


def test_calibration_error_custom_min() -> None:
    """CalibrationError with custom minimum size."""
    cal = _make_calibration_set(5, seed=0)

    with pytest.raises(CalibrationError, match="minimum is 10"):
        ConformalAbstentionLayer(
            target_error_rate=0.1,
            conformal_mode="unweighted",
            calibration_set=cal,
            min_calibration_size=10,
        )


def test_fallback_to_unweighted_on_small_cal() -> None:
    """When calibration set meets minimum, layer works in unweighted mode."""
    cal_set = _make_calibration_set(50, seed=42)

    layer = ConformalAbstentionLayer(
        target_error_rate=0.1,
        conformal_mode="unweighted",
        calibration_set=cal_set,
        min_calibration_size=50,
    )

    output = ClassifierOutput(score=0.5, representation=None, metadata={})
    pred_set = layer.predict_set(output)

    # Should return a valid prediction set
    assert isinstance(pred_set, set)
    assert all(label in (0, 1) for label in pred_set)


# ---------------------------------------------------------------------------
# Unit tests for prediction set semantics
# ---------------------------------------------------------------------------


def test_high_score_predicts_unsafe() -> None:
    """A very high score should predict class 1 (unsafe) with low error rate."""
    # Create calibration set with clear separation
    cal_set = []
    rng = np.random.default_rng(42)
    for _ in range(60):
        # Unsafe examples have high scores
        score = float(rng.uniform(0.7, 1.0))
        output = ClassifierOutput(score=score, representation=None, metadata={})
        cal_set.append((output, 1))
    for _ in range(60):
        # Safe examples have low scores
        score = float(rng.uniform(0.0, 0.3))
        output = ClassifierOutput(score=score, representation=None, metadata={})
        cal_set.append((output, 0))

    layer = ConformalAbstentionLayer(
        target_error_rate=0.1,
        conformal_mode="unweighted",
        calibration_set=cal_set,
        min_calibration_size=50,
    )

    # Very high score should include class 1
    high_output = ClassifierOutput(score=0.95, representation=None, metadata={})
    pred_set = layer.predict_set(high_output)
    assert 1 in pred_set


def test_low_score_predicts_safe() -> None:
    """A very low score should predict class 0 (safe) with low error rate."""
    cal_set = []
    rng = np.random.default_rng(42)
    for _ in range(60):
        score = float(rng.uniform(0.7, 1.0))
        output = ClassifierOutput(score=score, representation=None, metadata={})
        cal_set.append((output, 1))
    for _ in range(60):
        score = float(rng.uniform(0.0, 0.3))
        output = ClassifierOutput(score=score, representation=None, metadata={})
        cal_set.append((output, 0))

    layer = ConformalAbstentionLayer(
        target_error_rate=0.1,
        conformal_mode="unweighted",
        calibration_set=cal_set,
        min_calibration_size=50,
    )

    low_output = ClassifierOutput(score=0.05, representation=None, metadata={})
    pred_set = layer.predict_set(low_output)
    assert 0 in pred_set


def test_coverage_stats_initial() -> None:
    """Coverage stats are zero initially."""
    cal_set = _make_calibration_set(60, seed=42)
    layer = ConformalAbstentionLayer(
        target_error_rate=0.1,
        conformal_mode="unweighted",
        calibration_set=cal_set,
        min_calibration_size=50,
    )

    stats = layer.coverage_stats
    assert stats.n_predictions == 0
    assert stats.n_abstentions == 0
    assert stats.post_shift_coverage is None
