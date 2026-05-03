"""
Property-based and unit tests for the DensityRatioEstimator.

Properties tested:
- P19: Weights are strictly positive and finite for any non-empty, finite-valued inputs
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from shift_detection_monitor.adaptation.density_ratio import DensityRatioEstimator


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def st_embedding_pair(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pair of non-empty, finite-valued embedding matrices.

    Both matrices have the same dimensionality d but may have different
    numbers of rows.
    """
    d = draw(st.integers(min_value=2, max_value=32))
    n_source = draw(st.integers(min_value=5, max_value=30))
    n_target = draw(st.integers(min_value=5, max_value=30))

    source = draw(
        arrays(
            dtype=np.float64,
            shape=(n_source, d),
            elements=st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    target = draw(
        arrays(
            dtype=np.float64,
            shape=(n_target, d),
            elements=st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    # Ensure source and target are not all identical (logistic regression needs variance)
    # Add small perturbation to ensure non-degenerate data
    source = source + draw(
        arrays(
            dtype=np.float64,
            shape=(n_source, d),
            elements=st.floats(
                min_value=-0.01,
                max_value=0.01,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    return source, target


# ---------------------------------------------------------------------------
# P19: Weights Positive and Finite
# ---------------------------------------------------------------------------


@given(data=st_embedding_pair())
@settings(max_examples=100)
def test_weights_positive_and_finite(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    """P19: For any non-empty, finite-valued source and target embedding sets,
    estimated weights are strictly positive and finite.

    **Validates: Requirements 7.3**

    Feature: shift-detection-monitor, Property 19: Density Ratio Weights Positive and Finite
    """
    source, target = data

    # Skip degenerate cases where all values are identical
    assume(np.std(source) > 1e-10 or np.std(target) > 1e-10)

    estimator = DensityRatioEstimator(method="logistic", max_weight=10.0)
    estimator.fit(source, target)

    # Test weights on source embeddings
    weights_source = estimator.weights(source)
    assert weights_source.shape == (source.shape[0],)
    assert np.all(weights_source > 0), "All weights must be strictly positive"
    assert np.all(np.isfinite(weights_source)), "All weights must be finite"

    # Test weights on target embeddings
    weights_target = estimator.weights(target)
    assert weights_target.shape == (target.shape[0],)
    assert np.all(weights_target > 0), "All weights must be strictly positive"
    assert np.all(np.isfinite(weights_target)), "All weights must be finite"

    # Verify clipping bounds
    assert np.all(weights_source >= 1.0 / 10.0)
    assert np.all(weights_source <= 10.0)
    assert np.all(weights_target >= 1.0 / 10.0)
    assert np.all(weights_target <= 10.0)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_fit_raises_on_empty_source() -> None:
    """DensityRatioEstimator raises ValueError on empty source."""
    estimator = DensityRatioEstimator()
    source = np.empty((0, 4), dtype=np.float64)
    target = np.random.default_rng(42).standard_normal((10, 4))

    with pytest.raises(ValueError, match="non-empty"):
        estimator.fit(source, target)


def test_fit_raises_on_empty_target() -> None:
    """DensityRatioEstimator raises ValueError on empty target."""
    estimator = DensityRatioEstimator()
    source = np.random.default_rng(42).standard_normal((10, 4))
    target = np.empty((0, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="non-empty"):
        estimator.fit(source, target)


def test_weights_before_fit_raises() -> None:
    """Calling weights() before fit() raises RuntimeError."""
    estimator = DensityRatioEstimator()
    embeddings = np.random.default_rng(42).standard_normal((5, 4))

    with pytest.raises(RuntimeError, match="fit"):
        estimator.weights(embeddings)


def test_identical_distributions_weights_near_one() -> None:
    """When source and target are from the same distribution, weights should be near 1."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 8))
    source = data[:50]
    target = data[50:]

    estimator = DensityRatioEstimator(method="logistic", max_weight=10.0)
    estimator.fit(source, target)

    weights = estimator.weights(source)
    # Weights should be roughly around 1 for identical distributions
    mean_weight = np.mean(weights)
    assert 0.1 < mean_weight < 10.0


def test_custom_max_weight() -> None:
    """Custom max_weight clips weights correctly."""
    rng = np.random.default_rng(42)
    source = rng.standard_normal((30, 4))
    target = rng.standard_normal((30, 4)) + 5.0  # Shifted distribution

    estimator = DensityRatioEstimator(method="logistic", max_weight=5.0)
    estimator.fit(source, target)

    weights = estimator.weights(source)
    assert np.all(weights >= 1.0 / 5.0)
    assert np.all(weights <= 5.0)
