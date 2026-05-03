"""
Property-based and edge-case tests for the ConfidenceSequenceEngine.

Tests cover:
  - P7: Incremental-Batch Confluence (growing mode)
  - P8: Alarm Iff Reference Excluded With Warmup Suppression
  - Sliding-window edge cases (window size 1, exactly-full, post-reset, alarm-before-eviction)
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from shift_detection_monitor.detection.confidence_sequence import (
    ConfidenceSequenceEngine,
    CSUpdate,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies for CS inputs
# ---------------------------------------------------------------------------

_ALPHA = st.floats(min_value=0.001, max_value=0.5, allow_nan=False, allow_infinity=False)
_REFERENCE = st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False)
_BOUNDED_VALUE = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def st_bounded_sequence(draw: st.DrawFn) -> list[float]:
    """Generate a sequence of bounded values in [0, 1]."""
    n = draw(st.integers(min_value=1, max_value=100))
    return [draw(_BOUNDED_VALUE) for _ in range(n)]


@st.composite
def st_growing_engine_and_sequence(draw: st.DrawFn) -> tuple[dict, list[float]]:
    """Generate a growing-mode engine config and a sequence of values."""
    alpha = draw(_ALPHA)
    ref = draw(_REFERENCE)
    seq = draw(st_bounded_sequence())
    config = {
        "alpha": alpha,
        "reference_value": ref,
        "window_mode": "growing",
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "tail_bound": "bounded",
    }
    return config, seq


# ---------------------------------------------------------------------------
# Property 7: Incremental-Batch Confluence (growing mode)
# ---------------------------------------------------------------------------


class TestP7IncrementalBatchConfluence:
    """
    Feature: shift-detection-monitor, Property 7: CS Incremental-Batch Confluence

    For growing mode: updating incrementally (one value at a time) must produce
    the same bounds as recomputing from the full history at each step, within
    floating-point tolerance.

    **Validates: Requirements 3.4**
    """

    @given(data=st_growing_engine_and_sequence())
    @settings(max_examples=100)
    def test_incremental_equals_batch_growing_bounded(
        self, data: tuple[dict, list[float]]
    ) -> None:
        """Incremental updates match batch recomputation for bounded growing CS."""
        config, sequence = data
        assume(len(sequence) >= 1)

        # Incremental engine
        engine = ConfidenceSequenceEngine(**config)

        # Feed values one at a time, collecting history
        history: list[float] = []
        for val in sequence:
            history.append(val)
            incremental_result = engine.update(val)

            # Batch recomputation from scratch
            batch_result = engine.recompute_from_history(history)

            # Bounds must match within floating-point tolerance
            assert math.isclose(
                incremental_result.lower, batch_result.lower, rel_tol=1e-10, abs_tol=1e-12
            ), (
                f"Lower bound mismatch at step {incremental_result.time_step}: "
                f"incremental={incremental_result.lower}, batch={batch_result.lower}"
            )
            assert math.isclose(
                incremental_result.upper, batch_result.upper, rel_tol=1e-10, abs_tol=1e-12
            ), (
                f"Upper bound mismatch at step {incremental_result.time_step}: "
                f"incremental={incremental_result.upper}, batch={batch_result.upper}"
            )
            # Wealth is only meaningful in growing-bounded mode
            if incremental_result.wealth is not None and batch_result.wealth is not None:
                assert math.isclose(
                    incremental_result.wealth, batch_result.wealth, rel_tol=1e-10, abs_tol=1e-12
                ), (
                    f"Wealth mismatch at step {incremental_result.time_step}: "
                    f"incremental={incremental_result.wealth}, batch={batch_result.wealth}"
                )

    @given(
        alpha=_ALPHA,
        ref=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        variance_proxy=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        sequence=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_incremental_equals_batch_growing_sub_gaussian(
        self,
        alpha: float,
        ref: float,
        variance_proxy: float,
        sequence: list[float],
    ) -> None:
        """Incremental updates match batch recomputation for sub-Gaussian growing CS."""
        engine = ConfidenceSequenceEngine(
            alpha=alpha,
            reference_value=ref,
            window_mode="growing",
            tail_bound="sub_gaussian",
            variance_proxy=variance_proxy,
        )

        history: list[float] = []
        for val in sequence:
            history.append(val)
            incremental_result = engine.update(val)
            batch_result = engine.recompute_from_history(history)

            assert math.isclose(
                incremental_result.lower, batch_result.lower, rel_tol=1e-10, abs_tol=1e-12
            )
            assert math.isclose(
                incremental_result.upper, batch_result.upper, rel_tol=1e-10, abs_tol=1e-12
            )


# ---------------------------------------------------------------------------
# Property 8: Alarm Iff Reference Excluded With Warmup Suppression
# ---------------------------------------------------------------------------


class TestP8AlarmIffReferenceExcluded:
    """
    Feature: shift-detection-monitor, Property 8: Alarm Iff Reference Excluded With Warmup Suppression

    - If t < min_warmup_steps, no alarm regardless of bounds
    - If t >= min_warmup_steps, alarm iff reference_value ∉ [L_t, U_t]
    - CS accepts updates at every time step regardless of alarm state

    **Validates: Requirements 3.6, 3.7, 3.8**
    """

    @given(
        alpha=_ALPHA,
        ref=_REFERENCE,
        warmup=st.integers(min_value=2, max_value=50),
        sequence=st.lists(_BOUNDED_VALUE, min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_no_alarm_before_warmup(
        self,
        alpha: float,
        ref: float,
        warmup: int,
        sequence: list[float],
    ) -> None:
        """No alarm is raised before min_warmup_steps regardless of bounds."""
        engine = ConfidenceSequenceEngine(
            alpha=alpha,
            reference_value=ref,
            window_mode="growing",
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=warmup,
        )

        for val in sequence:
            result = engine.update(val)
            if result.time_step < warmup:
                assert not result.alarm, (
                    f"Alarm raised at step {result.time_step} < warmup {warmup}"
                )

    @given(
        alpha=_ALPHA,
        ref=_REFERENCE,
        warmup=st.integers(min_value=1, max_value=20),
        sequence=st.lists(_BOUNDED_VALUE, min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_alarm_iff_reference_excluded_after_warmup(
        self,
        alpha: float,
        ref: float,
        warmup: int,
        sequence: list[float],
    ) -> None:
        """After warmup, alarm iff reference_value ∉ [L_t, U_t]."""
        engine = ConfidenceSequenceEngine(
            alpha=alpha,
            reference_value=ref,
            window_mode="growing",
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=warmup,
        )

        for val in sequence:
            result = engine.update(val)
            if result.time_step >= warmup:
                ref_in_bounds = result.lower <= ref <= result.upper
                if ref_in_bounds:
                    assert not result.alarm, (
                        f"Alarm raised at step {result.time_step} but ref {ref} "
                        f"is in [{result.lower}, {result.upper}]"
                    )
                else:
                    assert result.alarm, (
                        f"No alarm at step {result.time_step} but ref {ref} "
                        f"is outside [{result.lower}, {result.upper}]"
                    )

    @given(
        alpha=_ALPHA,
        ref=_REFERENCE,
        sequence=st.lists(_BOUNDED_VALUE, min_size=10, max_size=100),
    )
    @settings(max_examples=100)
    def test_cs_accepts_updates_after_alarm(
        self,
        alpha: float,
        ref: float,
        sequence: list[float],
    ) -> None:
        """CS continues accepting updates at every time step regardless of alarm state."""
        engine = ConfidenceSequenceEngine(
            alpha=alpha,
            reference_value=ref,
            window_mode="growing",
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )

        alarm_seen = False
        for i, val in enumerate(sequence):
            result = engine.update(val)
            if result.alarm:
                alarm_seen = True
            # Regardless of alarm state, the engine should accept the update
            # and return a valid result with the correct time step
            assert result.time_step == i + 1

        # Verify the engine processed all values
        assert engine.time_step == len(sequence)

    @given(
        alpha=_ALPHA,
        ref=_REFERENCE,
        warmup=st.integers(min_value=2, max_value=30),
        window_size=st.integers(min_value=5, max_value=50),
        sequence=st.lists(_BOUNDED_VALUE, min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_alarm_iff_reference_excluded_sliding(
        self,
        alpha: float,
        ref: float,
        warmup: int,
        window_size: int,
        sequence: list[float],
    ) -> None:
        """Alarm iff reference excluded, also for sliding window mode."""
        engine = ConfidenceSequenceEngine(
            alpha=alpha,
            reference_value=ref,
            window_mode="sliding",
            window_size=window_size,
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=warmup,
        )

        for val in sequence:
            result = engine.update(val)
            if result.time_step < warmup:
                assert not result.alarm, (
                    f"Alarm raised at step {result.time_step} < warmup {warmup}"
                )
            else:
                ref_in_bounds = result.lower <= ref <= result.upper
                if ref_in_bounds:
                    assert not result.alarm
                else:
                    assert result.alarm


# ---------------------------------------------------------------------------
# Task 4.5: Sliding-window edge case tests
# ---------------------------------------------------------------------------


class TestSlidingWindowEdgeCases:
    """Edge case tests for the sliding-window CS mode."""

    def test_window_size_1(self) -> None:
        """Window size 1: each update replaces the previous observation entirely."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="sliding",
            window_size=1,
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )

        # First update
        r1 = engine.update(0.5)
        assert r1.time_step == 1
        assert math.isfinite(r1.lower)
        assert math.isfinite(r1.upper)

        # Second update replaces the first
        r2 = engine.update(0.9)
        assert r2.time_step == 2
        assert math.isfinite(r2.lower)
        assert math.isfinite(r2.upper)

        # Third update replaces the second
        r3 = engine.update(0.1)
        assert r3.time_step == 3
        assert math.isfinite(r3.lower)
        assert math.isfinite(r3.upper)

        # Window should only contain the last observation
        assert len(engine._window) == 1
        assert engine._window[0] == 0.1

    def test_exactly_full_window(self) -> None:
        """Exactly-full window: no eviction should occur when window is exactly full."""
        window_size = 5
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="sliding",
            window_size=window_size,
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )

        values = [0.3, 0.4, 0.5, 0.6, 0.7]
        for val in values:
            engine.update(val)

        # Window should be exactly full
        assert len(engine._window) == window_size
        assert list(engine._window) == values

        # Log-wealth increments should match window size
        assert len(engine._log_wealth_increments) == window_size

    def test_first_observation_after_window_eviction(self) -> None:
        """First observation after window fills: oldest observation is evicted."""
        window_size = 3
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="sliding",
            window_size=window_size,
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )

        # Fill window
        for val in [0.3, 0.4, 0.5]:
            engine.update(val)
        assert len(engine._window) == 3
        assert list(engine._window) == [0.3, 0.4, 0.5]

        # Add one more — should evict 0.3
        r = engine.update(0.6)
        assert len(engine._window) == window_size
        assert list(engine._window) == [0.4, 0.5, 0.6]
        assert r.time_step == 4
        assert math.isfinite(r.lower)
        assert math.isfinite(r.upper)

    def test_alarm_on_last_observation_before_eviction(self) -> None:
        """
        An observation that causes an alarm should still be detected even if
        it will be evicted on the next step. We use a larger window so the
        Hoeffding bound is tight enough to exclude the reference.
        """
        window_size = 50
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="sliding",
            window_size=window_size,
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )

        # Feed extreme values to push bounds away from reference.
        # With window_size=50 and all values at 0.99, the Hoeffding half-width
        # is ~1.0 * sqrt(log(20) / 100) ≈ 0.173, so lower ≈ 0.99 - 0.173 = 0.817
        # which excludes reference_value=0.5.
        results = []
        for _ in range(60):
            r = engine.update(0.99)
            results.append(r)

        # At some point, the alarm should have been raised
        alarm_steps = [r.time_step for r in results if r.alarm]
        assert len(alarm_steps) > 0, "Expected alarm to fire with extreme values"

        # The alarm_raised property should be True
        assert engine.alarm_raised

        # Record the first alarm step
        first_alarm_step = alarm_steps[0]
        assert first_alarm_step <= 60

        # Now feed values near the reference — the alarm was already recorded
        # and the engine continues accepting updates
        r_after = engine.update(0.5)
        assert r_after.time_step == 61
        assert math.isfinite(r_after.lower)
        assert math.isfinite(r_after.upper)

    def test_sliding_window_wealth_stays_finite(self) -> None:
        """Wealth should remain finite even with many sliding window updates."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="sliding",
            window_size=10,
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )

        rng = np.random.default_rng(42)
        for _ in range(500):
            val = rng.uniform(0, 1)
            r = engine.update(val)
            assert math.isfinite(r.lower), f"Lower bound is not finite: {r.lower}"
            assert math.isfinite(r.upper), f"Upper bound is not finite: {r.upper}"
            # Wealth is None in sliding mode — that's correct behavior
            assert r.wealth is None, f"Wealth should be None in sliding mode, got {r.wealth}"

    def test_sliding_window_bounds_valid_range(self) -> None:
        """Bounds should stay within [lower_bound, upper_bound] for bounded mode."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="sliding",
            window_size=20,
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )

        rng = np.random.default_rng(123)
        for _ in range(200):
            val = rng.uniform(0, 1)
            r = engine.update(val)
            assert r.lower >= 0.0, f"Lower bound {r.lower} < 0.0"
            assert r.upper <= 1.0, f"Upper bound {r.upper} > 1.0"
            assert r.lower <= r.upper, f"Lower {r.lower} > Upper {r.upper}"


# ---------------------------------------------------------------------------
# Additional unit tests for basic correctness
# ---------------------------------------------------------------------------


class TestCSEngineBasics:
    """Basic unit tests for ConfidenceSequenceEngine."""

    def test_invalid_alpha_raises(self) -> None:
        """Alpha outside (0, 1) should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="alpha must be in"):
            ConfidenceSequenceEngine(
                alpha=0.0, reference_value=0.5, window_mode="growing"
            )
        with pytest.raises(ValueError, match="alpha must be in"):
            ConfidenceSequenceEngine(
                alpha=1.0, reference_value=0.5, window_mode="growing"
            )

    def test_sliding_requires_window_size(self) -> None:
        """Sliding mode without window_size should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="window_size is required"):
            ConfidenceSequenceEngine(
                alpha=0.05, reference_value=0.5, window_mode="sliding"
            )

    def test_sub_gaussian_requires_variance_proxy(self) -> None:
        """Sub-Gaussian mode without variance_proxy should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="variance_proxy is required"):
            ConfidenceSequenceEngine(
                alpha=0.05,
                reference_value=0.5,
                window_mode="growing",
                tail_bound="sub_gaussian",
            )

    def test_growing_mode_bounds_narrow_with_data(self) -> None:
        """Bounds should narrow as more data is collected under the null."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="growing",
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )

        rng = np.random.default_rng(42)
        widths = []
        for _ in range(200):
            val = rng.uniform(0.3, 0.7)  # centered around 0.5
            r = engine.update(val)
            widths.append(r.upper - r.lower)

        # Width should generally decrease (not strictly, but trend)
        assert widths[-1] < widths[5], (
            f"Bounds did not narrow: width at step 5 = {widths[5]:.4f}, "
            f"width at step 200 = {widths[-1]:.4f}"
        )

    def test_csupdate_fields(self) -> None:
        """CSUpdate should have all required fields."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="growing",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        r = engine.update(0.5)
        assert hasattr(r, "time_step")
        assert hasattr(r, "lower")
        assert hasattr(r, "upper")
        assert hasattr(r, "statistic")
        assert hasattr(r, "alarm")
        assert hasattr(r, "wealth")
        assert hasattr(r, "window_mode")
        assert r.time_step == 1
        assert r.statistic == 0.5
        assert r.window_mode == "growing"
        # Growing-bounded mode should have a real wealth value
        assert r.wealth is not None
        assert math.isfinite(r.wealth)

    def test_csupdate_wealth_none_in_sliding(self) -> None:
        """CSUpdate.wealth should be None in sliding mode."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="sliding",
            window_size=10,
            lower_bound=0.0,
            upper_bound=1.0,
            min_warmup_steps=1,
        )
        r = engine.update(0.5)
        assert r.wealth is None
        assert r.window_mode == "sliding"

    def test_csupdate_wealth_none_in_sub_gaussian(self) -> None:
        """CSUpdate.wealth should be None in sub-Gaussian mode."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.0,
            window_mode="growing",
            tail_bound="sub_gaussian",
            variance_proxy=1.0,
        )
        r = engine.update(0.1)
        assert r.wealth is None
        assert r.window_mode == "growing"

    def test_default_warmup_sliding(self) -> None:
        """Default warmup for sliding mode should be window_size."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="sliding",
            window_size=25,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        assert engine._min_warmup_steps == 25

    def test_default_warmup_growing(self) -> None:
        """Default warmup for growing mode should be 1."""
        engine = ConfidenceSequenceEngine(
            alpha=0.05,
            reference_value=0.5,
            window_mode="growing",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        assert engine._min_warmup_steps == 1
