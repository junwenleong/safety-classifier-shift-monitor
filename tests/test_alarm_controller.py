"""
Property-based tests for the AlarmController.

Tests Properties 14, 15, 16, 17 from the design document.
"""

from __future__ import annotations

import math

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from shift_detection_monitor.detection.alarm_controller import (
    AlarmController,
    AlarmEvent,
)
from shift_detection_monitor.detection.confidence_sequence import CSUpdate


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ALPHA = st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)
_CORRECTION = st.sampled_from(["bonferroni", "sidak"])
_COMBINED_WINDOW = st.one_of(st.none(), st.integers(min_value=1, max_value=100))
_WINDOW_SIZE = st.integers(min_value=5, max_value=50)
_WARMUP = st.integers(min_value=1, max_value=20)


@st.composite
def st_alarm_controller_config(draw: st.DrawFn) -> dict:
    """Generate a valid AlarmController configuration."""
    alpha = draw(_ALPHA)
    correction = draw(_CORRECTION)
    combined_window = draw(_COMBINED_WINDOW)
    window_size = draw(_WINDOW_SIZE)
    warmup = draw(_WARMUP)
    return {
        "alpha": alpha,
        "correction_method": correction,
        "combined_window": combined_window,
        "window_mode": "sliding",
        "window_size": window_size,
        "min_warmup_steps": warmup,
        "tail_bound": "bounded",
        "lower_bound": 0.0,
        "upper_bound": 1.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_controller(config: dict) -> AlarmController:
    """Create an AlarmController from a config dict."""
    return AlarmController(**config)


def _feed_statistic(
    controller: AlarmController,
    name: str,
    value: float,
) -> CSUpdate:
    """Feed a statistic value to a detector's CS engine and report the update."""
    engine = controller._engines[name]
    cs_update = engine.update(value)
    controller.report_update(name, cs_update)
    return cs_update


# ---------------------------------------------------------------------------
# Property 14: Alarm Record Completeness and No Duplicates
# ---------------------------------------------------------------------------


@given(
    config=st_alarm_controller_config(),
    ref_mmd=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    ref_ks=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    n_steps=st.integers(min_value=30, max_value=100),
    shift_value=st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_p14_alarm_record_completeness_and_no_duplicates(
    config: dict,
    ref_mmd: float,
    ref_ks: float,
    n_steps: int,
    shift_value: float,
) -> None:
    """P14: Alarm records contain all required fields; each detector emits at most one alarm per shift.

    **Validates: Requirements 6.1, 6.4**

    For any alarm event, the AlarmRecord SHALL contain all required fields
    (time_step, detector, statistic_value, cs_lower, cs_upper, reference_value)
    with valid values. For any single detected shift, each detector SHALL emit
    at most one alarm.
    """
    # Feature: shift-detection-monitor, Property 14: Alarm Record Completeness and No Duplicates
    controller = _make_controller(config)
    controller.register_detector("mmd", ref_mmd)
    controller.register_detector("ks", ref_ks)

    all_alarms: list[AlarmEvent] = []
    detector_alarm_counts: dict[str, int] = {"mmd": 0, "ks": 0}

    for t in range(n_steps):
        # Feed values that may or may not trigger alarms
        # Use shift_value to push away from reference
        mmd_val = ref_mmd + shift_value
        ks_val = ref_ks + shift_value

        _feed_statistic(controller, "mmd", mmd_val)
        _feed_statistic(controller, "ks", ks_val)

        new_alarms = controller.check_alarms()
        for alarm in new_alarms:
            # Check all required fields are present and valid
            assert isinstance(alarm.time_step, int)
            assert alarm.time_step >= 0
            assert alarm.detector in ("mmd", "ks", "combined")
            assert isinstance(alarm.statistic_value, float)
            assert math.isfinite(alarm.statistic_value)
            assert isinstance(alarm.cs_lower, float)
            assert isinstance(alarm.cs_upper, float)
            assert isinstance(alarm.reference_value, float)
            assert math.isfinite(alarm.reference_value)

            if alarm.detector in ("mmd", "ks"):
                detector_alarm_counts[alarm.detector] += 1

        all_alarms.extend(new_alarms)

    # Each detector emits at most one alarm per shift
    assert detector_alarm_counts["mmd"] <= 1, (
        f"MMD detector emitted {detector_alarm_counts['mmd']} alarms, expected at most 1"
    )
    assert detector_alarm_counts["ks"] <= 1, (
        f"KS detector emitted {detector_alarm_counts['ks']} alarms, expected at most 1"
    )


# ---------------------------------------------------------------------------
# Property 15: Independent Detector Alarms
# ---------------------------------------------------------------------------


@given(
    config=st_alarm_controller_config(),
    ref_value=st.floats(min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False),
    n_steps=st.integers(min_value=30, max_value=80),
)
@settings(max_examples=100)
def test_p15_independent_detector_alarms(
    config: dict,
    ref_value: float,
    n_steps: int,
) -> None:
    """P15: When only one detector exceeds threshold, only that detector alarms.

    **Validates: Requirements 6.2**

    For any stream where only one detector's statistic exceeds its threshold,
    only that detector's alarm SHALL fire. The other detector's CS SHALL
    continue updating without alarming.
    """
    # Feature: shift-detection-monitor, Property 15: Independent Detector Alarms
    controller = _make_controller(config)
    controller.register_detector("mmd", ref_value)
    controller.register_detector("ks", ref_value)

    mmd_alarmed = False
    ks_alarmed = False

    for t in range(n_steps):
        # MMD: feed values far from reference to trigger alarm
        mmd_val = 0.99 if ref_value < 0.5 else 0.01
        # KS: feed values exactly at reference to NOT trigger alarm
        ks_val = ref_value

        mmd_update = _feed_statistic(controller, "mmd", mmd_val)
        ks_update = _feed_statistic(controller, "ks", ks_val)

        new_alarms = controller.check_alarms()
        for alarm in new_alarms:
            if alarm.detector == "mmd":
                mmd_alarmed = True
            elif alarm.detector == "ks":
                ks_alarmed = True

    # KS should not alarm since we fed it the reference value
    assert not ks_alarmed, "KS detector alarmed despite receiving reference values"

    # Verify KS engine continued updating (time_step advanced)
    if "ks" in controller._latest_updates:
        assert controller._latest_updates["ks"].time_step == n_steps


# ---------------------------------------------------------------------------
# Property 16: Combined Advisory Within Window
# ---------------------------------------------------------------------------


@given(
    alpha=_ALPHA,
    correction=_CORRECTION,
    combined_window=st.integers(min_value=1, max_value=50),
    warmup=st.integers(min_value=1, max_value=5),
    window_size=st.integers(min_value=5, max_value=20),
    mmd_alarm_step=st.integers(min_value=10, max_value=40),
    ks_alarm_offset=st.integers(min_value=0, max_value=60),
)
@settings(max_examples=100)
def test_p16_combined_advisory_within_window(
    alpha: float,
    correction: str,
    combined_window: int,
    warmup: int,
    window_size: int,
    mmd_alarm_step: int,
    ks_alarm_offset: int,
) -> None:
    """P16: Combined advisory emitted iff both detectors alarm within combined_advisory_window.

    **Validates: Requirements 6.3**

    For any pair of alarm events from the MMD and KS detectors, a combined
    advisory alarm SHALL be emitted if and only if the absolute difference
    between their alarm time steps is ≤ the configured combined_advisory_window.
    """
    # Feature: shift-detection-monitor, Property 16: Combined Advisory Within Window
    ref_value = 0.5

    controller = AlarmController(
        alpha=alpha,
        correction_method=correction,  # type: ignore[arg-type]
        combined_window=combined_window,
        window_mode="sliding",
        window_size=window_size,
        min_warmup_steps=warmup,
        tail_bound="bounded",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    controller.register_detector("mmd", ref_value)
    controller.register_detector("ks", ref_value)

    total_steps = mmd_alarm_step + ks_alarm_offset + 10
    ks_alarm_step = mmd_alarm_step + ks_alarm_offset

    all_alarms: list[AlarmEvent] = []

    for t in range(total_steps):
        # MMD: push far from reference starting at mmd_alarm_step
        if t >= mmd_alarm_step:
            mmd_val = 0.99
        else:
            mmd_val = ref_value

        # KS: push far from reference starting at ks_alarm_step
        if t >= ks_alarm_step:
            ks_val = 0.99
        else:
            ks_val = ref_value

        _feed_statistic(controller, "mmd", mmd_val)
        _feed_statistic(controller, "ks", ks_val)
        new_alarms = controller.check_alarms()
        all_alarms.extend(new_alarms)

    combined_alarms = [a for a in all_alarms if a.detector == "combined"]
    mmd_alarms = [a for a in all_alarms if a.detector == "mmd"]
    ks_alarms = [a for a in all_alarms if a.detector == "ks"]

    # If both detectors alarmed, check combined advisory logic
    if mmd_alarms and ks_alarms:
        actual_gap = abs(mmd_alarms[0].time_step - ks_alarms[0].time_step)
        if actual_gap <= combined_window:
            assert len(combined_alarms) == 1, (
                f"Expected combined advisory (gap={actual_gap} <= window={combined_window}), "
                f"but got {len(combined_alarms)} combined alarms"
            )
        else:
            assert len(combined_alarms) == 0, (
                f"Expected no combined advisory (gap={actual_gap} > window={combined_window}), "
                f"but got {len(combined_alarms)} combined alarms"
            )
    else:
        # If only one or no detector alarmed, no combined advisory
        assert len(combined_alarms) == 0, (
            f"Combined advisory emitted without both detectors alarming"
        )


# ---------------------------------------------------------------------------
# Property 17: Multiplicity Correction Formula
# ---------------------------------------------------------------------------


@given(
    alpha=st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False),
    correction=_CORRECTION,
    n_detectors=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_p17_multiplicity_correction_formula(
    alpha: float,
    correction: str,
    n_detectors: int,
) -> None:
    """P17: Per-detector α equals α/K (Bonferroni) or 1-(1-α)^(1/K) (Šidák).

    **Validates: Requirements 6.5**

    For any significance level α and number of detectors K, the per-detector α
    SHALL equal α/K (Bonferroni) or 1-(1-α)^(1/K) (Šidák). The family-wise
    rate SHALL not exceed α.
    """
    # Feature: shift-detection-monitor, Property 17: Multiplicity Correction Formula
    controller = AlarmController(
        alpha=alpha,
        correction_method=correction,  # type: ignore[arg-type]
        combined_window=None,
        window_mode="growing",
        min_warmup_steps=1,
        tail_bound="bounded",
        lower_bound=0.0,
        upper_bound=1.0,
    )

    # Register n_detectors
    for i in range(n_detectors):
        ref_val = 0.5
        controller.register_detector(f"det_{i}", ref_val)

    k = n_detectors
    corrected_alpha = controller.get_corrected_alpha()

    if correction == "bonferroni":
        expected = alpha / k
        assert abs(corrected_alpha - expected) < 1e-12, (
            f"Bonferroni: expected α/K = {expected}, got {corrected_alpha}"
        )
        # Family-wise rate: sum of per-detector alphas ≤ α
        family_wise = corrected_alpha * k
        assert family_wise <= alpha + 1e-12, (
            f"Bonferroni family-wise rate {family_wise} exceeds α={alpha}"
        )
    else:  # sidak
        expected = 1.0 - (1.0 - alpha) ** (1.0 / k)
        assert abs(corrected_alpha - expected) < 1e-12, (
            f"Šidák: expected 1-(1-α)^(1/K) = {expected}, got {corrected_alpha}"
        )
        # Family-wise rate: 1 - (1 - α_per)^K ≤ α
        family_wise = 1.0 - (1.0 - corrected_alpha) ** k
        assert family_wise <= alpha + 1e-12, (
            f"Šidák family-wise rate {family_wise} exceeds α={alpha}"
        )

    # Verify each engine has the corrected alpha
    for name, engine in controller._engines.items():
        assert abs(engine._alpha - corrected_alpha) < 1e-12, (
            f"Engine '{name}' has alpha={engine._alpha}, expected {corrected_alpha}"
        )
