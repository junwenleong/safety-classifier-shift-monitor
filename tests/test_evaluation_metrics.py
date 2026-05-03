"""Property tests for evaluation metrics.

Tests Properties 22 and 23 from the design document.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from shift_detection_monitor.evaluation.metrics import (
    compute_detection_latency,
    flag_control_result,
    generate_oc_curve,
)
from shift_detection_monitor.evaluation.results import AlarmRecord, CellResult, OCPoint
from tests.strategies import st_cell_result, st_alarm_record


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_UNIT_FLOAT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_LATENCY_FLOAT = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)


@st.composite
def st_cell_result_with_latency(draw: st.DrawFn) -> CellResult:
    """Generate a CellResult that has a detection_latency and false_alarm_rate."""
    return CellResult(
        classifier=draw(st.sampled_from(["deberta-v3-large", "llama-guard-3-8b"])),
        shift_condition=draw(st.sampled_from(["paraphrase", "code-switch"])),
        ground_truth_regime=draw(st.sampled_from(["regime_a", "regime_b", "regime_c"])),
        window_size=draw(st.integers(min_value=10, max_value=500)),
        seed=draw(st.integers(min_value=0, max_value=1000)),
        detection_latency=draw(_LATENCY_FLOAT),
        false_alarm_rate=draw(_UNIT_FLOAT),
        alarms=[],
        n_abstentions=0,
        n_predictions=100,
    )


# ---------------------------------------------------------------------------
# P22: OC curve is monotonically non-increasing
# ---------------------------------------------------------------------------


class TestOCCurveMonotonicity:
    """Feature: shift-detection-monitor, Property 22: OC Curve Monotonicity

    **Validates: Requirements 9.2**

    For any set of evaluation results for a single factorial cell, the generated
    Operating Characteristic Curve SHALL be monotonically non-increasing:
    as false_alarm_rate increases, detection_latency SHALL not increase.
    """

    @given(
        results=st.lists(st_cell_result_with_latency(), min_size=1, max_size=20)
    )
    @settings(max_examples=100)
    def test_oc_curve_monotonically_non_increasing(
        self, results: list[CellResult]
    ) -> None:
        """OC curve detection_latency is non-increasing as FAR increases."""
        oc_curve = generate_oc_curve(results)

        if len(oc_curve) < 2:
            return

        for i in range(len(oc_curve) - 1):
            assert oc_curve[i].false_alarm_rate <= oc_curve[i + 1].false_alarm_rate, (
                f"FAR not sorted: {oc_curve[i].false_alarm_rate} > {oc_curve[i + 1].false_alarm_rate}"
            )
            assert oc_curve[i].detection_latency >= oc_curve[i + 1].detection_latency, (
                f"Latency not non-increasing at index {i}: "
                f"{oc_curve[i].detection_latency} < {oc_curve[i + 1].detection_latency}"
            )

    @given(
        results=st.lists(st_cell_result_with_latency(), min_size=0, max_size=5)
    )
    @settings(max_examples=100)
    def test_oc_curve_preserves_point_count(
        self, results: list[CellResult]
    ) -> None:
        """OC curve has at most as many points as input results with latency."""
        oc_curve = generate_oc_curve(results)
        n_with_latency = sum(1 for r in results if r.detection_latency is not None)
        assert len(oc_curve) <= n_with_latency


# ---------------------------------------------------------------------------
# P23: Control result flagging correctness
# ---------------------------------------------------------------------------


class TestControlResultFlagging:
    """Feature: shift-detection-monitor, Property 23: Control Result Flagging

    **Validates: Requirements 9.6, 9.7**

    Negative-control alarm → is_false_positive=True;
    positive-control no-alarm within max_latency → is_missed_detection=True;
    no other results have these flags.
    """

    @given(
        max_latency=st.integers(min_value=10, max_value=1000),
        has_alarm=st.booleans(),
    )
    @settings(max_examples=100)
    def test_negative_control_alarm_is_false_positive(
        self, max_latency: int, has_alarm: bool
    ) -> None:
        """Negative control with alarm => is_false_positive=True."""
        alarms = []
        if has_alarm:
            alarms = [
                AlarmRecord(
                    time_step=50,
                    detector="mmd",
                    statistic_value=0.1,
                    cs_lower=0.0,
                    cs_upper=0.2,
                    reference_value=0.05,
                )
            ]

        result = CellResult(
            classifier="deberta-v3-large",
            shift_condition="paraphrase",
            ground_truth_regime="regime_a",
            window_size=200,
            seed=0,
            detection_latency=50.0 if has_alarm else None,
            false_alarm_rate=0.05,
            alarms=alarms,
            n_abstentions=0,
            n_predictions=100,
            is_negative_control=True,
            is_positive_control=False,
        )

        flagged = flag_control_result(result, max_latency)

        if has_alarm:
            assert flagged.is_false_positive is True
        else:
            assert flagged.is_false_positive is False

        # Negative controls should never be flagged as missed detection
        assert flagged.is_missed_detection is False

    @given(
        max_latency=st.integers(min_value=10, max_value=1000),
        detection_latency=st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100)
    def test_positive_control_no_alarm_is_missed_detection(
        self, max_latency: int, detection_latency: float | None
    ) -> None:
        """Positive control with no alarm or late alarm => is_missed_detection=True."""
        has_alarm = detection_latency is not None
        alarms = []
        if has_alarm:
            alarms = [
                AlarmRecord(
                    time_step=100,
                    detector="ks",
                    statistic_value=0.2,
                    cs_lower=0.0,
                    cs_upper=0.3,
                    reference_value=0.05,
                )
            ]

        result = CellResult(
            classifier="deberta-v3-large",
            shift_condition="paraphrase",
            ground_truth_regime="regime_a",
            window_size=200,
            seed=0,
            detection_latency=detection_latency,
            false_alarm_rate=0.05,
            alarms=alarms,
            n_abstentions=0,
            n_predictions=100,
            is_negative_control=False,
            is_positive_control=True,
        )

        flagged = flag_control_result(result, max_latency)

        if detection_latency is None or detection_latency > max_latency:
            assert flagged.is_missed_detection is True
        else:
            assert flagged.is_missed_detection is False

        # Positive controls should never be flagged as false positive
        assert flagged.is_false_positive is False

    @given(
        max_latency=st.integers(min_value=10, max_value=1000),
    )
    @settings(max_examples=100)
    def test_non_control_result_has_no_flags(self, max_latency: int) -> None:
        """Non-control results should not have false_positive or missed_detection flags."""
        result = CellResult(
            classifier="deberta-v3-large",
            shift_condition="paraphrase",
            ground_truth_regime="regime_a",
            window_size=200,
            seed=0,
            detection_latency=50.0,
            false_alarm_rate=0.05,
            alarms=[
                AlarmRecord(
                    time_step=50,
                    detector="mmd",
                    statistic_value=0.1,
                    cs_lower=0.0,
                    cs_upper=0.2,
                    reference_value=0.05,
                )
            ],
            n_abstentions=0,
            n_predictions=100,
            is_negative_control=False,
            is_positive_control=False,
        )

        flagged = flag_control_result(result, max_latency)
        assert flagged.is_false_positive is False
        assert flagged.is_missed_detection is False
