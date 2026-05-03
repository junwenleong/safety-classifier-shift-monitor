"""
Evaluation metrics for shift detection performance.

Provides:
- Detection latency computation
- Operating characteristic (OC) curve generation
- Control result flagging (false positive / missed detection)
- Regime-specific evaluation logic (A, B, C)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shift_detection_monitor.evaluation.results import AlarmRecord, CellResult, OCPoint


def compute_detection_latency(
    alarms: list[AlarmRecord],
    shift_onset_step: int,
) -> float | None:
    """Compute detection latency: time steps from shift onset to first alarm.

    Parameters
    ----------
    alarms : list[AlarmRecord]
        Alarm records from a detection run.
    shift_onset_step : int
        The time step at which the shift was injected.

    Returns
    -------
    float | None
        Number of time steps from onset to first alarm, or None if no alarm.
    """
    if not alarms:
        return None

    first_alarm_step = min(a.time_step for a in alarms)
    latency = float(first_alarm_step - shift_onset_step)
    return max(latency, 0.0)


def generate_oc_curve(cell_results: list[CellResult]) -> list[OCPoint]:
    """Generate an operating characteristic curve from multiple runs.

    The OC curve plots detection latency against false alarm rate.
    Results are sorted by false_alarm_rate in ascending order, and
    the curve is monotonically non-increasing in detection_latency.

    Parameters
    ----------
    cell_results : list[CellResult]
        Results from multiple runs at different thresholds / configurations.

    Returns
    -------
    list[OCPoint]
        OC curve points sorted by ascending false_alarm_rate with
        monotonically non-increasing detection_latency.
    """
    if not cell_results:
        return []

    # Collect (far, latency) pairs, skipping results with no detection
    raw_points: list[tuple[float, float]] = []
    for r in cell_results:
        if r.detection_latency is not None:
            raw_points.append((r.false_alarm_rate, r.detection_latency))

    if not raw_points:
        return []

    # Sort by false_alarm_rate ascending
    raw_points.sort(key=lambda p: p[0])

    # Enforce monotonically non-increasing detection_latency:
    # As FAR increases, latency should not increase.
    # Walk from highest FAR to lowest, carrying the minimum latency seen so far.
    enforced: list[tuple[float, float]] = list(raw_points)
    min_latency = enforced[-1][1]
    for i in range(len(enforced) - 1, -1, -1):
        far, lat = enforced[i]
        min_latency = min(min_latency, lat)
        # Walk backward from the end: at each point, latency should be >= min_latency
        # Actually, non-increasing means lat[i] >= lat[i+1].
        # We enforce from right to left: if lat[i] < lat[i+1], set lat[i] = lat[i+1]
    # Re-do: enforce non-increasing from left to right
    # lat[0] >= lat[1] >= ... >= lat[n-1]
    result_points: list[tuple[float, float]] = []
    # Start with the raw sorted points and enforce non-increasing
    running_min = float("inf")
    # Non-increasing means each subsequent value is <= the previous.
    # We process left to right, tracking the running minimum.
    for far, lat in raw_points:
        running_min = min(running_min, lat)
        result_points.append((far, running_min))

    return [
        OCPoint(false_alarm_rate=far, detection_latency=lat)
        for far, lat in result_points
    ]


def flag_control_result(
    result: CellResult,
    max_latency: int,
) -> CellResult:
    """Set is_false_positive or is_missed_detection flags on a control result.

    Parameters
    ----------
    result : CellResult
        A cell result from a control run.
    max_latency : int
        Maximum acceptable detection latency for positive controls.

    Returns
    -------
    CellResult
        A copy of the result with flags set appropriately.
    """
    is_false_positive = False
    is_missed_detection = False

    if result.is_negative_control:
        # Negative control: alarm raised => false positive
        if len(result.alarms) > 0:
            is_false_positive = True

    if result.is_positive_control:
        # Positive control: no alarm within max_latency => missed detection
        if result.detection_latency is None or result.detection_latency > max_latency:
            is_missed_detection = True

    return result.model_copy(
        update={
            "is_false_positive": is_false_positive,
            "is_missed_detection": is_missed_detection,
        }
    )


# ---------------------------------------------------------------------------
# Regime-specific evaluation logic
# ---------------------------------------------------------------------------


def evaluate_regime_a(
    alarms: list[AlarmRecord],
    known_onset: int,
) -> dict[str, Any]:
    """Regime A: detection latency computed against known synthetic onset step.

    Parameters
    ----------
    alarms : list[AlarmRecord]
        Alarm records from the detection run.
    known_onset : int
        The known shift onset time step.

    Returns
    -------
    dict
        Evaluation metrics including detection_latency and alarm_count.
    """
    latency = compute_detection_latency(alarms, known_onset)
    return {
        "regime": "regime_a",
        "known_onset": known_onset,
        "detection_latency": latency,
        "alarm_count": len(alarms),
        "detected": latency is not None,
    }


def evaluate_regime_b(
    alarms: list[AlarmRecord],
    timestamp_split: int,
    expert_labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Regime B: detection latency against timestamp split.

    Coverage metrics incorporate expert-label subset (~500 examples).

    Parameters
    ----------
    alarms : list[AlarmRecord]
        Alarm records from the detection run.
    timestamp_split : int
        The time step corresponding to the temporal split.
    expert_labels : list[dict] | None
        Expert-labeled examples for coverage computation.

    Returns
    -------
    dict
        Evaluation metrics including detection_latency and expert coverage info.
    """
    latency = compute_detection_latency(alarms, timestamp_split)
    n_expert = len(expert_labels) if expert_labels else 0

    return {
        "regime": "regime_b",
        "timestamp_split": timestamp_split,
        "detection_latency": latency,
        "alarm_count": len(alarms),
        "detected": latency is not None,
        "n_expert_labels": n_expert,
    }


def evaluate_regime_c(
    alarms: list[AlarmRecord],
    attack_success_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Regime C: alarm validity tied to suffix transfer success.

    Separate reporting for white-box vs. transfer-attack classifiers.

    Parameters
    ----------
    alarms : list[AlarmRecord]
        Alarm records from the detection run.
    attack_success_rates : dict[str, float] | None
        Mapping of classifier name to attack success rate.

    Returns
    -------
    dict
        Evaluation metrics including alarm validity and attack success info.
    """
    rates = attack_success_rates or {}
    # An alarm is considered valid if the attack actually succeeded
    # (i.e., the attack success rate is above a threshold)
    valid_alarm = len(alarms) > 0 and any(r > 0.0 for r in rates.values())

    return {
        "regime": "regime_c",
        "alarm_count": len(alarms),
        "detected": len(alarms) > 0,
        "valid_alarm": valid_alarm,
        "attack_success_rates": rates,
    }
