"""Property tests for result serialization round-trip (Property P20).

**Validates: Requirements 12.4**

P20: For any valid CellResult, parse(serialize(result)) == result,
including nested AlarmRecords and OC curve points. Min 100 iterations.
"""

from __future__ import annotations

from hypothesis import given, settings

from shift_detection_monitor.serialization.result_io import (
    parse_result,
    serialize_result,
)
from tests.strategies import st_cell_result


@settings(max_examples=100)
@given(result=st_cell_result())
def test_result_json_roundtrip(result):
    """parse(serialize(result)) == result for any valid CellResult."""
    json_line = serialize_result(result)
    restored = parse_result(json_line)
    assert restored == result


@settings(max_examples=100)
@given(result=st_cell_result())
def test_result_alarms_preserved(result):
    """Nested AlarmRecords survive round-trip."""
    json_line = serialize_result(result)
    restored = parse_result(json_line)
    assert len(restored.alarms) == len(result.alarms)
    for orig, rest in zip(result.alarms, restored.alarms):
        assert rest.time_step == orig.time_step
        assert rest.detector == orig.detector
        assert abs(rest.statistic_value - orig.statistic_value) < 1e-12
        assert abs(rest.cs_lower - orig.cs_lower) < 1e-12
        assert abs(rest.cs_upper - orig.cs_upper) < 1e-12
        assert abs(rest.reference_value - orig.reference_value) < 1e-12


@settings(max_examples=100)
@given(result=st_cell_result())
def test_result_oc_curve_preserved(result):
    """Nested OCPoint list survives round-trip."""
    json_line = serialize_result(result)
    restored = parse_result(json_line)
    assert len(restored.oc_curve) == len(result.oc_curve)
    for orig, rest in zip(result.oc_curve, restored.oc_curve):
        assert abs(rest.false_alarm_rate - orig.false_alarm_rate) < 1e-12
        assert abs(rest.detection_latency - orig.detection_latency) < 1e-12
