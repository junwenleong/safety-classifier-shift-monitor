"""Property tests for malformed JSONL handling (Property P21).

**Validates: Requirements 12.5**

P21: For any JSONL with mixed valid/malformed lines, parser returns
exactly the valid lines and skips malformed ones. Min 100 iterations.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shift_detection_monitor.serialization.result_io import (
    read_results,
    serialize_result,
    write_results,
)
from tests.strategies import st_cell_result

# Strategy for malformed lines — things that are definitely not valid CellResult JSON
_MALFORMED_LINES = st.one_of(
    st.just("not json at all"),
    st.just("{"),
    st.just('{"classifier": "missing_fields"}'),
    st.just("null"),
    st.just("42"),
    st.just('{"classifier": "x", "shift_condition": "y"}'),
    st.text(min_size=1, max_size=50).filter(lambda s: "\n" not in s),
)


@st.composite
def st_mixed_jsonl(draw: st.DrawFn) -> tuple[list[str], list[int]]:
    """Generate a mixed JSONL content with valid and malformed lines.

    Returns (lines, valid_indices) where valid_indices are the 0-based
    positions of valid CellResult lines in the list.
    """
    n_valid = draw(st.integers(min_value=0, max_value=5))
    n_malformed = draw(st.integers(min_value=0, max_value=5))

    valid_results = [draw(st_cell_result()) for _ in range(n_valid)]
    valid_lines = [serialize_result(r) for r in valid_results]
    malformed_lines = [draw(_MALFORMED_LINES) for _ in range(n_malformed)]

    # Build interleaved list with tracking
    entries: list[tuple[str, bool]] = []
    entries.extend((line, True) for line in valid_lines)
    entries.extend((line, False) for line in malformed_lines)

    # Shuffle deterministically using Hypothesis
    permuted = draw(st.permutations(entries))

    lines = [e[0] for e in permuted]
    valid_indices = [i for i, e in enumerate(permuted) if e[1]]

    return lines, valid_indices


@settings(max_examples=100)
@given(data=st_mixed_jsonl())
def test_malformed_lines_skipped(data):
    """Parser returns exactly the valid lines and skips malformed ones."""
    lines, valid_indices = data

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "mixed.jsonl"
        jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Suppress warnings during read — we only care about result count
        logging.getLogger("shift_detection_monitor.serialization.result_io").setLevel(
            logging.CRITICAL
        )
        try:
            results = read_results(jsonl_path)
        finally:
            logging.getLogger(
                "shift_detection_monitor.serialization.result_io"
            ).setLevel(logging.NOTSET)

    # The number of parsed results must equal the number of valid lines
    assert len(results) == len(valid_indices)


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st_mixed_jsonl())
def test_malformed_lines_produce_warnings(data, caplog):
    """Each malformed line produces a warning log entry."""
    lines, valid_indices = data
    n_malformed = len(lines) - len(valid_indices)

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "mixed.jsonl"
        jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            read_results(jsonl_path)

    warning_count = sum(
        1
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Skipping malformed line" in record.message
    )
    assert warning_count == n_malformed


@settings(max_examples=100)
@given(results=st.lists(st_cell_result(), min_size=1, max_size=5))
def test_write_read_roundtrip(results):
    """write_results then read_results recovers all results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "results.jsonl"
        write_results(results, jsonl_path)
        restored = read_results(jsonl_path)

    assert len(restored) == len(results)
    for orig, rest in zip(results, restored):
        assert rest == orig
