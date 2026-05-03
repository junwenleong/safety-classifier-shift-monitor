"""Result JSONL serialization and parsing.

Provides write/read for CellResult objects in JSONL format (one JSON
object per line). Malformed lines are skipped with a logged warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from shift_detection_monitor.evaluation.results import CellResult

logger = logging.getLogger(__name__)


def serialize_result(result: CellResult) -> str:
    """Serialize a single CellResult to a JSON line.

    Parameters
    ----------
    result : CellResult
        The result object to serialize.

    Returns
    -------
    str
        A single-line JSON string (no trailing newline).
    """
    return result.model_dump_json()


def parse_result(line: str) -> CellResult:
    """Parse a single JSON line into a CellResult.

    Parameters
    ----------
    line : str
        A JSON-formatted string representing a CellResult.

    Returns
    -------
    CellResult
        The parsed result object.

    Raises
    ------
    ValueError
        If the line is not valid JSON or does not conform to CellResult.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    try:
        return CellResult.model_validate(data)
    except Exception as exc:
        raise ValueError(f"CellResult validation failed: {exc}") from exc


def write_results(results: list[CellResult], path: Path) -> None:
    """Write a list of CellResults to a JSONL file.

    Parameters
    ----------
    results : list[CellResult]
        The result objects to write.
    path : Path
        Destination file path. Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for result in results:
            fh.write(serialize_result(result))
            fh.write("\n")


def read_results(path: Path) -> list[CellResult]:
    """Read CellResults from a JSONL file, skipping malformed lines.

    Malformed lines are logged as warnings with the line number and
    error message, then skipped.

    Parameters
    ----------
    path : Path
        Source JSONL file path.

    Returns
    -------
    list[CellResult]
        Successfully parsed results.
    """
    results: list[CellResult] = []
    with open(path, encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                results.append(parse_result(stripped))
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Skipping malformed line %d: %s", line_number, exc
                )
    return results
