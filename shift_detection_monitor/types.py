"""Shared types and exceptions for the Shift Detection Monitor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ClassifierOutput:
    """Output from a safety classifier for a single input."""

    score: float
    representation: np.ndarray | None
    metadata: dict[str, str] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassifierOutput):
            return NotImplemented
        if self.score != other.score or self.metadata != other.metadata:
            return False
        if self.representation is None and other.representation is None:
            return True
        if self.representation is None or other.representation is None:
            return False
        return np.array_equal(self.representation, other.representation)

    def __hash__(self) -> int:
        rep_hash = (
            self.representation.tobytes() if self.representation is not None else None
        )
        return hash((self.score, rep_hash, tuple(sorted(self.metadata.items()))))


@dataclass(frozen=True)
class StreamRecord:
    """A single timestamped record in the simulated stream."""

    time_step: int
    text: str
    score: float
    representation: np.ndarray | None
    ground_truth_label: int | None  # 1=unsafe, 0=safe, None=unlabeled
    is_shifted: bool
    source_dataset: str
    shift_condition: str | None  # e.g., "paraphrase", "code-switch", None for reference

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StreamRecord):
            return NotImplemented
        if (
            self.time_step != other.time_step
            or self.text != other.text
            or self.score != other.score
            or self.ground_truth_label != other.ground_truth_label
            or self.is_shifted != other.is_shifted
            or self.source_dataset != other.source_dataset
            or self.shift_condition != other.shift_condition
        ):
            return False
        if self.representation is None and other.representation is None:
            return True
        if self.representation is None or other.representation is None:
            return False
        return np.array_equal(self.representation, other.representation)

    def __hash__(self) -> int:
        rep_hash = (
            self.representation.tobytes() if self.representation is not None else None
        )
        return hash(
            (
                self.time_step,
                self.text,
                self.score,
                rep_hash,
                self.ground_truth_label,
                self.is_shifted,
                self.source_dataset,
                self.shift_condition,
            )
        )


# --- Custom Exceptions ---


class CalibrationError(Exception):
    """Raised when calibration data is insufficient or invalid."""


class ClassifierError(Exception):
    """Raised when a classifier fails to produce output (timeout, OOM, model error)."""


class ConfigValidationError(Exception):
    """Raised when a configuration file contains invalid or missing required fields."""
