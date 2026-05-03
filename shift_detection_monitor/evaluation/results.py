"""Result schema models for evaluation output serialization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AlarmRecord(BaseModel):
    """Serializable alarm event."""

    time_step: int
    detector: Literal["mmd", "ks", "combined"]
    statistic_value: float
    cs_lower: float
    cs_upper: float
    reference_value: float


class OCPoint(BaseModel):
    """A single point on the operating characteristic curve."""

    false_alarm_rate: float
    detection_latency: float


class CellResult(BaseModel):
    """Result for a single factorial cell x seed combination."""

    classifier: str
    shift_condition: str
    ground_truth_regime: str
    window_size: int
    seed: int
    detection_latency: float | None = None
    false_alarm_rate: float
    alarms: list[AlarmRecord] = Field(default_factory=list)
    conformal_coverage_pre: float | None = None
    conformal_coverage_post: float | None = None
    n_abstentions: int
    n_predictions: int
    is_negative_control: bool = False
    is_positive_control: bool = False
    is_false_positive: bool = False
    is_missed_detection: bool = False
    oc_curve: list[OCPoint] = Field(default_factory=list)
    active_detectors: list[str] = Field(
        default_factory=lambda: ["mmd", "ks"]
    )


class VarianceResult(BaseModel):
    """Serializable variance decomposition summary."""

    factor_variances: dict[str, float]
    interaction_variances: dict[str, float]
    effect_sizes: dict[str, dict[str, Any]]
    flagged_cells: list[str]
    residual_variance: float
