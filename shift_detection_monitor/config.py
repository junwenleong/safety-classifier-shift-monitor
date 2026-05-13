"""Pydantic v2 configuration models for the Shift Detection Monitor."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StreamConfig(BaseModel):
    """Configuration for the stream simulator."""

    reference_datasets: list[str] = Field(
        default_factory=lambda: ["wildguardmix", "toxicchat"]
    )
    shift_condition: str | None = None
    shift_onset_step: int = 500
    mixing_proportion: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int = 42


class DetectorConfig(BaseModel):
    """Configuration for shift detectors."""

    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    window_mode: Literal["sliding", "growing"] = "sliding"
    window_size: int = Field(default=200, ge=10)
    min_warmup_steps: int | None = None
    correction_method: Literal["bonferroni", "sidak"] = "bonferroni"
    combined_advisory_window: int | None = 50
    tail_bound: Literal["bounded", "sub_gaussian", "sub_exponential"] = "bounded"


class MMDConfig(BaseModel):
    """Configuration for MMD detector."""

    dim_reduction_threshold: int | None = 256
    n_bootstrap: int = Field(default=1000, ge=100)


class ConformalConfig(BaseModel):
    """Configuration for conformal abstention."""

    target_error_rate: float = Field(default=0.1, gt=0.0, lt=1.0)
    conformal_mode: Literal["unweighted", "weighted-on-alarm"] = "unweighted"
    min_calibration_size: int = Field(default=50, ge=10)
    density_ratio_method: Literal["logistic", "kliep"] = "logistic"


class ReferenceWindowConfig(BaseModel):
    """Configuration for reference window calibration."""

    size: int = Field(default=500, ge=50)
    min_size: int = Field(default=100, ge=50)


class FactorialConfig(BaseModel):
    """Configuration for the factorial evaluation design."""

    classifiers: list[str] = Field(
        default_factory=lambda: [
            "llama-guard-3-8b",
            "shieldgemma-9b",
            "roberta-hatespeech",
            "deberta-v3-large",
        ]
    )
    shift_conditions: list[str] = Field(
        default_factory=lambda: [
            "paraphrase",
            "code-switch",
            "adversarial-suffix",
            "compositional-long-context",
            "temporal",
        ]
    )
    ground_truth_regimes: list[Literal["regime_a", "regime_b", "regime_c"]] = Field(
        default_factory=lambda: ["regime_a", "regime_b", "regime_c"]
    )
    window_sizes: list[int] = Field(default_factory=lambda: [100, 200, 500])
    seeds: list[int] = Field(default_factory=lambda: list(range(20)))
    max_latency_positive_control: int = 200
    min_negative_control_runs: int = 20


class ControlConfig(BaseModel):
    """Configuration for negative and positive controls."""

    n_negative_runs: int = Field(default=20, ge=20)
    n_positive_runs: int = Field(default=20, ge=5)
    trivial_shift_mixing: float = Field(default=0.9, ge=0.5, le=1.0)
    max_latency: int = Field(default=200, ge=10)


class VarianceConfig(BaseModel):
    """Configuration for variance decomposition."""

    min_observations_per_cell: int = Field(default=10, ge=2)


class MonitorConfig(BaseModel):
    """Top-level configuration for the entire monitor."""

    stream: StreamConfig = Field(default_factory=StreamConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    mmd: MMDConfig = Field(default_factory=MMDConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    reference_window: ReferenceWindowConfig = Field(
        default_factory=ReferenceWindowConfig
    )
    factorial: FactorialConfig = Field(default_factory=FactorialConfig)
    controls: ControlConfig = Field(default_factory=ControlConfig)
    variance: VarianceConfig = Field(default_factory=VarianceConfig)
    output_dir: str = "results"

    @model_validator(mode="after")
    def validate_warmup(self) -> MonitorConfig:
        """Default min_warmup_steps to window_size if not explicitly set."""
        if self.detector.min_warmup_steps is None:
            self.detector = self.detector.model_copy(
                update={"min_warmup_steps": self.detector.window_size}
            )
        return self
