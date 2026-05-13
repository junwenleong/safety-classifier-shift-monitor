"""Tests to verify project scaffolding, types, config models, and strategies."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings

from shift_detection_monitor.config import (
    ConformalConfig,
    ControlConfig,
    DetectorConfig,
    FactorialConfig,
    MMDConfig,
    MonitorConfig,
    ReferenceWindowConfig,
    StreamConfig,
    VarianceConfig,
)
from shift_detection_monitor.evaluation.results import (
    AlarmRecord,
    CellResult,
    OCPoint,
    VarianceResult,
)
from shift_detection_monitor.types import (
    CalibrationError,
    ClassifierError,
    ClassifierOutput,
    ConfigValidationError,
    StreamRecord,
)

from tests.strategies import (
    st_alarm_record,
    st_cell_result,
    st_detector_config,
    st_embedding_matrix,
    st_monitor_config,
    st_oc_point,
    st_stream_config,
    st_stream_record,
)


# ---------------------------------------------------------------------------
# Task 1.2: Shared types
# ---------------------------------------------------------------------------


class TestClassifierOutput:
    def test_frozen(self, sample_classifier_output: ClassifierOutput) -> None:
        with pytest.raises(AttributeError):
            sample_classifier_output.score = 0.5  # type: ignore[misc]

    def test_none_representation(
        self, sample_classifier_output_no_repr: ClassifierOutput
    ) -> None:
        assert sample_classifier_output_no_repr.representation is None
        assert sample_classifier_output_no_repr.score == pytest.approx(0.42)

    def test_equality_with_arrays(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        a = ClassifierOutput(score=0.5, representation=arr, metadata={})
        b = ClassifierOutput(score=0.5, representation=arr.copy(), metadata={})
        assert a == b

    def test_equality_both_none(self) -> None:
        a = ClassifierOutput(score=0.5, representation=None, metadata={})
        b = ClassifierOutput(score=0.5, representation=None, metadata={})
        assert a == b

    def test_inequality_one_none(self) -> None:
        arr = np.array([1.0])
        a = ClassifierOutput(score=0.5, representation=arr, metadata={})
        b = ClassifierOutput(score=0.5, representation=None, metadata={})
        assert a != b

    def test_hash_consistency(self) -> None:
        arr = np.array([1.0, 2.0])
        a = ClassifierOutput(score=0.5, representation=arr, metadata={"k": "v"})
        b = ClassifierOutput(score=0.5, representation=arr.copy(), metadata={"k": "v"})
        assert hash(a) == hash(b)


class TestStreamRecord:
    def test_frozen(self, sample_stream_record: StreamRecord) -> None:
        with pytest.raises(AttributeError):
            sample_stream_record.time_step = 99  # type: ignore[misc]

    def test_fields(self, sample_stream_record: StreamRecord) -> None:
        assert sample_stream_record.time_step == 0
        assert sample_stream_record.is_shifted is False
        assert sample_stream_record.shift_condition is None
        assert sample_stream_record.ground_truth_label == 1

    def test_equality(self) -> None:
        arr = np.array([1.0, 2.0])
        a = StreamRecord(
            time_step=0, text="hi", score=0.5, representation=arr,
            ground_truth_label=None, is_shifted=False,
            source_dataset="test", shift_condition=None,
        )
        b = StreamRecord(
            time_step=0, text="hi", score=0.5, representation=arr.copy(),
            ground_truth_label=None, is_shifted=False,
            source_dataset="test", shift_condition=None,
        )
        assert a == b


class TestExceptions:
    def test_calibration_error(self) -> None:
        with pytest.raises(CalibrationError, match="too small"):
            raise CalibrationError("Reference window too small")

    def test_classifier_error(self) -> None:
        with pytest.raises(ClassifierError, match="timeout"):
            raise ClassifierError("Model timeout")

    def test_config_validation_error(self) -> None:
        with pytest.raises(ConfigValidationError, match="missing"):
            raise ConfigValidationError("Field 'alpha' missing")


# ---------------------------------------------------------------------------
# Task 1.3: ClassifierInterface Protocol
# ---------------------------------------------------------------------------


class TestClassifierInterface:
    def test_protocol_compliance(self) -> None:
        """Verify a concrete class can satisfy the protocol."""
        from shift_detection_monitor.classifiers.interface import ClassifierInterface

        class MockClassifier:
            @property
            def name(self) -> str:
                return "mock"

            @property
            def embedding_dim(self) -> int | None:
                return 16

            def predict(self, text: str) -> ClassifierOutput:
                return ClassifierOutput(score=0.5, representation=None, metadata={})

        clf: ClassifierInterface = MockClassifier()
        assert clf.name == "mock"
        assert clf.embedding_dim == 16
        out = clf.predict("test")
        assert out.score == 0.5


# ---------------------------------------------------------------------------
# Task 1.4: Config models
# ---------------------------------------------------------------------------


class TestConfigModels:
    def test_default_monitor_config(self) -> None:
        cfg = MonitorConfig()
        assert cfg.detector.alpha == pytest.approx(0.05)
        assert cfg.detector.window_size == 200
        assert cfg.output_dir == "results"

    def test_warmup_defaults_to_window_size(self) -> None:
        cfg = MonitorConfig()
        assert cfg.detector.min_warmup_steps == cfg.detector.window_size

    def test_warmup_explicit(self) -> None:
        cfg = MonitorConfig(
            detector=DetectorConfig(window_size=100, min_warmup_steps=50)
        )
        assert cfg.detector.min_warmup_steps == 50

    def test_factorial_default_classifiers(self) -> None:
        cfg = MonitorConfig()
        assert len(cfg.factorial.classifiers) == 4
        assert "llama-guard-3-8b" in cfg.factorial.classifiers
        assert "shieldgemma-9b" in cfg.factorial.classifiers
        assert "roberta-hatespeech" in cfg.factorial.classifiers
        assert "deberta-v3-large" in cfg.factorial.classifiers

    def test_invalid_alpha_too_high(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DetectorConfig(alpha=1.0)

    def test_invalid_alpha_too_low(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DetectorConfig(alpha=0.0)

    def test_invalid_window_size(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DetectorConfig(window_size=5)

    def test_stream_config_defaults(self) -> None:
        cfg = StreamConfig()
        assert cfg.shift_condition is None
        assert cfg.seed == 42

    def test_reference_window_config(self) -> None:
        cfg = ReferenceWindowConfig(size=200, min_size=100)
        assert cfg.size == 200
        assert cfg.min_size == 100

    def test_conformal_config(self) -> None:
        cfg = ConformalConfig()
        assert cfg.target_error_rate == pytest.approx(0.1)
        assert cfg.conformal_mode == "unweighted"

    def test_control_config_min_negative(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ControlConfig(n_negative_runs=10)  # must be >= 20


# ---------------------------------------------------------------------------
# Task 1.4: Result models
# ---------------------------------------------------------------------------


class TestResultModels:
    def test_alarm_record(self, sample_alarm_record: AlarmRecord) -> None:
        assert sample_alarm_record.time_step == 150
        assert sample_alarm_record.detector == "mmd"

    def test_cell_result(self, sample_cell_result: CellResult) -> None:
        assert sample_cell_result.classifier == "deberta-v3-large"
        assert len(sample_cell_result.alarms) == 1
        assert len(sample_cell_result.oc_curve) == 3

    def test_cell_result_defaults(self) -> None:
        cr = CellResult(
            classifier="test",
            shift_condition="paraphrase",
            ground_truth_regime="regime_a",
            window_size=100,
            seed=0,
            false_alarm_rate=0.0,
            n_abstentions=0,
            n_predictions=100,
        )
        assert cr.is_negative_control is False
        assert cr.is_false_positive is False
        assert cr.active_detectors == ["mmd", "ks"]

    def test_variance_result(self) -> None:
        vr = VarianceResult(
            factor_variances={"classifier": 0.3, "shift_type": 0.2},
            interaction_variances={"classifier:shift_type": 0.1},
            effect_sizes={
                "classifier": {
                    "estimate": 0.3,
                    "ci_lower": 0.1,
                    "ci_upper": 0.5,
                    "metric": "eta_squared",
                }
            },
            flagged_cells=["cell_1"],
            residual_variance=0.4,
        )
        assert vr.residual_variance == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Task 1.5: Hypothesis strategies generate valid objects
# ---------------------------------------------------------------------------


class TestStrategies:
    @given(st_stream_record())
    @settings(max_examples=20)
    def test_stream_record_strategy(self, record: StreamRecord) -> None:
        assert isinstance(record.time_step, int)
        assert isinstance(record.score, float)
        assert 0.0 <= record.score <= 1.0
        assert isinstance(record.is_shifted, bool)

    @given(st_stream_config())
    @settings(max_examples=20)
    def test_stream_config_strategy(self, cfg: StreamConfig) -> None:
        assert 0.0 <= cfg.mixing_proportion <= 1.0
        assert len(cfg.reference_datasets) >= 1

    @given(st_detector_config())
    @settings(max_examples=20)
    def test_detector_config_strategy(self, cfg: DetectorConfig) -> None:
        assert 0.0 < cfg.alpha < 1.0
        assert cfg.window_size >= 10

    @given(st_monitor_config())
    @settings(max_examples=20)
    def test_monitor_config_strategy(self, cfg: MonitorConfig) -> None:
        assert cfg.output_dir in ("results", "output", "eval_results")
        assert len(cfg.factorial.classifiers) >= 1

    @given(st_alarm_record())
    @settings(max_examples=20)
    def test_alarm_record_strategy(self, record: AlarmRecord) -> None:
        assert record.detector in ("mmd", "ks", "combined")

    @given(st_oc_point())
    @settings(max_examples=20)
    def test_oc_point_strategy(self, point: OCPoint) -> None:
        assert 0.0 <= point.false_alarm_rate <= 1.0

    @given(st_cell_result())
    @settings(max_examples=20)
    def test_cell_result_strategy(self, result: CellResult) -> None:
        assert result.n_abstentions <= result.n_predictions

    @given(st_embedding_matrix())
    @settings(max_examples=20)
    def test_embedding_matrix_strategy(self, mat: np.ndarray) -> None:
        assert mat.ndim == 2
        assert np.all(np.isfinite(mat))
