"""Shared fixtures for the Shift Detection Monitor test suite."""

from __future__ import annotations

import numpy as np
import pytest

from shift_detection_monitor.config import (
    DetectorConfig,
    MonitorConfig,
    ReferenceWindowConfig,
    StreamConfig,
)
from shift_detection_monitor.evaluation.results import AlarmRecord, CellResult, OCPoint
from shift_detection_monitor.types import ClassifierOutput, StreamRecord


@pytest.fixture()
def default_monitor_config() -> MonitorConfig:
    """A MonitorConfig with all defaults."""
    return MonitorConfig()


@pytest.fixture()
def small_monitor_config() -> MonitorConfig:
    """A MonitorConfig with small sizes for fast testing."""
    return MonitorConfig(
        stream=StreamConfig(seed=0),
        detector=DetectorConfig(
            alpha=0.05,
            window_size=20,
            min_warmup_steps=20,
        ),
        reference_window=ReferenceWindowConfig(size=50, min_size=50),
    )


@pytest.fixture()
def sample_embedding() -> np.ndarray:
    """A small deterministic embedding vector."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(16).astype(np.float64)


@pytest.fixture()
def sample_embedding_matrix() -> np.ndarray:
    """A small deterministic embedding matrix (10 x 16)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 16)).astype(np.float64)


@pytest.fixture()
def sample_classifier_output(sample_embedding: np.ndarray) -> ClassifierOutput:
    """A sample ClassifierOutput with an embedding."""
    return ClassifierOutput(
        score=0.85,
        representation=sample_embedding,
        metadata={"harm_category": "violence"},
    )


@pytest.fixture()
def sample_classifier_output_no_repr() -> ClassifierOutput:
    """A sample ClassifierOutput without an embedding (API-only classifier)."""
    return ClassifierOutput(
        score=0.42,
        representation=None,
        metadata={"source": "api"},
    )


@pytest.fixture()
def sample_stream_record(sample_embedding: np.ndarray) -> StreamRecord:
    """A sample StreamRecord with an embedding."""
    return StreamRecord(
        time_step=0,
        text="How do I make a bomb?",
        score=0.95,
        representation=sample_embedding,
        ground_truth_label=1,
        is_shifted=False,
        source_dataset="wildguardmix",
        shift_condition=None,
    )


@pytest.fixture()
def sample_stream_records(sample_embedding_matrix: np.ndarray) -> list[StreamRecord]:
    """A list of 10 sample StreamRecords with embeddings."""
    records = []
    for i in range(sample_embedding_matrix.shape[0]):
        records.append(
            StreamRecord(
                time_step=i,
                text=f"sample text {i}",
                score=float(i) / 10.0,
                representation=sample_embedding_matrix[i],
                ground_truth_label=1 if i % 3 == 0 else 0,
                is_shifted=False,
                source_dataset="wildguardmix",
                shift_condition=None,
            )
        )
    return records


@pytest.fixture()
def sample_alarm_record() -> AlarmRecord:
    """A sample AlarmRecord."""
    return AlarmRecord(
        time_step=150,
        detector="mmd",
        statistic_value=0.032,
        cs_lower=0.01,
        cs_upper=0.05,
        reference_value=0.005,
    )


@pytest.fixture()
def sample_cell_result(sample_alarm_record: AlarmRecord) -> CellResult:
    """A sample CellResult."""
    return CellResult(
        classifier="deberta-v3-large",
        shift_condition="paraphrase",
        ground_truth_regime="regime_a",
        window_size=200,
        seed=0,
        detection_latency=50.0,
        false_alarm_rate=0.03,
        alarms=[sample_alarm_record],
        conformal_coverage_pre=0.92,
        conformal_coverage_post=0.85,
        n_abstentions=12,
        n_predictions=500,
        oc_curve=[
            OCPoint(false_alarm_rate=0.01, detection_latency=100.0),
            OCPoint(false_alarm_rate=0.05, detection_latency=50.0),
            OCPoint(false_alarm_rate=0.10, detection_latency=30.0),
        ],
    )
