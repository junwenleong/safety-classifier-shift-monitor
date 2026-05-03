"""End-to-end integration test for the evaluation harness.

Runs a minimal factorial (1 classifier × 1 shift × 1 regime × 1 seed)
with a mock classifier and verifies valid JSONL output.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from shift_detection_monitor.config import (
    ControlConfig,
    DetectorConfig,
    FactorialConfig,
    MMDConfig,
    MonitorConfig,
    ReferenceWindowConfig,
    StreamConfig,
    VarianceConfig,
)
from shift_detection_monitor.evaluation.harness import EvaluationHarness
from shift_detection_monitor.evaluation.results import CellResult
from shift_detection_monitor.serialization.result_io import read_results
from shift_detection_monitor.types import ClassifierOutput


class MockClassifier:
    """Mock classifier for end-to-end testing."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "mock-classifier"

    @property
    def embedding_dim(self) -> int | None:
        return 8

    def predict(self, text: str) -> ClassifierOutput:
        h = hash(text) % 10000
        rng = np.random.default_rng(h)
        score = float(np.clip(rng.normal(0.4, 0.15), 0.0, 1.0))
        representation = rng.standard_normal(8).astype(np.float64)
        return ClassifierOutput(
            score=score,
            representation=representation,
            metadata={},
        )


def _make_examples(n: int, prefix: str, seed: int) -> list[dict]:
    """Generate test examples."""
    rng = np.random.default_rng(seed)
    return [
        {
            "text": f"{prefix} example {i} content {rng.integers(0, 10000)}",
            "label": int(rng.integers(0, 2)),
            "source_dataset": "wildguardmix",
        }
        for i in range(n)
    ]


class TestEndToEnd:
    """End-to-end test with mock classifier and minimal factorial."""

    def test_minimal_factorial_produces_valid_jsonl(self) -> None:
        """Run minimal factorial and verify JSONL output is valid."""
        config = MonitorConfig(
            stream=StreamConfig(
                reference_datasets=["wildguardmix"],
                shift_condition="paraphrase",
                shift_onset_step=30,
                mixing_proportion=0.5,
                seed=42,
            ),
            detector=DetectorConfig(
                alpha=0.05,
                window_mode="sliding",
                window_size=10,
                min_warmup_steps=10,
            ),
            mmd=MMDConfig(dim_reduction_threshold=None, n_bootstrap=100),
            reference_window=ReferenceWindowConfig(size=50, min_size=50),
            factorial=FactorialConfig(
                classifiers=["mock-classifier"],
                shift_conditions=["paraphrase"],
                ground_truth_regimes=["regime_a"],
                window_sizes=[10],
                seeds=[0],
                max_latency_positive_control=100,
                min_negative_control_runs=20,
            ),
            controls=ControlConfig(
                n_negative_runs=20,
                n_positive_runs=5,
                trivial_shift_mixing=0.9,
                max_latency=100,
            ),
            variance=VarianceConfig(min_observations_per_cell=2),
        )

        clf = MockClassifier(seed=42)
        ref_examples = _make_examples(200, "ref", 42)
        shifted_examples = _make_examples(200, "shifted", 99)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.jsonl"

            harness = EvaluationHarness(
                config=config,
                classifiers={"mock-classifier": clf},
                reference_examples=ref_examples,
                shifted_examples=shifted_examples,
            )

            results = harness.run(output_path)

            # Verify output file exists and is valid JSONL
            assert output_path.exists(), "Output file not created"

            parsed = read_results(output_path)
            assert len(parsed) == len(results), (
                f"Parsed {len(parsed)} results, expected {len(results)}"
            )

            # Verify each result has required fields
            for r in parsed:
                assert isinstance(r, CellResult)
                assert r.classifier == "mock-classifier"
                assert r.n_predictions >= 0

    def test_harness_run_cell(self) -> None:
        """Test running a single factorial cell."""
        from shift_detection_monitor.evaluation.harness import FactorialCell

        config = MonitorConfig(
            stream=StreamConfig(
                reference_datasets=["wildguardmix"],
                shift_condition="paraphrase",
                shift_onset_step=30,
                mixing_proportion=0.5,
                seed=0,
            ),
            detector=DetectorConfig(
                alpha=0.05,
                window_mode="sliding",
                window_size=10,
                min_warmup_steps=10,
            ),
            mmd=MMDConfig(dim_reduction_threshold=None, n_bootstrap=100),
            reference_window=ReferenceWindowConfig(size=50, min_size=50),
            factorial=FactorialConfig(
                classifiers=["mock-classifier"],
                shift_conditions=["paraphrase"],
                ground_truth_regimes=["regime_a"],
                window_sizes=[10],
                seeds=[0],
            ),
        )

        clf = MockClassifier(seed=42)
        ref_examples = _make_examples(200, "ref", 42)
        shifted_examples = _make_examples(200, "shifted", 99)

        harness = EvaluationHarness(
            config=config,
            classifiers={"mock-classifier": clf},
            reference_examples=ref_examples,
            shifted_examples=shifted_examples,
        )

        cell = FactorialCell(
            classifier_name="mock-classifier",
            shift_condition="paraphrase",
            ground_truth_regime="regime_a",
            window_size=10,
        )

        result = harness.run_cell(cell, seed=0)

        assert result.classifier == "mock-classifier"
        assert result.shift_condition == "paraphrase"
        assert result.ground_truth_regime == "regime_a"
        assert result.window_size == 10
        assert result.seed == 0
        assert result.n_predictions >= 0
        assert result.false_alarm_rate >= 0.0
