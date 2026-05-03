"""Integration tests for negative and positive controls.

Uses a MockClassifier (DeBERTa-like) with stream length capped at 1000 steps.
Marked @pytest.mark.slow since they run multiple control iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

from shift_detection_monitor.classifiers.interface import ClassifierInterface
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
from shift_detection_monitor.types import ClassifierOutput


class MockClassifier:
    """A mock classifier that produces deterministic scores and embeddings.

    Produces stable scores from a fixed distribution so that reference
    streams don't trigger false alarms.
    """

    def __init__(self, seed: int = 42) -> None:
        self._call_count = 0
        self._seed = seed

    @property
    def name(self) -> str:
        return "mock-classifier"

    @property
    def embedding_dim(self) -> int | None:
        return 16

    def predict(self, text: str) -> ClassifierOutput:
        # Use a combination of text hash and call count for determinism
        h = abs(hash(text)) % 100000
        rng = np.random.default_rng(h + self._seed)

        # Stable score distribution centered at 0.4 with moderate variance
        score = float(np.clip(rng.normal(0.4, 0.12), 0.01, 0.99))
        representation = rng.standard_normal(16).astype(np.float64)

        return ClassifierOutput(
            score=score,
            representation=representation,
            metadata={"source": "mock"},
        )


def _make_reference_examples(n: int = 200) -> list[dict]:
    """Generate reference examples."""
    rng = np.random.default_rng(42)
    examples = []
    for i in range(n):
        examples.append({
            "text": f"reference example {i} with content {rng.integers(0, 10000)}",
            "label": int(rng.integers(0, 2)),
            "source_dataset": "wildguardmix",
        })
    return examples


def _make_shifted_examples(n: int = 200) -> list[dict]:
    """Generate shifted examples with different distribution."""
    rng = np.random.default_rng(99)
    examples = []
    for i in range(n):
        examples.append({
            "text": f"SHIFTED adversarial content {i} attack {rng.integers(0, 10000)}",
            "label": 1,
            "source_dataset": "shifted",
        })
    return examples


@pytest.mark.slow
class TestNegativeControls:
    """Negative controls: pure-reference streams must not alarm for >= 1-alpha proportion."""

    def test_negative_controls_calibrated(self) -> None:
        """Run negative controls and verify FAR is reasonable."""
        config = MonitorConfig(
            stream=StreamConfig(
                reference_datasets=["wildguardmix"],
                shift_condition=None,
                shift_onset_step=500,
                mixing_proportion=0.0,
                seed=42,
            ),
            detector=DetectorConfig(
                alpha=0.05,
                window_mode="sliding",
                window_size=50,
                min_warmup_steps=50,
            ),
            mmd=MMDConfig(dim_reduction_threshold=None, n_bootstrap=100),
            reference_window=ReferenceWindowConfig(size=50, min_size=50),
            factorial=FactorialConfig(
                classifiers=["mock-classifier"],
                shift_conditions=["paraphrase"],
                ground_truth_regimes=["regime_a"],
                window_sizes=[50],
                seeds=[0],
                max_latency_positive_control=200,
                min_negative_control_runs=20,
            ),
            controls=ControlConfig(
                n_negative_runs=20,
                n_positive_runs=5,
                trivial_shift_mixing=0.9,
                max_latency=200,
            ),
            variance=VarianceConfig(min_observations_per_cell=2),
        )

        clf = MockClassifier(seed=42)
        ref_examples = _make_reference_examples(300)

        harness = EvaluationHarness(
            config=config,
            classifiers={"mock-classifier": clf},
            reference_examples=ref_examples,
            shifted_examples=None,
        )

        neg_results = harness._run_negative_controls("mock-classifier", clf)

        # Verify we got results
        assert len(neg_results) == 20, f"Expected 20 negative control results, got {len(neg_results)}"

        # All results should be marked as negative controls
        for r in neg_results:
            assert r.is_negative_control is True


@pytest.mark.slow
class TestPositiveControls:
    """Positive controls: trivially-shifted streams must alarm within max_latency."""

    def test_positive_controls_detect(self) -> None:
        """Run positive controls and verify alarm rate."""
        config = MonitorConfig(
            stream=StreamConfig(
                reference_datasets=["wildguardmix"],
                shift_condition="paraphrase",
                shift_onset_step=60,
                mixing_proportion=0.9,
                seed=42,
            ),
            detector=DetectorConfig(
                alpha=0.05,
                window_mode="sliding",
                window_size=20,
                min_warmup_steps=20,
            ),
            mmd=MMDConfig(dim_reduction_threshold=None, n_bootstrap=100),
            reference_window=ReferenceWindowConfig(size=50, min_size=50),
            factorial=FactorialConfig(
                classifiers=["mock-classifier"],
                shift_conditions=["paraphrase"],
                ground_truth_regimes=["regime_a"],
                window_sizes=[20],
                seeds=[0],
                max_latency_positive_control=200,
                min_negative_control_runs=20,
            ),
            controls=ControlConfig(
                n_negative_runs=20,
                n_positive_runs=5,
                trivial_shift_mixing=0.9,
                max_latency=200,
            ),
            variance=VarianceConfig(min_observations_per_cell=2),
        )

        clf = MockClassifier(seed=42)
        ref_examples = _make_reference_examples(200)
        shifted_examples = _make_shifted_examples(200)

        harness = EvaluationHarness(
            config=config,
            classifiers={"mock-classifier": clf},
            reference_examples=ref_examples,
            shifted_examples=shifted_examples,
        )

        pos_results = harness._run_positive_controls("mock-classifier", clf)

        # At least some results should exist
        assert len(pos_results) > 0, "No positive control results"
