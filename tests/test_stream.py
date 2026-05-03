"""
Property-based and unit tests for the ShiftDatasetBuilder and StreamSimulator.

Properties tested:
- P27: Dataset manifest completeness
- P28: Deterministic generation (same seed → identical output)
- P1: Stream deterministic replay (same seed → identical sequence)
- P2: Shift injection timing and proportion
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from shift_detection_monitor.classifiers.interface import ClassifierInterface
from shift_detection_monitor.config import StreamConfig
from shift_detection_monitor.stream.dataset_builder import (
    DatasetManifest,
    ShiftDatasetBuilder,
    ShiftDatasetConfig,
)
from shift_detection_monitor.stream.simulator import StreamSimulator
from shift_detection_monitor.types import ClassifierOutput, StreamRecord


# ---------------------------------------------------------------------------
# Mock classifier for testing
# ---------------------------------------------------------------------------


class MockClassifier:
    """A deterministic mock classifier for testing."""

    def __init__(self, embedding_dim: int = 8, seed: int = 42) -> None:
        self._embedding_dim = embedding_dim
        self._seed = seed
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock-classifier"

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def predict(self, text: str) -> ClassifierOutput:
        # Deterministic: hash the text to get a consistent score
        text_hash = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(text_hash)
        score = float(rng.uniform(0, 1))
        representation = rng.standard_normal(self._embedding_dim).astype(np.float64)
        return ClassifierOutput(
            score=score,
            representation=representation,
            metadata={"source": "mock"},
        )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_SHIFT_CONDITIONS = st.sampled_from(
    [
        "paraphrase",
        "code-switch",
        "adversarial-suffix",
        "compositional-long-context",
        "temporal",
    ]
)


@st.composite
def st_source_examples(draw: st.DrawFn) -> list[dict[str, Any]]:
    """Generate a list of source examples for the dataset builder."""
    n = draw(st.integers(min_value=1, max_value=20))
    examples = []
    for i in range(n):
        text = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(codec="utf-8", categories=("L", "N", "Z")),
            )
        )
        examples.append(
            {
                "text": text,
                "label": draw(st.sampled_from([0, 1])),
                "source_dataset": "test",
            }
        )
    return examples


# ---------------------------------------------------------------------------
# P27: Dataset Manifest Completeness
# ---------------------------------------------------------------------------


@given(
    shift_condition=_SHIFT_CONDITIONS,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    examples=st_source_examples(),
)
@settings(max_examples=100)
def test_manifest_contains_all_required_fields(
    shift_condition: str,
    seed: int,
    examples: list[dict[str, Any]],
) -> None:
    """P27: For any generated dataset, manifest contains all required fields.

    **Validates: Requirements 15.2**

    Feature: shift-detection-monitor, Property 27: Dataset Manifest Completeness
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        # Write source examples
        with open(source_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        builder = ShiftDatasetBuilder()
        manifest = builder.build(shift_condition, source_path, output_path, seed)

        # Check all required fields
        assert isinstance(manifest.generator_version, str)
        assert len(manifest.generator_version) > 0

        assert isinstance(manifest.model_used, str)
        assert len(manifest.model_used) > 0

        assert isinstance(manifest.generation_params, dict)

        assert isinstance(manifest.seed, int)
        assert manifest.seed == seed

        assert isinstance(manifest.n_examples, int)
        assert manifest.n_examples == len(examples)

        assert isinstance(manifest.shift_condition, str)
        assert manifest.shift_condition == shift_condition

        assert isinstance(manifest.created_at, str)
        assert len(manifest.created_at) > 0
        # Verify ISO 8601 format (should contain 'T' and timezone info)
        assert "T" in manifest.created_at or "+" in manifest.created_at

        assert isinstance(manifest.human_validation_flags, dict)


# ---------------------------------------------------------------------------
# P28: Deterministic Generation
# ---------------------------------------------------------------------------


@given(
    shift_condition=_SHIFT_CONDITIONS,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    examples=st_source_examples(),
)
@settings(max_examples=100)
def test_deterministic_generation(
    shift_condition: str,
    seed: int,
    examples: list[dict[str, Any]],
) -> None:
    """P28: For any shift condition, source inputs, and seed, running build() twice
    produces identical output.

    **Validates: Requirements 15.5**

    Feature: shift-detection-monitor, Property 28: Dataset Builder Deterministic Generation
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        output_path_1 = Path(tmpdir) / "output1.jsonl"
        output_path_2 = Path(tmpdir) / "output2.jsonl"

        # Write source examples
        with open(source_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, sort_keys=True) + "\n")

        builder = ShiftDatasetBuilder()

        # Build twice with same parameters
        manifest1 = builder.build(shift_condition, source_path, output_path_1, seed)
        manifest2 = builder.build(shift_condition, source_path, output_path_2, seed)

        # Read outputs
        with open(output_path_1) as f:
            content1 = f.read()
        with open(output_path_2) as f:
            content2 = f.read()

        # Outputs must be identical
        assert content1 == content2

        # Manifests must match (except created_at which is timestamped)
        assert manifest1.generator_version == manifest2.generator_version
        assert manifest1.model_used == manifest2.model_used
        assert manifest1.generation_params == manifest2.generation_params
        assert manifest1.seed == manifest2.seed
        assert manifest1.n_examples == manifest2.n_examples
        assert manifest1.shift_condition == manifest2.shift_condition
        assert manifest1.human_validation_flags == manifest2.human_validation_flags


# ---------------------------------------------------------------------------
# P1: Stream Deterministic Replay
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_ref=st.integers(min_value=2, max_value=20),
    n_shifted=st.integers(min_value=0, max_value=10),
    onset=st.integers(min_value=0, max_value=15),
    mixing=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_stream_deterministic_replay(
    seed: int,
    n_ref: int,
    n_shifted: int,
    onset: int,
    mixing: float,
) -> None:
    """P1: For any config and seed, replaying twice produces identical StreamRecord sequences.

    **Validates: Requirements 1.5**

    Feature: shift-detection-monitor, Property 1: Stream Deterministic Replay
    """
    ref_examples = [
        {"text": f"reference text {i}", "label": i % 2, "source_dataset": "ref"}
        for i in range(n_ref)
    ]
    shifted_examples = [
        {"text": f"shifted text {i}", "label": i % 2, "source_dataset": "shifted"}
        for i in range(n_shifted)
    ]

    has_shift = n_shifted > 0
    config = StreamConfig(
        shift_condition="paraphrase" if has_shift else None,
        shift_onset_step=onset,
        mixing_proportion=mixing,
        seed=seed,
    )

    classifier = MockClassifier(seed=seed)

    sim1 = StreamSimulator(
        config=config,
        classifier=classifier,
        seed=seed,
        reference_examples=ref_examples,
        shifted_examples=shifted_examples if has_shift else None,
    )
    records1 = list(sim1)

    sim2 = StreamSimulator(
        config=config,
        classifier=classifier,
        seed=seed,
        reference_examples=ref_examples,
        shifted_examples=shifted_examples if has_shift else None,
    )
    records2 = list(sim2)

    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1.time_step == r2.time_step
        assert r1.text == r2.text
        assert r1.score == r2.score
        assert r1.is_shifted == r2.is_shifted
        assert r1.shift_condition == r2.shift_condition
        assert r1.source_dataset == r2.source_dataset
        if r1.representation is not None and r2.representation is not None:
            assert np.array_equal(r1.representation, r2.representation)
        else:
            assert r1.representation is None and r2.representation is None


# ---------------------------------------------------------------------------
# P2: Shift Injection Timing and Proportion
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    onset=st.integers(min_value=5, max_value=20),
    mixing=st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_shift_injection_timing(
    seed: int,
    onset: int,
    mixing: float,
) -> None:
    """P2: Zero shifted records before onset; proportion converges after onset.

    **Validates: Requirements 1.2, 1.6**

    Feature: shift-detection-monitor, Property 2: Shift Injection Timing and Proportion
    """
    # Provide enough examples in both pools so neither is exhausted during the test.
    # After onset, each step draws shifted with probability `mixing`.
    # We need enough shifted examples to sustain the mixing proportion.
    post_onset_steps = 500
    n_shifted = int(post_onset_steps * mixing) + 100  # extra buffer
    n_ref = onset + int(post_onset_steps * (1 - mixing)) + 100  # extra buffer

    ref_examples = [
        {"text": f"reference text {i}", "label": i % 2, "source_dataset": "ref"}
        for i in range(n_ref)
    ]
    shifted_examples = [
        {"text": f"shifted text {i}", "label": i % 2, "source_dataset": "shifted"}
        for i in range(n_shifted)
    ]

    config = StreamConfig(
        shift_condition="paraphrase",
        shift_onset_step=onset,
        mixing_proportion=mixing,
        seed=seed,
    )

    classifier = MockClassifier(seed=seed)
    sim = StreamSimulator(
        config=config,
        classifier=classifier,
        seed=seed,
        reference_examples=ref_examples,
        shifted_examples=shifted_examples,
    )
    records = list(sim)

    # Check: zero shifted records before onset
    pre_onset = [r for r in records if r.time_step < onset]
    for r in pre_onset:
        assert not r.is_shifted, (
            f"Found shifted record at time_step={r.time_step} before onset={onset}"
        )

    # Check: proportion of shifted records after onset converges to mixing
    # Only consider the first post_onset_steps records after onset to avoid
    # pool exhaustion effects
    post_onset = [r for r in records if r.time_step >= onset]
    check_records = post_onset[:post_onset_steps]
    if len(check_records) >= 50:
        n_shifted_post = sum(1 for r in check_records if r.is_shifted)
        observed_proportion = n_shifted_post / len(check_records)
        # Use a generous tolerance: 3 * sqrt(p*(1-p)/n) + 0.05
        import math
        std_err = math.sqrt(mixing * (1 - mixing) / len(check_records))
        tolerance = max(3 * std_err + 0.05, 0.15)
        assert abs(observed_proportion - mixing) < tolerance, (
            f"Observed proportion {observed_proportion:.3f} too far from "
            f"expected {mixing:.3f} (n={len(check_records)}, tol={tolerance:.3f})"
        )


def test_no_shift_all_reference() -> None:
    """P2 (edge case): When shift_condition is None, all records have is_shifted=False.

    **Validates: Requirements 1.6**
    """
    ref_examples = [
        {"text": f"text {i}", "label": 0, "source_dataset": "ref"} for i in range(20)
    ]

    config = StreamConfig(
        shift_condition=None,
        shift_onset_step=5,
        mixing_proportion=0.5,
        seed=42,
    )

    classifier = MockClassifier(seed=42)
    sim = StreamSimulator(
        config=config,
        classifier=classifier,
        seed=42,
        reference_examples=ref_examples,
    )
    records = list(sim)

    assert len(records) > 0
    for r in records:
        assert not r.is_shifted
        assert r.shift_condition is None


# ---------------------------------------------------------------------------
# Unit tests for dataset builder
# ---------------------------------------------------------------------------


def test_builder_validate_correct_corpus() -> None:
    """Validate a correctly generated corpus passes validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        examples = [{"text": f"example {i}", "label": 0} for i in range(5)]
        with open(source_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        builder = ShiftDatasetBuilder()
        manifest = builder.build("paraphrase", source_path, output_path, seed=42)
        report = builder.validate(output_path, manifest)

        assert report.pass_rate == 1.0
        assert report.n_validated == 5
        assert report.n_passed == 5
        assert len(report.issues) == 0


def test_builder_invalid_condition() -> None:
    """Builder raises ValueError for unknown shift condition."""
    import pytest

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        with open(source_path, "w") as f:
            f.write(json.dumps({"text": "hello"}) + "\n")

        builder = ShiftDatasetBuilder()
        with pytest.raises(ValueError, match="Unknown shift condition"):
            builder.build("nonexistent", source_path, output_path, seed=0)
