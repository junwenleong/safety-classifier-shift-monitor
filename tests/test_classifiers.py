"""
Tests for classifier adapters.

Task 9.5: Smoke tests verifying all four adapters instantiate and conform
to the ClassifierInterface Protocol.

Task 9.6: Per-adapter representation vector validation tests using a
MockClassifier that returns deterministic outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shift_detection_monitor.classifiers.deberta import DeBERTaAdapter
from shift_detection_monitor.classifiers.gpt_oss_safeguard import RoBERTaHateSpeechAdapter
from shift_detection_monitor.classifiers.interface import ClassifierInterface
from shift_detection_monitor.classifiers.llama_guard import LlamaGuard3Adapter
from shift_detection_monitor.classifiers.shieldgemma import ShieldGemmaAdapter
from shift_detection_monitor.types import ClassifierOutput


# ---------------------------------------------------------------------------
# MockClassifier for testing without GPU/model weights
# ---------------------------------------------------------------------------


class MockClassifier:
    """A mock classifier that returns deterministic outputs.

    Conforms to ClassifierInterface. Returns a fixed score based on
    input text length and a deterministic representation vector.

    Parameters
    ----------
    classifier_name : str
        Name to return from the name property.
    dim : int | None
        Embedding dimensionality. None for API-only classifiers.
    """

    def __init__(
        self,
        classifier_name: str = "mock-classifier",
        dim: int | None = 64,
        model_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._name = classifier_name
        self._dim = dim
        self._rng = np.random.default_rng(42)

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int | None:
        return self._dim

    def predict(self, text: str) -> ClassifierOutput:
        # Deterministic score based on text hash
        score = (hash(text) % 1000) / 1000.0

        # Deterministic representation vector
        if self._dim is not None:
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            representation = rng.standard_normal(self._dim).astype(np.float64)
        else:
            representation = None

        return ClassifierOutput(
            score=score,
            representation=representation,
            metadata={"source": "mock"},
        )


# ---------------------------------------------------------------------------
# Task 9.5: Smoke tests — all adapters instantiate and conform to Protocol
# ---------------------------------------------------------------------------


class TestAdapterInstantiation:
    """Verify all four adapters can be instantiated without errors."""

    def test_llama_guard_instantiates(self) -> None:
        adapter = LlamaGuard3Adapter()
        assert adapter is not None

    def test_shieldgemma_instantiates(self) -> None:
        adapter = ShieldGemmaAdapter()
        assert adapter is not None

    def test_gpt_oss_safeguard_instantiates(self) -> None:
        adapter = RoBERTaHateSpeechAdapter()
        assert adapter is not None

    def test_deberta_instantiates(self) -> None:
        adapter = DeBERTaAdapter()
        assert adapter is not None

    def test_mock_classifier_instantiates(self) -> None:
        mock = MockClassifier()
        assert mock is not None


class TestProtocolConformance:
    """Verify all adapters conform to the ClassifierInterface Protocol."""

    @pytest.mark.parametrize(
        "adapter_cls",
        [LlamaGuard3Adapter, ShieldGemmaAdapter, RoBERTaHateSpeechAdapter, DeBERTaAdapter],
    )
    def test_has_name_property(self, adapter_cls: type) -> None:
        adapter = adapter_cls()
        assert isinstance(adapter.name, str)
        assert len(adapter.name) > 0

    @pytest.mark.parametrize(
        "adapter_cls",
        [LlamaGuard3Adapter, ShieldGemmaAdapter, RoBERTaHateSpeechAdapter, DeBERTaAdapter],
    )
    def test_has_embedding_dim_property(self, adapter_cls: type) -> None:
        adapter = adapter_cls()
        dim = adapter.embedding_dim
        assert dim is None or (isinstance(dim, int) and dim > 0)

    @pytest.mark.parametrize(
        "adapter_cls",
        [LlamaGuard3Adapter, ShieldGemmaAdapter, RoBERTaHateSpeechAdapter, DeBERTaAdapter],
    )
    def test_has_predict_method(self, adapter_cls: type) -> None:
        adapter = adapter_cls()
        assert callable(getattr(adapter, "predict", None))

    def test_mock_conforms_to_protocol(self) -> None:
        """MockClassifier should satisfy ClassifierInterface structurally."""
        mock = MockClassifier()
        # Structural protocol check: has name, embedding_dim, predict
        assert isinstance(mock.name, str)
        assert mock.embedding_dim is None or isinstance(mock.embedding_dim, int)
        assert callable(mock.predict)

    @pytest.mark.parametrize(
        "adapter_cls,expected_name",
        [
            (LlamaGuard3Adapter, "llama-guard-3-8b"),
            (ShieldGemmaAdapter, "shieldgemma-9b"),
            (RoBERTaHateSpeechAdapter, "roberta-hatespeech"),
            (DeBERTaAdapter, "deberta-v3-large"),
        ],
    )
    def test_adapter_names(self, adapter_cls: type, expected_name: str) -> None:
        adapter = adapter_cls()
        assert adapter.name == expected_name

    @pytest.mark.parametrize(
        "adapter_cls,expected_dim",
        [
            (LlamaGuard3Adapter, 4096),
            (ShieldGemmaAdapter, 3584),
            (RoBERTaHateSpeechAdapter, 768),
            (DeBERTaAdapter, 1024),
        ],
    )
    def test_adapter_embedding_dims(
        self, adapter_cls: type, expected_dim: int | None
    ) -> None:
        adapter = adapter_cls()
        assert adapter.embedding_dim == expected_dim


# ---------------------------------------------------------------------------
# Task 9.6: Per-adapter representation vector validation tests
# ---------------------------------------------------------------------------

_SAMPLE_INPUTS = [
    "How do I make a bomb?",
    "What is the weather like today?",
    "Tell me a joke about programming.",
]


class TestMockClassifierRepresentations:
    """Validate representation vectors from MockClassifier on sample inputs.

    Since actual model weights are not available in the test environment,
    we use MockClassifier to verify the output contract: shape, dtype,
    and finiteness of representation vectors.
    """

    @pytest.mark.parametrize(
        "classifier_name,dim",
        [
            ("llama-guard-3-8b", 4096),
            ("shieldgemma-9b", 3584),
            ("deberta-v3-large", 1024),
        ],
    )
    @pytest.mark.parametrize("text", _SAMPLE_INPUTS)
    def test_representation_shape(
        self, classifier_name: str, dim: int, text: str
    ) -> None:
        """Representation vector shape matches expected embedding_dim."""
        mock = MockClassifier(classifier_name=classifier_name, dim=dim)
        output = mock.predict(text)

        assert output.representation is not None
        assert output.representation.shape == (dim,), (
            f"Expected shape ({dim},), got {output.representation.shape}"
        )

    @pytest.mark.parametrize(
        "classifier_name,dim",
        [
            ("llama-guard-3-8b", 4096),
            ("shieldgemma-9b", 3584),
            ("deberta-v3-large", 1024),
        ],
    )
    @pytest.mark.parametrize("text", _SAMPLE_INPUTS)
    def test_representation_dtype(
        self, classifier_name: str, dim: int, text: str
    ) -> None:
        """Representation vector dtype is float64."""
        mock = MockClassifier(classifier_name=classifier_name, dim=dim)
        output = mock.predict(text)

        assert output.representation is not None
        assert output.representation.dtype == np.float64, (
            f"Expected dtype float64, got {output.representation.dtype}"
        )

    @pytest.mark.parametrize(
        "classifier_name,dim",
        [
            ("llama-guard-3-8b", 4096),
            ("shieldgemma-9b", 3584),
            ("deberta-v3-large", 1024),
        ],
    )
    @pytest.mark.parametrize("text", _SAMPLE_INPUTS)
    def test_representation_finiteness(
        self, classifier_name: str, dim: int, text: str
    ) -> None:
        """All elements of the representation vector are finite."""
        mock = MockClassifier(classifier_name=classifier_name, dim=dim)
        output = mock.predict(text)

        assert output.representation is not None
        assert np.all(np.isfinite(output.representation)), (
            "Representation vector contains non-finite values"
        )

    @pytest.mark.parametrize("text", _SAMPLE_INPUTS)
    def test_api_only_returns_none_representation(self, text: str) -> None:
        """Classifier with dim=None returns representation=None."""
        mock = MockClassifier(classifier_name="no-embedding-classifier", dim=None)
        output = mock.predict(text)

        assert output.representation is None

    @pytest.mark.parametrize("text", _SAMPLE_INPUTS)
    def test_score_in_valid_range(self, text: str) -> None:
        """Score is a finite float in [0, 1]."""
        mock = MockClassifier()
        output = mock.predict(text)

        assert isinstance(output.score, float)
        assert math.isfinite(output.score)
        assert 0.0 <= output.score <= 1.0

    @pytest.mark.parametrize("text", _SAMPLE_INPUTS)
    def test_metadata_is_dict(self, text: str) -> None:
        """Metadata is a dict with string keys and values."""
        mock = MockClassifier()
        output = mock.predict(text)

        assert isinstance(output.metadata, dict)
        for k, v in output.metadata.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_deterministic_output(self) -> None:
        """Same input produces same output (deterministic)."""
        mock = MockClassifier()
        text = "test input"
        output1 = mock.predict(text)
        output2 = mock.predict(text)

        assert output1.score == output2.score
        assert output1.metadata == output2.metadata
        if output1.representation is not None and output2.representation is not None:
            np.testing.assert_array_equal(
                output1.representation, output2.representation
            )
