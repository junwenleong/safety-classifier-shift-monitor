"""Abstract interface for safety classifier embedding extraction."""

from __future__ import annotations

from typing import Protocol

from shift_detection_monitor.types import ClassifierOutput


class ClassifierInterface(Protocol):
    """Abstract interface for safety classifier embedding extraction.

    This is the sole coupling point between the Monitor and any Safety_Classifier.
    Concrete adapters (LlamaGuard3Adapter, ShieldGemmaAdapter, etc.) must implement
    this protocol.
    """

    @property
    def name(self) -> str:
        """Unique classifier identifier."""
        ...

    @property
    def embedding_dim(self) -> int | None:
        """Dimensionality of representation vectors, None if unavailable."""
        ...

    def predict(self, text: str) -> ClassifierOutput:
        """Run classifier on input text.

        Returns score and optional representation vector.
        Raises ClassifierError on failure (timeout, OOM, model error).
        """
        ...
