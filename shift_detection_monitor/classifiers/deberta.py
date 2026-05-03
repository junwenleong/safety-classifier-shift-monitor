"""
DeBERTa-v3-large adapter for the ClassifierInterface.

Wraps a fine-tuned DeBERTa-v3-large model used as a reference safety
classifier. Uses HuggingFace Transformers for model loading and inference.

The representation vector is extracted from the penultimate transformer
layer's [CLS] token embedding.
"""

from __future__ import annotations

import logging

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)

# Expected penultimate-layer dimensionality for DeBERTa-v3-large
_DEBERTA_EMBEDDING_DIM = 1024


class DeBERTaAdapter:
    """Adapter for DeBERTa-v3-large safety classifier.

    Conforms to the ClassifierInterface protocol. Extracts a scalar safety
    score and a penultimate-layer [CLS] representation vector.

    Parameters
    ----------
    model_path : str | None
        Path or HuggingFace model ID for the fine-tuned DeBERTa model.
        Defaults to "microsoft/deberta-v3-large".
    device : str
        Device for inference ("cpu", "cuda", "cuda:0", etc.).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path or "microsoft/deberta-v3-large"
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazily load the model and tokenizer.

        Raises ClassifierError if dependencies are missing or model
        cannot be loaded.
        """
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading DeBERTa from %s", self._model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_path,
                output_hidden_states=True,
                num_labels=2,  # safe/unsafe binary classification
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info("DeBERTa loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(
                f"DeBERTa requires 'torch' and 'transformers' packages. "
                f"Install with: pip install torch transformers. Error: {e}"
            ) from e
        except Exception as e:
            raise ClassifierError(
                f"Failed to load DeBERTa from '{self._model_path}': {e}"
            ) from e

    @property
    def name(self) -> str:
        """Unique classifier identifier."""
        return "deberta-v3-large"

    @property
    def embedding_dim(self) -> int | None:
        """Dimensionality of representation vectors (penultimate layer [CLS])."""
        return _DEBERTA_EMBEDDING_DIM

    def predict(self, text: str) -> ClassifierOutput:
        """Run DeBERTa on input text.

        Returns a ClassifierOutput with:
        - score: probability of unsafe classification (0.0 = safe, 1.0 = unsafe)
        - representation: penultimate-layer [CLS] embedding (1024-dim)
        - metadata: classification label

        Raises ClassifierError on failure.
        """
        self._load_model()

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Extract penultimate-layer [CLS] representation
            hidden_states = outputs.hidden_states
            penultimate = hidden_states[-2]
            # [CLS] token is at position 0
            representation = penultimate[0, 0, :].cpu().numpy().astype(np.float64)

            # Extract safety score from logits
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
            # Assume label 1 = unsafe
            score = float(probs[1].cpu())

            metadata: dict[str, str] = {}
            if score > 0.5:
                metadata["classification"] = "unsafe"
            else:
                metadata["classification"] = "safe"

            return ClassifierOutput(
                score=score,
                representation=representation,
                metadata=metadata,
            )

        except ClassifierError:
            raise
        except Exception as e:
            raise ClassifierError(
                f"DeBERTa inference failed: {e}"
            ) from e
