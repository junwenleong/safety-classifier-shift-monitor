"""DeBERTa-v3-large adapter for the ClassifierInterface.

Uses HuggingFace Transformers with MPS acceleration (fallback to CPU).
Lazy-loads model on first predict() call. Extracts penultimate-layer
[CLS] representation via forward hook.

Supports loading a fine-tuned checkpoint via DEBERTA_CHECKPOINT_PATH env var.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)

_DEBERTA_EMBEDDING_DIM = 1024


def _get_device() -> Any:
    """Select MPS if available, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeBERTaAdapter:
    """Adapter for DeBERTa-v3-large safety classifier.

    Parameters
    ----------
    model_path : str | None
        Path or HuggingFace model ID. If DEBERTA_CHECKPOINT_PATH env var
        is set, uses that instead. Defaults to "microsoft/deberta-v3-large".
    device : str | None
        Device for inference. Defaults to auto-detected MPS/CPU.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        # Env var overrides constructor arg
        checkpoint = os.environ.get("DEBERTA_CHECKPOINT_PATH")
        self._model_path = checkpoint or model_path or "microsoft/deberta-v3-large"
        self._device_str = device
        self._model = None
        self._tokenizer = None
        self._device = None
        self._penultimate_output: np.ndarray | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._device = (
                torch.device(self._device_str) if self._device_str else _get_device()
            )
            logger.info("Loading DeBERTa from %s on %s", self._model_path, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_path,
                output_hidden_states=True,
                num_labels=2,
            )
            self._model.to(self._device)
            self._model.eval()

            # Register forward hook on penultimate encoder layer
            encoder_layers = self._model.deberta.encoder.layer
            encoder_layers[-2].register_forward_hook(self._hook)
            logger.info("DeBERTa loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(
                f"DeBERTa requires 'torch' and 'transformers'. Error: {e}"
            ) from e
        except Exception as e:
            raise ClassifierError(
                f"Failed to load DeBERTa from '{self._model_path}': {e}"
            ) from e

    def _hook(self, module: Any, input: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._penultimate_output = hidden.detach().cpu().float().numpy()

    @property
    def name(self) -> str:
        return "deberta-v3-large"

    @property
    def embedding_dim(self) -> int | None:
        return _DEBERTA_EMBEDDING_DIM

    def predict(self, text: str) -> ClassifierOutput:
        """Run inference on a single text."""
        self._load_model()
        try:
            import torch

            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # [CLS] token representation from penultimate layer
            representation = self._penultimate_output[0, 0, :].astype(np.float64)

            # Safety score from logits
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
            score = float(probs[1].cpu())  # label 1 = unsafe

            metadata = {"classification": "unsafe" if score > 0.5 else "safe"}
            return ClassifierOutput(score=score, representation=representation, metadata=metadata)
        except ClassifierError:
            raise
        except Exception as e:
            raise ClassifierError(f"DeBERTa inference failed: {e}") from e

    def predict_batch(self, texts: list[str]) -> list[ClassifierOutput]:
        """Run inference on a batch of texts."""
        return [self.predict(t) for t in texts]
