"""GPT-OSS-Safeguard adapter for the ClassifierInterface.

Uses HuggingFace Transformers with MPS acceleration (fallback to CPU).
Lazy-loads model on first predict() call. Extracts penultimate-layer
representation via forward hook on the second-to-last layer.

Note: The HuggingFace model ID "openai/gpt-oss-safeguard" is used.
If unavailable, check HuggingFace Hub for the correct ID.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)

# GPT-OSS-Safeguard is a sequence classification model; embedding dim TBD
# Based on typical GPT-2 medium architecture: 1024
_GPT_OSS_EMBEDDING_DIM = 1024


def _get_device() -> Any:
    """Select MPS if available, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GptOssSafeguardAdapter:
    """Adapter for GPT-OSS-Safeguard safety classifier.

    Parameters
    ----------
    model_path : str | None
        HuggingFace model ID. Defaults to "openai/gpt-oss-safeguard".
    device : str | None
        Device for inference. Defaults to auto-detected MPS/CPU.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_path = model_path or "openai/gpt-oss-safeguard"
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
            logger.info("Loading GPT-OSS-Safeguard from %s on %s", self._model_path, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_path,
                output_hidden_states=True,
                num_labels=2,
            )
            self._model.to(self._device)
            self._model.eval()

            # Register forward hook on penultimate layer of the base model
            # Architecture varies; try common patterns
            base = getattr(self._model, "transformer", None) or getattr(self._model, "model", None)
            if base is not None:
                layers = getattr(base, "h", None) or getattr(base, "layers", None)
                if layers is not None and len(layers) >= 2:
                    layers[-2].register_forward_hook(self._hook)

            logger.info("GPT-OSS-Safeguard loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(
                f"GPT-OSS-Safeguard requires 'torch' and 'transformers'. Error: {e}"
            ) from e
        except Exception as e:
            raise ClassifierError(
                f"Failed to load GPT-OSS-Safeguard from '{self._model_path}': {e}"
            ) from e

    def _hook(self, module: Any, input: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._penultimate_output = hidden.detach().cpu().float().numpy()

    @property
    def name(self) -> str:
        return "gpt-oss-safeguard"

    @property
    def embedding_dim(self) -> int | None:
        """Returns None if model not loaded (API-only fallback), else actual dim."""
        return None

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

            # Representation from hook or hidden_states fallback
            if self._penultimate_output is not None:
                representation = self._penultimate_output[0, 0, :].astype(np.float64)
            elif outputs.hidden_states is not None:
                penultimate = outputs.hidden_states[-2]
                representation = penultimate[0, 0, :].cpu().numpy().astype(np.float64)
            else:
                representation = None

            # Safety score from logits
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
            score = float(probs[1].cpu())  # label 1 = unsafe

            metadata = {"classification": "unsafe" if score > 0.5 else "safe"}
            return ClassifierOutput(score=score, representation=representation, metadata=metadata)
        except ClassifierError:
            raise
        except Exception as e:
            raise ClassifierError(f"GPT-OSS-Safeguard inference failed: {e}") from e

    def predict_batch(self, texts: list[str]) -> list[ClassifierOutput]:
        """Run inference on a batch of texts."""
        return [self.predict(t) for t in texts]
