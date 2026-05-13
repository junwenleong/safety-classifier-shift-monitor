"""RoBERTa Hate Speech adapter for the ClassifierInterface.

Uses facebook/roberta-hate-speech-dynabench-r4-target — a binary
hate speech classifier fine-tuned on DynaBench R4.

Replaces gpt-oss-safeguard in the factorial design (20B model too large).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)

_ROBERTA_EMBEDDING_DIM = 768


def _get_device() -> Any:
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RoBERTaHateSpeechAdapter:
    """Adapter for facebook/roberta-hate-speech-dynabench-r4-target.

    Parameters
    ----------
    model_path : str | None
        HuggingFace model ID. Defaults to the DynaBench R4 model.
    device : str | None
        Device for inference. Defaults to auto-detected MPS/CPU.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_path = model_path or "facebook/roberta-hate-speech-dynabench-r4-target"
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
            logger.info("Loading RoBERTa from %s on %s", self._model_path, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_path,
                output_hidden_states=True,
            )
            self._model.to(self._device)
            self._model.eval()

            # Forward hook on penultimate encoder layer
            encoder_layers = self._model.roberta.encoder.layer
            encoder_layers[-2].register_forward_hook(self._hook)
            logger.info("RoBERTa loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(
                f"RoBERTa requires 'torch' and 'transformers'. Error: {e}"
            ) from e
        except Exception as e:
            raise ClassifierError(
                f"Failed to load RoBERTa from '{self._model_path}': {e}"
            ) from e

    def _hook(self, module: Any, input: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._penultimate_output = hidden.detach().cpu().float().numpy()

    @property
    def name(self) -> str:
        return "roberta-hatespeech"

    @property
    def embedding_dim(self) -> int | None:
        return _ROBERTA_EMBEDDING_DIM

    def predict(self, text: str) -> ClassifierOutput:
        self._load_model()
        try:
            import torch

            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            representation = self._penultimate_output[0, 0, :].astype(np.float64)

            # Score: P(hate) — label 1 is "hate"
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
            score = float(probs[1].cpu())

            metadata = {"classification": "unsafe" if score > 0.5 else "safe"}
            return ClassifierOutput(score=score, representation=representation, metadata=metadata)
        except ClassifierError:
            raise
        except Exception as e:
            raise ClassifierError(f"RoBERTa inference failed: {e}") from e

    def predict_batch(self, texts: list[str]) -> list[ClassifierOutput]:
        return [self.predict(t) for t in texts]
