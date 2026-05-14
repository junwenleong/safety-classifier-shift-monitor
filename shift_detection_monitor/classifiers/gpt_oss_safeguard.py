"""KoalaAI Text-Moderation adapter for the ClassifierInterface.

Uses KoalaAI/Text-Moderation — a DeBERTa-v3-based multi-category content
moderation model covering: sexual, hate, violence, harassment, self-harm.
Score = 1 - P(OK), i.e. probability of any unsafe category.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768


def _get_device() -> Any:
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextModerationAdapter:
    """Adapter for KoalaAI/Text-Moderation safety classifier.

    Multi-category moderation (S, H, V, HR, SH, S3, H2, V2, OK).
    Score = 1 - P(OK) = probability of any unsafe category.

    Parameters
    ----------
    model_path : str | None
        HuggingFace model ID. Defaults to KoalaAI/Text-Moderation.
    device : str | None
        Device for inference. Defaults to auto-detected MPS/CPU.
    """

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self._model_path = model_path or "KoalaAI/Text-Moderation"
        self._device_str = device
        self._model = None
        self._tokenizer = None
        self._device = None
        self._penultimate_output: np.ndarray | None = None
        self._ok_label_id: int | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._device = (
                torch.device(self._device_str) if self._device_str else _get_device()
            )
            logger.info("Loading Text-Moderation from %s on %s", self._model_path, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_path, output_hidden_states=True,
            )
            self._model.to(self._device)
            self._model.eval()

            # Find the "OK" label index
            id2label = self._model.config.id2label
            for idx, label in id2label.items():
                if label == "OK":
                    self._ok_label_id = int(idx)
                    break

            # Forward hook on penultimate encoder layer
            encoder_layers = self._model.deberta.encoder.layer
            encoder_layers[-2].register_forward_hook(self._hook)
            logger.info("Text-Moderation loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(f"Text-Moderation requires 'torch' and 'transformers'. Error: {e}") from e
        except Exception as e:
            raise ClassifierError(f"Failed to load Text-Moderation from '{self._model_path}': {e}") from e

    def _hook(self, module: Any, input: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._penultimate_output = hidden.detach().cpu().float().numpy()

    @property
    def name(self) -> str:
        return "text-moderation"

    @property
    def embedding_dim(self) -> int | None:
        return _EMBEDDING_DIM

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

            # Score = 1 - P(OK) = probability of any unsafe category
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0)
            ok_prob = float(probs[self._ok_label_id].cpu())
            score = 1.0 - ok_prob

            metadata = {"classification": "unsafe" if score > 0.5 else "safe"}
            return ClassifierOutput(score=score, representation=representation, metadata=metadata)
        except ClassifierError:
            raise
        except Exception as e:
            raise ClassifierError(f"Text-Moderation inference failed: {e}") from e

    def predict_batch(self, texts: list[str]) -> list[ClassifierOutput]:
        return [self.predict(t) for t in texts]
