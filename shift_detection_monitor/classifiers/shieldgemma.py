"""ShieldGemma (9B) adapter for the ClassifierInterface.

Uses HuggingFace Transformers with MPS acceleration (fallback to CPU).
Lazy-loads model on first predict() call. Extracts penultimate-layer
representation via forward hook.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)

_SHIELDGEMMA_EMBEDDING_DIM = 3584


def _get_device() -> Any:
    """Select MPS if available, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ShieldGemmaAdapter:
    """Adapter for Google ShieldGemma (9B) safety classifier.

    Parameters
    ----------
    model_path : str | None
        HuggingFace model ID. Defaults to "google/shieldgemma-9b".
    device : str | None
        Device for inference. Defaults to auto-detected MPS/CPU.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_path = model_path or "google/shieldgemma-9b"
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
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = (
                torch.device(self._device_str) if self._device_str else _get_device()
            )
            logger.info("Loading ShieldGemma from %s on %s", self._model_path, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16 if str(self._device) != "cpu" else torch.float32,
                output_hidden_states=True,
            )
            self._model.to(self._device)
            self._model.eval()

            # Register forward hook on penultimate layer
            layers = self._model.model.layers
            layers[-2].register_forward_hook(self._hook)
            logger.info("ShieldGemma loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(
                f"ShieldGemma requires 'torch' and 'transformers'. Error: {e}"
            ) from e
        except Exception as e:
            raise ClassifierError(
                f"Failed to load ShieldGemma from '{self._model_path}': {e}"
            ) from e

    def _hook(self, module: Any, input: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._penultimate_output = hidden.detach().cpu().float().numpy()

    @property
    def name(self) -> str:
        return "shieldgemma-9b"

    @property
    def embedding_dim(self) -> int | None:
        return _SHIELDGEMMA_EMBEDDING_DIM

    def predict(self, text: str) -> ClassifierOutput:
        """Run inference on a single text."""
        self._load_model()
        try:
            import torch

            prompt = (
                "You are a policy expert trying to help determine whether a user "
                "prompt is in violation of the defined safety policies.\n\n"
                f"User Prompt: {text}\n\n"
                "Does the user prompt violate any of the above policies? "
                "Answer with 'Yes' or 'No'."
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Representation from hook: last token of penultimate layer
            representation = self._penultimate_output[0, -1, :].astype(np.float64)

            # Safety score from logits
            logits = outputs.logits[0, -1, :]
            yes_id = self._tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = self._tokenizer.encode("No", add_special_tokens=False)[0]
            probs = torch.softmax(logits[[no_id, yes_id]], dim=0)
            score = float(probs[1].cpu())  # P(Yes = violation)

            metadata = {"classification": "unsafe" if score > 0.5 else "safe"}
            return ClassifierOutput(score=score, representation=representation, metadata=metadata)
        except ClassifierError:
            raise
        except Exception as e:
            raise ClassifierError(f"ShieldGemma inference failed: {e}") from e

    def predict_batch(self, texts: list[str]) -> list[ClassifierOutput]:
        """Run inference on a batch of texts."""
        return [self.predict(t) for t in texts]
