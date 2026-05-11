"""Llama Guard 3 (8B) adapter for the ClassifierInterface.

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

_LLAMA_GUARD_3_EMBEDDING_DIM = 4096


def _get_device() -> Any:
    """Select MPS if available, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LlamaGuard3Adapter:
    """Adapter for Meta Llama Guard 3 (8B) safety classifier.

    Parameters
    ----------
    model_path : str | None
        HuggingFace model ID. Defaults to "meta-llama/Llama-Guard-3-8B".
    device : str
        Device for inference. Defaults to auto-detected MPS/CPU.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_path = model_path or "meta-llama/Llama-Guard-3-8B"
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
            logger.info("Loading Llama Guard 3 from %s on %s", self._model_path, self._device)
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
            logger.info("Llama Guard 3 loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(
                f"Llama Guard 3 requires 'torch' and 'transformers'. Error: {e}"
            ) from e
        except Exception as e:
            raise ClassifierError(
                f"Failed to load Llama Guard 3 from '{self._model_path}': {e}"
            ) from e

    def _hook(self, module: Any, input: Any, output: Any) -> None:
        # output is a tuple; first element is the hidden state tensor
        hidden = output[0] if isinstance(output, tuple) else output
        self._penultimate_output = hidden.detach().cpu().float().numpy()

    @property
    def name(self) -> str:
        return "llama-guard-3-8b"

    @property
    def embedding_dim(self) -> int | None:
        return _LLAMA_GUARD_3_EMBEDDING_DIM

    def predict(self, text: str) -> ClassifierOutput:
        """Run inference on a single text."""
        self._load_model()
        try:
            import torch

            chat = [{"role": "user", "content": text}]
            input_ids = self._tokenizer.apply_chat_template(
                chat, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(input_ids)

            # Representation from hook: last token of penultimate layer
            representation = self._penultimate_output[0, -1, :].astype(np.float64)

            # Safety score from logits
            logits = outputs.logits[0, -1, :]
            safe_id = self._tokenizer.encode("safe", add_special_tokens=False)[0]
            unsafe_id = self._tokenizer.encode("unsafe", add_special_tokens=False)[0]
            probs = torch.softmax(logits[[safe_id, unsafe_id]], dim=0)
            score = float(probs[1].cpu())

            metadata = {"classification": "unsafe" if score > 0.5 else "safe"}
            return ClassifierOutput(score=score, representation=representation, metadata=metadata)
        except ClassifierError:
            raise
        except Exception as e:
            raise ClassifierError(f"Llama Guard 3 inference failed: {e}") from e

    def predict_batch(self, texts: list[str]) -> list[ClassifierOutput]:
        """Run inference on a batch of texts."""
        return [self.predict(t) for t in texts]
