"""
Llama Guard 3 (8B) adapter for the ClassifierInterface.

Wraps Meta's Llama Guard 3 safety classifier to extract scores and
penultimate-layer representation vectors. Uses HuggingFace Transformers
for model loading and inference.

The model classifies text as safe/unsafe and provides harm category labels.
The representation vector is extracted from the penultimate transformer layer.
"""

from __future__ import annotations

import logging

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)

# Expected penultimate-layer dimensionality for Llama Guard 3 8B
_LLAMA_GUARD_3_EMBEDDING_DIM = 4096


class LlamaGuard3Adapter:
    """Adapter for Meta Llama Guard 3 (8B) safety classifier.

    Conforms to the ClassifierInterface protocol. Extracts a scalar safety
    score and a penultimate-layer representation vector from the model.

    Parameters
    ----------
    model_path : str | None
        Path or HuggingFace model ID for Llama Guard 3.
        Defaults to "meta-llama/Llama-Guard-3-8B".
    device : str
        Device for inference ("cpu", "cuda", "cuda:0", etc.).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path or "meta-llama/Llama-Guard-3-8B"
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
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading Llama Guard 3 from %s", self._model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16 if "cuda" in self._device else torch.float32,
                device_map=self._device,
                output_hidden_states=True,
            )
            self._model.eval()
            logger.info("Llama Guard 3 loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(
                f"Llama Guard 3 requires 'torch' and 'transformers' packages. "
                f"Install with: pip install torch transformers. Error: {e}"
            ) from e
        except Exception as e:
            raise ClassifierError(
                f"Failed to load Llama Guard 3 from '{self._model_path}': {e}"
            ) from e

    @property
    def name(self) -> str:
        """Unique classifier identifier."""
        return "llama-guard-3-8b"

    @property
    def embedding_dim(self) -> int | None:
        """Dimensionality of representation vectors (penultimate layer)."""
        return _LLAMA_GUARD_3_EMBEDDING_DIM

    def predict(self, text: str) -> ClassifierOutput:
        """Run Llama Guard 3 on input text.

        Returns a ClassifierOutput with:
        - score: probability of the 'unsafe' token (0.0 = safe, 1.0 = unsafe)
        - representation: penultimate-layer hidden state (4096-dim)
        - metadata: harm category if classified as unsafe

        Raises ClassifierError on failure.
        """
        self._load_model()

        try:
            import torch

            # Format input for Llama Guard 3
            chat = [{"role": "user", "content": text}]
            input_ids = self._tokenizer.apply_chat_template(
                chat, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(input_ids, output_hidden_states=True)

            # Extract penultimate-layer representation
            # hidden_states is a tuple of (n_layers + 1) tensors
            hidden_states = outputs.hidden_states
            penultimate = hidden_states[-2]  # second-to-last layer
            # Use the last token's representation as the sequence representation
            representation = penultimate[0, -1, :].cpu().numpy().astype(np.float64)

            # Extract safety score from logits
            logits = outputs.logits[0, -1, :]
            # Llama Guard uses "safe"/"unsafe" tokens
            safe_token_id = self._tokenizer.encode("safe", add_special_tokens=False)[0]
            unsafe_token_id = self._tokenizer.encode("unsafe", add_special_tokens=False)[0]
            probs = torch.softmax(
                logits[[safe_token_id, unsafe_token_id]], dim=0
            )
            score = float(probs[1].cpu())  # P(unsafe)

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
                f"Llama Guard 3 inference failed: {e}"
            ) from e
