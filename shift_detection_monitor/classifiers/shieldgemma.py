"""
ShieldGemma (9B) adapter for the ClassifierInterface.

Wraps Google's ShieldGemma safety classifier to extract scores and
penultimate-layer representation vectors. Uses HuggingFace Transformers
for model loading and inference.

ShieldGemma classifies text across multiple harm categories (dangerous
content, harassment, hate speech, sexually explicit content). The
representation vector is extracted from the penultimate transformer layer.
"""

from __future__ import annotations

import logging

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)

# Expected penultimate-layer dimensionality for ShieldGemma 9B
_SHIELDGEMMA_EMBEDDING_DIM = 3584


class ShieldGemmaAdapter:
    """Adapter for Google ShieldGemma (9B) safety classifier.

    Conforms to the ClassifierInterface protocol. Extracts a scalar safety
    score and a penultimate-layer representation vector from the model.

    Parameters
    ----------
    model_path : str | None
        Path or HuggingFace model ID for ShieldGemma.
        Defaults to "google/shieldgemma-9b".
    device : str
        Device for inference ("cpu", "cuda", "cuda:0", etc.).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path or "google/shieldgemma-9b"
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

            logger.info("Loading ShieldGemma from %s", self._model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16 if "cuda" in self._device else torch.float32,
                device_map=self._device,
                output_hidden_states=True,
            )
            self._model.eval()
            logger.info("ShieldGemma loaded on %s", self._device)
        except ImportError as e:
            raise ClassifierError(
                f"ShieldGemma requires 'torch' and 'transformers' packages. "
                f"Install with: pip install torch transformers. Error: {e}"
            ) from e
        except Exception as e:
            raise ClassifierError(
                f"Failed to load ShieldGemma from '{self._model_path}': {e}"
            ) from e

    @property
    def name(self) -> str:
        """Unique classifier identifier."""
        return "shieldgemma-9b"

    @property
    def embedding_dim(self) -> int | None:
        """Dimensionality of representation vectors (penultimate layer)."""
        return _SHIELDGEMMA_EMBEDDING_DIM

    def predict(self, text: str) -> ClassifierOutput:
        """Run ShieldGemma on input text.

        Returns a ClassifierOutput with:
        - score: probability of unsafe classification (0.0 = safe, 1.0 = unsafe)
        - representation: penultimate-layer hidden state (3584-dim)
        - metadata: harm categories detected

        Raises ClassifierError on failure.
        """
        self._load_model()

        try:
            import torch

            # Format input for ShieldGemma
            prompt = (
                "You are a policy expert trying to help determine whether a user "
                "prompt is in violation of the defined safety policies.\n\n"
                f"User Prompt: {text}\n\n"
                "Does the user prompt violate any of the above policies? "
                "Answer with 'Yes' or 'No'."
            )

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)

            # Extract penultimate-layer representation
            hidden_states = outputs.hidden_states
            penultimate = hidden_states[-2]
            representation = penultimate[0, -1, :].cpu().numpy().astype(np.float64)

            # Extract safety score from logits
            logits = outputs.logits[0, -1, :]
            yes_token_id = self._tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self._tokenizer.encode("No", add_special_tokens=False)[0]
            probs = torch.softmax(
                logits[[no_token_id, yes_token_id]], dim=0
            )
            score = float(probs[1].cpu())  # P(Yes = violation)

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
                f"ShieldGemma inference failed: {e}"
            ) from e
