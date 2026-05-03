"""
GPT-OSS-Safeguard adapter for the ClassifierInterface.

Wraps the GPT-OSS-Safeguard safety classifier. This is an API-only
classifier that does not expose internal embeddings or representation
vectors. The adapter returns representation=None, and the MMD detector
will skip this classifier for embedding-based detection.

Only the scalar safety score is available from the API response.
"""

from __future__ import annotations

import logging

import numpy as np

from shift_detection_monitor.types import ClassifierError, ClassifierOutput

logger = logging.getLogger(__name__)


class GptOssSafeguardAdapter:
    """Adapter for GPT-OSS-Safeguard safety classifier.

    Conforms to the ClassifierInterface protocol. This is an API-only
    classifier — no representation vectors are available.

    **Limitation**: Since GPT-OSS-Safeguard is accessed via API and does
    not expose internal model representations, ``representation`` is always
    ``None``. The MMD detector will skip this classifier; only the KS
    detector (which operates on scalar scores) will be active.

    Parameters
    ----------
    model_path : str | None
        API endpoint URL or model identifier.
        Defaults to "gpt-oss-safeguard".
    device : str
        Ignored for API-only classifier. Accepted for interface consistency.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path or "gpt-oss-safeguard"
        self._device = device  # Unused, kept for interface consistency
        self._api_client = None

    def _init_client(self) -> None:
        """Lazily initialize the API client.

        Raises ClassifierError if the API is not reachable or
        configuration is invalid.
        """
        if self._api_client is not None:
            return

        try:
            # In a real deployment, this would initialize an HTTP client
            # or SDK for the GPT-OSS-Safeguard API.
            logger.info(
                "Initializing GPT-OSS-Safeguard client for %s",
                self._model_path,
            )
            self._api_client = True  # Placeholder for actual client
        except Exception as e:
            raise ClassifierError(
                f"Failed to initialize GPT-OSS-Safeguard client: {e}"
            ) from e

    @property
    def name(self) -> str:
        """Unique classifier identifier."""
        return "gpt-oss-safeguard"

    @property
    def embedding_dim(self) -> int | None:
        """Dimensionality of representation vectors.

        Returns None because GPT-OSS-Safeguard is API-only and does not
        expose internal representations.
        """
        return None

    def predict(self, text: str) -> ClassifierOutput:
        """Run GPT-OSS-Safeguard on input text.

        Returns a ClassifierOutput with:
        - score: safety score from the API (0.0 = safe, 1.0 = unsafe)
        - representation: None (API-only, no embeddings available)
        - metadata: source information

        Raises ClassifierError on failure.
        """
        self._init_client()

        try:
            # In a real deployment, this would call the API endpoint.
            # For now, raise an error indicating the API is not configured.
            raise ClassifierError(
                "GPT-OSS-Safeguard API is not configured. "
                "Set the API endpoint and credentials to enable inference. "
                "This adapter requires a live API connection."
            )

        except ClassifierError:
            raise
        except Exception as e:
            raise ClassifierError(
                f"GPT-OSS-Safeguard inference failed: {e}"
            ) from e
