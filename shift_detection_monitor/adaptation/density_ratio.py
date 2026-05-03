"""
Density ratio estimation for weighted conformal prediction.

Estimates covariate shift weights p_target(x) / p_source(x) using logistic
regression on representation vectors.

Logistic regression approach:
1. Label source as 0, target as 1
2. Fit logistic regression
3. Density ratio = P(1|x) / P(0|x) = exp(w^T x + b)
4. Clip to [1/C, C]
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression


class DensityRatioEstimator:
    """Estimates covariate shift weights for weighted conformal prediction.

    Parameters
    ----------
    method : {"logistic", "kliep"}
        Estimation method. Currently only "logistic" is fully implemented.
    max_weight : float
        Maximum weight clipping value C. Weights are clipped to [1/C, C].
    """

    def __init__(
        self,
        method: Literal["logistic", "kliep"] = "logistic",
        max_weight: float = 10.0,
    ) -> None:
        self._method = method
        self._max_weight = max_weight
        self._model: LogisticRegression | None = None
        self._fitted = False

    def fit(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> None:
        """Fit the density ratio estimator.

        Parameters
        ----------
        source_embeddings : np.ndarray
            Reference (source) embeddings, shape (n_source, d).
        target_embeddings : np.ndarray
            Post-alarm (target) embeddings, shape (n_target, d).
        """
        if source_embeddings.shape[0] == 0 or target_embeddings.shape[0] == 0:
            raise ValueError("Both source and target embeddings must be non-empty.")

        if self._method == "logistic":
            self._fit_logistic(source_embeddings, target_embeddings)
        elif self._method == "kliep":
            # KLIEP is a placeholder — fall back to logistic for now
            self._fit_logistic(source_embeddings, target_embeddings)
        else:
            raise ValueError(f"Unknown method: {self._method!r}")

        self._fitted = True

    def _fit_logistic(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> None:
        """Fit logistic regression for density ratio estimation.

        Labels source as 0, target as 1. The density ratio is then
        r(x) = P(target|x) / P(source|x) = exp(w^T x + b).
        """
        X = np.vstack([source_embeddings, target_embeddings])
        y = np.concatenate(
            [
                np.zeros(source_embeddings.shape[0]),
                np.ones(target_embeddings.shape[0]),
            ]
        )

        self._model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
        )
        self._model.fit(X, y)

    def weights(self, embeddings: np.ndarray) -> np.ndarray:
        """Return density ratio weights for each embedding.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to compute weights for, shape (n, d).

        Returns
        -------
        np.ndarray
            Density ratio weights, shape (n,). Strictly positive and finite,
            clipped to [1/max_weight, max_weight].
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Must call fit() before weights().")

        # Compute log-odds: w^T x + b
        # For logistic regression: P(1|x) / P(0|x) = exp(w^T x + b)
        decision = self._model.decision_function(embeddings)

        # Density ratio = exp(decision)
        # Clip decision values to avoid overflow
        max_log = np.log(self._max_weight)
        min_log = -max_log
        clipped_decision = np.clip(decision, min_log, max_log)

        ratios = np.exp(clipped_decision)

        # Clip to [1/C, C] for stability
        ratios = np.clip(ratios, 1.0 / self._max_weight, self._max_weight)

        return ratios
