"""
Conformal abstention layer for adapting classifier decision thresholds post-shift.

Implements split-conformal prediction sets with optional covariate-shift weighting
(Tibshirani et al. 2019). Supports unweighted and weighted-on-alarm modes.

Algorithm (unweighted):
1. Compute nonconformity scores on calibration set: α_i = 1 - f(x_i)_{y_i}
2. Compute quantile q̂ at level ⌈(1-ε)(n+1)⌉/n
3. Prediction set: C(x) = {y : 1 - f(x)_y ≤ q̂}
4. Abstain if |C(x)| > 1 or |C(x)| = 0

For weighted mode on alarm:
1. Estimate density ratios via DensityRatioEstimator
2. Compute weighted quantile
3. Update threshold
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import CalibrationError, ClassifierOutput, StreamRecord


@dataclass(frozen=True)
class CoverageStats:
    """Coverage statistics for the conformal abstention layer."""

    pre_shift_coverage: float
    post_shift_coverage: float | None
    n_abstentions: int
    n_predictions: int


class ConformalAbstentionLayer:
    """Computes split-conformal prediction sets and adapts thresholds post-shift.

    Assumes binary classification: safe=0, unsafe=1.

    Parameters
    ----------
    target_error_rate : float
        Target miscoverage rate ε ∈ (0, 1).
    conformal_mode : {"unweighted", "weighted-on-alarm"}
        Whether to use weighted conformal prediction after alarm.
    calibration_set : list[tuple[ClassifierOutput, int]]
        Calibration data: (classifier_output, true_label) pairs.
    min_calibration_size : int
        Minimum calibration set size. Raises CalibrationError if too small.
    """

    def __init__(
        self,
        target_error_rate: float,
        conformal_mode: Literal["unweighted", "weighted-on-alarm"],
        calibration_set: list[tuple[ClassifierOutput, int]],
        min_calibration_size: int = 50,
    ) -> None:
        if len(calibration_set) < min_calibration_size:
            raise CalibrationError(
                f"Calibration set has {len(calibration_set)} examples, "
                f"but minimum is {min_calibration_size}."
            )

        self._target_error_rate = target_error_rate
        self._conformal_mode = conformal_mode
        self._calibration_set = calibration_set
        self._min_calibration_size = min_calibration_size

        # Compute nonconformity scores and initial threshold
        self._nonconformity_scores = self._compute_nonconformity_scores(
            calibration_set
        )
        self._threshold = self._compute_quantile(
            self._nonconformity_scores, target_error_rate
        )

        # Coverage tracking
        self._n_predictions = 0
        self._n_abstentions = 0
        self._n_correct_pre = 0
        self._n_total_pre = 0
        self._n_correct_post = 0
        self._n_total_post = 0
        self._post_alarm = False

    @staticmethod
    def _compute_nonconformity_scores(
        calibration_set: list[tuple[ClassifierOutput, int]],
    ) -> np.ndarray:
        """Compute nonconformity scores: α_i = 1 - f(x_i)_{y_i}.

        For binary classification (safe=0, unsafe=1):
        - If true label is 1 (unsafe): score for class 1 = classifier score
          → α = 1 - score
        - If true label is 0 (safe): score for class 0 = 1 - classifier score
          → α = 1 - (1 - score) = score
        """
        scores = []
        for output, label in calibration_set:
            if label == 1:
                # Classifier score is P(unsafe)
                alpha = 1.0 - output.score
            else:
                # P(safe) = 1 - P(unsafe)
                alpha = output.score
            scores.append(alpha)
        return np.array(scores, dtype=np.float64)

    @staticmethod
    def _compute_quantile(scores: np.ndarray, epsilon: float) -> float:
        """Compute the conformal quantile q̂.

        q̂ = ⌈(1-ε)(n+1)⌉/n-th quantile of the nonconformity scores.
        """
        n = len(scores)
        if n == 0:
            return float("inf")

        level = math.ceil((1.0 - epsilon) * (n + 1)) / n
        # Clamp to [0, 1] for np.quantile
        level = min(level, 1.0)
        level = max(level, 0.0)
        return float(np.quantile(scores, level))

    @staticmethod
    def _compute_weighted_quantile(
        scores: np.ndarray, weights: np.ndarray, epsilon: float
    ) -> float:
        """Compute the weighted conformal quantile.

        q̂_w = inf{q : Σ_{i: α_i ≤ q} w̃_i ≥ 1 - ε}
        where w̃_i = w_i / (Σ_j w_j + 1)
        """
        n = len(scores)
        if n == 0:
            return float("inf")

        # Normalize weights
        total_weight = np.sum(weights) + 1.0
        normalized_weights = weights / total_weight

        # Sort by nonconformity score
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = normalized_weights[sorted_indices]

        # Find the smallest q such that cumulative weight ≥ 1 - ε
        cumulative = 0.0
        target = 1.0 - epsilon
        for i in range(n):
            cumulative += sorted_weights[i]
            if cumulative >= target:
                return float(sorted_scores[i])

        # If we never reach the target, return the maximum score
        return float(sorted_scores[-1])

    def predict_set(self, output: ClassifierOutput) -> set[int]:
        """Return the conformal prediction set for a classifier output.

        For binary classification (safe=0, unsafe=1):
        - Include class y if 1 - f(x)_y ≤ q̂

        Returns {0}, {1}, {0, 1} (abstain), or {} (abstain, conservative).
        """
        self._n_predictions += 1

        prediction_set: set[int] = set()

        # Check class 1 (unsafe): nonconformity = 1 - score
        alpha_1 = 1.0 - output.score
        if alpha_1 <= self._threshold:
            prediction_set.add(1)

        # Check class 0 (safe): nonconformity = score (since P(safe) = 1 - score)
        alpha_0 = output.score
        if alpha_0 <= self._threshold:
            prediction_set.add(0)

        # Track abstentions
        if len(prediction_set) != 1:
            self._n_abstentions += 1

        return prediction_set

    def update_coverage(self, output: ClassifierOutput, true_label: int) -> None:
        """Update coverage tracking with a labeled example."""
        pred_set = self.predict_set(output)
        # Undo the prediction count from predict_set since we already counted
        self._n_predictions -= 1

        covered = true_label in pred_set

        if not self._post_alarm:
            self._n_total_pre += 1
            if covered:
                self._n_correct_pre += 1
        else:
            self._n_total_post += 1
            if covered:
                self._n_correct_post += 1

    def on_alarm(
        self,
        post_alarm_records: list[StreamRecord],
        frozen_stats: FrozenReferenceStats,
    ) -> None:
        """Recompute thresholds using weighted conformal prediction.

        Only active when conformal_mode == "weighted-on-alarm".
        Estimates density ratios via DensityRatioEstimator.
        """
        self._post_alarm = True

        if self._conformal_mode != "weighted-on-alarm":
            return

        # Need embeddings for density ratio estimation
        target_embeddings = []
        for record in post_alarm_records:
            if record.representation is not None:
                target_embeddings.append(record.representation)

        if len(target_embeddings) == 0:
            # No embeddings available, fall back to unweighted
            return

        source_embeddings = frozen_stats.reference_embeddings
        if source_embeddings.size == 0:
            return

        target_arr = np.array(target_embeddings, dtype=np.float64)

        # Import here to avoid circular dependency
        from shift_detection_monitor.adaptation.density_ratio import (
            DensityRatioEstimator,
        )

        estimator = DensityRatioEstimator(method="logistic")
        estimator.fit(source_embeddings, target_arr)

        # Compute weights for calibration examples
        cal_embeddings = []
        cal_indices = []
        for i, (output, _label) in enumerate(self._calibration_set):
            if output.representation is not None:
                cal_embeddings.append(output.representation)
                cal_indices.append(i)

        if len(cal_embeddings) == 0:
            return

        cal_arr = np.array(cal_embeddings, dtype=np.float64)
        weights = estimator.weights(cal_arr)

        # Recompute threshold with weights
        cal_scores = self._nonconformity_scores[cal_indices]
        self._threshold = self._compute_weighted_quantile(
            cal_scores, weights, self._target_error_rate
        )

    @property
    def coverage_stats(self) -> CoverageStats:
        """Return pre-shift and post-shift empirical coverage."""
        pre_coverage = (
            self._n_correct_pre / self._n_total_pre if self._n_total_pre > 0 else 0.0
        )
        post_coverage = (
            self._n_correct_post / self._n_total_post
            if self._n_total_post > 0
            else None
        )

        return CoverageStats(
            pre_shift_coverage=pre_coverage,
            post_shift_coverage=post_coverage,
            n_abstentions=self._n_abstentions,
            n_predictions=self._n_predictions,
        )

    @property
    def threshold(self) -> float:
        """Current conformal threshold."""
        return self._threshold
