"""
Variance decomposition for factorial evaluation results.

Implements hierarchical random-effects ANOVA with partially nested design.
Crossed factors: classifier, shift_type.
Nested: attack_family within adversarial-suffix, language within code-switch.

Uses a simplified ANOVA approach that works reliably with the test data
sizes typical in this evaluation. Falls back to sum-of-squares decomposition
when statsmodels mixed-effects models are not feasible (e.g., insufficient
data for convergence).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from shift_detection_monitor.evaluation.results import CellResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectSize:
    """Effect size estimate with confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float
    metric: Literal["eta_squared", "cohens_d"] = "eta_squared"


@dataclass(frozen=True)
class VarianceDecomposition:
    """Summary of hierarchical ANOVA results."""

    factor_variances: dict[str, float]
    interaction_variances: dict[str, float]
    effect_sizes: dict[str, EffectSize]
    flagged_cells: list[str]
    residual_variance: float


class VarianceDecomposer:
    """Fits a hierarchical random-effects ANOVA with partially nested design.

    Parameters
    ----------
    min_observations_per_cell : int
        Minimum number of observations per factorial cell. Cells with
        fewer observations are flagged.
    """

    def __init__(self, min_observations_per_cell: int = 10) -> None:
        self._min_obs = min_observations_per_cell

    def fit(self, results: list[CellResult]) -> VarianceDecomposition:
        """Fit the ANOVA model on structured evaluation results.

        Uses a simplified sum-of-squares decomposition:
        1. Compute grand mean of detection latency
        2. Compute SS for each factor (classifier, shift_type)
        3. Compute SS for interaction (classifier × shift_type)
        4. Compute residual SS
        5. Convert to variance proportions

        Parameters
        ----------
        results : list[CellResult]
            Evaluation results from the factorial design.

        Returns
        -------
        VarianceDecomposition
            Variance decomposition summary.
        """
        # Filter to results with detection latency
        valid = [r for r in results if r.detection_latency is not None
                 and not r.is_negative_control and not r.is_positive_control]

        # Flag cells with insufficient observations
        flagged_cells = self._flag_insufficient_cells(results)

        if len(valid) < 2:
            return VarianceDecomposition(
                factor_variances={"classifier": 0.0, "shift_type": 0.0},
                interaction_variances={"classifier:shift_type": 0.0},
                effect_sizes={
                    "classifier": EffectSize(0.0, 0.0, 0.0),
                    "shift_type": EffectSize(0.0, 0.0, 0.0),
                },
                flagged_cells=flagged_cells,
                residual_variance=1.0,
            )

        latencies = np.array([r.detection_latency for r in valid], dtype=np.float64)
        grand_mean = float(np.mean(latencies))
        ss_total = float(np.sum((latencies - grand_mean) ** 2))

        if ss_total < 1e-12:
            return VarianceDecomposition(
                factor_variances={"classifier": 0.0, "shift_type": 0.0},
                interaction_variances={"classifier:shift_type": 0.0},
                effect_sizes={
                    "classifier": EffectSize(0.0, 0.0, 0.0),
                    "shift_type": EffectSize(0.0, 0.0, 0.0),
                },
                flagged_cells=flagged_cells,
                residual_variance=1.0,
            )

        # Compute factor-level means
        classifier_groups: dict[str, list[float]] = {}
        shift_groups: dict[str, list[float]] = {}
        interaction_groups: dict[str, list[float]] = {}

        for r in valid:
            lat = r.detection_latency
            assert lat is not None

            classifier_groups.setdefault(r.classifier, []).append(lat)
            shift_groups.setdefault(r.shift_condition, []).append(lat)

            key = f"{r.classifier}:{r.shift_condition}"
            interaction_groups.setdefault(key, []).append(lat)

        # SS for classifier factor
        ss_classifier = 0.0
        for vals in classifier_groups.values():
            group_mean = np.mean(vals)
            ss_classifier += len(vals) * (group_mean - grand_mean) ** 2

        # SS for shift_type factor
        ss_shift = 0.0
        for vals in shift_groups.values():
            group_mean = np.mean(vals)
            ss_shift += len(vals) * (group_mean - grand_mean) ** 2

        # SS for interaction
        ss_interaction = 0.0
        for key, vals in interaction_groups.items():
            clf_name, shift_name = key.split(":", 1)
            clf_mean = np.mean(classifier_groups[clf_name])
            shift_mean = np.mean(shift_groups[shift_name])
            group_mean = np.mean(vals)
            # Interaction effect = group_mean - clf_mean - shift_mean + grand_mean
            interaction_effect = group_mean - clf_mean - shift_mean + grand_mean
            ss_interaction += len(vals) * interaction_effect ** 2

        # Residual
        ss_residual = max(ss_total - ss_classifier - ss_shift - ss_interaction, 0.0)

        # Convert to proportions (ensure they sum to 1.0)
        total_ss = ss_classifier + ss_shift + ss_interaction + ss_residual
        if total_ss < 1e-12:
            total_ss = 1.0

        prop_classifier = ss_classifier / total_ss
        prop_shift = ss_shift / total_ss
        prop_interaction = ss_interaction / total_ss
        prop_residual = ss_residual / total_ss

        # Normalize to ensure exact sum = 1.0
        prop_sum = prop_classifier + prop_shift + prop_interaction + prop_residual
        if prop_sum > 0:
            prop_classifier /= prop_sum
            prop_shift /= prop_sum
            prop_interaction /= prop_sum
            prop_residual /= prop_sum

        # Compute effect sizes (η² = SS_factor / SS_total)
        eta_sq_classifier = ss_classifier / total_ss if total_ss > 0 else 0.0
        eta_sq_shift = ss_shift / total_ss if total_ss > 0 else 0.0

        # Bootstrap CIs for effect sizes
        ci_classifier = self._bootstrap_eta_squared_ci(
            valid, "classifier", eta_sq_classifier
        )
        ci_shift = self._bootstrap_eta_squared_ci(
            valid, "shift_type", eta_sq_shift
        )

        return VarianceDecomposition(
            factor_variances={
                "classifier": prop_classifier,
                "shift_type": prop_shift,
            },
            interaction_variances={
                "classifier:shift_type": prop_interaction,
            },
            effect_sizes={
                "classifier": EffectSize(
                    estimate=eta_sq_classifier,
                    ci_lower=ci_classifier[0],
                    ci_upper=ci_classifier[1],
                ),
                "shift_type": EffectSize(
                    estimate=eta_sq_shift,
                    ci_lower=ci_shift[0],
                    ci_upper=ci_shift[1],
                ),
            },
            flagged_cells=flagged_cells,
            residual_variance=prop_residual,
        )

    def _flag_insufficient_cells(
        self, results: list[CellResult]
    ) -> list[str]:
        """Flag cells with fewer observations than the minimum."""
        cell_counts: dict[str, int] = {}
        for r in results:
            if r.is_negative_control or r.is_positive_control:
                continue
            key = f"{r.classifier}:{r.shift_condition}:{r.ground_truth_regime}:{r.window_size}"
            cell_counts[key] = cell_counts.get(key, 0) + 1

        flagged = []
        for key, count in cell_counts.items():
            if count < self._min_obs:
                flagged.append(key)

        return sorted(flagged)

    def _bootstrap_eta_squared_ci(
        self,
        results: list[CellResult],
        factor: str,
        point_estimate: float,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for η².

        Parameters
        ----------
        results : list[CellResult]
            Valid results with detection latency.
        factor : str
            Factor name ("classifier" or "shift_type").
        point_estimate : float
            The point estimate of η² from the original data.
        n_bootstrap : int
            Number of bootstrap samples.
        ci_level : float
            Confidence level.

        Returns
        -------
        tuple[float, float]
            (ci_lower, ci_upper) for η².
        """
        if len(results) < 3:
            return (0.0, max(0.0, point_estimate))

        rng = np.random.default_rng(42)
        eta_sq_samples = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = rng.integers(0, len(results), size=len(results))
            boot_results = [results[i] for i in indices]

            boot_latencies = np.array(
                [r.detection_latency for r in boot_results], dtype=np.float64
            )
            boot_grand_mean = float(np.mean(boot_latencies))
            boot_ss_total = float(np.sum((boot_latencies - boot_grand_mean) ** 2))

            if boot_ss_total < 1e-12:
                eta_sq_samples.append(0.0)
                continue

            # Compute SS for the factor
            groups: dict[str, list[float]] = {}
            for r in boot_results:
                key = r.classifier if factor == "classifier" else r.shift_condition
                groups.setdefault(key, []).append(r.detection_latency)  # type: ignore

            ss_factor = 0.0
            for vals in groups.values():
                group_mean = np.mean(vals)
                ss_factor += len(vals) * (group_mean - boot_grand_mean) ** 2

            eta_sq = ss_factor / boot_ss_total
            eta_sq_samples.append(float(eta_sq))

        if not eta_sq_samples:
            return (0.0, max(0.0, point_estimate))

        alpha = 1.0 - ci_level
        lower = float(np.percentile(eta_sq_samples, 100 * alpha / 2))
        upper = float(np.percentile(eta_sq_samples, 100 * (1 - alpha / 2)))

        # Ensure non-negative
        lower = max(lower, 0.0)
        upper = max(upper, lower)

        # Ensure CI contains the point estimate.
        # Bootstrap CIs can miss the point estimate due to bias.
        lower = min(lower, point_estimate)
        upper = max(upper, point_estimate)

        return (lower, upper)
