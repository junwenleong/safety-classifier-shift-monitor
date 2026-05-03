"""Property tests for the VarianceDecomposer.

Tests Properties 24, 25, and 26 from the design document.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from shift_detection_monitor.evaluation.results import CellResult
from shift_detection_monitor.evaluation.variance_decomposer import VarianceDecomposer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_CLASSIFIERS = ["deberta-v3-large", "llama-guard-3-8b", "shieldgemma-9b"]
_SHIFTS = ["paraphrase", "code-switch", "adversarial-suffix"]
_REGIMES = ["regime_a", "regime_b", "regime_c"]


@st.composite
def st_factorial_results(
    draw: st.DrawFn,
    min_per_cell: int = 3,
    max_per_cell: int = 8,
) -> list[CellResult]:
    """Generate a list of CellResults forming a factorial design.

    Ensures multiple classifiers and shift conditions with enough
    observations per cell for meaningful ANOVA.
    """
    classifiers = draw(
        st.lists(
            st.sampled_from(_CLASSIFIERS),
            min_size=2,
            max_size=3,
            unique=True,
        )
    )
    shifts = draw(
        st.lists(
            st.sampled_from(_SHIFTS),
            min_size=2,
            max_size=3,
            unique=True,
        )
    )

    results = []
    for clf in classifiers:
        for shift in shifts:
            n = draw(st.integers(min_value=min_per_cell, max_value=max_per_cell))
            for i in range(n):
                latency = draw(
                    st.floats(
                        min_value=1.0,
                        max_value=1000.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                results.append(
                    CellResult(
                        classifier=clf,
                        shift_condition=shift,
                        ground_truth_regime=draw(st.sampled_from(_REGIMES)),
                        window_size=200,
                        seed=i,
                        detection_latency=latency,
                        false_alarm_rate=0.05,
                        alarms=[],
                        n_abstentions=0,
                        n_predictions=100,
                    )
                )

    return results


@st.composite
def st_results_with_insufficient_cells(draw: st.DrawFn) -> tuple[list[CellResult], int]:
    """Generate results where some cells have fewer observations than min_obs."""
    min_obs = draw(st.integers(min_value=3, max_value=10))

    results = []
    # One cell with enough observations
    for i in range(min_obs + 2):
        results.append(
            CellResult(
                classifier="deberta-v3-large",
                shift_condition="paraphrase",
                ground_truth_regime="regime_a",
                window_size=200,
                seed=i,
                detection_latency=float(50 + i),
                false_alarm_rate=0.05,
                alarms=[],
                n_abstentions=0,
                n_predictions=100,
            )
        )

    # One cell with insufficient observations
    n_insufficient = draw(st.integers(min_value=1, max_value=max(min_obs - 1, 1)))
    for i in range(n_insufficient):
        results.append(
            CellResult(
                classifier="llama-guard-3-8b",
                shift_condition="code-switch",
                ground_truth_regime="regime_b",
                window_size=100,
                seed=i,
                detection_latency=float(80 + i),
                false_alarm_rate=0.03,
                alarms=[],
                n_abstentions=0,
                n_predictions=100,
            )
        )

    return results, min_obs


# ---------------------------------------------------------------------------
# P24: Variance proportions sum to 1.0 within 1e-6
# ---------------------------------------------------------------------------


class TestVarianceProportionsSum:
    """Feature: shift-detection-monitor, Property 24: Variance Proportions Sum to Unity

    **Validates: Requirements 10.2**

    For any valid results with sufficient observations, variance proportions
    (all factors + interactions + residual) sum to 1.0 within 1e-6 tolerance.
    """

    @given(results=st_factorial_results(min_per_cell=3, max_per_cell=8))
    @settings(max_examples=100)
    def test_variance_proportions_sum_to_one(
        self, results: list[CellResult]
    ) -> None:
        """All variance proportions sum to 1.0."""
        assume(len(results) >= 4)

        decomposer = VarianceDecomposer(min_observations_per_cell=2)
        decomp = decomposer.fit(results)

        total = sum(decomp.factor_variances.values())
        total += sum(decomp.interaction_variances.values())
        total += decomp.residual_variance

        assert abs(total - 1.0) < 1e-6, (
            f"Variance proportions sum to {total}, expected 1.0. "
            f"Factors: {decomp.factor_variances}, "
            f"Interactions: {decomp.interaction_variances}, "
            f"Residual: {decomp.residual_variance}"
        )

    @given(results=st_factorial_results(min_per_cell=3, max_per_cell=8))
    @settings(max_examples=100)
    def test_all_proportions_non_negative(
        self, results: list[CellResult]
    ) -> None:
        """All variance proportions are non-negative."""
        assume(len(results) >= 4)

        decomposer = VarianceDecomposer(min_observations_per_cell=2)
        decomp = decomposer.fit(results)

        for name, prop in decomp.factor_variances.items():
            assert prop >= 0.0, f"Factor {name} has negative proportion: {prop}"

        for name, prop in decomp.interaction_variances.items():
            assert prop >= 0.0, f"Interaction {name} has negative proportion: {prop}"

        assert decomp.residual_variance >= 0.0, (
            f"Residual variance is negative: {decomp.residual_variance}"
        )


# ---------------------------------------------------------------------------
# P25: Insufficient cells flagged
# ---------------------------------------------------------------------------


class TestInsufficientCellFlagging:
    """Feature: shift-detection-monitor, Property 25: Insufficient Cell Flagging

    **Validates: Requirements 10.4**

    For any cell with fewer observations than configured minimum,
    that cell appears in flagged_cells.
    """

    @given(data=st_results_with_insufficient_cells())
    @settings(max_examples=100)
    def test_insufficient_cells_are_flagged(
        self, data: tuple[list[CellResult], int]
    ) -> None:
        """Cells with fewer observations than min_obs are flagged."""
        results, min_obs = data

        decomposer = VarianceDecomposer(min_observations_per_cell=min_obs)
        decomp = decomposer.fit(results)

        # Count observations per cell
        cell_counts: dict[str, int] = {}
        for r in results:
            if r.is_negative_control or r.is_positive_control:
                continue
            key = f"{r.classifier}:{r.shift_condition}:{r.ground_truth_regime}:{r.window_size}"
            cell_counts[key] = cell_counts.get(key, 0) + 1

        for key, count in cell_counts.items():
            if count < min_obs:
                assert key in decomp.flagged_cells, (
                    f"Cell {key} has {count} observations (< {min_obs}) "
                    f"but was not flagged. Flagged: {decomp.flagged_cells}"
                )


# ---------------------------------------------------------------------------
# P26: Effect sizes non-negative with valid CIs
# ---------------------------------------------------------------------------


class TestEffectSizesValid:
    """Feature: shift-detection-monitor, Property 26: Effect Sizes Non-Negative With Valid CIs

    **Validates: Requirements 9.5**

    For any factor, η² ≥ 0 and ci_lower ≤ estimate ≤ ci_upper.
    """

    @given(results=st_factorial_results(min_per_cell=3, max_per_cell=8))
    @settings(max_examples=100)
    def test_effect_sizes_non_negative(
        self, results: list[CellResult]
    ) -> None:
        """All effect size estimates are non-negative."""
        assume(len(results) >= 4)

        decomposer = VarianceDecomposer(min_observations_per_cell=2)
        decomp = decomposer.fit(results)

        for name, es in decomp.effect_sizes.items():
            assert es.estimate >= 0.0, (
                f"Effect size for {name} is negative: {es.estimate}"
            )

    @given(results=st_factorial_results(min_per_cell=3, max_per_cell=8))
    @settings(max_examples=100)
    def test_ci_contains_estimate(
        self, results: list[CellResult]
    ) -> None:
        """CI contains the point estimate: ci_lower ≤ estimate ≤ ci_upper."""
        assume(len(results) >= 4)

        decomposer = VarianceDecomposer(min_observations_per_cell=2)
        decomp = decomposer.fit(results)

        for name, es in decomp.effect_sizes.items():
            assert es.ci_lower <= es.estimate, (
                f"CI lower ({es.ci_lower}) > estimate ({es.estimate}) for {name}"
            )
            assert es.estimate <= es.ci_upper, (
                f"Estimate ({es.estimate}) > CI upper ({es.ci_upper}) for {name}"
            )
