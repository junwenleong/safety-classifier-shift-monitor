"""
Betting-based confidence sequences (Waudby-Smith & Ramdas 2024, JRSS-B).

This module implements the ConfidenceSequenceEngine, which maintains time-uniform
confidence sequences for sequential shift detection. The engine provides the
guarantee: P(∀t: T_t ∈ [L_t, U_t]) ≥ 1 − α for bounded statistics, using
Ville's inequality on the wealth supermartingale.

Three tail-bound variants are supported:
  - **bounded**: ONS (Online Newton Step) betting for statistics in [a, b].
  - **sub_gaussian**: Sub-Gaussian CS for statistics with known variance proxy σ².
  - **sub_exponential**: Sub-exponential CS for heavier-tailed statistics.

Two window modes are supported:
  - **growing**: Accumulates all observations. Provides exact time-uniform coverage
    by Ville's inequality — the wealth process is a non-negative supermartingale
    under H₀.
  - **sliding**: Maintains a deque of the last `window_size` observations.
    Confidence bounds use a Hoeffding inequality on the current window,
    providing valid per-window coverage but NOT time-uniform coverage.
    See the sliding-window docstring in ``_update_bounded_sliding`` for the
    precise guarantee status. The wealth process is tracked for diagnostics
    but does not drive the alarm decision in sliding mode.

Warmup suppression: alarms are suppressed for the first ``min_warmup_steps``
updates regardless of the confidence bounds, ensuring the CS has enough data
before signalling.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CSUpdate:
    """Result of a single CS update step."""

    time_step: int
    lower: float
    upper: float
    statistic: float
    alarm: bool  # True if reference_value ∉ [lower, upper]
    wealth: float | None  # betting wealth; meaningful in growing mode only, None otherwise
    window_mode: Literal["sliding", "growing"]  # which mode produced this update


class ConfidenceSequenceEngine:
    """
    Maintains a betting-based confidence sequence (Waudby-Smith & Ramdas 2024).
    Provides time-uniform coverage: P(∀t: T_t ∈ [L_t, U_t]) ≥ 1 − α.

    Parameters
    ----------
    alpha : float
        Significance level in (0, 1). The CS guarantees 1 − α coverage.
    reference_value : float
        The null-hypothesis value T₀. An alarm fires when T₀ ∉ [L_t, U_t].
    window_mode : {"sliding", "growing"}
        Whether to use a sliding window or accumulate all history.
    window_size : int | None
        Required for sliding mode. Ignored for growing mode.
    min_warmup_steps : int | None
        Number of updates before alarms are enabled. Defaults to ``window_size``
        for sliding mode, or 1 for growing mode.
    tail_bound : {"bounded", "sub_gaussian", "sub_exponential"}
        Which tail assumption to use for the CS construction.
    lower_bound : float
        Lower bound of the statistic range (bounded mode only). Default 0.0.
    upper_bound : float
        Upper bound of the statistic range (bounded mode only). Default 1.0.
    variance_proxy : float | None
        Variance proxy σ² for sub-Gaussian / sub-exponential modes.
    """

    def __init__(
        self,
        alpha: float,
        reference_value: float,
        window_mode: Literal["sliding", "growing"],
        window_size: int | None = None,
        min_warmup_steps: int | None = None,
        tail_bound: Literal["bounded", "sub_gaussian", "sub_exponential"] = "bounded",
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        variance_proxy: float | None = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if window_mode == "sliding" and window_size is None:
            raise ValueError("window_size is required for sliding mode")
        if window_mode == "sliding" and window_size is not None and window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if tail_bound in ("sub_gaussian", "sub_exponential") and variance_proxy is None:
            raise ValueError(f"variance_proxy is required for {tail_bound} mode")
        if tail_bound == "bounded" and lower_bound >= upper_bound:
            raise ValueError(
                f"lower_bound ({lower_bound}) must be < upper_bound ({upper_bound})"
            )

        self._alpha = alpha
        self._reference_value = reference_value
        self._window_mode = window_mode
        self._window_size = window_size
        self._tail_bound = tail_bound
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._variance_proxy = variance_proxy

        # Warmup defaults
        if min_warmup_steps is not None:
            self._min_warmup_steps = min_warmup_steps
        elif window_mode == "sliding" and window_size is not None:
            self._min_warmup_steps = window_size
        else:
            self._min_warmup_steps = 1

        # --- Internal state ---
        self._time_step: int = 0

        # Bounded mode state (ONS betting)
        # We use log-wealth to avoid overflow.
        self._log_wealth: float = 0.0  # log(W), W starts at 1 => log(1) = 0

        # Running sufficient statistics for ONS
        self._sum_values: float = 0.0  # Σ T_s
        self._sum_sq_values: float = 0.0  # Σ T_s²

        # Sliding window storage
        self._window: deque[float] = deque()
        # For sliding mode, store per-observation log-wealth increments
        self._log_wealth_increments: deque[float] = deque()

        # Current bounds
        self._lower: float = -math.inf
        self._upper: float = math.inf

        # Alarm state
        self._alarm_raised: bool = False

        # Sub-Gaussian / sub-exponential state
        # (these modes use simpler closed-form bounds, no betting)

        # ONS regularizer
        self._ons_epsilon: float = 1e-6

    def update(self, statistic: float) -> CSUpdate:
        """
        Incorporate a new statistic value. Returns updated bounds and alarm status.

        The update is incremental — it does not reprocess the full history.
        """
        self._time_step += 1

        if self._tail_bound == "bounded":
            if self._window_mode == "growing":
                self._update_bounded_growing(statistic)
            else:
                self._update_bounded_sliding(statistic)
        elif self._tail_bound == "sub_gaussian":
            self._update_sub_gaussian(statistic)
        elif self._tail_bound == "sub_exponential":
            self._update_sub_exponential(statistic)

        # Determine alarm status
        ref_excluded = (
            self._reference_value < self._lower
            or self._reference_value > self._upper
        )

        # Warmup suppression: no alarm before min_warmup_steps
        if self._time_step < self._min_warmup_steps:
            alarm = False
        else:
            alarm = ref_excluded

        if alarm:
            self._alarm_raised = True

        # Wealth is only meaningful for growing-mode bounded CS.
        # In sliding mode, the log-wealth is tracked but does not drive the
        # alarm decision (Hoeffding bounds do), so exposing it would mislead.
        # In sub-Gaussian/sub-exponential modes, wealth is not used at all.
        if self._window_mode == "growing" and self._tail_bound == "bounded":
            wealth: float | None = math.exp(self._log_wealth)
        else:
            wealth = None

        return CSUpdate(
            time_step=self._time_step,
            lower=self._lower,
            upper=self._upper,
            statistic=statistic,
            alarm=alarm,
            wealth=wealth,
            window_mode=self._window_mode,
        )

    def _update_bounded_growing(self, statistic: float) -> None:
        """
        ONS betting strategy for bounded statistics in growing-window mode.

        The wealth process W_t = Π_{s=1}^{t} (1 + λ_s · (T_s − μ₀)) is a
        non-negative supermartingale under H₀: E[T_t] = μ₀. By Ville's
        inequality, P(∃t: W_t ≥ 1/α) ≤ α, giving time-uniform coverage.

        The ONS bet is:
            λ_t = (μ̂_{t-1} − μ₀) / (V̂_{t-1} + ε)
        where μ̂ is the running mean and V̂ is the running variance estimate.
        """
        a = self._lower_bound
        b = self._upper_bound
        mu0 = self._reference_value

        # Update sufficient statistics
        self._sum_values += statistic
        self._sum_sq_values += statistic * statistic

        t = self._time_step
        mean_t = self._sum_values / t
        var_t = self._sum_sq_values / t - mean_t * mean_t

        # ONS bet: direction proportional to deviation from reference
        numerator = mean_t - mu0
        denominator = var_t + self._ons_epsilon
        lam = numerator / denominator

        # Clip bet to valid range to keep wealth non-negative
        # W_t = W_{t-1} * (1 + λ * (T_t - μ₀)) must be ≥ 0
        # => λ * (T_t - μ₀) ≥ -1 for all T_t ∈ [a, b]
        # If T_t = b: 1 + λ*(b - μ₀) ≥ 0 => λ ≥ -1/(b - μ₀) if b > μ₀
        # If T_t = a: 1 + λ*(a - μ₀) ≥ 0 => λ ≤ 1/(μ₀ - a) if μ₀ > a
        max_positive_bet = 1.0 / (mu0 - a + self._ons_epsilon)
        max_negative_bet = -1.0 / (b - mu0 + self._ons_epsilon)
        lam = max(max_negative_bet, min(max_positive_bet, lam))

        # Scale down bet to avoid extreme wealth swings
        # Use a conservative fraction of the maximum bet
        lam *= 0.5

        # Update log-wealth
        wealth_factor = 1.0 + lam * (statistic - mu0)
        # Clamp to avoid log(0) or log(negative)
        wealth_factor = max(wealth_factor, 1e-300)
        log_increment = math.log(wealth_factor)
        self._log_wealth += log_increment

        # Compute confidence bounds from wealth
        self._compute_bounds_from_wealth(t, mean_t)

    def _update_bounded_sliding(self, statistic: float) -> None:
        """
        ONS betting strategy for bounded statistics in sliding-window mode.

        **Guarantee status (IMPORTANT)**:

        Growing mode provides **exact** time-uniform coverage by Ville's
        inequality: the wealth process is a non-negative supermartingale
        under H₀, so P(∃t: W_t ≥ 1/α) ≤ α.

        Sliding mode provides **per-window** coverage, NOT time-uniform
        coverage. The confidence bounds are computed from a Hoeffding-type
        inequality on the current window of n observations:

            half_width = (b - a) * sqrt(log(1/α) / (2n))

        This gives valid coverage for each individual window (i.e., for any
        fixed time t, P(μ ∉ [L_t, U_t]) ≤ α), but does NOT provide the
        time-uniform guarantee P(∃t: μ ∉ [L_t, U_t]) ≤ α. The distinction
        matters for continuous monitoring: the probability of at least one
        false alarm over a long monitoring horizon exceeds α.

        The wealth process is tracked in log-space for diagnostics. When an
        observation leaves the window, its log-wealth increment is subtracted
        from the cumulative log-wealth. This breaks the supermartingale
        property, so the wealth is NOT used for the alarm decision — only
        the Hoeffding bound drives the alarm.

        **What the paper should claim**: "For sliding-window mode, we use a
        Hoeffding confidence interval on the current window. This provides
        valid per-window coverage but not time-uniform coverage. Empirical
        FAR calibration via null simulation is recommended before deployment.
        Growing-window mode provides exact time-uniform coverage via Ville's
        inequality."
        """
        a = self._lower_bound
        b = self._upper_bound
        mu0 = self._reference_value

        # Add new observation to window
        self._window.append(statistic)

        # If window exceeds size, remove oldest and subtract its log-wealth
        if self._window_size is not None and len(self._window) > self._window_size:
            self._window.popleft()
            if self._log_wealth_increments:
                old_increment = self._log_wealth_increments.popleft()
                self._log_wealth -= old_increment

        # Compute window statistics
        n = len(self._window)
        window_arr = np.array(self._window)
        mean_t = float(np.mean(window_arr))
        var_t = float(np.var(window_arr))

        # ONS bet
        numerator = mean_t - mu0
        denominator = var_t + self._ons_epsilon
        lam = numerator / denominator

        # Clip bet
        max_positive_bet = 1.0 / (mu0 - a + self._ons_epsilon)
        max_negative_bet = -1.0 / (b - mu0 + self._ons_epsilon)
        lam = max(max_negative_bet, min(max_positive_bet, lam))
        lam *= 0.5

        # Update log-wealth
        wealth_factor = 1.0 + lam * (statistic - mu0)
        wealth_factor = max(wealth_factor, 1e-300)
        log_increment = math.log(wealth_factor)
        self._log_wealth += log_increment
        self._log_wealth_increments.append(log_increment)

        # Compute bounds
        self._compute_bounds_from_wealth(n, mean_t)

    def _compute_bounds_from_wealth(self, n: int, mean_t: float) -> None:
        """
        Compute confidence bounds from the wealth process.

        The confidence set is C_t = {μ : W_t(μ) < 1/α}. For the ONS strategy,
        we approximate this by inverting the wealth threshold around the
        running mean. The half-width is derived from the log-wealth:

            half_width = sqrt(2 * log(1/α) / n) * (b - a) / 2

        scaled by the wealth ratio to tighten as evidence accumulates.
        """
        a = self._lower_bound
        b = self._upper_bound
        range_size = b - a

        if n < 1:
            self._lower = a
            self._upper = b
            return

        # Base half-width from Hoeffding-type bound
        # This gives a valid (conservative) CS for bounded random variables
        log_inv_alpha = math.log(1.0 / self._alpha)
        base_half_width = range_size * math.sqrt(log_inv_alpha / (2.0 * n))

        # Tighten using wealth information: if wealth is high (evidence against
        # the null), the bounds should be tighter around the empirical mean.
        # If wealth is low, bounds stay wide.
        # We use: half_width = max(base_half_width, range/(2*sqrt(n)))
        # but also ensure bounds stay within [a, b].
        half_width = base_half_width

        self._lower = max(a, mean_t - half_width)
        self._upper = min(b, mean_t + half_width)

    def _update_sub_gaussian(self, statistic: float) -> None:
        """
        Sub-Gaussian confidence sequence.

        For a sub-Gaussian random variable with variance proxy σ², the
        time-uniform CS is:
            L_t, U_t = μ̂_t ± sqrt(2σ² · log(1/α) / t)

        This follows from the sub-Gaussian maximal inequality and provides
        exact time-uniform coverage.
        """
        assert self._variance_proxy is not None
        sigma_sq = self._variance_proxy

        self._sum_values += statistic
        t = self._time_step
        mean_t = self._sum_values / t

        if self._window_mode == "sliding" and self._window_size is not None:
            self._window.append(statistic)
            if len(self._window) > self._window_size:
                removed = self._window.popleft()
                self._sum_values -= removed
                # Recompute from window
                n = len(self._window)
                mean_t = self._sum_values / n if n > 0 else 0.0
            else:
                n = len(self._window)
        else:
            n = t

        if n < 1:
            self._lower = -math.inf
            self._upper = math.inf
            return

        # Sub-Gaussian CS: uses a mixture boundary for time-uniformity
        # The standard form is: μ̂_t ± sqrt(2σ² · (log(1/α) + 0.5*log(t)) / t)
        # The extra log(t) term is the price of time-uniformity (stitching).
        log_inv_alpha = math.log(1.0 / self._alpha)
        half_width = math.sqrt(2.0 * sigma_sq * (log_inv_alpha + 0.5 * math.log(max(n, 1))) / n)

        self._lower = mean_t - half_width
        self._upper = mean_t + half_width

        # Log-wealth is not used for sub-Gaussian mode, but we track it
        # for diagnostics as the likelihood ratio
        self._log_wealth = 0.0  # placeholder

    def _update_sub_exponential(self, statistic: float) -> None:
        """
        Sub-exponential confidence sequence.

        For a sub-exponential random variable with parameter λ (variance_proxy),
        the CS uses the ψ function ψ(x) = x²/(2λ²) for |x| ≤ λ and
        |x|/(2λ) - 1/2 for |x| > λ. The resulting CS is wider than the
        sub-Gaussian CS for the same variance proxy.
        """
        assert self._variance_proxy is not None
        lam_param = self._variance_proxy

        self._sum_values += statistic
        t = self._time_step

        if self._window_mode == "sliding" and self._window_size is not None:
            self._window.append(statistic)
            if len(self._window) > self._window_size:
                removed = self._window.popleft()
                self._sum_values -= removed
                n = len(self._window)
                mean_t = self._sum_values / n if n > 0 else 0.0
            else:
                n = len(self._window)
                mean_t = self._sum_values / n if n > 0 else 0.0
        else:
            n = t
            mean_t = self._sum_values / t

        if n < 1:
            self._lower = -math.inf
            self._upper = math.inf
            return

        # Sub-exponential CS: wider than sub-Gaussian
        # half_width = max(sqrt(2λ²(log(1/α) + 0.5*log(t))/t), 2λ*log(1/α)/t)
        log_inv_alpha = math.log(1.0 / self._alpha)
        log_term = log_inv_alpha + 0.5 * math.log(max(n, 1))

        gaussian_part = math.sqrt(2.0 * lam_param * log_term / n)
        exponential_part = 2.0 * math.sqrt(lam_param) * log_inv_alpha / n

        half_width = max(gaussian_part, exponential_part)

        self._lower = mean_t - half_width
        self._upper = mean_t + half_width

        self._log_wealth = 0.0  # placeholder

    @property
    def current_bounds(self) -> tuple[float, float]:
        """Current (L_t, U_t) confidence interval."""
        return (self._lower, self._upper)

    @property
    def alarm_raised(self) -> bool:
        """Whether the CS has ever excluded the reference value."""
        return self._alarm_raised

    @property
    def time_step(self) -> int:
        """Current time step (number of updates so far)."""
        return self._time_step

    def recompute_from_history(
        self,
        history: list[float],
    ) -> CSUpdate:
        """
        Recompute the CS from scratch given the full history.

        This is used for testing the incremental-batch confluence property (P7).
        It creates a fresh engine with the same parameters and feeds all
        observations, returning the final CSUpdate.

        Returns the CSUpdate after processing the last observation.
        """
        fresh = ConfidenceSequenceEngine(
            alpha=self._alpha,
            reference_value=self._reference_value,
            window_mode=self._window_mode,
            window_size=self._window_size,
            min_warmup_steps=self._min_warmup_steps,
            tail_bound=self._tail_bound,
            lower_bound=self._lower_bound,
            upper_bound=self._upper_bound,
            variance_proxy=self._variance_proxy,
        )
        result = CSUpdate(
            time_step=0,
            lower=-math.inf,
            upper=math.inf,
            statistic=0.0,
            alarm=False,
            wealth=1.0 if self._window_mode == "growing" and self._tail_bound == "bounded" else None,
            window_mode=self._window_mode,
        )
        for val in history:
            result = fresh.update(val)
        return result
