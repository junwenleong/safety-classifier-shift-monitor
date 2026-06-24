"""
Conformal test martingale detectors for anytime-valid shift monitoring.

Implements betting martingales on conformal p-values derived from the frozen
reference CDF. Three variants with different memory/accumulation properties:

- **PointMartingale**: Single accumulator from t=0. Simple but diluted by
  pre-shift observations — best for known-changepoint settings.
- **CUSUMMartingale**: Resets wealth to 1 when it drops below 1 (Page's rule).
  Adaptive to unknown changepoints; bounded memory in practice.
- **ScanMartingale**: Maintains W concurrent sub-martingales, one started per
  step. Alarms if any reaches threshold log(W/α) (union bound). Best for
  unknown changepoints with formal guarantee.

All three provide anytime-valid Type I error control:
  P(∃t: alarm under H₀) ≤ α

under the assumption that the stream scores are exchangeable with the reference
(i.e., drawn from the same distribution). The conformal p-values are valid
under exchangeability; if the stream is non-exchangeable (even benignly),
the guarantee degrades (see AV5 in FOLLOW_UP_EXPERIMENTS.md).

References:
  - Vovk, V. (2021). Testing randomness online. Statistical Science.
  - Howard, S.R., Ramdas, A., McAuliffe, J., Sekhon, J. (2021).
    Time-uniform, nonparametric, nonasymptotic confidence sequences. AoS.
  - Shin, J., Ramdas, A., Rinaldo, A. (2022). Nonparametric iterated-
    logarithm extensions of the sequential probability ratio test. JASA.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import CalibrationError, StreamRecord


@dataclass(frozen=True)
class MartingaleAlarm:
    """Structured alarm result from a martingale detector."""

    time_step: int
    log_wealth: float  # max log-wealth across all active sub-martingales
    threshold: float  # log-threshold for alarm
    alarmed: bool
    p_value: float  # conformal p-value at this step


def _conformal_p_two_sided(score: float, reference_cdf: np.ndarray) -> float:
    """Compute two-sided conformal p-value against the empirical reference CDF.

    Under H₀ (score is exchangeable with reference), the p-value is
    uniformly distributed on (0, 1). Under H₁ (shifted), it concentrates
    near 0 (score is extreme relative to reference).

    Two-sided: p = 2 * min(p_upper, p_lower), clipped to [0, 1].
    The +1 in numerator/denominator ensures p ∈ (0, 1] (never exactly 0),
    which is required for the log-betting to be finite.

    Parameters
    ----------
    score : float
        The observed score to test.
    reference_cdf : np.ndarray
        Sorted array of reference scores (the empirical CDF).

    Returns
    -------
    float
        Two-sided conformal p-value in (0, 1].
    """
    n = len(reference_cdf)
    # p_upper = P(ref >= score) with continuity correction
    n_geq = np.searchsorted(reference_cdf, score, side="left")
    p_upper = (n - n_geq + 1) / (n + 1)
    # p_lower = P(ref <= score) with continuity correction
    n_leq = np.searchsorted(reference_cdf, score, side="right")
    p_lower = (n_leq + 1) / (n + 1)
    return min(2.0 * min(p_upper, p_lower), 1.0)


def _bet_log_increment(p: float, epsilon: float) -> float:
    """Compute the log-wealth increment for a single bet.

    The betting function is the power method: f(p) = ε * p^(ε-1).
    Under H₀ (p ~ Uniform[0,1]), E[f(p)] = 1 (fair game).
    Under H₁ (p concentrated near 0), E[f(p)] > 1 (profitable).

    The log-increment is: log(ε) + (ε - 1) * log(p).

    Parameters
    ----------
    p : float
        Conformal p-value. Must be > 0.
    epsilon : float
        Betting parameter in (0, 1). Smaller = more conservative but
        more powerful against small p-values.

    Returns
    -------
    float
        Log of the wealth multiplicative factor.
    """
    p = max(p, 1e-300)  # prevent log(0)
    return math.log(epsilon) + (epsilon - 1.0) * math.log(p)


class PointMartingale:
    """Single-accumulator martingale from t=0.

    Accumulates log-wealth from the start. Alarms when log_wealth ≥ log(1/α).
    Simple but weak for unknown changepoints: pre-shift observations
    (which produce p ≈ Uniform) dilute the accumulated evidence.

    Parameters
    ----------
    frozen_stats : FrozenReferenceStats
        Reference statistics. Uses reference_cdf for conformal p-values.
    alpha : float
        Significance level. Alarm fires when wealth ≥ 1/α.
    epsilon : float
        Betting parameter in (0, 1). Default 0.3 (validated in Gate B).
    """

    def __init__(
        self,
        frozen_stats: FrozenReferenceStats,
        alpha: float = 0.05,
        epsilon: float = 0.3,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < epsilon < 1.0):
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if frozen_stats.n_reference < 2:
            raise CalibrationError(
                f"Need at least 2 reference scores, got {frozen_stats.n_reference}"
            )

        self._reference_cdf = frozen_stats.reference_cdf
        self._alpha = alpha
        self._epsilon = epsilon
        self._log_threshold = math.log(1.0 / alpha)
        self._log_wealth: float = 0.0
        self._time_step: int = 0
        self._alarm_step: int | None = None

    def update(self, record: StreamRecord) -> float:
        """Process one record. Returns current log-wealth (detection statistic).

        A return value ≥ log(1/α) indicates an alarm.
        """
        self._time_step += 1
        p = _conformal_p_two_sided(record.score, self._reference_cdf)
        self._log_wealth += _bet_log_increment(p, self._epsilon)

        if self._alarm_step is None and self._log_wealth >= self._log_threshold:
            self._alarm_step = self._time_step

        return self._log_wealth

    def update_detailed(self, record: StreamRecord) -> MartingaleAlarm:
        """Process one record with full diagnostic output."""
        p = _conformal_p_two_sided(record.score, self._reference_cdf)
        self._time_step += 1
        self._log_wealth += _bet_log_increment(p, self._epsilon)

        alarmed = self._log_wealth >= self._log_threshold
        if self._alarm_step is None and alarmed:
            self._alarm_step = self._time_step

        return MartingaleAlarm(
            time_step=self._time_step,
            log_wealth=self._log_wealth,
            threshold=self._log_threshold,
            alarmed=alarmed,
            p_value=p,
        )

    @property
    def alarm_step(self) -> int | None:
        """Time step at which alarm first fired, or None."""
        return self._alarm_step

    @property
    def log_wealth(self) -> float:
        """Current cumulative log-wealth."""
        return self._log_wealth

    @property
    def threshold(self) -> float:
        """Log-threshold for alarm."""
        return self._log_threshold


class CUSUMMartingale:
    """CUSUM-style martingale with reset at wealth=1.

    Resets log-wealth to 0 whenever it drops below 0 (Page's rule).
    Equivalent to taking the maximum over all possible changepoint locations
    of the post-change likelihood ratio. Provides adaptive detection
    without the dilution problem of PointMartingale.

    NOTE ON FAR GUARANTEE: The Page's reset gives the CUSUM multiple
    independent "starts," so the simple threshold log(1/α) does NOT
    control FAR at α. The actual FAR depends on stream length. We use
    threshold log(horizon/α) as an approximate correction, where horizon
    is the expected monitoring duration. For infinite-horizon guarantee,
    use ScanMartingale instead.

    Parameters
    ----------
    frozen_stats : FrozenReferenceStats
        Reference statistics. Uses reference_cdf for conformal p-values.
    alpha : float
        Significance level.
    epsilon : float
        Betting parameter in (0, 1).
    horizon : int
        Expected monitoring duration (used for threshold correction).
        Default 1000.
    """

    def __init__(
        self,
        frozen_stats: FrozenReferenceStats,
        alpha: float = 0.05,
        epsilon: float = 0.3,
        horizon: int = 1000,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < epsilon < 1.0):
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if frozen_stats.n_reference < 2:
            raise CalibrationError(
                f"Need at least 2 reference scores, got {frozen_stats.n_reference}"
            )

        self._reference_cdf = frozen_stats.reference_cdf
        self._alpha = alpha
        self._epsilon = epsilon
        # Threshold correction: the CUSUM restarts O(T) times over T steps.
        # Each restart is an independent test at level exp(-threshold).
        # By union bound over ~T restarts: threshold = log(T/α).
        self._log_threshold = math.log(horizon / alpha)
        self._log_wealth: float = 0.0
        self._time_step: int = 0
        self._alarm_step: int | None = None

    def update(self, record: StreamRecord) -> float:
        """Process one record. Returns current log-wealth (detection statistic).

        Log-wealth is reset to 0 when it goes negative (CUSUM rule).
        A return value ≥ log(1/α) indicates an alarm.
        """
        self._time_step += 1
        p = _conformal_p_two_sided(record.score, self._reference_cdf)
        self._log_wealth += _bet_log_increment(p, self._epsilon)
        # Page's reset: if wealth drops below 1 (log < 0), restart
        if self._log_wealth < 0.0:
            self._log_wealth = 0.0

        if self._alarm_step is None and self._log_wealth >= self._log_threshold:
            self._alarm_step = self._time_step

        return self._log_wealth

    def update_detailed(self, record: StreamRecord) -> MartingaleAlarm:
        """Process one record with full diagnostic output."""
        p = _conformal_p_two_sided(record.score, self._reference_cdf)
        self._time_step += 1
        self._log_wealth += _bet_log_increment(p, self._epsilon)
        if self._log_wealth < 0.0:
            self._log_wealth = 0.0

        alarmed = self._log_wealth >= self._log_threshold
        if self._alarm_step is None and alarmed:
            self._alarm_step = self._time_step

        return MartingaleAlarm(
            time_step=self._time_step,
            log_wealth=self._log_wealth,
            threshold=self._log_threshold,
            alarmed=alarmed,
            p_value=p,
        )

    @property
    def alarm_step(self) -> int | None:
        return self._alarm_step

    @property
    def log_wealth(self) -> float:
        return self._log_wealth

    @property
    def threshold(self) -> float:
        return self._log_threshold


class ScanMartingale:
    """Scan martingale: union of W concurrent sub-martingales.

    Maintains a rolling window of W sub-martingales, one started at each
    step. Alarms if any sub-martingale's log-wealth exceeds log(W/α),
    where the extra log(W) is the union-bound correction over W simultaneous
    tests.

    This is the recommended default for unknown-changepoint detection:
    whichever sub-martingale starts near the true changepoint accumulates
    evidence fastest, avoiding the dilution problem. The bounded memory (O(W))
    makes it practical for continuous monitoring.

    The union bound is tight: for W independent tests each at level α/W,
    P(∃ at least one alarm under H₀) ≤ W · (α/W) = α.

    Parameters
    ----------
    frozen_stats : FrozenReferenceStats
        Reference statistics. Uses reference_cdf for conformal p-values.
    alpha : float
        Significance level.
    window : int
        Number of concurrent sub-martingales (= scan window width W).
        Default 50, validated in Gate B.
    epsilon : float
        Betting parameter in (0, 1). Default 0.3.
    """

    def __init__(
        self,
        frozen_stats: FrozenReferenceStats,
        alpha: float = 0.05,
        window: int = 50,
        epsilon: float = 0.3,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < epsilon < 1.0):
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if frozen_stats.n_reference < 2:
            raise CalibrationError(
                f"Need at least 2 reference scores, got {frozen_stats.n_reference}"
            )

        self._reference_cdf = frozen_stats.reference_cdf
        self._alpha = alpha
        self._window = window
        self._epsilon = epsilon
        # Union bound: each sub-martingale tested at level α/W
        self._log_threshold = math.log(window / alpha)
        self._log_wealths: deque[float] = deque()
        self._time_step: int = 0
        self._alarm_step: int | None = None

    def update(self, record: StreamRecord) -> float:
        """Process one record. Returns max log-wealth across active sub-martingales.

        A return value ≥ log(W/α) indicates an alarm.
        """
        self._time_step += 1
        p = _conformal_p_two_sided(record.score, self._reference_cdf)
        log_inc = _bet_log_increment(p, self._epsilon)

        # Update all existing sub-martingales
        for i in range(len(self._log_wealths)):
            self._log_wealths[i] += log_inc

        # Start a new sub-martingale at this step
        self._log_wealths.append(log_inc)

        # Evict oldest if beyond window
        if len(self._log_wealths) > self._window:
            self._log_wealths.popleft()

        max_log_wealth = max(self._log_wealths) if self._log_wealths else 0.0

        if self._alarm_step is None and max_log_wealth >= self._log_threshold:
            self._alarm_step = self._time_step

        return max_log_wealth

    def update_detailed(self, record: StreamRecord) -> MartingaleAlarm:
        """Process one record with full diagnostic output."""
        self._time_step += 1
        p = _conformal_p_two_sided(record.score, self._reference_cdf)
        log_inc = _bet_log_increment(p, self._epsilon)

        for i in range(len(self._log_wealths)):
            self._log_wealths[i] += log_inc
        self._log_wealths.append(log_inc)
        if len(self._log_wealths) > self._window:
            self._log_wealths.popleft()

        max_log_wealth = max(self._log_wealths) if self._log_wealths else 0.0
        alarmed = max_log_wealth >= self._log_threshold

        if self._alarm_step is None and alarmed:
            self._alarm_step = self._time_step

        return MartingaleAlarm(
            time_step=self._time_step,
            log_wealth=max_log_wealth,
            threshold=self._log_threshold,
            alarmed=alarmed,
            p_value=p,
        )

    @property
    def alarm_step(self) -> int | None:
        return self._alarm_step

    @property
    def log_wealth(self) -> float:
        """Max log-wealth across active sub-martingales."""
        return max(self._log_wealths) if self._log_wealths else 0.0

    @property
    def threshold(self) -> float:
        return self._log_threshold

    @property
    def n_active(self) -> int:
        """Number of currently active sub-martingales."""
        return len(self._log_wealths)
