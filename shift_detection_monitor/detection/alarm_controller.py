"""
Alarm controller for managing parallel shift detectors with multiplicity correction.

The AlarmController coordinates alarms from multiple parallel detectors (MMD, KS),
applying Bonferroni or Šidák correction to control the family-wise error rate.
It deduplicates alarms (each detector fires at most once per shift) and emits
a combined advisory when both detectors alarm within a configurable time window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shift_detection_monitor.detection.confidence_sequence import (
    ConfidenceSequenceEngine,
    CSUpdate,
)


@dataclass(frozen=True)
class AlarmEvent:
    """Record of a shift alarm."""

    time_step: int
    detector: Literal["mmd", "ks", "combined"]
    statistic_value: float
    cs_lower: float
    cs_upper: float
    reference_value: float


class AlarmController:
    """
    Manages alarms from parallel detectors with multiplicity correction.

    Each registered detector gets its own ConfidenceSequenceEngine with a
    corrected significance level so that the family-wise error rate does
    not exceed the configured α.

    Parameters
    ----------
    alpha : float
        Family-wise significance level in (0, 1).
    correction_method : {"bonferroni", "sidak"}
        Multiplicity correction method.
    combined_window : int | None
        If set, emit a combined advisory alarm when both detectors alarm
        within this many time steps of each other.
    window_mode : {"sliding", "growing"}
        Window mode for the underlying confidence sequences.
    window_size : int | None
        Window size for sliding mode.
    min_warmup_steps : int | None
        Minimum warmup steps before alarms are enabled.
    tail_bound : {"bounded", "sub_gaussian", "sub_exponential"}
        Tail bound assumption for the CS construction.
    lower_bound : float
        Lower bound of the statistic range (bounded mode).
    upper_bound : float
        Upper bound of the statistic range (bounded mode).
    variance_proxy : float | None
        Variance proxy for sub-Gaussian/sub-exponential modes.
    """

    def __init__(
        self,
        alpha: float,
        correction_method: Literal["bonferroni", "sidak"],
        combined_window: int | None = None,
        window_mode: Literal["sliding", "growing"] = "sliding",
        window_size: int | None = None,
        min_warmup_steps: int | None = None,
        tail_bound: Literal["bounded", "sub_gaussian", "sub_exponential"] = "bounded",
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        variance_proxy: float | None = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self._alpha = alpha
        self._correction_method = correction_method
        self._combined_window = combined_window
        self._window_mode = window_mode
        self._window_size = window_size
        self._min_warmup_steps = min_warmup_steps
        self._tail_bound = tail_bound
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._variance_proxy = variance_proxy

        # Registered detectors: name -> CS engine
        self._engines: dict[str, ConfidenceSequenceEngine] = {}
        # Reference values per detector
        self._reference_values: dict[str, float] = {}
        # Latest CSUpdate per detector
        self._latest_updates: dict[str, CSUpdate] = {}
        # Track which detectors have already alarmed (deduplication)
        self._alarmed_detectors: set[str] = set()
        # Track alarm time steps for combined advisory logic
        self._alarm_time_steps: dict[str, int] = {}
        # Track whether combined advisory has been emitted
        self._combined_advisory_emitted: bool = False
        # Pending alarms since last check_alarms() call
        self._pending_alarms: list[AlarmEvent] = []

    @property
    def alpha(self) -> float:
        """Family-wise significance level."""
        return self._alpha

    @property
    def correction_method(self) -> Literal["bonferroni", "sidak"]:
        """Multiplicity correction method."""
        return self._correction_method

    @property
    def num_detectors(self) -> int:
        """Number of registered detectors."""
        return len(self._engines)

    def _compute_corrected_alpha(self, k: int) -> float:
        """Compute per-detector alpha for k detectors.

        Bonferroni: α_per = α / k
        Šidák:      α_per = 1 - (1 - α)^(1/k)
        """
        if k < 1:
            raise ValueError("Must have at least 1 detector")

        if self._correction_method == "bonferroni":
            return self._alpha / k
        else:  # sidak
            return 1.0 - (1.0 - self._alpha) ** (1.0 / k)

    def get_corrected_alpha(self) -> float:
        """Return the per-detector corrected alpha for the current number of detectors."""
        return self._compute_corrected_alpha(len(self._engines))

    def register_detector(
        self, name: str, reference_value: float
    ) -> ConfidenceSequenceEngine:
        """Register a detector and return a CS engine with corrected alpha.

        The per-detector alpha is computed based on the total number of
        registered detectors (including this one). When a new detector is
        registered, all existing engines are recreated with the updated
        corrected alpha.

        Parameters
        ----------
        name : str
            Unique detector name (e.g., "mmd", "ks").
        reference_value : float
            The null-hypothesis reference value for this detector's statistic.

        Returns
        -------
        ConfidenceSequenceEngine
            A CS engine configured with the corrected alpha.
        """
        if name in self._engines:
            raise ValueError(f"Detector '{name}' is already registered")

        # Store reference value
        self._reference_values[name] = reference_value

        # Compute corrected alpha for the new total number of detectors
        k = len(self._engines) + 1
        corrected_alpha = self._compute_corrected_alpha(k)

        # Recreate all existing engines with updated corrected alpha
        for existing_name in list(self._engines.keys()):
            self._engines[existing_name] = self._create_engine(
                corrected_alpha, self._reference_values[existing_name]
            )

        # Create engine for the new detector
        engine = self._create_engine(corrected_alpha, reference_value)
        self._engines[name] = engine

        return engine

    def _create_engine(
        self, corrected_alpha: float, reference_value: float
    ) -> ConfidenceSequenceEngine:
        """Create a ConfidenceSequenceEngine with the given parameters."""
        return ConfidenceSequenceEngine(
            alpha=corrected_alpha,
            reference_value=reference_value,
            window_mode=self._window_mode,
            window_size=self._window_size,
            min_warmup_steps=self._min_warmup_steps,
            tail_bound=self._tail_bound,
            lower_bound=self._lower_bound,
            upper_bound=self._upper_bound,
            variance_proxy=self._variance_proxy,
        )

    def report_update(self, name: str, cs_update: CSUpdate) -> None:
        """Record the latest CSUpdate from a detector.

        This method is called after each detector computes a new statistic
        and updates its CS engine. It checks for new alarms and queues
        them for the next check_alarms() call.

        Parameters
        ----------
        name : str
            The detector name.
        cs_update : CSUpdate
            The latest CS update from the detector.
        """
        if name not in self._engines:
            raise ValueError(f"Detector '{name}' is not registered")

        self._latest_updates[name] = cs_update

        # Check if this detector has a new alarm (not previously alarmed)
        if cs_update.alarm and name not in self._alarmed_detectors:
            self._alarmed_detectors.add(name)
            self._alarm_time_steps[name] = cs_update.time_step

            alarm_event = AlarmEvent(
                time_step=cs_update.time_step,
                detector=name,  # type: ignore[arg-type]
                statistic_value=cs_update.statistic,
                cs_lower=cs_update.lower,
                cs_upper=cs_update.upper,
                reference_value=self._reference_values[name],
            )
            self._pending_alarms.append(alarm_event)

            # Check for combined advisory
            self._check_combined_advisory()

    def _check_combined_advisory(self) -> None:
        """Check if a combined advisory should be emitted.

        A combined advisory is emitted when both detectors alarm within
        the combined_window time steps of each other. It is emitted at
        most once.
        """
        if self._combined_advisory_emitted:
            return
        if self._combined_window is None:
            return

        # Need at least 2 alarmed detectors
        if len(self._alarm_time_steps) < 2:
            return

        # Get all alarm time steps
        time_steps = list(self._alarm_time_steps.values())

        # Check if any pair of alarms is within the combined window
        for i in range(len(time_steps)):
            for j in range(i + 1, len(time_steps)):
                if abs(time_steps[i] - time_steps[j]) <= self._combined_window:
                    # Emit combined advisory at the later alarm time
                    combined_time = max(time_steps[i], time_steps[j])

                    # Use the latest update from the most recently alarmed detector
                    # for the statistic values
                    latest_name = max(
                        self._alarm_time_steps,
                        key=lambda n: self._alarm_time_steps[n],
                    )
                    latest_update = self._latest_updates[latest_name]

                    combined_event = AlarmEvent(
                        time_step=combined_time,
                        detector="combined",
                        statistic_value=latest_update.statistic,
                        cs_lower=latest_update.lower,
                        cs_upper=latest_update.upper,
                        reference_value=self._reference_values[latest_name],
                    )
                    self._pending_alarms.append(combined_event)
                    self._combined_advisory_emitted = True
                    return

    def check_alarms(self) -> list[AlarmEvent]:
        """Return new alarm events since the last call.

        Deduplicates: each detector emits at most one alarm per shift.
        Emits combined advisory if both detectors alarm within
        combined_window time steps.

        Returns
        -------
        list[AlarmEvent]
            New alarm events since the last check.
        """
        alarms = list(self._pending_alarms)
        self._pending_alarms.clear()
        return alarms

    def reset(self) -> None:
        """Reset alarm state for a new shift detection run.

        Clears all alarm history and pending alarms, but keeps
        registered detectors and their engines.
        """
        self._alarmed_detectors.clear()
        self._alarm_time_steps.clear()
        self._combined_advisory_emitted = False
        self._pending_alarms.clear()
        self._latest_updates.clear()
