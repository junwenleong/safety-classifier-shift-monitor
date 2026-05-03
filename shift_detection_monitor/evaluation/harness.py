"""
Evaluation harness for the full factorial evaluation.

Orchestrates:
1. Negative + positive controls for each classifier
2. Calibration check (halt if negative controls fail)
3. Factorial cell execution
4. JSONL result output
5. Variance decomposition
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shift_detection_monitor.classifiers.interface import ClassifierInterface
from shift_detection_monitor.config import MonitorConfig, StreamConfig
from shift_detection_monitor.detection.alarm_controller import AlarmController
from shift_detection_monitor.detection.confidence_sequence import (
    ConfidenceSequenceEngine,
)
from shift_detection_monitor.detection.ks_detector import KSDetector
from shift_detection_monitor.detection.mmd_detector import MMDDetector
from shift_detection_monitor.detection.reference_window import (
    FrozenReferenceStats,
    ReferenceWindow,
)
from shift_detection_monitor.evaluation.metrics import (
    compute_detection_latency,
    flag_control_result,
)
from shift_detection_monitor.evaluation.results import AlarmRecord, CellResult
from shift_detection_monitor.serialization.result_io import write_results
from shift_detection_monitor.stream.simulator import StreamSimulator
from shift_detection_monitor.types import StreamRecord

logger = logging.getLogger(__name__)


@dataclass
class FactorialCell:
    """A single cell in the factorial evaluation design."""

    classifier_name: str
    shift_condition: str
    ground_truth_regime: str
    window_size: int


class EvaluationHarness:
    """Orchestrates the full factorial evaluation.

    Parameters
    ----------
    config : MonitorConfig
        Full monitor configuration.
    classifiers : dict[str, ClassifierInterface]
        Mapping of classifier name to classifier instance.
    reference_examples : list[dict] | None
        In-memory reference examples for testing.
    shifted_examples : list[dict] | None
        In-memory shifted examples for testing.
    """

    def __init__(
        self,
        config: MonitorConfig,
        classifiers: dict[str, ClassifierInterface],
        reference_examples: list[dict[str, Any]] | None = None,
        shifted_examples: list[dict[str, Any]] | None = None,
    ) -> None:
        self._config = config
        self._classifiers = classifiers
        self._reference_examples = reference_examples
        self._shifted_examples = shifted_examples

    def run(self, output_path: Path) -> list[CellResult]:
        """Execute the full evaluation.

        1. Run negative + positive controls for each classifier
        2. Halt if negative controls fail calibration check
        3. Run factorial cells
        4. Write JSONL results
        5. Return all results

        Parameters
        ----------
        output_path : Path
            Path to write JSONL results.

        Returns
        -------
        list[CellResult]
            All evaluation results.
        """
        all_results: list[CellResult] = []

        # 1. Run controls for each classifier
        for clf_name, clf in self._classifiers.items():
            logger.info("Running controls for classifier: %s", clf_name)

            # Negative controls
            neg_results = self._run_negative_controls(clf_name, clf)
            all_results.extend(neg_results)

            # Check calibration
            n_false_positives = sum(1 for r in neg_results if r.is_false_positive)
            far = n_false_positives / len(neg_results) if neg_results else 0.0
            alpha = self._config.detector.alpha

            if far > alpha:
                logger.error(
                    "Calibration failure for %s: FAR=%.3f > alpha=%.3f. "
                    "Halting factorial evaluation.",
                    clf_name,
                    far,
                    alpha,
                )
                write_results(all_results, output_path)
                return all_results

            # Positive controls
            pos_results = self._run_positive_controls(clf_name, clf)
            all_results.extend(pos_results)

        # 3. Run factorial cells
        for clf_name in self._config.factorial.classifiers:
            if clf_name not in self._classifiers:
                logger.warning("Classifier %s not available, skipping", clf_name)
                continue

            clf = self._classifiers[clf_name]

            for shift_cond in self._config.factorial.shift_conditions:
                for regime in self._config.factorial.ground_truth_regimes:
                    for ws in self._config.factorial.window_sizes:
                        cell = FactorialCell(
                            classifier_name=clf_name,
                            shift_condition=shift_cond,
                            ground_truth_regime=regime,
                            window_size=ws,
                        )
                        for seed in self._config.factorial.seeds:
                            try:
                                result = self.run_cell(cell, seed)
                                all_results.append(result)
                            except Exception:
                                logger.exception(
                                    "Failed cell: %s/%s/%s/ws=%d/seed=%d",
                                    clf_name,
                                    shift_cond,
                                    regime,
                                    ws,
                                    seed,
                                )

        # 4. Write JSONL results
        write_results(all_results, output_path)

        return all_results

    def run_cell(self, cell: FactorialCell, seed: int) -> CellResult:
        """Run a single factorial cell with a given seed.

        Parameters
        ----------
        cell : FactorialCell
            The factorial cell specification.
        seed : int
            Random seed for this run.

        Returns
        -------
        CellResult
            Result for this cell.
        """
        clf = self._classifiers[cell.classifier_name]

        # Build stream config for this cell
        stream_config = StreamConfig(
            reference_datasets=self._config.stream.reference_datasets,
            shift_condition=cell.shift_condition,
            shift_onset_step=self._config.stream.shift_onset_step,
            mixing_proportion=self._config.stream.mixing_proportion,
            seed=seed,
        )

        # Create stream simulator
        simulator = StreamSimulator(
            config=stream_config,
            classifier=clf,
            seed=seed,
            reference_examples=self._reference_examples,
            shifted_examples=self._shifted_examples,
        )

        # Collect stream records
        records: list[StreamRecord] = list(simulator)

        if not records:
            return CellResult(
                classifier=cell.classifier_name,
                shift_condition=cell.shift_condition,
                ground_truth_regime=cell.ground_truth_regime,
                window_size=cell.window_size,
                seed=seed,
                detection_latency=None,
                false_alarm_rate=0.0,
                alarms=[],
                n_abstentions=0,
                n_predictions=len(records),
            )

        # Build reference window from initial records
        ref_size = min(
            self._config.reference_window.min_size, len(records)
        )
        ref_window = ReferenceWindow(
            min_size=ref_size,
            dim_reduction_threshold=self._config.mmd.dim_reduction_threshold,
            n_bootstrap=min(self._config.mmd.n_bootstrap, 100),  # cap for speed
        )

        for rec in records[:ref_size]:
            ref_window.add(rec)

        frozen_stats = ref_window.freeze()

        # Set up detectors
        active_detectors: list[str] = []
        has_embeddings = frozen_stats.reference_embeddings.size > 0

        alarm_controller = AlarmController(
            alpha=self._config.detector.alpha,
            correction_method=self._config.detector.correction_method,
            combined_window=self._config.detector.combined_advisory_window,
            window_mode=self._config.detector.window_mode,
            window_size=cell.window_size,
            min_warmup_steps=self._config.detector.min_warmup_steps,
        )

        mmd_detector = None
        ks_detector = KSDetector(
            frozen_stats=frozen_stats,
            window_size=cell.window_size,
        )

        # Register KS detector
        ks_engine = alarm_controller.register_detector(
            "ks", reference_value=0.0
        )
        active_detectors.append("ks")

        if has_embeddings:
            mmd_detector = MMDDetector(
                frozen_stats=frozen_stats,
                window_size=cell.window_size,
            )
            mmd_engine = alarm_controller.register_detector(
                "mmd", reference_value=frozen_stats.mmd_reference_value
            )
            active_detectors.append("mmd")

        # Process stream records after reference window
        alarms: list[AlarmRecord] = []
        for rec in records[ref_size:]:
            # KS update
            ks_stat = ks_detector.update(rec)
            ks_update = ks_engine.update(ks_stat)
            alarm_controller.report_update("ks", ks_update)

            # MMD update
            if mmd_detector is not None:
                mmd_stat = mmd_detector.update(rec)
                if mmd_stat is not None:
                    mmd_update = mmd_engine.update(mmd_stat)
                    alarm_controller.report_update("mmd", mmd_update)

            # Check for new alarms
            new_alarms = alarm_controller.check_alarms()
            for ae in new_alarms:
                alarms.append(
                    AlarmRecord(
                        time_step=ae.time_step,
                        detector=ae.detector,
                        statistic_value=ae.statistic_value,
                        cs_lower=ae.cs_lower,
                        cs_upper=ae.cs_upper,
                        reference_value=ae.reference_value,
                    )
                )

        # Compute metrics
        onset = self._config.stream.shift_onset_step
        detection_latency = compute_detection_latency(alarms, onset)

        # Compute false alarm rate (alarms before onset / steps before onset)
        pre_onset_alarms = sum(1 for a in alarms if a.time_step < onset)
        steps_before_onset = max(onset - ref_size, 1)
        far = pre_onset_alarms / steps_before_onset

        return CellResult(
            classifier=cell.classifier_name,
            shift_condition=cell.shift_condition,
            ground_truth_regime=cell.ground_truth_regime,
            window_size=cell.window_size,
            seed=seed,
            detection_latency=detection_latency,
            false_alarm_rate=far,
            alarms=alarms,
            n_abstentions=0,
            n_predictions=len(records) - ref_size,
            active_detectors=active_detectors,
        )

    def _run_negative_controls(
        self,
        clf_name: str,
        clf: ClassifierInterface,
    ) -> list[CellResult]:
        """Run negative controls: pure-reference streams that must not alarm."""
        results: list[CellResult] = []
        n_runs = self._config.controls.n_negative_runs
        max_latency = self._config.controls.max_latency

        for i in range(n_runs):
            seed = 10000 + i  # Distinct seed range for controls

            stream_config = StreamConfig(
                reference_datasets=self._config.stream.reference_datasets,
                shift_condition=None,  # Pure reference
                shift_onset_step=0,
                mixing_proportion=0.0,
                seed=seed,
            )

            simulator = StreamSimulator(
                config=stream_config,
                classifier=clf,
                seed=seed,
                reference_examples=self._reference_examples,
                shifted_examples=None,
            )

            records = list(simulator)
            if not records:
                continue

            # Run detection pipeline
            cell = FactorialCell(
                classifier_name=clf_name,
                shift_condition="none",
                ground_truth_regime="regime_a",
                window_size=self._config.detector.window_size,
            )

            result = self._run_detection_pipeline(
                cell, seed, records, is_negative_control=True
            )
            result = flag_control_result(result, max_latency)
            results.append(result)

        return results

    def _run_positive_controls(
        self,
        clf_name: str,
        clf: ClassifierInterface,
    ) -> list[CellResult]:
        """Run positive controls: trivially-shifted streams that must alarm quickly."""
        results: list[CellResult] = []
        n_runs = self._config.controls.n_positive_runs
        max_latency = self._config.controls.max_latency

        for i in range(n_runs):
            seed = 20000 + i

            stream_config = StreamConfig(
                reference_datasets=self._config.stream.reference_datasets,
                shift_condition="paraphrase",  # Use a simple shift
                shift_onset_step=self._config.stream.shift_onset_step,
                mixing_proportion=self._config.controls.trivial_shift_mixing,
                seed=seed,
            )

            simulator = StreamSimulator(
                config=stream_config,
                classifier=clf,
                seed=seed,
                reference_examples=self._reference_examples,
                shifted_examples=self._shifted_examples,
            )

            records = list(simulator)
            if not records:
                continue

            cell = FactorialCell(
                classifier_name=clf_name,
                shift_condition="paraphrase",
                ground_truth_regime="regime_a",
                window_size=self._config.detector.window_size,
            )

            result = self._run_detection_pipeline(
                cell, seed, records, is_positive_control=True
            )
            result = flag_control_result(result, max_latency)
            results.append(result)

        return results

    def _run_detection_pipeline(
        self,
        cell: FactorialCell,
        seed: int,
        records: list[StreamRecord],
        is_negative_control: bool = False,
        is_positive_control: bool = False,
    ) -> CellResult:
        """Run the detection pipeline on pre-collected records."""
        ref_size = min(
            self._config.reference_window.min_size, len(records)
        )

        ref_window = ReferenceWindow(
            min_size=ref_size,
            dim_reduction_threshold=self._config.mmd.dim_reduction_threshold,
            n_bootstrap=min(self._config.mmd.n_bootstrap, 100),
        )

        for rec in records[:ref_size]:
            ref_window.add(rec)

        frozen_stats = ref_window.freeze()

        has_embeddings = frozen_stats.reference_embeddings.size > 0

        alarm_controller = AlarmController(
            alpha=self._config.detector.alpha,
            correction_method=self._config.detector.correction_method,
            combined_window=self._config.detector.combined_advisory_window,
            window_mode=self._config.detector.window_mode,
            window_size=cell.window_size,
            min_warmup_steps=self._config.detector.min_warmup_steps,
        )

        ks_detector = KSDetector(
            frozen_stats=frozen_stats,
            window_size=cell.window_size,
        )
        ks_engine = alarm_controller.register_detector("ks", reference_value=0.0)
        active_detectors = ["ks"]

        mmd_detector = None
        mmd_engine = None
        if has_embeddings:
            mmd_detector = MMDDetector(
                frozen_stats=frozen_stats,
                window_size=cell.window_size,
            )
            mmd_engine = alarm_controller.register_detector(
                "mmd", reference_value=frozen_stats.mmd_reference_value
            )
            active_detectors.append("mmd")

        alarms: list[AlarmRecord] = []
        for rec in records[ref_size:]:
            ks_stat = ks_detector.update(rec)
            ks_update = ks_engine.update(ks_stat)
            alarm_controller.report_update("ks", ks_update)

            if mmd_detector is not None and mmd_engine is not None:
                mmd_stat = mmd_detector.update(rec)
                if mmd_stat is not None:
                    mmd_update = mmd_engine.update(mmd_stat)
                    alarm_controller.report_update("mmd", mmd_update)

            new_alarms = alarm_controller.check_alarms()
            for ae in new_alarms:
                alarms.append(
                    AlarmRecord(
                        time_step=ae.time_step,
                        detector=ae.detector,
                        statistic_value=ae.statistic_value,
                        cs_lower=ae.cs_lower,
                        cs_upper=ae.cs_upper,
                        reference_value=ae.reference_value,
                    )
                )

        onset = self._config.stream.shift_onset_step
        detection_latency = compute_detection_latency(alarms, onset) if not is_negative_control else None
        if is_negative_control and alarms:
            detection_latency = float(alarms[0].time_step)

        pre_onset_alarms = sum(1 for a in alarms if a.time_step < onset)
        steps_before_onset = max(onset - ref_size, 1)
        far = pre_onset_alarms / steps_before_onset if not is_negative_control else (
            len(alarms) / max(len(records) - ref_size, 1)
        )

        return CellResult(
            classifier=cell.classifier_name,
            shift_condition=cell.shift_condition,
            ground_truth_regime=cell.ground_truth_regime,
            window_size=cell.window_size,
            seed=seed,
            detection_latency=detection_latency,
            false_alarm_rate=far,
            alarms=alarms,
            n_abstentions=0,
            n_predictions=len(records) - ref_size,
            is_negative_control=is_negative_control,
            is_positive_control=is_positive_control,
            active_detectors=active_detectors,
        )
