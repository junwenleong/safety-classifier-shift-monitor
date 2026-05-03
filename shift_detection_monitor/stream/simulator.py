"""
Stream simulator for replaying pre-built corpora as a time-ordered stream
with controllable shift injection.

The simulator reads reference examples and shifted examples, interleaves them
based on shift_onset_step and mixing_proportion, passes each through a classifier
to get score + representation, and yields StreamRecords in time order.

Supports deterministic replay via seed and an in-memory mode for testing.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from shift_detection_monitor.classifiers.interface import ClassifierInterface
from shift_detection_monitor.config import StreamConfig
from shift_detection_monitor.types import StreamRecord


class StreamSimulator:
    """Replays pre-built corpora as a time-ordered stream with controllable shift injection.

    Parameters
    ----------
    config : StreamConfig
        Stream configuration (onset, mixing proportion, seed, etc.).
    classifier : ClassifierInterface
        Classifier to score each example.
    seed : int
        Random seed for deterministic replay.
    reference_examples : list[dict] | None
        In-memory reference examples. If provided, reference_datasets paths are ignored.
    shifted_examples : list[dict] | None
        In-memory shifted examples. If provided, shift corpus paths are ignored.
    """

    def __init__(
        self,
        config: StreamConfig,
        classifier: ClassifierInterface,
        seed: int,
        reference_examples: list[dict[str, Any]] | None = None,
        shifted_examples: list[dict[str, Any]] | None = None,
    ) -> None:
        self._config = config
        self._classifier = classifier
        self._seed = seed
        self._reference_examples = reference_examples or []
        self._shifted_examples = shifted_examples or []
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[StreamRecord]:
        """Yield stream records in time order."""
        rng = random.Random(self._seed)
        ref_examples = list(self._reference_examples)
        shifted_examples = list(self._shifted_examples)

        if not ref_examples:
            return

        onset = self._config.shift_onset_step
        mixing = self._config.mixing_proportion
        has_shift = (
            self._config.shift_condition is not None and len(shifted_examples) > 0
        )

        # Shuffle both pools deterministically
        rng.shuffle(ref_examples)
        if shifted_examples:
            rng.shuffle(shifted_examples)

        ref_idx = 0
        shifted_idx = 0
        total_examples = len(ref_examples) + len(shifted_examples)

        for t in range(total_examples):
            use_shifted = False

            if has_shift and t >= onset:
                # After onset, use shifted with probability = mixing_proportion
                use_shifted = rng.random() < mixing

            if use_shifted and shifted_idx < len(shifted_examples):
                ex = shifted_examples[shifted_idx]
                shifted_idx += 1
                is_shifted = True
                shift_condition = self._config.shift_condition
                source_dataset = ex.get("source_dataset", "shifted")
            elif ref_idx < len(ref_examples):
                ex = ref_examples[ref_idx]
                ref_idx += 1
                is_shifted = False
                shift_condition = None
                source_dataset = ex.get("source_dataset", "reference")
            else:
                # Both pools may be exhausted; stop if nothing left
                if use_shifted and shifted_idx < len(shifted_examples):
                    ex = shifted_examples[shifted_idx]
                    shifted_idx += 1
                    is_shifted = True
                    shift_condition = self._config.shift_condition
                    source_dataset = ex.get("source_dataset", "shifted")
                elif ref_idx < len(ref_examples):
                    ex = ref_examples[ref_idx]
                    ref_idx += 1
                    is_shifted = False
                    shift_condition = None
                    source_dataset = ex.get("source_dataset", "reference")
                else:
                    break

            text = ex.get("text", "")
            ground_truth = ex.get("label", ex.get("ground_truth_label", None))

            # Pass through classifier
            output = self._classifier.predict(text)

            yield StreamRecord(
                time_step=t,
                text=text,
                score=output.score,
                representation=output.representation,
                ground_truth_label=ground_truth,
                is_shifted=is_shifted,
                source_dataset=source_dataset,
                shift_condition=shift_condition,
            )

    def reset(self, seed: int) -> None:
        """Reset to initial state with a new seed for deterministic replay."""
        self._seed = seed
        self._rng = random.Random(seed)
