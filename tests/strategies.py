"""Hypothesis strategies for all data models in the Shift Detection Monitor."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from shift_detection_monitor.config import (
    ConformalConfig,
    ControlConfig,
    DetectorConfig,
    FactorialConfig,
    MMDConfig,
    MonitorConfig,
    ReferenceWindowConfig,
    StreamConfig,
    VarianceConfig,
)
from shift_detection_monitor.evaluation.results import (
    AlarmRecord,
    CellResult,
    OCPoint,
)
from shift_detection_monitor.types import ClassifierOutput, StreamRecord

# ---------------------------------------------------------------------------
# Primitive strategies
# ---------------------------------------------------------------------------

_FINITE_FLOAT = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_UNIT_FLOAT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_SCORE_FLOAT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_SMALL_TEXT = st.text(min_size=1, max_size=100, alphabet=st.characters(codec="utf-8", categories=("L", "N", "P", "Z")))
_DATASET_NAME = st.sampled_from(["wildguardmix", "toxicchat", "llamaguard_eval"])
_SHIFT_CONDITIONS = st.sampled_from(
    [
        "paraphrase",
        "code-switch",
        "adversarial-suffix",
        "compositional-long-context",
        "temporal",
    ]
)
_CLASSIFIER_NAMES = st.sampled_from(
    [
        "llama-guard-3-8b",
        "shieldgemma-9b",
        "gpt-oss-safeguard",
        "deberta-v3-large",
    ]
)
_REGIMES = st.sampled_from(["regime_a", "regime_b", "regime_c"])
_DETECTOR_NAMES = st.sampled_from(["mmd", "ks", "combined"])


# ---------------------------------------------------------------------------
# Embedding matrix strategy
# ---------------------------------------------------------------------------


@st.composite
def st_embedding_matrix(
    draw: st.DrawFn,
    n: int | None = None,
    d: int | None = None,
) -> np.ndarray:
    """Generate an (n, d) numpy array of finite floats."""
    rows = n if n is not None else draw(st.integers(min_value=2, max_value=50))
    cols = d if d is not None else draw(st.integers(min_value=2, max_value=64))
    return draw(
        arrays(
            dtype=np.float64,
            shape=(rows, cols),
            elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        )
    )


# ---------------------------------------------------------------------------
# StreamRecord strategy
# ---------------------------------------------------------------------------


@st.composite
def st_stream_record(draw: st.DrawFn) -> StreamRecord:
    """Generate a valid StreamRecord."""
    has_repr = draw(st.booleans())
    representation = (
        draw(
            arrays(
                dtype=np.float64,
                shape=(draw(st.integers(min_value=2, max_value=64)),),
                elements=st.floats(
                    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
                ),
            )
        )
        if has_repr
        else None
    )
    is_shifted = draw(st.booleans())
    return StreamRecord(
        time_step=draw(st.integers(min_value=0, max_value=100_000)),
        text=draw(_SMALL_TEXT),
        score=draw(_SCORE_FLOAT),
        representation=representation,
        ground_truth_label=draw(st.one_of(st.none(), st.sampled_from([0, 1]))),
        is_shifted=is_shifted,
        source_dataset=draw(_DATASET_NAME),
        shift_condition=draw(_SHIFT_CONDITIONS) if is_shifted else None,
    )


# ---------------------------------------------------------------------------
# Config strategies
# ---------------------------------------------------------------------------


@st.composite
def st_stream_config(draw: st.DrawFn) -> StreamConfig:
    """Generate a valid StreamConfig."""
    has_shift = draw(st.booleans())
    return StreamConfig(
        reference_datasets=draw(
            st.lists(_DATASET_NAME, min_size=1, max_size=3, unique=True)
        ),
        shift_condition=draw(_SHIFT_CONDITIONS) if has_shift else None,
        shift_onset_step=draw(st.integers(min_value=0, max_value=10_000)),
        mixing_proportion=draw(_UNIT_FLOAT),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    )


@st.composite
def st_detector_config(draw: st.DrawFn) -> DetectorConfig:
    """Generate a valid DetectorConfig."""
    alpha = draw(st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False))
    window_size = draw(st.integers(min_value=10, max_value=1000))
    warmup = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=1000)))
    return DetectorConfig(
        alpha=alpha,
        window_mode=draw(st.sampled_from(["sliding", "growing"])),
        window_size=window_size,
        min_warmup_steps=warmup,
        correction_method=draw(st.sampled_from(["bonferroni", "sidak"])),
        combined_advisory_window=draw(
            st.one_of(st.none(), st.integers(min_value=1, max_value=500))
        ),
        tail_bound=draw(
            st.sampled_from(["bounded", "sub_gaussian", "sub_exponential"])
        ),
    )


@st.composite
def st_monitor_config(draw: st.DrawFn) -> MonitorConfig:
    """Generate a valid MonitorConfig."""
    detector = draw(st_detector_config())
    ref_size = draw(st.integers(min_value=50, max_value=2000))
    ref_min = draw(st.integers(min_value=50, max_value=ref_size))
    return MonitorConfig(
        stream=draw(st_stream_config()),
        detector=detector,
        mmd=MMDConfig(
            dim_reduction_threshold=draw(
                st.one_of(st.none(), st.integers(min_value=2, max_value=512))
            ),
            n_bootstrap=draw(st.integers(min_value=100, max_value=5000)),
        ),
        conformal=ConformalConfig(
            target_error_rate=draw(
                st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False)
            ),
            conformal_mode=draw(
                st.sampled_from(["unweighted", "weighted-on-alarm"])
            ),
            min_calibration_size=draw(st.integers(min_value=10, max_value=500)),
            density_ratio_method=draw(st.sampled_from(["logistic", "kliep"])),
        ),
        reference_window=ReferenceWindowConfig(size=ref_size, min_size=ref_min),
        factorial=FactorialConfig(
            classifiers=draw(
                st.lists(_CLASSIFIER_NAMES, min_size=1, max_size=4, unique=True)
            ),
            shift_conditions=draw(
                st.lists(_SHIFT_CONDITIONS, min_size=1, max_size=5, unique=True)
            ),
            ground_truth_regimes=draw(
                st.lists(_REGIMES, min_size=1, max_size=3, unique=True)
            ),
            window_sizes=draw(
                st.lists(
                    st.integers(min_value=10, max_value=1000),
                    min_size=1,
                    max_size=5,
                    unique=True,
                )
            ),
            seeds=draw(
                st.lists(
                    st.integers(min_value=0, max_value=1000),
                    min_size=1,
                    max_size=20,
                    unique=True,
                )
            ),
            max_latency_positive_control=draw(
                st.integers(min_value=10, max_value=1000)
            ),
            min_negative_control_runs=draw(st.integers(min_value=20, max_value=100)),
        ),
        controls=ControlConfig(
            n_negative_runs=draw(st.integers(min_value=20, max_value=100)),
            n_positive_runs=draw(st.integers(min_value=5, max_value=100)),
            trivial_shift_mixing=draw(
                st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
            ),
            max_latency=draw(st.integers(min_value=10, max_value=1000)),
        ),
        variance=VarianceConfig(
            min_observations_per_cell=draw(st.integers(min_value=2, max_value=50))
        ),
        output_dir=draw(st.sampled_from(["results", "output", "eval_results"])),
    )


# ---------------------------------------------------------------------------
# Result model strategies
# ---------------------------------------------------------------------------


@st.composite
def st_alarm_record(draw: st.DrawFn) -> AlarmRecord:
    """Generate a valid AlarmRecord."""
    return AlarmRecord(
        time_step=draw(st.integers(min_value=0, max_value=100_000)),
        detector=draw(_DETECTOR_NAMES),
        statistic_value=draw(_FINITE_FLOAT),
        cs_lower=draw(_FINITE_FLOAT),
        cs_upper=draw(_FINITE_FLOAT),
        reference_value=draw(_FINITE_FLOAT),
    )


@st.composite
def st_oc_point(draw: st.DrawFn) -> OCPoint:
    """Generate a valid OCPoint."""
    return OCPoint(
        false_alarm_rate=draw(_UNIT_FLOAT),
        detection_latency=draw(
            st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def st_cell_result(draw: st.DrawFn) -> CellResult:
    """Generate a valid CellResult."""
    is_neg = draw(st.booleans())
    is_pos = not is_neg and draw(st.booleans())
    has_alarm = draw(st.booleans())
    alarms = draw(st.lists(st_alarm_record(), min_size=0, max_size=3)) if has_alarm else []
    detection_latency = (
        draw(
            st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)
        )
        if has_alarm
        else None
    )
    n_predictions = draw(st.integers(min_value=1, max_value=10_000))
    n_abstentions = draw(st.integers(min_value=0, max_value=n_predictions))
    return CellResult(
        classifier=draw(_CLASSIFIER_NAMES),
        shift_condition=draw(_SHIFT_CONDITIONS),
        ground_truth_regime=draw(_REGIMES),
        window_size=draw(st.integers(min_value=10, max_value=1000)),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        detection_latency=detection_latency,
        false_alarm_rate=draw(_UNIT_FLOAT),
        alarms=alarms,
        conformal_coverage_pre=draw(st.one_of(st.none(), _UNIT_FLOAT)),
        conformal_coverage_post=draw(st.one_of(st.none(), _UNIT_FLOAT)),
        n_abstentions=n_abstentions,
        n_predictions=n_predictions,
        is_negative_control=is_neg,
        is_positive_control=is_pos,
        is_false_positive=is_neg and has_alarm,
        is_missed_detection=is_pos and not has_alarm,
        oc_curve=draw(st.lists(st_oc_point(), min_size=0, max_size=10)),
        active_detectors=draw(
            st.lists(
                st.sampled_from(["mmd", "ks"]),
                min_size=1,
                max_size=2,
                unique=True,
            )
        ),
    )
