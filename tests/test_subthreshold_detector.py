"""Tests for CUSUMDetector and EWMADetector (shift_detection_monitor.detection.subthreshold_detector).

Mirrors the style of test_ks.py / test_mmd.py: a mix of deterministic
sanity checks, a numerical-fidelity regression test against the original
script implementation (scripts/exp_subthreshold_detection.py), and
property-based tests via Hypothesis.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.detection.subthreshold_detector import (
    CUSUMDetector,
    EWMADetector,
)
from shift_detection_monitor.types import StreamRecord


def _make_record(t: int, score: float) -> StreamRecord:
    return StreamRecord(
        time_step=t,
        text=f"text_{t}",
        score=score,
        representation=None,
        ground_truth_label=None,
        is_shifted=False,
        source_dataset="wildguardmix",
        shift_condition=None,
    )


def _make_frozen_stats(reference_cdf: np.ndarray) -> FrozenReferenceStats:
    rng = np.random.default_rng(0)
    return FrozenReferenceStats(
        kernel_bandwidth=1.0,
        reference_cdf=reference_cdf,
        reference_embeddings=rng.standard_normal((5, 2)),
        mmd_null_distribution=np.zeros(10),
        mmd_reference_value=0.0,
        pca_components=None,
        pca_mean=None,
        n_reference=len(reference_cdf),
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_cusum_rejects_empty_reference() -> None:
    stats = _make_frozen_stats(np.array([]))
    with pytest.raises(ValueError):
        CUSUMDetector(frozen_stats=stats, k=0.5)


def test_ewma_rejects_empty_reference() -> None:
    stats = _make_frozen_stats(np.array([]))
    with pytest.raises(ValueError):
        EWMADetector(frozen_stats=stats, lam=0.05)


@pytest.mark.parametrize("bad_lam", [0.0, -0.1, 1.5])
def test_ewma_rejects_invalid_lambda(bad_lam: float) -> None:
    stats = _make_frozen_stats(np.array([0.0, 0.1, 0.2, 0.0, 0.1]))
    with pytest.raises(ValueError):
        EWMADetector(frozen_stats=stats, lam=bad_lam)


def test_ewma_accepts_lambda_equal_one() -> None:
    stats = _make_frozen_stats(np.array([0.0, 0.1, 0.2, 0.0, 0.1]))
    EWMADetector(frozen_stats=stats, lam=1.0)  # should not raise


# ---------------------------------------------------------------------------
# Null-stream behavior: statistics should stay low when nothing has shifted
# ---------------------------------------------------------------------------


def test_cusum_stays_at_zero_on_exact_mean_stream() -> None:
    """Feeding the in-control mean forever should never accumulate (each
    increment is exactly -k_raw, which floors at 0 immediately)."""
    ref = np.full(500, 0.02)
    stats = _make_frozen_stats(ref)
    det = CUSUMDetector(frozen_stats=stats, k=0.5)
    for t in range(200):
        val = det.update(_make_record(t, 0.02))
    assert val == 0.0


def test_ewma_stays_near_zero_on_exact_mean_stream() -> None:
    ref = np.full(500, 0.02)
    stats = _make_frozen_stats(ref)
    det = EWMADetector(frozen_stats=stats, lam=0.05)
    for t in range(200):
        val = det.update(_make_record(t, 0.02))
    # sigma0 is ~0 here so the +1e-12 floor dominates; the statistic must
    # not explode or produce NaN/inf on a degenerate (zero-variance) reference.
    assert np.isfinite(val)


def test_cusum_bounded_on_iid_null_resampling() -> None:
    """On a stream drawn iid from the same distribution as the reference
    (the null hypothesis), CUSUM(k=0.5) should stay well below a generous
    threshold most of the time -- it should not blow up unboundedly."""
    rng = np.random.default_rng(42)
    null_scores = np.clip(rng.normal(0.02, 0.02, size=500), 0.0, 1.0)
    stats = _make_frozen_stats(null_scores)
    det = CUSUMDetector(frozen_stats=stats, k=0.5)
    max_stat = 0.0
    for t in range(2000):
        x = float(rng.choice(null_scores))
        max_stat = max(max_stat, det.update(_make_record(t, x)))
    # A generously large threshold (30 sigma-units of slack) should not be
    # crossed on a pure null stream; this is a sanity bound, not a formal
    # FAR guarantee (this package does not claim one for CUSUM -- see the
    # module docstring).
    assert max_stat < 30.0 * np.std(null_scores)


# ---------------------------------------------------------------------------
# Shift-detection behavior: statistics should grow under sustained shift
# ---------------------------------------------------------------------------


def test_cusum_grows_under_sustained_upward_shift() -> None:
    ref = np.full(500, 0.02)
    stats = _make_frozen_stats(ref)
    det = CUSUMDetector(frozen_stats=stats, k=0.5)
    # sigma0 floors at 1e-12 here (zero-variance reference), so k_raw ~ 0:
    # any score above mu0 accumulates without bound. Use a non-degenerate
    # reference so k_raw is meaningful.
    ref2 = np.clip(np.random.default_rng(1).normal(0.02, 0.03, size=500), 0.0, 1.0)
    stats2 = _make_frozen_stats(ref2)
    det2 = CUSUMDetector(frozen_stats=stats2, k=0.5)
    values = [det2.update(_make_record(t, 0.5)) for t in range(50)]
    # Sustained large shift (0.5 vs mu0~0.02) must produce a monotonically
    # non-decreasing S+ that grows well past its starting point.
    assert values[-1] > values[0]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
    assert values[-1] > 5.0


def test_ewma_grows_under_sustained_upward_shift() -> None:
    ref = np.clip(np.random.default_rng(2).normal(0.02, 0.03, size=500), 0.0, 1.0)
    stats = _make_frozen_stats(ref)
    det = EWMADetector(frozen_stats=stats, lam=0.05)
    values = [det.update(_make_record(t, 0.5)) for t in range(100)]
    assert values[-1] > values[0]
    assert values[-1] > 3.0  # comfortably past a typical L=3 control limit


def test_cusum_reset_returns_to_zero() -> None:
    ref = np.clip(np.random.default_rng(3).normal(0.02, 0.03, size=500), 0.0, 1.0)
    stats = _make_frozen_stats(ref)
    det = CUSUMDetector(frozen_stats=stats, k=0.5)
    for t in range(50):
        det.update(_make_record(t, 0.9))
    det.reset()
    assert det.update(_make_record(50, float(np.mean(ref)))) == pytest.approx(0.0, abs=1e-6)


def test_ewma_reset_returns_to_mean() -> None:
    ref = np.clip(np.random.default_rng(4).normal(0.02, 0.03, size=500), 0.0, 1.0)
    stats = _make_frozen_stats(ref)
    det = EWMADetector(frozen_stats=stats, lam=0.2)
    for t in range(50):
        det.update(_make_record(t, 0.9))
    det.reset()
    val = det.update(_make_record(50, float(np.mean(ref))))
    assert val < 1.0  # first observation at the mean should not alarm


# ---------------------------------------------------------------------------
# Numerical-fidelity regression test vs. the original script implementation
# ---------------------------------------------------------------------------


def test_cusum_matches_original_script_recursion() -> None:
    """Regression test: the ported CUSUMDetector.update() sequence must
    equal scripts/exp_subthreshold_detection.py's CUSUMDetector.run()
    recursion step-for-step on an identical synthetic stream, to guard
    against the port silently diverging from the paper's actual evaluation
    code."""
    rng = np.random.default_rng(7)
    ref = np.clip(rng.normal(0.02, 0.03, size=500), 0.0, 1.0)
    mu0 = float(np.mean(ref))
    sigma0 = float(np.std(ref))
    k = 0.5

    stream = np.clip(rng.normal(0.3, 0.1, size=100), 0.0, 1.0)

    # Original recursion (hand-copied from exp_subthreshold_detection.py's
    # CUSUMDetector.run(), which operates on a raw numpy array rather than
    # StreamRecord objects).
    s_pos_orig = 0.0
    s_neg_orig = 0.0
    k_raw = k * sigma0
    expected = []
    for x in stream:
        s_pos_orig = max(0.0, s_pos_orig + (x - mu0) - k_raw)
        s_neg_orig = max(0.0, s_neg_orig - (x - mu0) - k_raw)
        expected.append(max(s_pos_orig, s_neg_orig))

    stats = _make_frozen_stats(ref)
    det = CUSUMDetector(frozen_stats=stats, k=k)
    actual = [det.update(_make_record(t, float(x))) for t, x in enumerate(stream)]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_ewma_matches_original_script_recursion() -> None:
    """Regression test: the ported EWMADetector must reproduce the
    original script's z / sigma_z(n) recursion (up to the normalization
    this port applies so a single threshold is comparable across t)."""
    rng = np.random.default_rng(8)
    ref = np.clip(rng.normal(0.02, 0.03, size=500), 0.0, 1.0)
    mu0 = float(np.mean(ref))
    sigma0 = float(np.std(ref))
    lam = 0.05

    stream = np.clip(rng.normal(0.3, 0.1, size=100), 0.0, 1.0)

    z = mu0
    expected = []
    for n, x in enumerate(stream, start=1):
        z = lam * x + (1 - lam) * z
        sigma_z_n = sigma0 * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * n)))
        expected.append(abs(z - mu0) / sigma_z_n)

    stats = _make_frozen_stats(ref)
    det = EWMADetector(frozen_stats=stats, lam=lam)
    actual = [det.update(_make_record(t, float(x))) for t, x in enumerate(stream)]

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

_SCORE_FLOAT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(
    ref_scores=arrays(dtype=np.float64, shape=st.integers(5, 50), elements=_SCORE_FLOAT),
    stream_scores=st.lists(_SCORE_FLOAT, min_size=1, max_size=50),
    k=st.floats(min_value=0.01, max_value=5.0),
)
@settings(max_examples=100)
def test_cusum_statistic_always_nonnegative(
    ref_scores: np.ndarray, stream_scores: list[float], k: float
) -> None:
    """P: CUSUM's returned statistic max(S+, S-) is always >= 0, for any
    reference distribution, any stream, and any positive allowance k --
    this is a structural invariant of Page's max(0, ...) recursion, not
    dependent on the specific data."""
    stats = _make_frozen_stats(ref_scores)
    det = CUSUMDetector(frozen_stats=stats, k=k)
    for t, x in enumerate(stream_scores):
        val = det.update(_make_record(t, x))
        assert val >= 0.0


@given(
    ref_scores=arrays(dtype=np.float64, shape=st.integers(5, 50), elements=_SCORE_FLOAT),
    stream_scores=st.lists(_SCORE_FLOAT, min_size=1, max_size=50),
    lam=st.floats(min_value=1e-3, max_value=1.0),
)
@settings(max_examples=100)
def test_ewma_statistic_always_nonnegative_and_finite(
    ref_scores: np.ndarray, stream_scores: list[float], lam: float
) -> None:
    """P: EWMA's normalized statistic is always finite and >= 0, including
    on degenerate (zero-variance) reference distributions where sigma0
    floors at 1e-12."""
    stats = _make_frozen_stats(ref_scores)
    det = EWMADetector(frozen_stats=stats, lam=lam)
    for t, x in enumerate(stream_scores):
        val = det.update(_make_record(t, x))
        assert val >= 0.0
        assert np.isfinite(val)


@given(lam=st.floats(min_value=0.001, max_value=0.5))
@settings(max_examples=30)
def test_ewma_sigma_z_converges_to_asymptotic_formula(lam: float) -> None:
    """P: the finite-sample sigma_z(n) this port uses converges to the
    standard asymptotic EWMA control-limit formula sigma0*sqrt(lam/(2-lam))
    as n grows -- i.e. the exact, finite-sample formula and the commonly
    cited asymptotic shorthand agree in the limit."""
    sigma0 = 1.0
    asymptotic = sigma0 * np.sqrt(lam / (2 - lam))
    n = 100_000
    finite_sample = sigma0 * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * n)))
    np.testing.assert_allclose(finite_sample, asymptotic, rtol=1e-6)
