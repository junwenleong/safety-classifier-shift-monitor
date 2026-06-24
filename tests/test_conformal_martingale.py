"""Tests for the conformal martingale detectors.

Tests:
  1. Conformal p-value correctness (uniform under null)
  2. FAR control on null streams (all three variants)
  3. Detection on shifted streams
  4. Alarm ordering (scan ≥ CUSUM ≥ point on unknown changepoints)
  5. Input validation (CalibrationError, ValueError)
  6. Boundary cases (constant scores, single reference point)
  7. update_detailed consistency with update
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shift_detection_monitor.detection.conformal_martingale import (
    CUSUMMartingale,
    MartingaleAlarm,
    PointMartingale,
    ScanMartingale,
    _bet_log_increment,
    _conformal_p_two_sided,
)
from shift_detection_monitor.detection.reference_window import FrozenReferenceStats
from shift_detection_monitor.types import CalibrationError, StreamRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frozen_stats(ref_scores: np.ndarray) -> FrozenReferenceStats:
    """Build minimal FrozenReferenceStats from an array of reference scores."""
    n = len(ref_scores)
    return FrozenReferenceStats(
        kernel_bandwidth=1.0,
        reference_cdf=np.sort(ref_scores),
        reference_embeddings=np.zeros((n, 2)),
        mmd_null_distribution=np.zeros(10),
        mmd_reference_value=0.0,
        pca_components=None,
        pca_mean=None,
        n_reference=n,
    )


def _make_record(score: float, time_step: int = 0) -> StreamRecord:
    return StreamRecord(
        time_step=time_step,
        text="",
        score=score,
        representation=None,
        ground_truth_label=None,
        is_shifted=False,
        source_dataset="test",
        shift_condition=None,
    )


# ---------------------------------------------------------------------------
# Test: _conformal_p_two_sided
# ---------------------------------------------------------------------------


class TestConformalPValue:
    def test_score_in_middle_of_reference(self):
        """Score at median of reference → p near 1.0."""
        ref = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00
        p = _conformal_p_two_sided(0.5, ref)
        # At median, both tails are large → p ≈ 1.0
        assert p > 0.9

    def test_score_at_extreme(self):
        """Score far outside reference → p near 0."""
        ref = np.linspace(0.3, 0.7, 100)
        p = _conformal_p_two_sided(0.99, ref)
        assert p < 0.1

    def test_p_always_positive(self):
        """P-value is always > 0 (due to +1 correction)."""
        ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        p = _conformal_p_two_sided(999.0, ref)
        assert p > 0

    def test_p_at_most_one(self):
        """P-value is at most 1.0."""
        ref = np.linspace(0, 1, 1000)
        p = _conformal_p_two_sided(0.5, ref)
        assert p <= 1.0

    @given(st.floats(min_value=0, max_value=1, allow_nan=False))
    @settings(max_examples=50)
    def test_p_bounded_for_any_score(self, score: float):
        """P-value is in (0, 1] for any valid score."""
        ref = np.linspace(0, 1, 50)
        p = _conformal_p_two_sided(score, ref)
        assert 0 < p <= 1.0

    def test_uniformity_under_null(self):
        """Under H₀ (scores from same distribution), p-values are uniform."""
        rng = np.random.default_rng(42)
        ref = np.sort(rng.uniform(0, 1, 500))
        # Draw test scores from the same distribution
        test_scores = rng.uniform(0, 1, 1000)
        p_values = [_conformal_p_two_sided(s, ref) for s in test_scores]
        # KS test for uniformity
        from scipy.stats import kstest
        stat, pval = kstest(p_values, "uniform")
        # Should not reject uniformity at 0.01 level
        assert pval > 0.01, f"P-values not uniform: KS stat={stat:.4f}, p={pval:.4f}"


# ---------------------------------------------------------------------------
# Test: _bet_log_increment
# ---------------------------------------------------------------------------


class TestBetLogIncrement:
    def test_fair_game_under_uniform(self):
        """E[log_increment] ≈ 0 when p ~ Uniform (H₀ holds)."""
        rng = np.random.default_rng(42)
        p_values = rng.uniform(0, 1, 10000)
        increments = [_bet_log_increment(p, 0.3) for p in p_values]
        # Under H₀, mean log-increment should be ≈ 0 (or slightly negative
        # because log(ε·p^(ε-1)) has E = ε·∫p^(ε-1)dp under U[0,1] = 1,
        # but log is concave so E[log] ≤ log(E) = 0)
        mean_inc = np.mean(increments)
        assert mean_inc < 0.1  # should be near 0 or negative

    def test_positive_for_small_p(self):
        """Small p → positive increment (evidence against H₀)."""
        inc = _bet_log_increment(0.01, 0.3)
        assert inc > 0

    def test_negative_for_large_p(self):
        """Large p → negative increment (consistent with H₀)."""
        inc = _bet_log_increment(0.9, 0.3)
        assert inc < 0


# ---------------------------------------------------------------------------
# Test: FAR control on null streams
# ---------------------------------------------------------------------------


class TestFARControl:
    """All three detectors should have FAR ≤ α on iid null streams."""

    @pytest.fixture()
    def reference_scores(self):
        rng = np.random.default_rng(42)
        return np.sort(rng.uniform(0, 1, 200))

    def _run_null_stream(self, detector, ref_scores, n_stream=800, seed=0):
        """Run a null stream (iid from reference) and return whether alarm fired."""
        rng = np.random.default_rng(seed)
        for t in range(n_stream):
            score = rng.choice(ref_scores)
            record = _make_record(score, t)
            stat = detector.update(record)
            if detector.alarm_step is not None:
                return True
        return False

    def test_point_far(self, reference_scores):
        """PointMartingale FAR ≤ α over many null runs."""
        stats = _make_frozen_stats(reference_scores)
        n_runs = 200
        alarms = 0
        for seed in range(n_runs):
            det = PointMartingale(stats, alpha=0.05, epsilon=0.3)
            if self._run_null_stream(det, reference_scores, seed=seed):
                alarms += 1
        far = alarms / n_runs
        # By Ville's inequality, FAR should be ≤ 0.05
        # Allow some statistical slack (Wilson UB at 200 trials)
        assert far <= 0.10, f"PointMartingale FAR={far:.2%} exceeds tolerance"

    def test_cusum_far(self, reference_scores):
        """CUSUMMartingale FAR ≤ α over many null runs (with horizon correction)."""
        stats = _make_frozen_stats(reference_scores)
        n_runs = 200
        alarms = 0
        for seed in range(n_runs):
            det = CUSUMMartingale(stats, alpha=0.05, epsilon=0.3, horizon=800)
            if self._run_null_stream(det, reference_scores, seed=seed):
                alarms += 1
        far = alarms / n_runs
        assert far <= 0.10, f"CUSUMMartingale FAR={far:.2%} exceeds tolerance"

    def test_scan_far(self, reference_scores):
        """ScanMartingale FAR ≤ α over many null runs."""
        stats = _make_frozen_stats(reference_scores)
        n_runs = 200
        alarms = 0
        for seed in range(n_runs):
            det = ScanMartingale(stats, alpha=0.05, window=50, epsilon=0.3)
            if self._run_null_stream(det, reference_scores, seed=seed):
                alarms += 1
        far = alarms / n_runs
        assert far <= 0.10, f"ScanMartingale FAR={far:.2%} exceeds tolerance"


# ---------------------------------------------------------------------------
# Test: Detection on shifted streams
# ---------------------------------------------------------------------------


class TestDetection:
    """All detectors should detect obvious shifts."""

    @pytest.fixture()
    def reference_scores(self):
        rng = np.random.default_rng(42)
        return np.sort(rng.uniform(0, 0.3, 200))

    def _run_shifted_stream(self, detector, ref_scores, shift_onset=200, n_post=300):
        """Run stream with shift after onset. Returns latency or None."""
        rng = np.random.default_rng(99)
        # Pre-shift: iid from reference
        for t in range(shift_onset):
            score = rng.choice(ref_scores)
            detector.update(_make_record(score, t))

        # Post-shift: scores from shifted distribution
        for t in range(n_post):
            score = rng.uniform(0.7, 1.0)  # large shift
            detector.update(_make_record(score, shift_onset + t))
            if detector.alarm_step is not None:
                return detector.alarm_step - shift_onset

        return None

    def test_scan_detects_large_shift(self, reference_scores):
        stats = _make_frozen_stats(reference_scores)
        det = ScanMartingale(stats, alpha=0.05, window=50, epsilon=0.3)
        latency = self._run_shifted_stream(det, reference_scores)
        assert latency is not None, "ScanMartingale failed to detect large shift"
        assert latency < 100, f"ScanMartingale too slow: latency={latency}"

    def test_cusum_detects_large_shift(self, reference_scores):
        stats = _make_frozen_stats(reference_scores)
        det = CUSUMMartingale(stats, alpha=0.05, epsilon=0.3)
        latency = self._run_shifted_stream(det, reference_scores)
        assert latency is not None, "CUSUMMartingale failed to detect large shift"
        assert latency < 100

    def test_point_detects_large_shift(self, reference_scores):
        stats = _make_frozen_stats(reference_scores)
        det = PointMartingale(stats, alpha=0.05, epsilon=0.3)
        latency = self._run_shifted_stream(det, reference_scores)
        # Point may be slower due to dilution but should still detect
        assert latency is not None, "PointMartingale failed to detect large shift"

    def test_scan_faster_than_point_on_late_shift(self, reference_scores):
        """Scan should be faster than Point when changepoint is late (dilution)."""
        stats = _make_frozen_stats(reference_scores)
        # Long pre-shift to maximize dilution
        det_scan = ScanMartingale(stats, alpha=0.05, window=50, epsilon=0.3)
        det_point = PointMartingale(stats, alpha=0.05, epsilon=0.3)
        lat_scan = self._run_shifted_stream(det_scan, reference_scores, shift_onset=500)
        lat_point = self._run_shifted_stream(det_point, reference_scores, shift_onset=500)
        # Scan should be no worse than point
        if lat_scan is not None and lat_point is not None:
            assert lat_scan <= lat_point + 50  # allow some slack


# ---------------------------------------------------------------------------
# Test: Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_alpha_zero_raises(self):
        stats = _make_frozen_stats(np.linspace(0, 1, 10))
        with pytest.raises(ValueError, match="alpha"):
            ScanMartingale(stats, alpha=0.0)

    def test_alpha_one_raises(self):
        stats = _make_frozen_stats(np.linspace(0, 1, 10))
        with pytest.raises(ValueError, match="alpha"):
            ScanMartingale(stats, alpha=1.0)

    def test_epsilon_zero_raises(self):
        stats = _make_frozen_stats(np.linspace(0, 1, 10))
        with pytest.raises(ValueError, match="epsilon"):
            ScanMartingale(stats, epsilon=0.0)

    def test_epsilon_one_raises(self):
        stats = _make_frozen_stats(np.linspace(0, 1, 10))
        with pytest.raises(ValueError, match="epsilon"):
            ScanMartingale(stats, epsilon=1.0)

    def test_window_zero_raises(self):
        stats = _make_frozen_stats(np.linspace(0, 1, 10))
        with pytest.raises(ValueError, match="window"):
            ScanMartingale(stats, window=0)

    def test_insufficient_reference_raises(self):
        stats = _make_frozen_stats(np.array([0.5]))  # only 1 score
        with pytest.raises(CalibrationError):
            ScanMartingale(stats)

    def test_two_reference_scores_ok(self):
        """Minimum 2 reference scores should work."""
        stats = _make_frozen_stats(np.array([0.3, 0.7]))
        det = ScanMartingale(stats)
        result = det.update(_make_record(0.5))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Test: update_detailed consistency
# ---------------------------------------------------------------------------


class TestDetailedUpdate:
    def test_detailed_returns_martingale_alarm(self):
        stats = _make_frozen_stats(np.linspace(0, 1, 50))
        det = ScanMartingale(stats)
        result = det.update_detailed(_make_record(0.5))
        assert isinstance(result, MartingaleAlarm)
        assert result.time_step == 1
        assert result.threshold == det.threshold
        assert 0 < result.p_value <= 1.0

    def test_detailed_matches_update(self):
        """update() and update_detailed() should agree on log_wealth."""
        rng = np.random.default_rng(42)
        ref = np.sort(rng.uniform(0, 1, 100))
        stats = _make_frozen_stats(ref)

        det1 = ScanMartingale(stats, alpha=0.05, window=50, epsilon=0.3)
        det2 = ScanMartingale(stats, alpha=0.05, window=50, epsilon=0.3)

        for t in range(50):
            score = rng.uniform(0, 1)
            stat1 = det1.update(_make_record(score, t))
            alarm2 = det2.update_detailed(_make_record(score, t))
            assert math.isclose(stat1, alarm2.log_wealth, rel_tol=1e-12)

    def test_cusum_detailed_reset(self):
        """CUSUM reset should be visible in detailed output."""
        ref = np.linspace(0, 1, 100)
        stats = _make_frozen_stats(ref)
        det = CUSUMMartingale(stats)
        # Feed scores near median → log_wealth should stay near 0
        for t in range(20):
            alarm = det.update_detailed(_make_record(0.5, t))
        # After many neutral observations, wealth should have been reset
        assert alarm.log_wealth >= 0.0  # CUSUM never goes below 0


# ---------------------------------------------------------------------------
# Test: Scan window mechanics
# ---------------------------------------------------------------------------


class TestScanWindow:
    def test_n_active_grows_to_window(self):
        """Number of active sub-martingales should grow to window size."""
        stats = _make_frozen_stats(np.linspace(0, 1, 50))
        det = ScanMartingale(stats, window=10)
        for t in range(20):
            det.update(_make_record(0.5, t))
        assert det.n_active == 10

    def test_n_active_starts_at_one(self):
        stats = _make_frozen_stats(np.linspace(0, 1, 50))
        det = ScanMartingale(stats, window=10)
        det.update(_make_record(0.5))
        assert det.n_active == 1

    def test_scan_threshold_includes_union_bound(self):
        """Scan threshold should be log(W/α), CUSUM uses log(horizon/α)."""
        stats = _make_frozen_stats(np.linspace(0, 1, 50))
        det_scan = ScanMartingale(stats, alpha=0.05, window=50)
        det_cusum = CUSUMMartingale(stats, alpha=0.05, horizon=1000)
        # Both use union-bound-style thresholds
        assert math.isclose(det_scan.threshold, math.log(50 / 0.05), rel_tol=1e-10)
        assert math.isclose(det_cusum.threshold, math.log(1000 / 0.05), rel_tol=1e-10)
