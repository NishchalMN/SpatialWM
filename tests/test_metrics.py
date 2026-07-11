"""
Tests for spatialwm.eval.metrics — RED now, green on impl.

Contracts defended:
- chamfer: symmetric and zero on identical clouds
- miou: known value on a tiny pred/gt
- depth_rmse: known value on simple arrays
- depth_delta: known fraction for a trivial case
- occupancy_iou: known value for identical/disjoint grids
- reliability_curve: correct output shape; boundary properties
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialwm.eval.metrics import (
    chamfer,
    depth_delta,
    depth_rmse,
    miou,
    occupancy_iou,
    reliability_curve,
)

# ---------------------------------------------------------------------------
# chamfer
# ---------------------------------------------------------------------------

class TestChamfer:
    def test_zero_on_identical_clouds(self):
        """chamfer(a, a) == 0."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((50, 3))
        assert chamfer(a, a) == pytest.approx(0.0, abs=1e-8)

    def test_symmetric(self):
        """chamfer(a, b) == chamfer(b, a)."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((30, 3))
        b = rng.standard_normal((40, 3))
        assert chamfer(a, b) == pytest.approx(chamfer(b, a), rel=1e-6)

    def test_grows_with_distance(self):
        """Chamfer distance is larger when clouds are further apart."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal((20, 3))
        b_near = a + rng.normal(0, 0.1, a.shape)
        b_far = a + 10.0

        assert chamfer(a, b_near) < chamfer(a, b_far), (
            "Chamfer should be smaller for the nearby cloud"
        )

    def test_nonnegative(self):
        """Chamfer distance is always >= 0."""
        rng = np.random.default_rng(3)
        a = rng.standard_normal((15, 3))
        b = rng.standard_normal((15, 3))
        assert chamfer(a, b) >= 0.0

    def test_known_value_unit_offset(self):
        """
        chamfer([[0,0,0]], [[1,0,0]]) == 1.0
        (nearest distance from each point to the other set is 1).
        """
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        # Symmetric Chamfer = avg(nn_a->b + nn_b->a) = avg(1 + 1) = 1
        result = chamfer(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# miou
# ---------------------------------------------------------------------------

class TestMiou:
    def test_miou_perfect_prediction(self):
        """miou == 1.0 when pred == gt."""
        pred = np.array([0, 1, 2, 0, 1, 2])
        gt = np.array([0, 1, 2, 0, 1, 2])
        result = miou(pred, gt, num_classes=3)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_miou_zero_no_overlap(self):
        """miou == 0.0 when pred and gt have no overlap per class."""
        pred = np.array([0, 0, 0, 0])
        gt = np.array([1, 1, 1, 1])
        result = miou(pred, gt, num_classes=2)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_miou_known_tiny(self):
        """
        Hand-computed case:
        pred = [0, 0, 1, 1]
        gt   = [0, 1, 0, 1]
        class 0: TP=1, FP=1, FN=1 -> IoU = 1/3
        class 1: TP=1, FP=1, FN=1 -> IoU = 1/3
        mIoU = 1/3
        """
        pred = np.array([0, 0, 1, 1])
        gt = np.array([0, 1, 0, 1])
        result = miou(pred, gt, num_classes=2)
        assert result == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_miou_in_unit_interval(self):
        """miou is always in [0, 1]."""
        rng = np.random.default_rng(0)
        pred = rng.integers(0, 5, 100)
        gt = rng.integers(0, 5, 100)
        result = miou(pred, gt, num_classes=5)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# depth_rmse
# ---------------------------------------------------------------------------

class TestDepthRmse:
    def test_zero_on_identical_maps(self):
        """depth_rmse(d, d) == 0."""
        d = np.random.rand(10, 10).astype(np.float32) + 0.1
        assert depth_rmse(d, d) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        """RMSE of constant offset == that offset."""
        pred = np.ones((4, 4)) * 2.0
        gt = np.ones((4, 4)) * 1.0   # all differ by 1
        assert depth_rmse(pred, gt) == pytest.approx(1.0, abs=1e-6)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        pred = rng.uniform(0.1, 5.0, (8, 8))
        gt = rng.uniform(0.1, 5.0, (8, 8))
        assert depth_rmse(pred, gt) >= 0.0


# ---------------------------------------------------------------------------
# depth_delta
# ---------------------------------------------------------------------------

class TestDepthDelta:
    def test_perfect_depth_gives_delta_1(self):
        """pred == gt -> all pixels pass threshold -> delta==1.0."""
        d = np.ones((6, 6)) * 3.0
        assert depth_delta(d, d, thresh=1.25) == pytest.approx(1.0, abs=1e-6)

    def test_no_pixel_passes_gives_delta_0(self):
        """
        pred/gt = 2.0 and gt=1.0 -> ratio=2.0 > 1.25 -> no pixel passes.
        """
        pred = np.full((4, 4), 2.0)
        gt = np.full((4, 4), 1.0)
        assert depth_delta(pred, gt, thresh=1.25) == pytest.approx(0.0, abs=1e-6)

    def test_half_pixels_pass(self):
        """Half the pixels have ratio 1.0, half have ratio > thresh -> delta=0.5."""
        pred = np.array([1.0, 1.0, 2.0, 2.0])
        gt = np.array([1.0, 1.0, 1.0, 1.0])
        result = depth_delta(pred, gt, thresh=1.25)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_in_unit_interval(self):
        rng = np.random.default_rng(1)
        pred = rng.uniform(0.5, 4.0, 100)
        gt = rng.uniform(0.5, 4.0, 100)
        assert 0.0 <= depth_delta(pred, gt) <= 1.0


# ---------------------------------------------------------------------------
# occupancy_iou
# ---------------------------------------------------------------------------

class TestOccupancyIou:
    def test_identical_gives_1(self):
        """IoU of identical binary grid == 1.0."""
        grid = np.array([1, 0, 1, 1, 0], dtype=float)
        assert occupancy_iou(grid, grid) == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_gives_0(self):
        """IoU of disjoint grids == 0.0."""
        pred = np.array([1, 1, 0, 0], dtype=float)
        gt = np.array([0, 0, 1, 1], dtype=float)
        assert occupancy_iou(pred, gt) == pytest.approx(0.0, abs=1e-6)

    def test_known_partial_overlap(self):
        """
        pred = [1,1,0,0], gt = [1,0,1,0]
        intersection = [1,0,0,0] -> 1
        union        = [1,1,1,0] -> 3
        IoU = 1/3
        """
        pred = np.array([1, 1, 0, 0], dtype=float)
        gt = np.array([1, 0, 1, 0], dtype=float)
        assert occupancy_iou(pred, gt) == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_nonnegative_and_leq_1(self):
        rng = np.random.default_rng(2)
        pred = (rng.uniform(size=100) > 0.5).astype(float)
        gt = (rng.uniform(size=100) > 0.5).astype(float)
        result = occupancy_iou(pred, gt)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# reliability_curve
# ---------------------------------------------------------------------------

class TestReliabilityCurve:
    def test_output_shapes(self):
        """reliability_curve returns two arrays of length == bins."""
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 200)
        labels = (rng.uniform(0, 1, 200) > 0.5).astype(float)
        conf, acc = reliability_curve(probs, labels, bins=10)
        assert len(conf) == 10
        assert len(acc) == 10

    def test_perfectly_calibrated_model(self):
        """
        If every probability is matched by the ground-truth label (labels ~= probs),
        the reliability curve is approximately the diagonal (conf ≈ acc per bin).
        We allow 0.1 per-bin deviation to account for sampling variance.
        """
        rng = np.random.default_rng(5)
        probs = rng.uniform(0, 1, 2000)
        # Use Bernoulli draws with probs as the true probability
        labels = (rng.uniform(0, 1, 2000) < probs).astype(float)
        conf, acc = reliability_curve(probs, labels, bins=10)
        for c, a in zip(conf, acc):
            if not np.isnan(c) and not np.isnan(a):
                assert abs(c - a) < 0.15, (
                    f"Calibration gap at conf={c:.2f}: acc={a:.2f}, diff={abs(c-a):.3f}"
                )

    def test_values_in_unit_interval(self):
        """All returned confidence and accuracy values are in [0, 1]."""
        rng = np.random.default_rng(3)
        probs = rng.uniform(0, 1, 100)
        labels = (rng.uniform(size=100) > 0.5).astype(float)
        conf, acc = reliability_curve(probs, labels, bins=5)
        for c, a in zip(conf, acc):
            if not np.isnan(c):
                assert 0.0 <= c <= 1.0
            if not np.isnan(a):
                assert 0.0 <= a <= 1.0
