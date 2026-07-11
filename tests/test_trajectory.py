"""
Tests for spatialwm.eval.trajectory — RED now, green on impl.

Contracts defended:
1. umeyama recovers a known Sim(3) (R, t, s) from synthetic src/dst pair.
2. ate ≈ 0 on two identical aligned trajectories.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spatialwm.eval.trajectory import ate, umeyama

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_angle_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_rel = R_est @ R_gt.T
    cos_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def _make_sim3_pair(rng, n: int = 50, scale: float = 1.5):
    """Construct src, dst = s * R @ src + t (with noise=0 for exact recovery check)."""
    src = rng.standard_normal((n, 3))
    R_gt = Rotation.from_euler("xyz", [12.0, -7.0, 20.0], degrees=True).as_matrix()
    t_gt = np.array([1.0, -2.0, 0.5])
    dst = scale * (R_gt @ src.T).T + t_gt
    return src, dst, R_gt, t_gt, scale


# ---------------------------------------------------------------------------
# umeyama
# ---------------------------------------------------------------------------

class TestUmeyama:
    def test_recovers_known_rotation(self):
        """umeyama recovers ground-truth R to within 0.1°."""
        rng = np.random.default_rng(7)
        src, dst, R_gt, t_gt, s_gt = _make_sim3_pair(rng, n=50, scale=1.5)
        R_est, t_est, s_est = umeyama(src, dst, with_scale=True)
        err = _rotation_angle_deg(R_est, R_gt)
        assert err < 0.1, f"Rotation error {err:.4f}° >= 0.1°"

    def test_recovers_known_translation(self):
        """umeyama recovers ground-truth t to within 1e-4."""
        rng = np.random.default_rng(7)
        src, dst, R_gt, t_gt, s_gt = _make_sim3_pair(rng, n=50, scale=1.5)
        R_est, t_est, s_est = umeyama(src, dst, with_scale=True)
        np.testing.assert_allclose(t_est, t_gt, atol=1e-4)

    def test_recovers_known_scale(self):
        """umeyama recovers ground-truth scale to within 1%."""
        rng = np.random.default_rng(7)
        src, dst, R_gt, t_gt, s_gt = _make_sim3_pair(rng, n=50, scale=1.5)
        R_est, t_est, s_est = umeyama(src, dst, with_scale=True)
        assert abs(s_est - s_gt) / s_gt < 0.01, (
            f"Scale error: est={s_est:.4f}, gt={s_gt:.4f}"
        )

    def test_without_scale_returns_unit_scale(self):
        """with_scale=False must return s == 1.0 (pure SE(3) alignment)."""
        rng = np.random.default_rng(9)
        src, dst, _, _, _ = _make_sim3_pair(rng, n=30, scale=1.0)
        R_est, t_est, s_est = umeyama(src, dst, with_scale=False)
        assert abs(s_est - 1.0) < 1e-9, f"Expected s=1 with with_scale=False, got {s_est}"

    def test_output_shapes(self):
        """umeyama returns (3,3), (3,), float."""
        rng = np.random.default_rng(0)
        src = rng.standard_normal((20, 3))
        dst = rng.standard_normal((20, 3))
        R_est, t_est, s_est = umeyama(src, dst)
        assert R_est.shape == (3, 3)
        assert np.asarray(t_est).shape == (3,)
        assert isinstance(s_est, float) or np.ndim(s_est) == 0


# ---------------------------------------------------------------------------
# ate
# ---------------------------------------------------------------------------

class TestAte:
    def test_ate_zero_on_identical_trajectories(self):
        """ATE is 0 when estimated == ground truth (after Umeyama alignment)."""
        rng = np.random.default_rng(11)
        traj = rng.standard_normal((40, 3))
        error = ate(traj, traj)
        assert error == pytest.approx(0.0, abs=1e-6), (
            f"ATE should be 0 on identical trajectories, got {error}"
        )

    def test_ate_increases_with_drift(self):
        """ATE grows as trajectory drift increases."""
        rng = np.random.default_rng(12)
        traj_gt = np.cumsum(rng.standard_normal((30, 3)) * 0.1, axis=0)

        noise_small = rng.standard_normal((30, 3)) * 0.01
        noise_large = rng.standard_normal((30, 3)) * 1.0

        ate_small = ate(traj_gt + noise_small, traj_gt)
        ate_large = ate(traj_gt + noise_large, traj_gt)

        assert ate_small < ate_large, (
            f"Expected ate_small < ate_large, got {ate_small:.4f} vs {ate_large:.4f}"
        )

    def test_ate_nonnegative(self):
        """ATE is always >= 0."""
        rng = np.random.default_rng(13)
        traj_gt = rng.standard_normal((20, 3))
        traj_est = traj_gt + rng.standard_normal((20, 3)) * 0.5
        assert ate(traj_est, traj_gt) >= 0.0
