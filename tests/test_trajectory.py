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

from spatialwm.eval.trajectory import ate, rpe, umeyama

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


# ---------------------------------------------------------------------------
# ate poses
# ---------------------------------------------------------------------------

class TestAtePoses:
    def test_ate_poses_zero_on_identical(self):
        """ATE is 0 on identical (N, 4, 4) pose trajectories."""
        rng = np.random.default_rng(14)
        n = 25
        traj = np.array([np.eye(4) for _ in range(n)])
        traj[:, :3, 3] = rng.standard_normal((n, 3))
        assert ate(traj, traj) == pytest.approx(0.0, abs=1e-6)

    def test_ate_poses_matches_positions(self):
        """ATE on poses matches ATE on extracted positions."""
        rng = np.random.default_rng(15)
        n = 30
        traj_gt = np.array([np.eye(4) for _ in range(n)])
        traj_gt[:, :3, 3] = np.cumsum(rng.standard_normal((n, 3)) * 0.1, axis=0)
        
        traj_est = np.array([np.eye(4) for _ in range(n)])
        traj_est[:, :3, 3] = traj_gt[:, :3, 3] + rng.standard_normal((n, 3)) * 0.05
        
        err_poses = ate(traj_est, traj_gt)
        err_pos = ate(traj_est[:, :3, 3], traj_gt[:, :3, 3])
        assert err_poses == pytest.approx(err_pos, abs=1e-9)


# ---------------------------------------------------------------------------
# rpe
# ---------------------------------------------------------------------------

class TestRpe:
    def test_rpe_zero_on_identical(self):
        """RPE is 0 when estimated == ground truth."""
        n = 20
        traj = np.array([np.eye(4) for _ in range(n)])
        for i in range(n):
            traj[i, :3, 3] = [float(i), 0.0, 0.0]
        assert rpe(traj, traj, delta=1) == pytest.approx(0.0, abs=1e-6)
        assert rpe(traj, traj, delta=3) == pytest.approx(0.0, abs=1e-6)

    def test_rpe_increases_with_drift(self):
        """RPE translation error increases as drift increases."""
        rng = np.random.default_rng(16)
        n = 40
        traj_gt = np.array([np.eye(4) for _ in range(n)])
        for i in range(n):
            traj_gt[i, :3, 3] = [float(i) * 0.1, 0.0, 0.0]
            
        traj_est_small = traj_gt.copy()
        traj_est_large = traj_gt.copy()
        
        traj_est_small[:, :3, 3] += rng.standard_normal((n, 3)) * 0.005
        traj_est_large[:, :3, 3] += rng.standard_normal((n, 3)) * 0.1
        
        rpe_small = rpe(traj_est_small, traj_gt, delta=2)
        rpe_large = rpe(traj_est_large, traj_gt, delta=2)
        assert rpe_small < rpe_large


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

class TestTrajectoryValidation:
    def test_umeyama_invalid_shape(self):
        with pytest.raises(ValueError, match="shape"):
            umeyama(np.zeros((5, 2)), np.zeros((5, 2)))
        with pytest.raises(ValueError, match="shape"):
            umeyama(np.zeros((5, 3)), np.zeros((5, 2)))
        with pytest.raises(ValueError, match="match"):
            umeyama(np.zeros((5, 3)), np.zeros((6, 3)))

    def test_umeyama_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            umeyama(np.array([[np.nan, 0, 0], [0, 0, 0], [0, 0, 0]]), np.zeros((3, 3)))
        with pytest.raises(ValueError, match="non-finite"):
            umeyama(np.zeros((3, 3)), np.array([[np.inf, 0, 0], [0, 0, 0], [0, 0, 0]]))

    def test_umeyama_too_few_points(self):
        with pytest.raises(ValueError, match="points"):
            umeyama(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_umeyama_degeneracy(self):
        # Collinear points (rank 1)
        collinear_src = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        collinear_dst = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        with pytest.raises(ValueError, match="degenerate"):
            umeyama(collinear_src, collinear_dst)
            
        # Coincident points (rank 0)
        coincident_src = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        coincident_dst = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="degenerate"):
            umeyama(coincident_src, coincident_dst)

    def test_ate_invalid_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            ate(np.zeros((5, 4)), np.zeros((5, 3)))
        with pytest.raises(ValueError, match="shape"):
            ate(np.zeros((5, 4, 3)), np.zeros((5, 4, 4)))

    def test_ate_mismatched_frames(self):
        with pytest.raises(ValueError, match="same number of frames"):
            ate(np.zeros((5, 3)), np.zeros((6, 3)))

    def test_ate_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            ate(np.array([[np.nan, 0, 0], [0, 0, 0], [0, 0, 0]]), np.zeros((3, 3)))

    def test_rpe_invalid_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            rpe(np.zeros((5, 3)), np.zeros((5, 4, 4)))
        with pytest.raises(ValueError, match="shape"):
            rpe(np.zeros((5, 4, 4)), np.zeros((5, 4, 3)))

    def test_rpe_mismatched_frames(self):
        with pytest.raises(ValueError, match="same number of frames"):
            rpe(np.zeros((5, 4, 4)), np.zeros((6, 4, 4)))

    def test_rpe_non_finite(self):
        traj_est = np.array([np.eye(4) for _ in range(5)])
        traj_gt = np.array([np.eye(4) for _ in range(5)])
        traj_est[2, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            rpe(traj_est, traj_gt)

    def test_rpe_invalid_delta(self):
        traj = np.array([np.eye(4) for _ in range(5)])
        with pytest.raises(TypeError, match="integer"):
            rpe(traj, traj, delta="1") # type: ignore
        with pytest.raises(ValueError, match="at least 1"):
            rpe(traj, traj, delta=0)
        with pytest.raises(ValueError, match="strictly less than"):
            rpe(traj, traj, delta=5)
        with pytest.raises(ValueError, match="strictly less than"):
            rpe(traj, traj, delta=6)
