"""Tests for spatialwm.geometry.lidar_odometry — RED now, green on impl.

Contracts defended:
1. voxel_downsample reduces point count and stays within input bounds.
2. odometry recovers a known constant translation between scans.
3. first pose is identity; output shape is (K,4,4).
4. transform accumulation orientation, shape, finite SE(3), invalid inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialwm.geometry.lidar_odometry import (
    lidar_odometry,
    lidar_odometry_detailed,
    voxel_downsample,
)


def _grid_cloud(n_side: int = 20) -> np.ndarray:
    lin = np.linspace(-1.0, 1.0, n_side)
    xx, yy = np.meshgrid(lin, lin)
    zz = 0.1 * np.sin(3 * xx) * np.cos(3 * yy)  # some structure for ICP
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


class TestVoxelDownsample:
    def test_reduces_point_count(self):
        pts = np.random.default_rng(0).standard_normal((5000, 3))
        out = voxel_downsample(pts, voxel=0.5)
        assert out.shape[1] == 3
        assert out.shape[0] <= pts.shape[0]

    def test_stays_within_bounds(self):
        pts = np.random.default_rng(1).standard_normal((5000, 3))
        out = voxel_downsample(pts, voxel=0.5)
        assert np.all(out.min(axis=0) >= pts.min(axis=0) - 1e-6)
        assert np.all(out.max(axis=0) <= pts.max(axis=0) + 1e-6)

    def test_invalid_inputs(self):
        # 1. Non-numpy array
        with pytest.raises(TypeError):
            voxel_downsample([[1.0, 2.0, 3.0]], voxel=0.5)
        # 2. Wrong dimensions (ndim != 2)
        with pytest.raises(ValueError):
            voxel_downsample(np.array([1.0, 2.0, 3.0]), voxel=0.5)
        # 3. Wrong shape (N, 4)
        with pytest.raises(ValueError):
            voxel_downsample(np.zeros((10, 4)), voxel=0.5)
        # 4. Non-floating type
        with pytest.raises(TypeError):
            voxel_downsample(np.zeros((10, 3), dtype=np.int32), voxel=0.5)
        # 5. Non-finite values
        pts_nan = np.zeros((10, 3))
        pts_nan[0, 0] = np.nan
        with pytest.raises(ValueError):
            voxel_downsample(pts_nan, voxel=0.5)
        # 6. Non-positive voxel size
        pts = np.zeros((10, 3))
        with pytest.raises(ValueError):
            voxel_downsample(pts, voxel=0.0)
        with pytest.raises(ValueError):
            voxel_downsample(pts, voxel=-0.1)


class TestLidarOdometry:
    def test_scan_to_submap_returns_diagnostics_and_metric_poses(self):
        displacement = np.array([0.04, -0.01, 0.0])
        base = _grid_cloud(n_side=30)
        scans = [base + index * displacement for index in range(6)]
        result = lidar_odometry_detailed(
            scans,
            method="scan_to_submap",
            voxel=0.08,
            submap_size=3,
            submap_voxel=0.1,
        )
        assert result.poses_scan_to_world.shape == (6, 4, 4)
        assert len(result.diagnostics) == 5
        assert all(item.method == "scan_to_submap" for item in result.diagnostics)
        assert all(item.fitness > 0.1 for item in result.diagnostics)
        np.testing.assert_allclose(
            result.poses_scan_to_world[-1, :3, 3], -5 * displacement, atol=0.03
        )

    def test_recovers_constant_translation(self):
        d = np.array([0.05, 0.0, 0.0])
        base = _grid_cloud()
        scans = [base + i * d for i in range(4)]
        poses = lidar_odometry(scans, voxel=0.1)

        # global translation of pose k should track k*(-d) or k*d depending on
        # convention; magnitude per step must match |d| within tolerance.
        steps = np.diff(poses[:, :3, 3], axis=0)
        step_mags = np.linalg.norm(steps, axis=1)
        np.testing.assert_allclose(step_mags, np.linalg.norm(d), atol=0.02)

    def test_first_pose_is_identity(self):
        base = _grid_cloud()
        scans = [base, base + np.array([0.05, 0.0, 0.0])]
        poses = lidar_odometry(scans, voxel=0.1)
        np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-9)

    def test_output_shape(self):
        base = _grid_cloud()
        scans = [base, base + 0.05, base + 0.1]
        poses = lidar_odometry(scans, voxel=0.1)
        assert poses.shape == (3, 4, 4)

    def test_finite_se3(self):
        d = np.array([0.03, -0.02, 0.01])
        base = _grid_cloud()
        scans = [base + i * d for i in range(3)]
        poses = lidar_odometry(scans, voxel=0.1)
        
        for P in poses:
            assert np.all(np.isfinite(P))
            # Check bottom row is [0, 0, 0, 1]
            np.testing.assert_allclose(P[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
            # Check rotation part is orthonormal
            R = P[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
            assert abs(np.linalg.det(R) - 1.0) < 1e-5

    def test_transform_accumulation_orientation(self):
        d = np.array([0.04, 0.0, 0.0])
        base = _grid_cloud()
        scans = [base + i * d for i in range(4)]
        poses = lidar_odometry(scans, voxel=0.1)
        
        # Mapping scan k to scan 0: P_k translation should be close to -k * d
        for k in range(4):
            np.testing.assert_allclose(poses[k, :3, 3], -k * d, atol=0.01)

    def test_invalid_inputs(self):
        base = _grid_cloud()
        # 1. Non-list/tuple scans
        with pytest.raises(TypeError):
            lidar_odometry(base)
        # 2. Empty list of scans
        with pytest.raises(ValueError):
            lidar_odometry([])
        # 3. List containing non-numpy array
        with pytest.raises(TypeError):
            lidar_odometry([base, "not an array"])
        # 4. Invalid voxel
        with pytest.raises(ValueError):
            lidar_odometry([base, base + 0.05], voxel=-0.5)
        # 5. Invalid max_iters
        with pytest.raises(ValueError):
            lidar_odometry([base, base + 0.05], max_iters=0)
        with pytest.raises(ValueError):
            lidar_odometry([base, base + 0.05], max_correspondence_distance=-1.0)
        # 7. Invalid min_fitness type
        with pytest.raises(TypeError):
            lidar_odometry([base, base + 0.05], min_fitness="not_a_float")
        # 8. Invalid min_fitness value (negative)
        with pytest.raises(ValueError):
            lidar_odometry([base, base + 0.05], min_fitness=-0.1)
        # 9. Invalid min_fitness value (greater than 1)
        with pytest.raises(ValueError):
            lidar_odometry([base, base + 0.05], min_fitness=1.1)

    def test_max_correspondence_distance_effect(self):
        d = np.array([0.05, 0.0, 0.0])
        base = _grid_cloud()
        scans = [base, base + d]
        
        # If correspondence distance is very small, registration fails to find
        # enough/correct matches, leading to low/zero fitness, which should raise a RuntimeError.
        with pytest.raises(RuntimeError) as exc_info:
            lidar_odometry(
                scans, voxel=0.1, max_correspondence_distance=0.001, min_fitness=0.1
            )
        
        assert "fitness" in str(exc_info.value)
        assert "is below min_fitness" in str(exc_info.value)

        # Boundary validation: min_fitness = 0.0 should not raise error even with poor registration.
        poses_zero = lidar_odometry(
            scans, voxel=0.1, max_correspondence_distance=0.001, min_fitness=0.0
        )
        np.testing.assert_allclose(poses_zero[1], np.eye(4), atol=1e-5)
