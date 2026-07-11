"""Tests for spatialwm.geometry.lidar_odometry — RED now, green on impl.

Contracts defended:
1. voxel_downsample reduces point count and stays within input bounds.
2. odometry recovers a known constant translation between scans.
3. first pose is identity; output shape is (K,4,4).
"""

from __future__ import annotations

import numpy as np

from spatialwm.geometry.lidar_odometry import lidar_odometry, voxel_downsample


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


class TestLidarOdometry:
    def test_recovers_constant_translation(self):
        d = np.array([0.1, 0.0, 0.0])
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
        scans = [base, base + np.array([0.1, 0.0, 0.0])]
        poses = lidar_odometry(scans, voxel=0.1)
        np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-9)

    def test_output_shape(self):
        base = _grid_cloud()
        scans = [base, base + 0.1, base + 0.2]
        poses = lidar_odometry(scans, voxel=0.1)
        assert poses.shape == (3, 4, 4)
