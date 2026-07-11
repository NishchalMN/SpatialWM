"""
Tests for spatialwm.geometry.camera — raises NotImplementedError until implemented.

Contracts defended:
1. round-trip: unproject(project(X_cam)) ≈ X_cam to 1e-6
2. camera_center: -R.T @ t for arbitrary pose
"""

from __future__ import annotations

import numpy as np

from spatialwm.geometry.camera import camera_center, project, unproject

# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------

K = np.array(
    [[800.0, 0.0, 320.0],
     [0.0, 800.0, 240.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# Identity camera (cam == world for cam1)
R_id = np.eye(3, dtype=np.float64)
t_zero = np.zeros((3, 1), dtype=np.float64)


def _make_pts(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    X[:, 2] = rng.uniform(2.0, 8.0, n)  # strictly positive depth
    return X


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestUnprojectProjectRoundTrip:
    """project then unproject should recover original camera-frame points."""

    def test_identity_camera_round_trip(self):
        """unproject(project(X_cam)) == X_cam for identity pose to 1e-6."""
        X_cam = _make_pts(20)
        # Depth = z-coordinate of the camera-frame points
        uv = project(K, R_id, t_zero, X_cam)
        depth = X_cam[:, 2]
        X_recovered = unproject(K, uv, depth)
        np.testing.assert_allclose(X_recovered, X_cam, atol=1e-6)

    def test_nontrivial_pose_round_trip(self):
        """Round-trip holds for a camera with non-zero rotation and translation."""
        from scipy.spatial.transform import Rotation

        R = Rotation.from_euler("xyz", [5.0, -10.0, 3.0], degrees=True).as_matrix()
        t = np.array([[0.3], [-0.1], [0.5]])

        # World points
        X_world = _make_pts(15, seed=1)
        # Camera-frame points (what unproject should return)
        X_cam = (R @ X_world.T + t).T

        # Depths are the z-coordinates in camera frame
        depth = X_cam[:, 2]
        uv = project(K, R, t, X_world)
        X_recovered = unproject(K, uv, depth)
        np.testing.assert_allclose(X_recovered, X_cam, atol=1e-6)

    def test_single_point_round_trip(self):
        """Round-trip works for a single point (N=1 edge case)."""
        X = np.array([[0.2, -0.3, 5.0]])
        uv = project(K, R_id, t_zero, X)
        depth = X[:, 2]
        X_rec = unproject(K, uv, depth)
        np.testing.assert_allclose(X_rec, X, atol=1e-6)


# ---------------------------------------------------------------------------
# camera_center
# ---------------------------------------------------------------------------

class TestCameraCenter:
    def test_identity_pose_center_at_origin(self):
        """Camera with R=I, t=0 has center at the world origin."""
        C = camera_center(R_id, t_zero)
        np.testing.assert_allclose(C.ravel(), np.zeros(3), atol=1e-12)

    def test_known_translation(self):
        """camera_center == -R.T @ t for a pure translation."""
        t = np.array([[1.0], [2.0], [3.0]])
        C = camera_center(R_id, t)
        expected = -R_id.T @ t
        np.testing.assert_allclose(C.ravel(), expected.ravel(), atol=1e-12)

    def test_general_pose(self):
        """camera_center == -R.T @ t for an arbitrary SE(3) pose."""
        from scipy.spatial.transform import Rotation

        R = Rotation.from_euler("y", 37.0, degrees=True).as_matrix()
        t = np.array([[0.7], [-0.4], [1.2]])
        C = camera_center(R, t)
        expected = -R.T @ t
        np.testing.assert_allclose(C.ravel(), expected.ravel(), atol=1e-12)

    def test_output_shape(self):
        """camera_center returns something broadcastable to (3,) or (3,1)."""
        C = camera_center(R_id, t_zero)
        assert np.asarray(C).size == 3
