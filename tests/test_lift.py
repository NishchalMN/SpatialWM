"""
Tests for spatialwm.perception.lift — RED now, green on impl.

Contract defended:
    lift_features output shapes: points_xyz (M, 3), feats (M, C)
    consistent with input depth/feat_map dims.
"""

from __future__ import annotations

import numpy as np

from spatialwm.perception.lift import lift_features

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLiftFeatures:
    """Shape contracts for lift_features on synthetic inputs."""

    def _make_inputs(self, H: int = 32, W: int = 48, C: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        depth = rng.uniform(0.5, 5.0, (H, W)).astype(np.float32)
        pose = np.eye(4, dtype=np.float32)
        K = np.array(
            [[200.0, 0.0, 24.0],
             [0.0, 200.0, 16.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        feat_map = rng.standard_normal((H, W, C)).astype(np.float32)
        return depth, pose, K, feat_map

    def test_output_ndim(self):
        """points_xyz and feats are 2-D arrays."""
        depth, pose, K, feat_map = self._make_inputs()
        pts, feats = lift_features(depth, pose, K, feat_map)
        assert pts.ndim == 2, f"points_xyz must be 2-D, got {pts.ndim}-D"
        assert feats.ndim == 2, f"feats must be 2-D, got {feats.ndim}-D"

    def test_output_channel_dim(self):
        """feats second dimension equals C (feature channels)."""
        H, W, C = 16, 24, 32
        depth, pose, K, feat_map = self._make_inputs(H, W, C)
        pts, feats = lift_features(depth, pose, K, feat_map)
        assert feats.shape[1] == C, (
            f"Expected feats.shape[1]=={C}, got {feats.shape[1]}"
        )

    def test_xyz_dim(self):
        """points_xyz second dimension is 3 (x, y, z)."""
        depth, pose, K, feat_map = self._make_inputs()
        pts, feats = lift_features(depth, pose, K, feat_map)
        assert pts.shape[1] == 3, (
            f"Expected points_xyz.shape[1]==3, got {pts.shape[1]}"
        )

    def test_consistent_n_points(self):
        """Number of lifted points equals number of features (M matches)."""
        depth, pose, K, feat_map = self._make_inputs(H=16, W=24, C=8)
        pts, feats = lift_features(depth, pose, K, feat_map)
        assert pts.shape[0] == feats.shape[0], (
            f"points_xyz has {pts.shape[0]} points but feats has {feats.shape[0]}"
        )

    def test_at_most_hwc_points(self):
        """Lifted points cannot exceed H*W (one per pixel max)."""
        H, W, C = 8, 12, 4
        depth, pose, K, feat_map = self._make_inputs(H, W, C)
        pts, feats = lift_features(depth, pose, K, feat_map)
        assert pts.shape[0] <= H * W, (
            f"More points than pixels: {pts.shape[0]} > {H * W}"
        )

    def test_non_identity_pose_changes_xyz(self):
        """
        Applying a non-identity pose must change the world-frame point coordinates
        relative to using an identity pose.
        """
        import numpy as np
        from scipy.spatial.transform import Rotation

        H, W, C = 8, 12, 4
        rng = np.random.default_rng(77)
        depth = rng.uniform(1.0, 3.0, (H, W)).astype(np.float32)
        K = np.array([[100.0, 0, 6.0], [0, 100.0, 4.0], [0, 0, 1.0]], np.float32)
        feat_map = rng.standard_normal((H, W, C)).astype(np.float32)

        pose_id = np.eye(4, dtype=np.float32)

        R = Rotation.from_euler("y", 30.0, degrees=True).as_matrix().astype(np.float32)
        pose_rot = np.eye(4, dtype=np.float32)
        pose_rot[:3, :3] = R
        pose_rot[:3, 3] = [1.0, 0.0, 0.0]

        pts_id, _ = lift_features(depth, pose_id, K, feat_map)
        pts_rot, _ = lift_features(depth, pose_rot, K, feat_map)

        # Not identical (different poses mean different world coords)
        assert not np.allclose(pts_id, pts_rot, atol=1e-3), (
            "pose=identity and pose=rotation gave identical xyz — pose is not being applied"
        )
