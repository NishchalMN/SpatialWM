"""Feature lifting: unproject depth + features to 3D point cloud."""

import numpy as np


def lift_features(
    depth: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    feat_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth to 3D, attach per-pixel features, multi-view aggregation by voxel-average.

    Args:
        depth: (H, W) depth map
        pose: (4, 4) camera pose matrix
        K: (3, 3) intrinsic matrix
        feat_map: (H, W, C) feature map

    Returns:
        points_xyz: (N, 3) 3D points
        feats: (N, C) features per point
    """
    raise NotImplementedError


def _demo():
    """Demo: lift synthetic depth/pose/K/feat_map."""
    H, W = 480, 640
    C = 64

    depth = np.random.rand(H, W).astype(np.float32) * 5.0
    pose = np.eye(4, dtype=np.float32)
    K = np.array(
        [[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    feat_map = np.random.randn(H, W, C).astype(np.float32)

    try:
        points_xyz, feats = lift_features(depth, pose, K, feat_map)
        print(f"Lifted points: {points_xyz.shape}, features: {feats.shape}")
    except NotImplementedError:
        print("lift_features not implemented yet")


if __name__ == "__main__":
    _demo()
