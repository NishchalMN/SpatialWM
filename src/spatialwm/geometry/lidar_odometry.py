"""LiDAR scan-to-scan odometry.

Reuses `icp_point2point` from `icp.py`; the only new logic is voxel
downsampling of each scan and accumulation of relative poses into a global
trajectory.
"""

from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel: float = 0.2) -> np.ndarray:
    """Downsample a point cloud to one point per occupied voxel.

    Bin points into a regular grid of side `voxel`, keep one representative
    (e.g. the centroid) per non-empty cell. Reduces cost and improves ICP
    conditioning by equalizing point density before registration.

    Args:
        points: (N,3) LiDAR points.
        voxel: cell size in meters.

    Returns:
        (M,3) downsampled points, M <= N.
    """
    raise NotImplementedError


def lidar_odometry(
    scans: list[np.ndarray],
    voxel: float = 0.2,
    max_iters: int = 50,
) -> np.ndarray:
    """Estimate a trajectory from a sequence of LiDAR scans.

    For each consecutive pair, voxel-downsample both scans, register scan k+1
    to scan k with `icp_point2point` to get the relative transform T_k, then
    accumulate global poses:  P_0 = I,  P_{k+1} = P_k @ T_k.

    Drift accumulates because each T_k carries registration error that
    compounds multiplicatively along the chain (no loop closure / global map).

    Args:
        scans: list of (N_i,3) point clouds, temporally ordered.
        voxel: downsample cell size passed to `voxel_downsample`.
        max_iters: ICP iteration cap per pair.

    Returns:
        (K,4,4) array of global SE(3) poses, one per scan (P_0 = identity).
    """
    raise NotImplementedError


def _demo() -> None:
    """Demo: build two synthetic scans and run one odometry step (raises)."""
    rng = np.random.default_rng(0)
    scan0 = rng.standard_normal((500, 3))
    scan1 = scan0 + np.array([0.1, 0.0, 0.0])  # small forward motion
    lidar_odometry([scan0, scan1])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    if ap.parse_args().demo:
        _demo()
