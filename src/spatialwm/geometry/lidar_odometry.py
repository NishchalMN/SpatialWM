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
    import open3d as o3d

    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N,3) array")
    if not np.issubdtype(points.dtype, np.floating):
        raise TypeError("points must be of floating type")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")

    if not isinstance(voxel, (int, float, np.floating, np.integer)) or voxel <= 0:
        raise ValueError("voxel must be positive")

    # Conversion to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=float(voxel))

    return np.asarray(pcd_down.points)


def lidar_odometry(
    scans: list[np.ndarray],
    voxel: float = 0.2,
    max_iters: int = 50,
    max_correspondence_distance: float = 1.0,
    min_fitness: float = 0.1,
) -> np.ndarray:
    """Estimate a trajectory from a sequence of LiDAR scans.

    For each consecutive pair, voxel-downsample both scans, register scan k+1
    to scan k with `register_point_clouds` to get the relative transform T_k, then
    accumulate global poses:  P_0 = I,  P_{k+1} = P_k @ T_k.

    Drift accumulates because each T_k carries registration error that
    compounds multiplicatively along the chain (no loop closure / global map).

    Args:
        scans: list of (N_i,3) point clouds, temporally ordered.
        voxel: downsample cell size passed to `voxel_downsample`.
        max_iters: ICP iteration cap per pair.
        max_correspondence_distance: Max correspondence distance for ICP registration.
        min_fitness: Minimum acceptable registration fitness (overlap ratio), default is 0.1.

    Returns:
        (K,4,4) array of global SE(3) poses, one per scan (P_0 = identity).
    """
    from spatialwm.geometry.icp import register_point_clouds

    if not isinstance(scans, (list, tuple)):
        raise TypeError("scans must be a list or tuple of numpy arrays")
    if len(scans) == 0:
        raise ValueError("scans must not be empty")

    for i, scan in enumerate(scans):
        if not isinstance(scan, np.ndarray):
            raise TypeError(f"scan at index {i} must be a numpy array")

    if not isinstance(voxel, (int, float, np.floating, np.integer)) or voxel <= 0:
        raise ValueError("voxel must be positive")
    if not isinstance(max_iters, (int, np.integer)) or max_iters <= 0:
        raise ValueError("max_iters must be a positive integer")
    if (
        not isinstance(max_correspondence_distance, (int, float, np.floating, np.integer))
        or max_correspondence_distance <= 0
    ):
        raise ValueError("max_correspondence_distance must be positive")
    if not isinstance(min_fitness, (int, float, np.floating, np.integer)):
        raise TypeError("min_fitness must be a float or integer")
    if not (0.0 <= min_fitness <= 1.0):
        raise ValueError("min_fitness must be between 0.0 and 1.0 inclusive")

    # Downsample each scan
    downsampled_scans = [voxel_downsample(scan, voxel) for scan in scans]

    K = len(scans)
    poses = np.zeros((K, 4, 4))
    poses[0] = np.eye(4)

    current_pose = np.eye(4)
    for k in range(K - 1):
        reg_result = register_point_clouds(
            src=downsampled_scans[k + 1],
            dst=downsampled_scans[k],
            max_correspondence_distance=max_correspondence_distance,
            max_iters=max_iters,
        )
        T_k = reg_result.transformation
        fitness = reg_result.fitness
        rmse = reg_result.inlier_rmse

        if not (np.all(np.isfinite(T_k)) and np.isfinite(fitness) and np.isfinite(rmse)):
            raise RuntimeError(
                f"Registration failed for pair {k+1} -> {k} due to non-finite outputs:\n"
                f"fitness={fitness}, rmse={rmse}, transformation=\n{T_k}"
            )
        if fitness < min_fitness:
            raise RuntimeError(
                f"Registration failed for pair {k+1} -> {k}: fitness {fitness} "
                f"is below min_fitness {min_fitness}.\n"
                f"rmse={rmse}, transformation=\n{T_k}"
            )

        current_pose = current_pose @ T_k
        poses[k + 1] = current_pose

    return poses


def _demo() -> None:
    """Demo: build two synthetic scans and run one odometry step."""
    n_side = 20
    lin = np.linspace(-1.0, 1.0, n_side)
    xx, yy = np.meshgrid(lin, lin)
    zz = 0.1 * np.sin(3 * xx) * np.cos(3 * yy)
    scan0 = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    scan1 = scan0 + np.array([0.05, 0.0, 0.0])  # small forward motion

    poses = lidar_odometry([scan0, scan1], voxel=0.1)
    print("Estimated poses:")
    for i, P in enumerate(poses):
        print(f"Pose {i}:\n{P}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    if ap.parse_args().demo:
        _demo()
