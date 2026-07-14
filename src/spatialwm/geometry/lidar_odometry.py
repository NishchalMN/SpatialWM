"""LiDAR scan-to-scan odometry.

Reuses `icp_point2point` from `icp.py`; the only new logic is voxel
downsampling of each scan and accumulation of relative poses into a global
trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarRegistrationDiagnostic:
    """Per-frame registration evidence for drift and confidence analysis."""

    source_index: int
    target_index: int
    method: str
    fitness: float
    inlier_rmse_m: float
    source_points: int
    target_points: int
    correction_translation_m: float
    correction_rotation_deg: float


@dataclass(frozen=True)
class LidarOdometryResult:
    """Metric LiDAR trajectory and registration diagnostics."""

    poses_scan_to_world: np.ndarray
    diagnostics: tuple[LidarRegistrationDiagnostic, ...]
    method: str
    voxel_m: float
    submap_size: int


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
    return lidar_odometry_detailed(
        scans,
        method="scan_to_scan",
        voxel=voxel,
        max_iters=max_iters,
        max_correspondence_distance=max_correspondence_distance,
        min_fitness=min_fitness,
    ).poses_scan_to_world


def _validate_odometry_inputs(
    scans: list[np.ndarray],
    voxel: float,
    max_iters: int,
    max_correspondence_distance: float,
    min_fitness: float,
) -> None:

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


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _rotation_angle_deg(transform: np.ndarray) -> float:
    cosine = np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def lidar_odometry_detailed(
    scans: list[np.ndarray],
    *,
    method: str = "scan_to_scan",
    voxel: float = 0.2,
    max_iters: int = 50,
    max_correspondence_distance: float = 1.0,
    min_fitness: float = 0.1,
    submap_size: int = 5,
    submap_voxel: float = 0.3,
    max_submap_points: int = 120_000,
) -> LidarOdometryResult:
    """Estimate scan-to-world poses using pairwise or local-submap ICP.

    ``scan_to_scan`` registers each scan against its predecessor.  ``scan_to_submap``
    predicts the next global pose with a constant-velocity model, transforms the
    current scan into the world frame, then estimates a bounded correction against
    the last ``submap_size`` accepted scans.
    """
    from spatialwm.geometry.icp import register_point_clouds

    _validate_odometry_inputs(
        scans, voxel, max_iters, max_correspondence_distance, min_fitness
    )
    if method not in {"scan_to_scan", "scan_to_submap"}:
        raise ValueError("method must be scan_to_scan or scan_to_submap")
    if not isinstance(submap_size, (int, np.integer)) or submap_size < 1:
        raise ValueError("submap_size must be a positive integer")
    if submap_voxel <= 0.0:
        raise ValueError("submap_voxel must be positive")
    if not isinstance(max_submap_points, (int, np.integer)) or max_submap_points < 3:
        raise ValueError("max_submap_points must be an integer >= 3")

    downsampled_scans = [voxel_downsample(scan, voxel) for scan in scans]
    poses = np.zeros((len(scans), 4, 4), dtype=np.float64)
    poses[0] = np.eye(4)
    diagnostics: list[LidarRegistrationDiagnostic] = []

    for k in range(len(scans) - 1):
        if k == 0:
            relative_prediction = np.eye(4)
        else:
            relative_prediction = np.linalg.inv(poses[k - 1]) @ poses[k]

        if method == "scan_to_scan":
            source = downsampled_scans[k + 1]
            target = downsampled_scans[k]
            reg_result = register_point_clouds(
                src=source,
                dst=target,
                max_correspondence_distance=max_correspondence_distance,
                max_iters=max_iters,
                init=relative_prediction,
            )
            step_transform = reg_result.transformation
            next_pose = poses[k] @ step_transform
            correction = step_transform @ np.linalg.inv(relative_prediction)
            target_index = k
        else:
            predicted_pose = poses[k] @ relative_prediction
            source = _transform_points(downsampled_scans[k + 1], predicted_pose)
            first_submap_scan = max(0, k - submap_size + 1)
            target = np.vstack(
                [
                    _transform_points(downsampled_scans[index], poses[index])
                    for index in range(first_submap_scan, k + 1)
                ]
            )
            target = voxel_downsample(target, submap_voxel)
            if len(target) > max_submap_points:
                sample_ids = np.linspace(
                    0, len(target) - 1, max_submap_points, dtype=np.int64
                )
                target = target[sample_ids]
            reg_result = register_point_clouds(
                src=source,
                dst=target,
                max_correspondence_distance=max_correspondence_distance,
                max_iters=max_iters,
                init=np.eye(4),
            )
            correction = reg_result.transformation
            next_pose = correction @ predicted_pose
            target_index = first_submap_scan

        transform = reg_result.transformation
        fitness = reg_result.fitness
        rmse = reg_result.inlier_rmse
        if not (
            np.all(np.isfinite(transform))
            and np.all(np.isfinite(next_pose))
            and np.isfinite(fitness)
            and np.isfinite(rmse)
        ):
            raise RuntimeError(
                f"Registration failed for pair {k+1} -> {k} due to non-finite outputs:\n"
                f"fitness={fitness}, rmse={rmse}, transformation=\n{transform}"
            )
        if fitness < min_fitness:
            raise RuntimeError(
                f"Registration failed for pair {k+1} -> {k}: fitness {fitness} "
                f"is below min_fitness {min_fitness}.\n"
                f"rmse={rmse}, transformation=\n{transform}"
            )
        poses[k + 1] = next_pose
        diagnostics.append(
            LidarRegistrationDiagnostic(
                source_index=k + 1,
                target_index=target_index,
                method=method,
                fitness=float(fitness),
                inlier_rmse_m=float(rmse),
                source_points=len(source),
                target_points=len(target),
                correction_translation_m=float(np.linalg.norm(correction[:3, 3])),
                correction_rotation_deg=_rotation_angle_deg(correction),
            )
        )

    return LidarOdometryResult(
        poses_scan_to_world=poses,
        diagnostics=tuple(diagnostics),
        method=method,
        voxel_m=float(voxel),
        submap_size=submap_size if method == "scan_to_submap" else 1,
    )


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
