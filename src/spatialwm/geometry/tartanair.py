"""TartanAir geometry coordinate transformations and pair selection helpers.

This module provides tools to parse TartanAir poses, compute relative transforms,
select frame pairs matching baseline criteria, and compute registration errors.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def parse_pose_to_transform(pose_row: np.ndarray) -> np.ndarray:
    """Parse a 7-value pose row (tx, ty, tz, qx, qy, qz, qw) into an explicit
    4x4 camera-to-world transform matrix in standard camera coordinates (RDF).

    TartanAir poses are natively defined in the NED (North-East-Down) frame.
    We convert the pose to standard camera coordinates (RDF: Right-Down-Forward)
    using the official TartanAir transformation convention.

    Source URL for coordinate conventions:
    https://github.com/castacks/tartanair_tools/blob/master/evaluation/trajectory_transform.py

    Args:
        pose_row: Array of shape (7,) containing [tx, ty, tz, qx, qy, qz, qw].

    Returns:
        T_cam_to_world: 4x4 camera-to-world transform matrix mapping camera-local
            OpenCV RDF coordinates to the world frame.
    """
    if not isinstance(pose_row, np.ndarray):
        raise TypeError("pose_row must be a numpy array")
    if pose_row.shape != (7,):
        raise ValueError(f"pose_row must have shape (7,), got {pose_row.shape}")

    tx, ty, tz, qx, qy, qz, qw = pose_row

    # 1. Construct the 4x4 transformation matrix in the NED frame.
    # The translation is in the NED frame, and the quaternion describes the
    # rotation of the camera's local NED frame relative to the world NED frame.
    t_ned = np.array([tx, ty, tz], dtype=float)

    # Normalize quaternion to avoid errors
    q = np.array([qx, qy, qz, qw], dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-8:
        q = q / q_norm
    else:
        q = np.array([0.0, 0.0, 0.0, 1.0])

    R_ned = Rotation.from_quat(q).as_matrix()

    T_ned = np.identity(4, dtype=float)
    T_ned[:3, :3] = R_ned
    T_ned[:3, 3] = t_ned

    # 2. Transform the camera's local frame from NED to standard OpenCV RDF frame.
    # T_ned2cam maps NED (X-Forward, Y-Right, Z-Down) to OpenCV (X-Right, Y-Down, Z-Forward).
    # Rotation part R_ned2cam maps:
    #   X_cam = Y_ned (East -> Right)
    #   Y_cam = Z_ned (Down -> Down)
    #   Z_cam = X_ned (North -> Forward)
    T_ned2cam = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

    T_cam2ned = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

    T_cam = T_ned2cam @ T_ned @ T_cam2ned
    return T_cam


def derive_relative_transform(T_source: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """Derive the relative transform source_camera -> target_camera.

    Args:
        T_source: 4x4 source camera-to-world transform.
        T_target: 4x4 target camera-to-world transform.

    Returns:
        T_source_to_target: 4x4 relative transform matrix.
    """
    if not isinstance(T_source, np.ndarray) or T_source.shape != (4, 4):
        raise ValueError("T_source must be a 4x4 numpy array")
    if not isinstance(T_target, np.ndarray) or T_target.shape != (4, 4):
        raise ValueError("T_target must be a 4x4 numpy array")

    # P_world = T_source @ P_source
    # P_world = T_target @ P_target
    # -> P_target = T_target^-1 @ T_source @ P_source
    return np.linalg.inv(T_target) @ T_source


def compute_se3_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Compute SE(3) registration error between estimated and ground-truth transforms.

    Args:
        T_est: 4x4 estimated transform matrix.
        T_gt: 4x4 ground truth transform matrix.

    Returns:
        translation_error: Translation norm error in meters.
        rotation_error: Rotation angle error in degrees.
    """
    if not isinstance(T_est, np.ndarray) or T_est.shape != (4, 4):
        raise ValueError("T_est must be a 4x4 numpy array")
    if not isinstance(T_gt, np.ndarray) or T_gt.shape != (4, 4):
        raise ValueError("T_gt must be a 4x4 numpy array")

    T_diff = np.linalg.inv(T_gt) @ T_est
    t_err = np.linalg.norm(T_diff[:3, 3])

    R_diff = T_diff[:3, :3]
    cos_theta = (np.trace(R_diff) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    r_err = np.degrees(np.arccos(cos_theta))

    return float(t_err), float(r_err)


def select_target_frames(
    poses: np.ndarray,
    source_idx: int,
    target_baselines: list[float],
    max_search_frames: int | None = None,
) -> list[dict]:
    """Select forward frame indices nearest requested translation targets.

    Args:
        poses: Array of shape (N, 7) containing poses [tx, ty, tz, qx, qy, qz, qw].
        source_idx: Start frame index.
        target_baselines: List of requested baseline targets in meters.
        max_search_frames: Bounded search limit. If None, search to the end of poses.

    Returns:
        A list of dictionaries describing selected pairs, with keys:
            - 'target_idx': The selected target frame index.
            - 'requested_baseline': The requested baseline target.
            - 'actual_baseline': The actual baseline translation distance.
            - 'rotation_deg': The actual relative rotation angle in degrees.
    """
    if not isinstance(poses, np.ndarray) or poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("poses must be a numpy array of shape (N, 7)")
    if source_idx < 0 or source_idx >= len(poses):
        raise ValueError(
            f"source_idx {source_idx} out of bounds for trajectory of length {len(poses)}"
        )
    if not isinstance(target_baselines, list) or len(target_baselines) == 0:
        raise ValueError("target_baselines must be a non-empty list of floats")
    for b in target_baselines:
        if b <= 0.0:
            raise ValueError(f"target baseline must be positive, got {b}")

    T_source = parse_pose_to_transform(poses[source_idx])
    t_source = T_source[:3, 3]

    max_idx = len(poses)
    if max_search_frames is not None:
        if max_search_frames <= 0:
            raise ValueError("max_search_frames must be positive")
        max_idx = min(max_idx, source_idx + max_search_frames + 1)

    results = []
    for target_b in target_baselines:
        best_idx = -1
        best_diff = float("inf")
        best_dist = 0.0
        best_rot = 0.0

        # Search forward: target_idx > source_idx
        for idx in range(source_idx + 1, max_idx):
            T_target = parse_pose_to_transform(poses[idx])
            t_target = T_target[:3, 3]

            dist = float(np.linalg.norm(t_target - t_source))
            diff = abs(dist - target_b)

            # Deterministic tie-breaker: strict inequality (<) keeps the smaller index
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
                best_dist = dist

                T_rel = derive_relative_transform(T_source, T_target)
                R_diff = T_rel[:3, :3]
                cos_theta = (np.trace(R_diff) - 1.0) / 2.0
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                best_rot = float(np.degrees(np.arccos(cos_theta)))

        if best_idx == -1:
            raise ValueError(f"No valid forward frame found for baseline target {target_b}")

        results.append({
            "target_idx": best_idx,
            "requested_baseline": target_b,
            "actual_baseline": best_dist,
            "rotation_deg": best_rot,
        })

    return results
