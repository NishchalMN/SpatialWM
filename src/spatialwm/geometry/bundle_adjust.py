"""Sparse bundle adjustment for calibrated pinhole cameras.

Camera poses use a six-vector [rotation_vector, translation] and map world
points into a camera frame. Observations are rows of
[camera_index, point_index, u, v].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def _validate_problem(
    poses: np.ndarray,
    points: np.ndarray,
    K: np.ndarray,
    obs: np.ndarray,
) -> None:
    """Validate the public bundle-adjustment array contract."""
    arrays = {
        "poses0": poses,
        "X0": points,
        "K": K,
        "obs": obs,
    }
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")

    if poses.ndim != 2 or poses.shape[1] != 6 or len(poses) == 0:
        raise ValueError("poses0 must have shape (M, 6) with M >= 1")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("X0 must have shape (N, 3) with N >= 1")
    if K.shape != (3, 3):
        raise ValueError("K must have shape (3, 3)")
    if obs.ndim != 2 or obs.shape[1] != 4 or len(obs) == 0:
        raise ValueError("obs must have shape (P, 4) with P >= 1")

    cam_ids = obs[:, 0]
    point_ids = obs[:, 1]
    if np.any(cam_ids != np.floor(cam_ids)) or np.any(point_ids != np.floor(point_ids)):
        raise ValueError("camera and point indices in obs must be integers")
    if np.any(cam_ids < 0) or np.any(cam_ids >= len(poses)):
        raise ValueError("obs contains a camera index outside poses0")
    if np.any(point_ids < 0) or np.any(point_ids >= len(points)):
        raise ValueError("obs contains a point index outside X0")


def _project_observations(
    poses: np.ndarray,
    points: np.ndarray,
    K: np.ndarray,
    obs: np.ndarray,
) -> np.ndarray:
    """Project the point referenced by every observation through its camera."""
    camera_ids = obs[:, 0].astype(np.int64)
    point_ids = obs[:, 1].astype(np.int64)

    rotations = Rotation.from_rotvec(poses[camera_ids, :3])
    points_camera = rotations.apply(points[point_ids]) + poses[camera_ids, 3:]

    homogeneous = points_camera @ K.T
    depth = homogeneous[:, 2]
    safe_depth = np.where(
        np.abs(depth) < 1e-8,
        np.where(depth >= 0.0, 1e-8, -1e-8),
        depth,
    )
    return homogeneous[:, :2] / safe_depth[:, None]


def reprojection_residuals(
    params: np.ndarray,
    n_cams: int,
    n_pts: int,
    K: np.ndarray,
    obs: np.ndarray,
) -> np.ndarray:
    """Return flattened predicted-minus-observed pixel residuals.

    Args:
        params: Full parameter vector: 6 * n_cams pose values followed by
            3 * n_pts world-point values.
        n_cams: Number of camera poses.
        n_pts: Number of 3D points.
        K: Shared calibrated pinhole intrinsics, shape (3, 3).
        obs: Observation rows [camera_index, point_index, u, v], shape (P, 4).

    Returns:
        A vector of shape (2 * P,) ordered as
        [du_0, dv_0, du_1, dv_1, ...].
    """
    if not isinstance(params, np.ndarray):
        raise TypeError("params must be a numpy array")
    if not isinstance(n_cams, (int, np.integer)) or n_cams < 1:
        raise ValueError("n_cams must be a positive integer")
    if not isinstance(n_pts, (int, np.integer)) or n_pts < 1:
        raise ValueError("n_pts must be a positive integer")

    expected = 6 * int(n_cams) + 3 * int(n_pts)
    if params.ndim != 1 or len(params) != expected:
        raise ValueError(f"params must have shape ({expected},)")

    poses = params[: 6 * n_cams].reshape(n_cams, 6)
    points = params[6 * n_cams :].reshape(n_pts, 3)
    _validate_problem(poses, points, K, obs)

    projected = _project_observations(poses, points, K, obs)
    return (projected - obs[:, 2:4]).ravel()


def _pack_free_parameters(poses: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Pack parameters after removing the seven fixed similarity-gauge values."""
    return np.concatenate(
        [
            poses[1:].ravel(),
            points[0, :2],
            points[1:].ravel(),
        ]
    )


def _unpack_free_parameters(
    params: np.ndarray,
    fixed_first_pose: np.ndarray,
    fixed_first_point_z: float,
    n_cams: int,
    n_pts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore full pose/point arrays from the gauge-fixed parameterization."""
    pose_values = 6 * (n_cams - 1)

    poses = np.empty((n_cams, 6), dtype=np.float64)
    poses[0] = fixed_first_pose
    if n_cams > 1:
        poses[1:] = params[:pose_values].reshape(n_cams - 1, 6)

    points = np.empty((n_pts, 3), dtype=np.float64)
    points[0, :2] = params[pose_values : pose_values + 2]
    points[0, 2] = fixed_first_point_z
    if n_pts > 1:
        points[1:] = params[pose_values + 2 :].reshape(n_pts - 1, 3)

    return poses, points


def _free_jacobian_sparsity(
    obs: np.ndarray,
    n_cams: int,
    n_pts: int,
) -> lil_matrix:
    """Build the observation-to-variable dependency pattern."""
    n_residuals = 2 * len(obs)
    n_variables = 6 * (n_cams - 1) + 2 + 3 * (n_pts - 1)
    sparsity = lil_matrix((n_residuals, n_variables), dtype=np.int8)
    point_offset = 6 * (n_cams - 1)

    for observation_index, row in enumerate(obs):
        camera_id = int(row[0])
        point_id = int(row[1])
        residual_rows = slice(2 * observation_index, 2 * observation_index + 2)

        if camera_id > 0:
            camera_start = 6 * (camera_id - 1)
            sparsity[residual_rows, camera_start : camera_start + 6] = 1

        if point_id == 0:
            sparsity[residual_rows, point_offset : point_offset + 2] = 1
        else:
            point_start = point_offset + 2 + 3 * (point_id - 1)
            sparsity[residual_rows, point_start : point_start + 3] = 1

    return sparsity


def bundle_adjust(
    poses0: np.ndarray,
    X0: np.ndarray,
    K: np.ndarray,
    obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly refine calibrated camera poses and 3D points.

    The first camera pose and the first point's world z coordinate are held
    fixed. Those seven constraints remove the global similarity gauge: six
    rigid-frame degrees of freedom plus monocular scale. This chooses a
    coordinate system; it does not add scene information.

    SciPy's trust-region reflective solver receives the exact block sparsity
    pattern: each 2D observation depends on one camera and one point. A
    soft-L1 loss reduces the influence of occasional bad observations.

    Args:
        poses0: Initial world-to-camera poses, shape (M, 6), represented as
            rotation vector followed by translation.
        X0: Initial world points, shape (N, 3).
        K: Shared camera intrinsics, shape (3, 3).
        obs: Observation rows [camera_index, point_index, u, v], shape (P, 4).

    Returns:
        (poses, points) with the same shapes as poses0 and X0.
    """
    _validate_problem(poses0, X0, K, obs)

    poses_initial = np.asarray(poses0, dtype=np.float64)
    points_initial = np.asarray(X0, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    observations = np.asarray(obs, dtype=np.float64)

    n_cams = len(poses_initial)
    n_pts = len(points_initial)
    fixed_first_pose = poses_initial[0].copy()
    fixed_first_point_z = float(points_initial[0, 2])
    params0 = _pack_free_parameters(poses_initial, points_initial)

    def residual_function(free_params: np.ndarray) -> np.ndarray:
        poses, points = _unpack_free_parameters(
            free_params,
            fixed_first_pose,
            fixed_first_point_z,
            n_cams,
            n_pts,
        )
        projected = _project_observations(poses, points, intrinsics, observations)
        return (projected - observations[:, 2:4]).ravel()

    result = least_squares(
        residual_function,
        params0,
        method="trf",
        jac_sparsity=_free_jacobian_sparsity(observations, n_cams, n_pts),
        tr_solver="lsmr",
        x_scale="jac",
        loss="soft_l1",
        f_scale=1.0,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=300,
    )

    poses, points = _unpack_free_parameters(
        result.x,
        fixed_first_pose,
        fixed_first_point_z,
        n_cams,
        n_pts,
    )
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(points)):
        raise RuntimeError("bundle adjustment produced non-finite parameters")

    return poses, points


def _demo() -> None:
    """Run a deterministic two-camera synthetic bundle-adjustment example."""
    rng = np.random.default_rng(7)
    K = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    poses_gt = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.08, 0.0, 0.4, 0.0, 0.0],
        ]
    )
    points_gt = rng.uniform(-1.0, 1.0, (30, 3))
    points_gt[:, 2] += 4.0

    rows = []
    for camera_id in range(len(poses_gt)):
        camera_rows = np.column_stack(
            [
                np.full(len(points_gt), camera_id),
                np.arange(len(points_gt)),
                np.zeros((len(points_gt), 2)),
            ]
        )
        projected = _project_observations(poses_gt, points_gt, K, camera_rows)
        camera_rows[:, 2:] = projected + rng.normal(0.0, 0.3, projected.shape)
        rows.append(camera_rows)
    observations = np.vstack(rows)

    poses0 = poses_gt + rng.normal(0.0, 0.03, poses_gt.shape)
    points0 = points_gt + rng.normal(0.0, 0.15, points_gt.shape)
    params0 = np.concatenate([poses0.ravel(), points0.ravel()])
    before = np.mean(
        np.linalg.norm(
            reprojection_residuals(params0, 2, len(points0), K, observations).reshape(-1, 2),
            axis=1,
        )
    )
    poses, points = bundle_adjust(poses0, points0, K, observations)
    params = np.concatenate([poses.ravel(), points.ravel()])
    after = np.mean(
        np.linalg.norm(
            reprojection_residuals(params, 2, len(points), K, observations).reshape(-1, 2),
            axis=1,
        )
    )
    print(f"Mean reprojection error: {before:.3f}px -> {after:.3f}px")


if __name__ == "__main__":
    _demo()
