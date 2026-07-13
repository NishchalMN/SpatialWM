"""
Pytest fixtures for the SpatialWM test suite.

All geometry is computed with raw numpy — no spatialwm imports — so fixtures
are independent of the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

DEFERRED_TEST_MODULES = {
    "test_elevation.py": "Deferred by the 2026-07-13 geometry-first scope reset",
    "test_jepa_min.py": "Deferred until the Phase C world-model signal check",
    "test_lift.py": "Deferred by the 2026-07-13 geometry-first scope reset",
    "test_metrics.py": "Deferred until the probes/evaluation phase that consumes these metrics",
    "test_pointnet.py": "Deferred by the 2026-07-13 geometry-first scope reset",
}


def pytest_collection_modifyitems(items):
    """Keep deferred contracts visible without making the active suite falsely red."""
    for item in items:
        reason = DEFERRED_TEST_MODULES.get(item.path.name)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))

# ---------------------------------------------------------------------------
# Helpers (private to this module)
# ---------------------------------------------------------------------------

def _project_numpy(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Raw K[R|t]X projection — must NOT import spatialwm."""
    # t must be (3,1) or (3,)
    t = np.asarray(t).reshape(3, 1)
    # X: (N,3) -> (3,N)
    Xc = R @ X.T + t          # (3, N)
    xh = K @ Xc               # (3, N)
    uv = (xh[:2] / xh[2]).T  # (N, 2)
    return uv


def _homogeneous(pts: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,4) by appending ones."""
    return np.hstack([pts, np.ones((len(pts), 1))])


# ---------------------------------------------------------------------------
# synthetic_two_view
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_two_view():
    """
    Known-geometry two-view setup.

    Camera 1: R1 = I, t1 = 0.
    Camera 2: small rotation + translation, chosen so all points are in front.

    Returns a dict with keys:
        K   — (3,3) intrinsic matrix
        R   — (3,3) rotation of camera 2
        t   — (3,1) translation of camera 2
        X   — (N,3) world-frame 3D points
        x1  — (N,2) projections in camera 1
        x2  — (N,2) projections in camera 2
    """
    rng = np.random.default_rng(42)

    K = np.array(
        [[800.0, 0.0, 320.0],
         [0.0, 800.0, 240.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Camera 2: 10° rotation around y-axis + translation
    R2 = Rotation.from_euler("y", 10, degrees=True).as_matrix()
    t2 = np.array([[0.5], [0.0], [0.0]], dtype=np.float64)

    # 30 random 3D points in front of BOTH cameras (z in [3, 7])
    N = 30
    X = rng.uniform(-1, 1, (N, 3))
    X[:, 2] = rng.uniform(3.0, 7.0, N)

    # Identity camera 1
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))

    x1 = _project_numpy(K, R1, t1, X)
    x2 = _project_numpy(K, R2, t2, X)

    return dict(K=K, R=R2, t=t2, X=X, x1=x1, x2=x2)


# ---------------------------------------------------------------------------
# perturbed_bunny
# ---------------------------------------------------------------------------

@pytest.fixture
def perturbed_bunny():
    """
    ~200 random 3D points with a known SE(3) applied.

    Returns a dict with keys:
        src  — (N,3) source cloud
        dst  — (N,3) dst = R @ src.T + t (no noise, exact correspondence)
        R    — (3,3) ground-truth rotation
        t    — (3,) ground-truth translation
        T    — (4,4) ground-truth SE(3) matrix
    """
    rng = np.random.default_rng(7)

    N = 200
    src = rng.standard_normal((N, 3)).astype(np.float64)

    # Small rotation (~5°) so ICP converges easily
    R = Rotation.from_euler("xyz", [3.0, 4.0, 2.0], degrees=True).as_matrix()
    t = np.array([0.05, -0.03, 0.02], dtype=np.float64)

    dst = (R @ src.T).T + t

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return dict(src=src, dst=dst, R=R, t=t, T=T)


# ---------------------------------------------------------------------------
# noisy_ba_problem
# ---------------------------------------------------------------------------

@pytest.fixture
def noisy_ba_problem():
    """
    5-camera / 100-point bundle-adjustment problem.

    Ground-truth setup → project → add pixel noise → add pose/point noise
    for the initial estimate passed to bundle_adjust.

    Returns a dict with keys:
        K        — (3,3) shared intrinsic matrix
        poses_gt — (5,6) ground-truth poses (rvec + t, each row)
        X_gt     — (100,3) ground-truth 3D points
        poses0   — (5,6) noisy initial poses
        X0       — (100,3) noisy initial 3D points
        obs      — (500,4) observation array: [cam_idx, pt_idx, u, v]
    """
    rng = np.random.default_rng(99)

    n_cams = 5
    n_pts = 100

    K = np.array(
        [[800.0, 0.0, 320.0],
         [0.0, 800.0, 240.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Ground-truth 3D points — in front of all cameras
    X_gt = rng.uniform(-1.5, 1.5, (n_pts, 3))
    X_gt[:, 2] += 5.0  # push in front

    # Ground-truth camera poses: small rotations + translations
    poses_gt = np.zeros((n_cams, 6))
    for i in range(n_cams):
        angle_deg = i * 8.0  # spread cameras
        R_i = Rotation.from_euler("y", angle_deg, degrees=True).as_matrix()
        t_i = np.array([i * 0.3, 0.0, 0.0])
        rvec = Rotation.from_matrix(R_i).as_rotvec()
        poses_gt[i, :3] = rvec
        poses_gt[i, 3:] = t_i

    # Build observations (every point seen by every camera)
    obs_rows = []
    for ci in range(n_cams):
        rvec = poses_gt[ci, :3]
        t_cam = poses_gt[ci, 3:]
        R_cam = Rotation.from_rotvec(rvec).as_matrix()
        uv = _project_numpy(K, R_cam, t_cam.reshape(3, 1), X_gt)
        # add 0.5-pixel pixel noise
        uv += rng.normal(0, 0.5, uv.shape)
        for pi in range(n_pts):
            obs_rows.append([ci, pi, uv[pi, 0], uv[pi, 1]])

    obs = np.array(obs_rows, dtype=np.float64)

    # Noisy initials: poses ± 0.05 rad/m, points ± 0.2 m
    poses0 = poses_gt + rng.normal(0, 0.05, poses_gt.shape)
    X0 = X_gt + rng.normal(0, 0.2, X_gt.shape)

    return dict(
        K=K,
        poses_gt=poses_gt,
        X_gt=X_gt,
        poses0=poses0,
        X0=X0,
        obs=obs,
    )
