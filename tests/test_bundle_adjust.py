"""
Tests for spatialwm.geometry.bundle_adjust — raises NotImplementedError until implemented.

Contract defended:
    Mean reprojection error after bundle_adjust drops > 5× compared to
    the initial noisy estimate on the noisy_ba_problem fixture.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from spatialwm.geometry.bundle_adjust import bundle_adjust

# ---------------------------------------------------------------------------
# Helper: compute mean reprojection error given poses, points, K, obs
# ---------------------------------------------------------------------------

def _reprojection_error(poses: np.ndarray, X: np.ndarray, K: np.ndarray, obs: np.ndarray) -> float:
    """
    Compute mean pixel reprojection error.

    poses: (M,6) rvec+t
    X:     (N,3) world points
    K:     (3,3) intrinsics
    obs:   (P,4) [cam_idx, pt_idx, u, v]
    """
    total_err = 0.0
    for row in obs:
        ci, pi = int(row[0]), int(row[1])
        u_obs, v_obs = row[2], row[3]

        rvec = poses[ci, :3]
        t_cam = poses[ci, 3:]
        R = Rotation.from_rotvec(rvec).as_matrix()
        Xc = R @ X[pi] + t_cam
        if Xc[2] <= 0:
            total_err += 1e3   # degenerate point behind camera
            continue
        xh = K @ Xc
        u_proj = xh[0] / xh[2]
        v_proj = xh[1] / xh[2]
        total_err += np.sqrt((u_proj - u_obs) ** 2 + (v_proj - v_obs) ** 2)

    return total_err / len(obs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBundleAdjust:
    def test_reprojection_error_drops_5x(self, noisy_ba_problem):
        """
        bundle_adjust must reduce mean reprojection error by at least 5× over
        the noisy initial estimate.
        """
        d = noisy_ba_problem
        K = d["K"]
        obs = d["obs"]

        error_before = _reprojection_error(d["poses0"], d["X0"], K, obs)

        poses_opt, X_opt = bundle_adjust(d["poses0"], d["X0"], K, obs)

        error_after = _reprojection_error(poses_opt, X_opt, K, obs)

        ratio = error_before / (error_after + 1e-12)
        assert ratio > 5.0, (
            f"Reprojection error did not drop 5×: before={error_before:.3f}, "
            f"after={error_after:.3f}, ratio={ratio:.2f}"
        )

    def test_output_shapes_preserved(self, noisy_ba_problem):
        """bundle_adjust returns poses and points with same shape as input."""
        d = noisy_ba_problem
        poses_opt, X_opt = bundle_adjust(d["poses0"], d["X0"], d["K"], d["obs"])

        assert poses_opt.shape == d["poses0"].shape, (
            f"poses shape changed: {d['poses0'].shape} -> {poses_opt.shape}"
        )
        assert X_opt.shape == d["X0"].shape, (
            f"X shape changed: {d['X0'].shape} -> {X_opt.shape}"
        )

    def test_error_after_lower_than_before(self, noisy_ba_problem):
        """
        At minimum, the optimized error must be no worse than initial
        (even if 5× criterion is tight for some random seeds).
        """
        d = noisy_ba_problem
        error_before = _reprojection_error(d["poses0"], d["X0"], d["K"], d["obs"])
        poses_opt, X_opt = bundle_adjust(d["poses0"], d["X0"], d["K"], d["obs"])
        error_after = _reprojection_error(poses_opt, X_opt, d["K"], d["obs"])

        assert error_after <= error_before, (
            f"bundle_adjust made reprojection WORSE: {error_before:.3f} -> {error_after:.3f}"
        )
