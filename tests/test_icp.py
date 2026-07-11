"""
Tests for spatialwm.geometry.icp — raises NotImplementedError until implemented.

Contracts defended:
1. icp_point2point recovers known SE(3) to < 0.5° rotation and < 1% relative translation.
2. Returned error list is monotonically non-increasing.
"""

from __future__ import annotations

import math

import numpy as np

from spatialwm.geometry.icp import icp_point2point

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIcpPointToPoint:
    def test_recovers_rotation_within_half_degree(self, perturbed_bunny):
        """ICP rotation error < 0.5° for a small-rotation perturbation."""
        d = perturbed_bunny
        T_est, _ = icp_point2point(d["src"], d["dst"])

        R_est = T_est[:3, :3]
        R_gt = d["R"]

        R_rel = R_est @ R_gt.T
        cos_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_val))

        assert angle_deg < 0.5, f"Rotation error {angle_deg:.4f}° >= 0.5°"

    def test_recovers_translation_within_1_percent(self, perturbed_bunny):
        """ICP translation error < 1% of the translation magnitude."""
        d = perturbed_bunny
        T_est, _ = icp_point2point(d["src"], d["dst"])

        t_est = T_est[:3, 3]
        t_gt = d["t"]

        t_norm = np.linalg.norm(t_gt)
        err = np.linalg.norm(t_est - t_gt)

        if t_norm > 1e-8:
            rel_err = err / t_norm
            assert rel_err < 0.01, f"Translation relative error {rel_err:.4f} >= 0.01"
        else:
            assert err < 1e-6

    def test_error_curve_monotonically_nonincreasing(self, perturbed_bunny):
        """
        Each iteration's error is <= the previous iteration's error
        (ICP must converge, not diverge).
        """
        d = perturbed_bunny
        _, errors = icp_point2point(d["src"], d["dst"])

        assert len(errors) >= 1, "errors list must be non-empty"
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] + 1e-9, (
                f"Error increased at iter {i}: {errors[i - 1]:.6f} -> {errors[i]:.6f}"
            )

    def test_output_transformation_is_4x4(self, perturbed_bunny):
        """icp_point2point returns a 4×4 SE(3) matrix."""
        d = perturbed_bunny
        T_est, errors = icp_point2point(d["src"], d["dst"])

        assert T_est.shape == (4, 4), f"Expected (4,4) T, got {T_est.shape}"
        assert isinstance(errors, list), "errors should be a list"

    def test_rotation_part_is_valid(self, perturbed_bunny):
        """R block of returned T must be orthonormal with det +1."""
        d = perturbed_bunny
        T_est, _ = icp_point2point(d["src"], d["dst"])

        R = T_est[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_identical_clouds_gives_identity(self):
        """ICP of a cloud against itself returns identity transform and ~0 error."""
        rng = np.random.default_rng(55)
        pts = rng.standard_normal((100, 3))
        T_est, errors = icp_point2point(pts, pts)

        np.testing.assert_allclose(T_est[:3, :3], np.eye(3), atol=1e-4)
        np.testing.assert_allclose(T_est[:3, 3], np.zeros(3), atol=1e-4)
        assert errors[-1] < 1e-4
