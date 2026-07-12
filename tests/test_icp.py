"""
Tests for spatialwm.geometry.icp using Open3D ICP point-to-point.

Contracts defended:
1. icp_point2point recovers known SE(3) to < 0.5° rotation and < 1% relative translation.
2. Returned error list contains a single final RMSE value.
"""

from __future__ import annotations

import math

import numpy as np
import open3d as o3d
import pytest

from spatialwm.geometry.icp import icp_point2point, register_point_clouds

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

    def test_errors_contains_final_rmse(self, perturbed_bunny):
        """
        The errors list must be a single-element list containing the final
        inlier RMSE value from the registration result.
        """
        d = perturbed_bunny
        _, errors = icp_point2point(d["src"], d["dst"])

        assert isinstance(errors, list), "errors should be a list"
        assert len(errors) == 1, f"Expected exactly 1 error entry, got {len(errors)}"
        assert isinstance(errors[0], float)
        assert errors[0] >= 0.0

    def test_validation_errors(self):
        """Test input validations in register_point_clouds."""
        src = np.ones((10, 3), dtype=np.float64)
        dst = np.ones((10, 3), dtype=np.float64)

        # Non-numpy arrays
        with pytest.raises(TypeError):
            register_point_clouds([[1.0, 2.0, 3.0]], dst)
        with pytest.raises(TypeError):
            register_point_clouds(src, [[1.0, 2.0, 3.0]])

        # Dimension checks
        with pytest.raises(ValueError):
            register_point_clouds(src.flatten(), dst)
        with pytest.raises(ValueError):
            register_point_clouds(src, dst[:, :2])

        # Point count check
        with pytest.raises(ValueError):
            register_point_clouds(src[:2], dst)
        with pytest.raises(ValueError):
            register_point_clouds(src, dst[:2])

        # Non-floating type
        with pytest.raises(TypeError):
            register_point_clouds(src.astype(np.int32), dst)

        # Non-finite values
        src_inf = src.copy()
        src_inf[0, 0] = np.inf
        with pytest.raises(ValueError):
            register_point_clouds(src_inf, dst)

        # Invalid max_correspondence_distance
        with pytest.raises(ValueError):
            register_point_clouds(src, dst, max_correspondence_distance=0)
        with pytest.raises(ValueError):
            register_point_clouds(src, dst, max_correspondence_distance=-1.0)

        # Invalid max_iters
        with pytest.raises(ValueError):
            register_point_clouds(src, dst, max_iters=0)
        with pytest.raises(ValueError):
            register_point_clouds(src, dst, max_iters=-5)

        # Invalid tol
        with pytest.raises(ValueError):
            register_point_clouds(src, dst, tol=-1e-6)

        # Invalid init
        with pytest.raises(TypeError):
            register_point_clouds(src, dst, init=[[1.0]*4]*4)
        with pytest.raises(ValueError):
            register_point_clouds(src, dst, init=np.eye(3))
        init_inf = np.eye(4)
        init_inf[0, 0] = np.nan
        with pytest.raises(ValueError):
            register_point_clouds(src, dst, init=init_inf)

    def test_register_point_clouds_rich_result(self, perturbed_bunny):
        """Test register_point_clouds directly and verify rich Open3D result."""
        d = perturbed_bunny
        result = register_point_clouds(
            d["src"],
            d["dst"],
            max_correspondence_distance=0.5,
            max_iters=50,
            tol=1e-6,
        )

        assert isinstance(result, o3d.pipelines.registration.RegistrationResult)
        assert hasattr(result, "transformation")
        assert hasattr(result, "fitness")
        assert hasattr(result, "inlier_rmse")
        assert hasattr(result, "correspondence_set")

        assert result.fitness > 0.99
        assert result.inlier_rmse < 1e-4
        assert result.transformation.shape == (4, 4)

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
