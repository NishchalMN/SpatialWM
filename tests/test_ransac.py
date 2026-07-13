"""Tests for spatialwm.geometry.ransac.

Contracts defended:
1. OpenCV-backed fundamental RANSAC recovers a plausible fundamental matrix with outliers.
2. The compatibility ransac wrapper behaves correctly.
3. Edge cases and invalid inputs raise appropriate exceptions.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialwm.geometry.ransac import RansacResult, fundamental_ransac, ransac


def _make_inlier_outlier_data(rng, n_inliers: int, n_outliers: int):
    """Build (x1, x2) stacked for RANSAC where the first n_inliers rows are
    consistent with a known fundamental geometry and the last n_outliers
    rows are random noise correspondences.

    Returns stacked data of shape (N, 4) — [u1, v1, u2, v2] — and ground_F.
    """
    from scipy.spatial.transform import Rotation

    K = np.array(
        [[800.0, 0.0, 320.0],
         [0.0, 800.0, 240.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    R_true = Rotation.from_euler("y", 10.0, degrees=True).as_matrix()
    t_true = np.array([[0.5], [0.0], [0.0]])

    # Inlier 3D points
    X = rng.uniform(-1.0, 1.0, (n_inliers, 3))
    X[:, 2] = rng.uniform(3.0, 7.0, n_inliers)

    def proj(R, t, pts):
        Xc = R @ pts.T + t
        xh = K @ Xc
        return (xh[:2] / xh[2]).T

    x1_in = proj(np.eye(3), np.zeros((3, 1)), X)
    x2_in = proj(R_true, t_true, X)

    # Outlier correspondences: random pixels in image space
    x1_out = rng.uniform([0, 0], [640, 480], (n_outliers, 2))
    x2_out = rng.uniform([0, 0], [640, 480], (n_outliers, 2))

    x1 = np.vstack([x1_in, x1_out])
    x2 = np.vstack([x2_in, x2_out])

    data = np.hstack([x1, x2])
    return data, K


class TestFundamentalRansac:
    """Tests for the new OpenCV-backed fundamental_ransac API."""

    def test_inlier_ratio_plausible_with_40_percent_outliers(self):
        """recovers plausible inlier ratio on contaminated F correspondences"""
        rng = np.random.default_rng(42)
        n_in, n_out = 60, 40   # 40% outlier rate
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)
        x1, x2 = data[:, :2], data[:, 2:]

        result = fundamental_ransac(
            x1,
            x2,
            thresh=1.0,
            p_success=0.99,
            max_iters=5000,
            method='usac_magsac'
        )

        assert isinstance(result, RansacResult)
        assert result.inlier_ratio >= 0.50, (
            f"inlier_ratio {result.inlier_ratio:.3f} below 0.50 "
            f"(expected ~0.60 for 40% outlier data)"
        )

    def test_returned_inlier_mask_shape_dtype(self):
        """returned inlier mask shape/dtype are correct"""
        rng = np.random.default_rng(0)
        n_in, n_out = 60, 40
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)
        x1, x2 = data[:, :2], data[:, 2:]

        result = fundamental_ransac(x1, x2, thresh=1.0)

        assert result.inliers.shape == (x1.shape[0],)
        assert result.inliers.dtype == bool or result.inliers.dtype == np.bool_

    def test_returned_model_shape(self):
        """returned model shape is (3, 3)"""
        rng = np.random.default_rng(1)
        n_in, n_out = 60, 40
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)
        x1, x2 = data[:, :2], data[:, 2:]

        result = fundamental_ransac(x1, x2, thresh=1.0)
        assert result.model.shape == (3, 3)
        assert result.model.dtype == np.float64

    def test_bad_shapes_and_too_few_correspondences_fail(self):
        """bad shapes / too few correspondences fail clearly"""
        # Too few points
        x1_few = np.random.uniform(0, 100, (7, 2))
        x2_few = np.random.uniform(0, 100, (7, 2))
        with pytest.raises(ValueError, match="At least 8 correspondences"):
            fundamental_ransac(x1_few, x2_few)

        # Mismatched length
        x1_mis = np.random.uniform(0, 100, (10, 2))
        x2_mis = np.random.uniform(0, 100, (11, 2))
        with pytest.raises(ValueError, match="same length"):
            fundamental_ransac(x1_mis, x2_mis)

        # Invalid shape
        x1_invalid = np.random.uniform(0, 100, (10, 3))
        x2_invalid = np.random.uniform(0, 100, (10, 2))
        with pytest.raises(ValueError, match="shape"):
            fundamental_ransac(x1_invalid, x2_invalid)

        # Non-finite inputs
        x1_nan = np.random.uniform(0, 100, (10, 2))
        x1_nan[0, 0] = np.nan
        x2_nan = np.random.uniform(0, 100, (10, 2))
        with pytest.raises(ValueError, match="finite"):
            fundamental_ransac(x1_nan, x2_nan)

    def test_invalid_parameters_fail(self):
        """invalid parameter values raise ValueError"""
        x1 = np.random.uniform(0, 100, (10, 2))
        x2 = np.random.uniform(0, 100, (10, 2))

        with pytest.raises(ValueError, match="Threshold"):
            fundamental_ransac(x1, x2, thresh=-1.0)

        with pytest.raises(ValueError, match="p_success"):
            fundamental_ransac(x1, x2, p_success=1.5)

        with pytest.raises(ValueError, match="max_iters"):
            fundamental_ransac(x1, x2, max_iters=0)

        with pytest.raises(ValueError, match="Unknown RANSAC method"):
            fundamental_ransac(x1, x2, method="invalid_method")

    def test_runtime_error_on_failure(self):
        """raises RuntimeError if OpenCV cannot find a valid model"""
        x1 = np.zeros((10, 2))
        x2 = np.zeros((10, 2))
        with pytest.raises(RuntimeError, match="failed to find a valid model"):
            fundamental_ransac(x1, x2)


class TestRansacCompatibility:
    """Tests for the compatibility ransac wrapper."""

    def test_compatibility_ransac_success(self):
        """compatibility ransac path works for Nx4 point correspondences"""
        rng = np.random.default_rng(42)
        n_in, n_out = 60, 40
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)

        result = ransac(
            data,
            thresh=1.0,
            p_success=0.99,
            max_iters=5000,
        )

        assert isinstance(result, RansacResult)
        assert result.model.shape == (3, 3)
        assert result.inliers.shape == (data.shape[0],)
        assert result.inlier_ratio >= 0.50

    def test_compatibility_ransac_raises_not_implemented(self):
        """compatibility ransac raises NotImplementedError for non-Nx4 data"""
        # Nx3 data
        data_invalid = np.random.uniform(0, 100, (10, 3))
        with pytest.raises(NotImplementedError, match="Generic scratch RANSAC"):
            ransac(data_invalid)
