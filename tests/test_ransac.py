"""
Tests for spatialwm.geometry.ransac — RED now, green on impl.

Contracts defended:
1. RANSAC with 40% outliers recovers a fundamental matrix; inlier_ratio is plausible.
2. Adaptive iteration count N = log(1-p)/log(1-w^s) is consistent with reported n_iters.
"""

from __future__ import annotations

import math

import numpy as np

from spatialwm.geometry.ransac import RansacResult, ransac

# ---------------------------------------------------------------------------
# Helpers: build contaminated correspondence data
# ---------------------------------------------------------------------------

def _make_inlier_outlier_data(rng, n_inliers: int, n_outliers: int):
    """
    Build (x1, x2) stacked for RANSAC where the first n_inliers rows are
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
    x1_out = rng.uniform(0, 640, (n_outliers, 2))
    x2_out = rng.uniform(0, 480, (n_outliers, 2))

    x1 = np.vstack([x1_in, x1_out])
    x2 = np.vstack([x2_in, x2_out])

    data = np.hstack([x1, x2])
    return data, K


def _ransac_fit_fn(samples):
    """Adapter: fit_fn receives min_samples rows of [u1,v1,u2,v2]."""
    from spatialwm.geometry.two_view import fundamental_8pt

    x1 = samples[:, :2]
    x2 = samples[:, 2:]
    return fundamental_8pt(x1, x2)


def _ransac_score_fn(F, data):
    """score_fn: return per-point Sampson distances for the given F and data."""
    from spatialwm.geometry.two_view import sampson_distance

    x1 = data[:, :2]
    x2 = data[:, 2:]
    return sampson_distance(F, x1, x2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRansacInlierRecovery:
    """RANSAC with 40% outliers should identify the inlier set."""

    def test_inlier_ratio_plausible_with_40_percent_outliers(self):
        """inlier_ratio >= 0.55 when true inlier rate is 0.60."""
        rng = np.random.default_rng(42)
        n_in, n_out = 60, 40   # 40% outlier rate
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)

        result: RansacResult = ransac(
            data,
            fit_fn=_ransac_fit_fn,
            score_fn=_ransac_score_fn,
            min_samples=8,
            thresh=1.0,      # Sampson pixel^2 threshold
            p_success=0.99,
            max_iters=5000,
        )

        assert isinstance(result, RansacResult)
        assert result.inlier_ratio >= 0.55, (
            f"inlier_ratio {result.inlier_ratio:.3f} below 0.55 "
            f"(expected ~0.60 for 40% outlier data)"
        )

    def test_inlier_mask_length_matches_data(self):
        """result.inliers is a boolean mask of the same length as data."""
        rng = np.random.default_rng(0)
        n_in, n_out = 60, 40
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)

        result: RansacResult = ransac(
            data,
            fit_fn=_ransac_fit_fn,
            score_fn=_ransac_score_fn,
            min_samples=8,
            thresh=1.0,
            p_success=0.99,
            max_iters=5000,
        )

        assert len(result.inliers) == len(data)
        assert result.inliers.dtype == bool or result.inliers.dtype == np.bool_

    def test_recovered_model_is_3x3(self):
        """RANSAC returns a 3×3 fundamental matrix."""
        rng = np.random.default_rng(1)
        n_in, n_out = 60, 40
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)

        result: RansacResult = ransac(
            data,
            fit_fn=_ransac_fit_fn,
            score_fn=_ransac_score_fn,
            min_samples=8,
            thresh=1.0,
        )
        assert result.model.shape == (3, 3)

    def test_recovered_inliers_mostly_true_inliers(self):
        """
        With known inlier labeling (first 60 rows), most RANSAC inliers should
        overlap with the true inlier set.  We accept >= 80% precision.
        """
        rng = np.random.default_rng(3)
        n_in, n_out = 60, 40
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)

        result: RansacResult = ransac(
            data,
            fit_fn=_ransac_fit_fn,
            score_fn=_ransac_score_fn,
            min_samples=8,
            thresh=1.0,
            p_success=0.99,
            max_iters=5000,
        )

        mask = result.inliers
        true_inlier_flags = np.zeros(len(data), dtype=bool)
        true_inlier_flags[:n_in] = True

        n_detected = mask.sum()
        if n_detected > 0:
            precision = (mask & true_inlier_flags).sum() / n_detected
            assert precision >= 0.80, f"RANSAC inlier precision {precision:.3f} < 0.80"


class TestRansacAdaptiveIterCount:
    """
    N_adaptive = ceil(log(1-p) / log(1-w^s)) for known w (inlier fraction)
    and s (min_samples).

    The reported n_iters must be <= N_adaptive(w_hat, s) + a small tolerance
    to account for the adaptive updating strategy.
    """

    def test_adaptive_iter_count_consistent_with_formula(self):
        """n_iters is consistent with the adaptive RANSAC formula."""
        rng = np.random.default_rng(10)
        n_in, n_out = 80, 20   # 80% inliers -> known w ≈ 0.80
        data, _ = _make_inlier_outlier_data(rng, n_in, n_out)

        p = 0.99
        s = 8
        w_approx = 0.80
        N_formula = math.ceil(math.log(1.0 - p) / math.log(1.0 - w_approx ** s))

        result: RansacResult = ransac(
            data,
            fit_fn=_ransac_fit_fn,
            score_fn=_ransac_score_fn,
            min_samples=s,
            thresh=1.0,
            p_success=p,
            max_iters=5000,
        )

        # Adaptive RANSAC updates N as it finds better inlier ratios;
        # final n_iters should be << 5000 and close to N_formula (within 5×).
        assert result.n_iters <= max(N_formula * 5, 200), (
            f"n_iters={result.n_iters} far exceeds adaptive formula ~{N_formula}"
        )
        assert result.n_iters >= 1, "n_iters must be at least 1"
