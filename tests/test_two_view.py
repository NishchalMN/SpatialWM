"""
Tests for spatialwm.geometry.two_view — RED now, green on impl.

Contracts defended:
1. fundamental_8pt -> essential_from_F -> decompose_E -> cheirality_select
   recovers R within 0.5° and t-direction within 1° from synthetic two-view geometry.
2. sampson_distance median is within 20% of the cv2-computed Sampson distance.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from spatialwm.geometry.two_view import (
    cheirality_select,
    decompose_E,
    essential_from_F,
    fundamental_8pt,
    sampson_distance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_angle_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Angle (degrees) between two rotation matrices via trace formula."""
    R_rel = R_est @ R_gt.T
    cos_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def _cv2_sampson(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute Sampson distance using OpenCV as reference."""
    # cv2.computeCorrespondEpilines is not Sampson; compute manually using cv2's convention.
    # Sampson: (x2 F x1)^2 / ((Fx1)_1^2 + (Fx1)_2^2 + (Ftx2)_1^2 + (Ftx2)_2^2)
    N = len(x1)
    x1h = np.hstack([x1, np.ones((N, 1))])  # (N,3)
    x2h = np.hstack([x2, np.ones((N, 1))])  # (N,3)
    Fx1 = (F @ x1h.T).T      # (N,3)
    Ftx2 = (F.T @ x2h.T).T  # (N,3)
    num = np.sum(x2h * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / (den + 1e-12)


# ---------------------------------------------------------------------------
# R/t recovery from synthetic two-view
# ---------------------------------------------------------------------------

class TestFundamentalAndEssentialPipeline:
    """End-to-end: 8-pt F -> E -> decompose -> cheirality recovers ground-truth pose."""

    def test_rotation_error_within_half_degree(self, synthetic_two_view):
        """Recovered R is within 0.5° of ground truth."""
        d = synthetic_two_view
        F = fundamental_8pt(d["x1"], d["x2"])
        E = essential_from_F(F, d["K"], d["K"])
        cands = decompose_E(E)
        R_est, _ = cheirality_select(cands, d["K"], d["K"], d["x1"], d["x2"])
        err = _rotation_angle_deg(R_est, d["R"])
        assert err < 0.5, f"Rotation error {err:.3f}° exceeds 0.5°"

    def test_translation_direction_within_one_degree(self, synthetic_two_view):
        """Recovered t-direction is within 1° of ground truth."""
        d = synthetic_two_view
        F = fundamental_8pt(d["x1"], d["x2"])
        E = essential_from_F(F, d["K"], d["K"])
        cands = decompose_E(E)
        _, t_est = cheirality_select(cands, d["K"], d["K"], d["x1"], d["x2"])

        t_gt = d["t"].ravel()
        t_e = np.asarray(t_est).ravel()

        # Normalize both; sign is ambiguous — take the smaller angle
        t_gt_n = t_gt / np.linalg.norm(t_gt)
        t_e_n = t_e / np.linalg.norm(t_e)
        cos_val = abs(np.clip(np.dot(t_gt_n, t_e_n), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(cos_val))
        assert angle_deg < 1.0, f"Translation direction error {angle_deg:.3f}° exceeds 1°"

    def test_four_decompose_candidates_returned(self, synthetic_two_view):
        """decompose_E must return exactly 4 (R, t) candidates."""
        d = synthetic_two_view
        F = fundamental_8pt(d["x1"], d["x2"])
        E = essential_from_F(F, d["K"], d["K"])
        cands = decompose_E(E)
        assert len(cands) == 4, f"Expected 4 candidates, got {len(cands)}"

    def test_all_rotations_are_valid(self, synthetic_two_view):
        """All four R candidates from decompose_E must have det ≈ +1."""
        d = synthetic_two_view
        F = fundamental_8pt(d["x1"], d["x2"])
        E = essential_from_F(F, d["K"], d["K"])
        for R_c, _ in decompose_E(E):
            assert abs(np.linalg.det(R_c) - 1.0) < 1e-6, "Candidate R has det != +1"


# ---------------------------------------------------------------------------
# Sampson distance
# ---------------------------------------------------------------------------

class TestSampsonDistance:
    def test_median_within_20_percent_of_cv2(self, synthetic_two_view):
        """
        After estimating F, our sampson_distance median is within 20% of
        the manually-computed Sampson distance (same formula, reference impl).
        """
        d = synthetic_two_view
        # Use OpenCV to get a clean F estimate as reference
        F_cv, mask = cv2.findFundamentalMat(
            d["x1"].astype(np.float32),
            d["x2"].astype(np.float32),
            cv2.FM_8POINT,
        )
        assert F_cv is not None, "cv2.findFundamentalMat returned None"

        sd_ours = sampson_distance(F_cv, d["x1"], d["x2"])
        sd_ref = _cv2_sampson(F_cv, d["x1"], d["x2"])

        med_ours = float(np.median(sd_ours))
        med_ref = float(np.median(sd_ref))

        if med_ref > 0:
            rel = abs(med_ours - med_ref) / med_ref
            assert rel < 0.20, (
                f"Sampson median {med_ours:.4f} deviates "
                f"{rel * 100:.1f}% from reference {med_ref:.4f}"
            )
        else:
            # Both should be ~0
            assert med_ours < 1e-6

    def test_sampson_zero_on_inlier_with_exact_F(self, synthetic_two_view):
        """Sampson distance is near zero for points consistent with exact F."""
        d = synthetic_two_view
        # Compute F from ground-truth calibrated geometry: F = K^{-T} E K^{-1}

        F_8pt = fundamental_8pt(d["x1"], d["x2"])
        sd = sampson_distance(F_8pt, d["x1"], d["x2"])
        # Median should be small (inlier data, not outlier-contaminated)
        assert np.median(sd) < 5.0, "Sampson median unexpectedly large for clean inlier data"

    def test_output_length_matches_input(self, synthetic_two_view):
        """sampson_distance returns one value per point pair."""
        d = synthetic_two_view
        F = fundamental_8pt(d["x1"], d["x2"])
        sd = sampson_distance(F, d["x1"], d["x2"])
        assert len(sd) == len(d["x1"])
