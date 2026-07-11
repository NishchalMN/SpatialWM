"""
Tests for spatialwm.geometry.sfm_toy — SKIPPED (needs real image directory / data).

Collectible but skipped until Week 2 image data is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialwm.geometry.sfm_toy import run_sfm


@pytest.mark.skip(reason="Week 2 — needs image directory with real images")
def test_run_sfm_output_shapes(tmp_path):
    """
    run_sfm returns points (N,3) and poses (M,4,4) for M images.
    Requires a real image directory.
    """
    K = np.array(
        [[800.0, 0.0, 320.0],
         [0.0, 800.0, 240.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points, poses = run_sfm(str(tmp_path), K)
    assert points.ndim == 2 and points.shape[1] == 3
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4)


@pytest.mark.skip(reason="Week 2 — needs image directory with real images")
def test_run_sfm_poses_are_se3(tmp_path):
    """All recovered camera poses must be valid SE(3) matrices (det(R)==1)."""
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 800.0
    K[0, 2] = 320.0
    K[1, 2] = 240.0

    _, poses = run_sfm(str(tmp_path), K)
    for T in poses:
        R = T[:3, :3]
        assert abs(np.linalg.det(R) - 1.0) < 1e-5, "Pose R has det != +1"
