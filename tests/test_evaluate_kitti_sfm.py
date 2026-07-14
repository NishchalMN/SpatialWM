from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

scripts_path = str(Path(__file__).parent.parent / "scripts")
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from evaluate_kitti_sfm import (  # noqa: E402
    camera_centres_from_world_to_camera,
    similarity_align_reconstruction,
)


def test_camera_centres_from_world_to_camera():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, :3, 3] = [-1.0, 0.0, 0.0]
    poses[2, :3, 3] = [-2.0, 0.0, 0.0]
    centres = camera_centres_from_world_to_camera(poses)
    np.testing.assert_allclose(centres, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_similarity_alignment_recovers_scaled_path_and_points():
    estimated = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    )
    ground_truth = 2.5 * estimated + np.array([4.0, -3.0, 1.0])
    points = np.array([[0.5, 0.5, 2.0], [2.0, 1.0, 3.0]])
    aligned_centres, aligned_points, scale, ate = similarity_align_reconstruction(
        estimated, ground_truth, points
    )
    np.testing.assert_allclose(aligned_centres, ground_truth, atol=1e-10)
    np.testing.assert_allclose(
        aligned_points, 2.5 * points + np.array([4.0, -3.0, 1.0]), atol=1e-10
    )
    assert scale == pytest.approx(2.5)
    assert ate < 1e-10
