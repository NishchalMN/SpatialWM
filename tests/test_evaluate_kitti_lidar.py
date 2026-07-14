"""Tests for scripts/evaluate_kitti_lidar.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest

# Add scripts directory to path to import evaluate_kitti_lidar
scripts_path = str(Path(__file__).parent.parent / "scripts")
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import evaluate_kitti_lidar  # noqa: E402


def test_normalize_poses():
    """Verify that normalize_poses shifts the first pose to identity and computes

    correct relative transforms (normalization direction).
    """
    # Create simple translations: T_0 is translation along x by 1.0, T_1 is translation by 3.0
    T_0 = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    T_1 = np.array(
        [
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    poses = [T_0, T_1]
    normalized = evaluate_kitti_lidar.normalize_poses(poses)

    # First pose must be identity
    np.testing.assert_allclose(normalized[0], np.eye(4), atol=1e-7)

    # Second pose must represent the correct relative transform: T_0_inv @ T_1
    # which is translation along x of (3.0 - 1.0) = 2.0
    expected_P1 = np.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(normalized[1], expected_P1, atol=1e-7)

    # Verify input validation
    with pytest.raises(ValueError, match="T_w_velo must have shape"):
        evaluate_kitti_lidar.normalize_poses(np.ones((2, 3, 3)))


def test_make_json_safe():
    """Verify NumPy structures are correctly converted to JSON-native formats."""
    raw_data = {
        "float": np.float64(1.23),
        "int": np.int32(42),
        "array": np.array([[1, 2], [3, 4]]),
        "nested": {"val": np.float32(5.5)},
        "plain_list": [1, 2, np.int64(3)],
    }

    safe_data = evaluate_kitti_lidar.make_json_safe(raw_data)

    # Check type conversions
    assert isinstance(safe_data["float"], float)
    assert isinstance(safe_data["int"], int)
    assert isinstance(safe_data["array"], list)
    assert isinstance(safe_data["array"][0][0], int)
    assert isinstance(safe_data["nested"]["val"], float)
    assert isinstance(safe_data["plain_list"][2], int)

    # Check that it actually serializes successfully
    serialized = json.dumps(safe_data)
    assert isinstance(serialized, str)


def test_compute_metrics():
    """Verify compute_metrics returns a valid metric dictionary or rejects short inputs."""
    # Build simple trajectories of 3 frames (minimum requirement)
    poses_est = np.stack([np.eye(4), np.eye(4), np.eye(4)])
    poses_est[1, :3, 3] = [1.0, 0.0, 0.0]
    poses_est[2, :3, 3] = [0.0, 1.0, 0.0]

    poses_gt = np.stack([np.eye(4), np.eye(4), np.eye(4)])
    poses_gt[1, :3, 3] = [1.0, 0.0, 0.0]
    poses_gt[2, :3, 3] = [0.0, 1.0, 0.0]

    metrics = evaluate_kitti_lidar.compute_metrics(poses_est, poses_gt)
    assert "ate" in metrics
    assert "rpe" in metrics
    assert isinstance(metrics["ate"], float)
    assert isinstance(metrics["rpe"], float)

    # Rejecting trajectory of length < 3
    short_est = np.stack([np.eye(4), np.eye(4)])
    short_gt = np.stack([np.eye(4), np.eye(4)])
    with pytest.raises(ValueError, match="At least 3 frames are required"):
        evaluate_kitti_lidar.compute_metrics(short_est, short_gt)


def test_compute_metrics_does_not_hide_metric_scale_error():
    """Primary LiDAR ATE must not use the free scale allowed by monocular data."""
    poses_gt = np.repeat(np.eye(4)[None], 4, axis=0)
    poses_gt[:, :3, 3] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    )
    poses_est = poses_gt.copy()
    poses_est[:, :3, 3] *= 2.0

    metrics = evaluate_kitti_lidar.compute_metrics(poses_est, poses_gt)
    assert metrics["rigid_aligned_ate_rmse_m"] > 0.1
    assert metrics["sim3_aligned_ate_rmse_m"] < 1e-10
    assert metrics["sim3_scale"] == pytest.approx(0.5)


def test_generate_diagnostic_plot(tmp_path):
    """Verify that generate_diagnostic_plot successfully creates a figure and saves it."""
    # Synthetic inputs (no dataset on disk required)
    points_source = np.random.standard_normal((100, 3))
    points_target = np.random.standard_normal((120, 3))
    T_pair = np.eye(4)
    poses_est = np.stack([np.eye(4), np.eye(4), np.eye(4)])
    poses_gt = np.stack([np.eye(4), np.eye(4), np.eye(4)])
    frame_ids = [0, 1, 2]
    voxel = 0.2
    max_corr = 1.0
    output_png = tmp_path / "test_diag.png"

    # Call plotting helper
    fig = evaluate_kitti_lidar.generate_diagnostic_plot(
        points_source=points_source,
        points_target=points_target,
        T_pair=T_pair,
        poses_est=poses_est,
        poses_gt=poses_gt,
        frame_ids=frame_ids,
        voxel=voxel,
        max_corr=max_corr,
        pair_source_idx=1,
        pair_target_idx=0,
        output_path=str(output_png),
    )

    # Verify return type and file generation
    assert isinstance(fig, matplotlib.figure.Figure)
    assert output_png.exists()


def test_generate_curated_trajectory_and_bev_figures(tmp_path):
    poses_gt = np.repeat(np.eye(4)[None], 4, axis=0)
    poses_gt[:, :3, 3] = np.array(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.1, 0.0], [1.2, 0.2, 0.0]]
    )
    poses_est = poses_gt.copy()
    poses_est[:, 0, 3] *= 1.02
    metrics = evaluate_kitti_lidar.compute_metrics(poses_est, poses_gt)
    trajectory_path = tmp_path / "trajectory.png"
    trajectory_figure = evaluate_kitti_lidar.generate_trajectory_figure(
        poses_est,
        poses_gt,
        metrics,
        [0, 1, 2, 3],
        str(trajectory_path),
    )
    assert isinstance(trajectory_figure, matplotlib.figure.Figure)
    assert trajectory_path.exists()

    rng = np.random.default_rng(8)
    scans = []
    base = rng.uniform([0.0, -5.0, -2.0], [15.0, 5.0, 1.0], (1000, 3))
    for frame in range(4):
        scans.append(base - np.array([0.4 * frame, 0.0, 0.0]))
    bev_path = tmp_path / "bev.png"
    bev_figure, bev_metrics = evaluate_kitti_lidar.generate_bev_figure(
        scans,
        poses_est,
        str(bev_path),
        cell=0.2,
    )
    assert isinstance(bev_figure, matplotlib.figure.Figure)
    assert bev_path.exists()
    assert bev_metrics["accumulated_occupied_cells"] >= (
        bev_metrics["single_scan_occupied_cells"]
    )
