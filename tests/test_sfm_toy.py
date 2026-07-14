"""Deterministic integration tests for the bounded sparse-SfM story."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from spatialwm.geometry.sfm_toy import SfmResult, run_sfm, run_sfm_detailed


@pytest.fixture(scope="module")
def synthetic_sequence(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    """Render repeatable point-attached textures under perspective motion."""
    directory = tmp_path_factory.mktemp("sfm_sequence")
    rng = np.random.default_rng(12)
    K = np.array(
        [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
    )
    points = rng.uniform([-2.5, -1.6, 4.0], [2.5, 1.6, 9.0], (180, 3))
    patches = []
    for _ in points:
        patch = rng.integers(0, 256, (17, 17), dtype=np.uint8)
        patch = cv2.GaussianBlur(patch, (3, 3), 0.0)
        cv2.rectangle(patch, (0, 0), (16, 16), 255, 1)
        patches.append(patch)

    for camera_id, translation_x in enumerate([0.0, -0.15, -0.30, -0.45]):
        image = np.full((480, 640), 24, dtype=np.uint8)
        points_camera = points + np.array([translation_x, 0.0, 0.0])
        projected = points_camera @ K.T
        projected = projected[:, :2] / projected[:, 2:3]
        for patch, (u_float, v_float) in zip(patches, projected):
            u, v = int(round(u_float)), int(round(v_float))
            half_size = 8
            if half_size <= u < 640 - half_size - 1 and half_size <= v < 480 - half_size - 1:
                region = image[
                    v - half_size : v + half_size + 1,
                    u - half_size : u + half_size + 1,
                ]
                np.maximum(region, patch, out=region)
        assert cv2.imwrite(str(directory / f"{camera_id:03d}.png"), image)
    return directory, K


@pytest.fixture(scope="module")
def reconstruction(synthetic_sequence: tuple[Path, np.ndarray]) -> SfmResult:
    directory, K = synthetic_sequence
    return run_sfm_detailed(
        directory,
        K,
        max_images=4,
        max_features=5000,
        ratio=0.8,
        min_initial_points=20,
        min_pnp_points=8,
        refine=True,
    )


def test_run_sfm_registers_sequence_and_refines(reconstruction: SfmResult) -> None:
    assert reconstruction.points.shape[1] == 3
    assert len(reconstruction.points) >= 50
    assert reconstruction.poses_world_to_camera.shape == (4, 4, 4)
    assert np.array_equal(reconstruction.registered_image_indices, np.arange(4))
    assert reconstruction.reprojection_rmse_after_px <= (
        reconstruction.reprojection_rmse_before_px + 1e-10
    )
    assert reconstruction.reprojection_rmse_after_px < 0.5
    assert np.all(reconstruction.track_lengths >= 2)
    assert np.any(reconstruction.track_lengths == 4)
    assert len(reconstruction.points) > reconstruction.initial_landmark_count
    assert reconstruction.triangulation_sources.shape == (len(reconstruction.points), 2)
    assert np.all(np.isfinite(reconstruction.landmark_confidence))
    assert np.all(reconstruction.landmark_confidence > 0.0)


def test_landmark_expansion_adds_verified_two_view_tracks(
    synthetic_sequence: tuple[Path, np.ndarray],
) -> None:
    directory, K = synthetic_sequence
    fixed = run_sfm_detailed(
        directory,
        K,
        max_images=4,
        max_features=5000,
        ratio=0.8,
        min_initial_points=20,
        min_pnp_points=8,
        refine=False,
        expand_landmarks=False,
    )
    expanded = run_sfm_detailed(
        directory,
        K,
        max_images=4,
        max_features=5000,
        ratio=0.8,
        min_initial_points=20,
        min_pnp_points=8,
        refine=False,
        expand_landmarks=True,
    )
    assert len(expanded.points) > len(fixed.points)
    assert expanded.initial_landmark_count == len(fixed.points)
    assert np.all(expanded.track_lengths >= 2)


def test_run_sfm_poses_are_world_to_camera_se3(reconstruction: SfmResult) -> None:
    poses = reconstruction.poses_world_to_camera
    for transform in poses:
        rotation = transform[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)
        assert np.array_equal(transform[3], np.array([0.0, 0.0, 0.0, 1.0]))

    centers = np.array(
        [-pose[:3, :3].T @ pose[:3, 3] for pose in poses]
    )
    assert np.all(np.diff(centers[:, 0]) > 0.0)


def test_run_sfm_wrapper_matches_detailed_contract(
    synthetic_sequence: tuple[Path, np.ndarray],
) -> None:
    directory, K = synthetic_sequence
    points, poses = run_sfm(
        directory,
        K,
        max_images=2,
        max_features=5000,
        ratio=0.8,
        min_initial_points=20,
        min_pnp_points=8,
        refine=False,
    )
    assert points.ndim == 2 and points.shape[1] == 3
    assert poses.shape == (2, 4, 4)


def test_run_sfm_rejects_missing_or_short_input(tmp_path: Path) -> None:
    K = np.eye(3)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_sfm_detailed(tmp_path / "missing", K)
    with pytest.raises(ValueError, match="at least two"):
        run_sfm_detailed(tmp_path, K)


def test_run_sfm_reports_featureless_initialization_failure(tmp_path: Path) -> None:
    K = np.array(
        [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
    )
    blank = np.full((480, 640), 127, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "000.png"), blank)
    assert cv2.imwrite(str(tmp_path / "001.png"), blank)
    with pytest.raises(RuntimeError, match="no image pair"):
        run_sfm_detailed(tmp_path, K, min_initial_points=8)
