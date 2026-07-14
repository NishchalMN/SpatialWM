"""A small, inspectable incremental Structure-from-Motion pipeline.

The implementation deliberately targets a bounded image sequence rather than
trying to be a production SfM system. It connects the project's classical 3D
building blocks into one story: feature matching, epipolar verification,
two-view initialization, PnP registration, and bundle adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from spatialwm.geometry.bundle_adjust import bundle_adjust
from spatialwm.geometry.features import FeatureMatchResult, match_features
from spatialwm.geometry.ransac import fundamental_ransac
from spatialwm.geometry.two_view import (
    cheirality_select,
    decompose_E,
    essential_from_F,
    triangulate_dlt,
)


@dataclass(frozen=True)
class SfmResult:
    """Sparse reconstruction plus diagnostics needed for verification.

    Poses are world-to-camera transforms. The first registered camera defines
    the world frame and the initial translation has unit length, so the result
    has an arbitrary monocular scale.
    """

    points: np.ndarray
    poses_world_to_camera: np.ndarray
    observations: np.ndarray
    image_paths: tuple[str, ...]
    registered_image_indices: np.ndarray
    initial_pair: tuple[int, int]
    reprojection_rmse_before_px: float
    reprojection_rmse_after_px: float
    track_lengths: np.ndarray
    initial_landmark_count: int
    triangulation_sources: np.ndarray
    landmark_confidence: np.ndarray


@dataclass(frozen=True)
class _TwoViewInitialization:
    second_index: int
    rotation: np.ndarray
    translation: np.ndarray
    points: np.ndarray
    reference_pixels: np.ndarray
    second_pixels: np.ndarray
    score: float


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _validate_intrinsics(K: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(K, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("K must have shape (3, 3)")
    if not np.all(np.isfinite(intrinsics)):
        raise ValueError("K must contain only finite values")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("K focal lengths must be positive")
    if abs(np.linalg.det(intrinsics)) <= np.finfo(float).eps:
        raise ValueError("K must be invertible")
    return intrinsics


def _load_images(
    image_dir: str | Path,
    *,
    start: int,
    stride: int,
    max_images: int,
) -> tuple[list[np.ndarray], list[Path], np.ndarray]:
    directory = Path(image_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    if start < 0 or stride < 1 or max_images < 2:
        raise ValueError("start must be nonnegative, stride positive, and max_images >= 2")

    paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES
    )
    selected_indices = np.arange(start, len(paths), stride, dtype=np.int64)[:max_images]
    if len(selected_indices) < 2:
        raise ValueError("the requested slice must contain at least two images")

    selected_paths = [paths[int(index)] for index in selected_indices]
    images = []
    for path in selected_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")
        images.append(image)
    return images, selected_paths, selected_indices


def _project(points: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    points_camera = points @ rotation.T + pose[3:]
    homogeneous = points_camera @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _reprojection_rmse(
    poses: np.ndarray,
    points: np.ndarray,
    K: np.ndarray,
    observations: np.ndarray,
) -> float:
    residuals = []
    for camera_id in np.unique(observations[:, 0].astype(np.int64)):
        rows = observations[observations[:, 0] == camera_id]
        point_ids = rows[:, 1].astype(np.int64)
        predicted = _project(points[point_ids], poses[camera_id], K)
        residuals.append(predicted - rows[:, 2:4])
    stacked = np.vstack(residuals)
    return float(np.sqrt(np.mean(np.sum(stacked**2, axis=1))))


def _recover_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    candidate_index: int,
    K: np.ndarray,
    *,
    detector: str,
    ratio: float,
    max_features: int,
    ransac_threshold_px: float,
    min_initial_points: int,
    seed: int,
) -> _TwoViewInitialization | None:
    matches = match_features(
        reference,
        candidate,
        detector=detector,
        ratio=ratio,
        mutual=True,
        max_features=max_features,
        seed=seed,
    )
    if len(matches.points1) < max(8, min_initial_points):
        return None

    try:
        geometry = fundamental_ransac(
            matches.points1,
            matches.points2,
            thresh=ransac_threshold_px,
            p_success=0.999,
            max_iters=10_000,
        )
    except (RuntimeError, ValueError):
        return None
    x1 = matches.points1[geometry.inliers]
    x2 = matches.points2[geometry.inliers]
    if len(x1) < min_initial_points:
        return None

    essential = essential_from_F(geometry.model, K, K)
    rotation, translation = cheirality_select(decompose_E(essential), K, K, x1, x2)
    projection1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projection2 = K @ np.hstack((rotation, translation[:, None]))
    points = triangulate_dlt(projection1, projection2, x1, x2)

    depth1 = points[:, 2]
    depth2 = (points @ rotation.T + translation)[:, 2]
    pose1 = np.zeros(6, dtype=np.float64)
    pose2 = np.concatenate((Rotation.from_matrix(rotation).as_rotvec(), translation))
    error1 = np.linalg.norm(_project(points, pose1, K) - x1, axis=1)
    error2 = np.linalg.norm(_project(points, pose2, K) - x2, axis=1)
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(error1)
        & np.isfinite(error2)
        & (depth1 > 0.0)
        & (depth2 > 0.0)
        & (error1 < 3.0 * ransac_threshold_px)
        & (error2 < 3.0 * ransac_threshold_px)
    )
    if np.count_nonzero(valid) < min_initial_points:
        return None

    points = points[valid]
    x1 = x1[valid]
    x2 = x2[valid]
    displacement = np.linalg.norm(x2 - x1, axis=1)
    score = float(len(points) * np.median(displacement))
    return _TwoViewInitialization(
        second_index=candidate_index,
        rotation=rotation,
        translation=translation,
        points=points,
        reference_pixels=x1,
        second_pixels=x2,
        score=score,
    )


def _associate_reference_tracks(
    reference_tracks: np.ndarray,
    matches: FeatureMatchResult,
    inlier_mask: np.ndarray,
    *,
    tolerance_px: float,
    reference_point_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate pairwise reference matches with initialized landmark tracks."""
    query = matches.points1[inlier_mask]
    current = matches.points2[inlier_mask]
    if len(query) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float64)

    distances, track_ids = cKDTree(reference_tracks).query(query, k=1)
    if reference_point_ids is None:
        reference_point_ids = np.arange(len(reference_tracks), dtype=np.int64)
    reference_point_ids = np.asarray(reference_point_ids, dtype=np.int64)
    if len(reference_point_ids) != len(reference_tracks):
        raise ValueError("reference_point_ids must match reference_tracks")
    candidates = [
        (float(distance), int(reference_point_ids[track_id]), current_pixel)
        for distance, track_id, current_pixel in zip(distances, track_ids, current)
        if distance <= tolerance_px
    ]
    candidates.sort(key=lambda item: item[0])
    used_points: set[int] = set()
    selected_ids = []
    selected_pixels = []
    for _, point_id, pixel in candidates:
        if point_id in used_points:
            continue
        used_points.add(point_id)
        selected_ids.append(point_id)
        selected_pixels.append(pixel)
    return np.asarray(selected_ids, dtype=np.int64), np.asarray(
        selected_pixels, dtype=np.float64
    ).reshape(-1, 2)


def _register_view(
    reference: np.ndarray,
    image: np.ndarray,
    points: np.ndarray,
    reference_tracks: np.ndarray,
    reference_point_ids: np.ndarray,
    K: np.ndarray,
    *,
    detector: str,
    ratio: float,
    max_features: int,
    ransac_threshold_px: float,
    track_tolerance_px: float,
    min_pnp_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    matches = match_features(
        reference,
        image,
        detector=detector,
        ratio=ratio,
        mutual=True,
        max_features=max_features,
        seed=seed,
    )
    if len(matches.points1) < 8:
        return None
    try:
        geometry = fundamental_ransac(
            matches.points1,
            matches.points2,
            thresh=ransac_threshold_px,
            p_success=0.999,
            max_iters=10_000,
        )
    except (RuntimeError, ValueError):
        return None

    point_ids, pixels = _associate_reference_tracks(
        reference_tracks,
        matches,
        geometry.inliers,
        tolerance_px=track_tolerance_px,
        reference_point_ids=reference_point_ids,
    )
    if len(point_ids) < min_pnp_points:
        return None

    cv2.setRNGSeed(int(seed))
    success, rotation_vector, translation, pnp_inliers = cv2.solvePnPRansac(
        objectPoints=np.ascontiguousarray(points[point_ids], dtype=np.float64),
        imagePoints=np.ascontiguousarray(pixels, dtype=np.float64),
        cameraMatrix=K,
        distCoeffs=None,
        iterationsCount=2000,
        reprojectionError=max(2.0, 2.0 * ransac_threshold_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or pnp_inliers is None or len(pnp_inliers) < min_pnp_points:
        return None
    keep = pnp_inliers.ravel()
    point_ids = point_ids[keep]
    pixels = pixels[keep]
    try:
        rotation_vector, translation = cv2.solvePnPRefineLM(
            objectPoints=np.ascontiguousarray(points[point_ids], dtype=np.float64),
            imagePoints=np.ascontiguousarray(pixels, dtype=np.float64),
            cameraMatrix=K,
            distCoeffs=None,
            rvec=rotation_vector,
            tvec=translation,
        )
    except cv2.error:
        pass

    pose = np.concatenate((rotation_vector.ravel(), translation.ravel()))
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    depths = (points[point_ids] @ rotation.T + pose[3:])[:, 2]
    positive = depths > 0.0
    if np.count_nonzero(positive) < min_pnp_points:
        return None
    return pose, point_ids[positive], pixels[positive]


def _triangulate_new_landmarks(
    anchor: np.ndarray,
    image: np.ndarray,
    anchor_pose: np.ndarray,
    image_pose: np.ndarray,
    anchor_observations: dict[int, np.ndarray],
    image_observations: dict[int, np.ndarray],
    K: np.ndarray,
    *,
    detector: str,
    ratio: float,
    max_features: int,
    ransac_threshold_px: float,
    track_tolerance_px: float,
    min_triangulation_angle_deg: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate verified matches not already assigned to landmarks."""
    matches = match_features(
        anchor,
        image,
        detector=detector,
        ratio=ratio,
        mutual=True,
        max_features=max_features,
        seed=seed,
    )
    if len(matches.points1) < 8:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
    try:
        geometry = fundamental_ransac(
            matches.points1,
            matches.points2,
            thresh=ransac_threshold_px,
            p_success=0.999,
            max_iters=10_000,
        )
    except (RuntimeError, ValueError):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )

    pixels_anchor = matches.points1[geometry.inliers]
    pixels_image = matches.points2[geometry.inliers]
    if anchor_observations:
        existing_anchor = np.vstack(list(anchor_observations.values()))
        distance_anchor, _ = cKDTree(existing_anchor).query(pixels_anchor, k=1)
    else:
        distance_anchor = np.full(len(pixels_anchor), np.inf)
    if image_observations:
        existing_image = np.vstack(list(image_observations.values()))
        distance_image, _ = cKDTree(existing_image).query(pixels_image, k=1)
    else:
        distance_image = np.full(len(pixels_image), np.inf)
    novel = (distance_anchor > track_tolerance_px) & (distance_image > track_tolerance_px)
    pixels_anchor = pixels_anchor[novel]
    pixels_image = pixels_image[novel]
    if not len(pixels_anchor):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )

    anchor_rotation = Rotation.from_rotvec(anchor_pose[:3]).as_matrix()
    image_rotation = Rotation.from_rotvec(image_pose[:3]).as_matrix()
    projection_anchor = K @ np.hstack((anchor_rotation, anchor_pose[3:, None]))
    projection_image = K @ np.hstack((image_rotation, image_pose[3:, None]))
    points = triangulate_dlt(
        projection_anchor, projection_image, pixels_anchor, pixels_image
    )
    depth_anchor = (points @ anchor_rotation.T + anchor_pose[3:])[:, 2]
    depth_image = (points @ image_rotation.T + image_pose[3:])[:, 2]
    error_anchor = np.linalg.norm(_project(points, anchor_pose, K) - pixels_anchor, axis=1)
    error_image = np.linalg.norm(_project(points, image_pose, K) - pixels_image, axis=1)

    center_anchor = -anchor_rotation.T @ anchor_pose[3:]
    center_image = -image_rotation.T @ image_pose[3:]
    ray_anchor = points - center_anchor
    ray_image = points - center_image
    ray_cosine = np.sum(ray_anchor * ray_image, axis=1) / (
        np.linalg.norm(ray_anchor, axis=1) * np.linalg.norm(ray_image, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(ray_cosine, -1.0, 1.0)))
    valid = (
        np.all(np.isfinite(points), axis=1)
        & (depth_anchor > 0.0)
        & (depth_image > 0.0)
        & (angles >= min_triangulation_angle_deg)
        & (error_anchor < 3.0 * ransac_threshold_px)
        & (error_image < 3.0 * ransac_threshold_px)
    )
    return points[valid], pixels_anchor[valid], pixels_image[valid]


def _poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    matrices[:, :3, :3] = Rotation.from_rotvec(poses[:, :3]).as_matrix()
    matrices[:, :3, 3] = poses[:, 3:]
    return matrices


def run_sfm_detailed(
    image_dir: str | Path,
    K: np.ndarray,
    *,
    start: int = 0,
    stride: int = 1,
    max_images: int = 8,
    detector: str = "sift",
    ratio: float = 0.75,
    max_features: int = 4000,
    ransac_threshold_px: float = 1.0,
    track_tolerance_px: float = 1.5,
    min_initial_points: int = 30,
    min_pnp_points: int = 12,
    expand_landmarks: bool = True,
    min_triangulation_angle_deg: float = 0.5,
    seed: int = 0,
    refine: bool = True,
) -> SfmResult:
    """Reconstruct a bounded sequence with incremental landmark expansion.

    The best reference-to-candidate initialization is chosen by the number of
    cheirality-valid points and their median image displacement. Each remaining
    camera is registered from a nearby registered view using PnP RANSAC. New,
    verified feature tracks are then triangulated between the two known poses.
    """
    intrinsics = _validate_intrinsics(K)
    images, paths, absolute_indices = _load_images(
        image_dir, start=start, stride=stride, max_images=max_images
    )
    if min_initial_points < 8 or min_pnp_points < 4:
        raise ValueError("min_initial_points must be >= 8 and min_pnp_points >= 4")
    if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0.0:
        raise ValueError("ransac_threshold_px must be finite and positive")
    if not np.isfinite(track_tolerance_px) or track_tolerance_px <= 0.0:
        raise ValueError("track_tolerance_px must be finite and positive")
    if not np.isfinite(min_triangulation_angle_deg) or min_triangulation_angle_deg <= 0.0:
        raise ValueError("min_triangulation_angle_deg must be finite and positive")

    initializations = [
        recovered
        for candidate_index, image in enumerate(images[1:], start=1)
        if (
            recovered := _recover_pair(
                images[0],
                image,
                candidate_index,
                intrinsics,
                detector=detector,
                ratio=ratio,
                max_features=max_features,
                ransac_threshold_px=ransac_threshold_px,
                min_initial_points=min_initial_points,
                seed=seed,
            )
        )
        is not None
    ]
    if not initializations:
        raise RuntimeError("no image pair produced a valid two-view initialization")
    initialization = max(initializations, key=lambda item: item.score)

    points = initialization.points.copy()
    initial_landmark_count = len(points)
    camera_records: list[tuple[int, np.ndarray]] = [
        (0, np.zeros(6, dtype=np.float64)),
        (
            initialization.second_index,
            np.concatenate(
                (
                    Rotation.from_matrix(initialization.rotation).as_rotvec(),
                    initialization.translation,
                )
            ),
        ),
    ]
    image_observations: dict[int, dict[int, np.ndarray]] = {
        0: {},
        initialization.second_index: {},
    }
    triangulation_sources: list[tuple[int, int]] = []
    for point_id, (pixel1, pixel2) in enumerate(
        zip(initialization.reference_pixels, initialization.second_pixels)
    ):
        image_observations[0][point_id] = pixel1
        image_observations[initialization.second_index][point_id] = pixel2
        triangulation_sources.append((0, initialization.second_index))

    for image_index, image in enumerate(images[1:], start=1):
        if image_index == initialization.second_index:
            continue
        registered_by_index = dict(camera_records)
        candidates = sorted(
            registered_by_index,
            key=lambda registered_index: abs(registered_index - image_index),
        )
        accepted = None
        anchor_index = -1
        for candidate_index in candidates:
            candidate_observations = image_observations[candidate_index]
            if len(candidate_observations) < min_pnp_points:
                continue
            candidate_point_ids = np.fromiter(
                candidate_observations.keys(), dtype=np.int64
            )
            candidate_pixels = np.vstack(list(candidate_observations.values()))
            registration = _register_view(
                images[candidate_index],
                image,
                points,
                candidate_pixels,
                candidate_point_ids,
                intrinsics,
                detector=detector,
                ratio=ratio,
                max_features=max_features,
                ransac_threshold_px=ransac_threshold_px,
                track_tolerance_px=track_tolerance_px,
                min_pnp_points=min_pnp_points,
                seed=seed,
            )
            if registration is not None:
                accepted = registration
                anchor_index = candidate_index
                break
        if accepted is None:
            continue
        pose, point_ids, pixels = accepted
        camera_records.append((image_index, pose))
        image_observations[image_index] = {
            int(point_id): pixel for point_id, pixel in zip(point_ids, pixels)
        }

        if expand_landmarks:
            new_points, anchor_pixels, image_pixels = _triangulate_new_landmarks(
                images[anchor_index],
                image,
                registered_by_index[anchor_index],
                pose,
                image_observations[anchor_index],
                image_observations[image_index],
                intrinsics,
                detector=detector,
                ratio=ratio,
                max_features=max_features,
                ransac_threshold_px=ransac_threshold_px,
                track_tolerance_px=track_tolerance_px,
                min_triangulation_angle_deg=min_triangulation_angle_deg,
                seed=seed,
            )
            first_new_id = len(points)
            if len(new_points):
                points = np.vstack((points, new_points))
            for offset, (anchor_pixel, image_pixel) in enumerate(
                zip(anchor_pixels, image_pixels)
            ):
                point_id = first_new_id + offset
                image_observations[anchor_index][point_id] = anchor_pixel
                image_observations[image_index][point_id] = image_pixel
                triangulation_sources.append((anchor_index, image_index))

    camera_id_by_image = {
        image_index: camera_id
        for camera_id, image_index in enumerate(sorted(dict(camera_records)))
    }
    observation_records: list[tuple[int, int, float, float]] = []
    for image_index, point_pixels in image_observations.items():
        camera_id = camera_id_by_image[image_index]
        for point_id, pixel in point_pixels.items():
            observation_records.append((camera_id, point_id, pixel[0], pixel[1]))

    order = np.argsort([record[0] for record in camera_records])
    registered_relative = np.array(
        [camera_records[int(old_id)][0] for old_id in order], dtype=np.int64
    )
    poses_initial = np.vstack([camera_records[int(old_id)][1] for old_id in order])
    observations = np.asarray(observation_records, dtype=np.float64)
    observation_order = np.lexsort((observations[:, 1], observations[:, 0]))
    observations = observations[observation_order]

    rmse_before = _reprojection_rmse(
        poses_initial, points, intrinsics, observations
    )
    if refine:
        poses_final, points_final = bundle_adjust(
            poses_initial, points, intrinsics, observations
        )
    else:
        poses_final, points_final = poses_initial, points
    rmse_after = _reprojection_rmse(poses_final, points_final, intrinsics, observations)

    track_lengths = np.bincount(
        observations[:, 1].astype(np.int64), minlength=len(points_final)
    )
    point_errors = np.zeros(len(points_final), dtype=np.float64)
    for point_id in range(len(points_final)):
        rows = observations[observations[:, 1] == point_id]
        errors = []
        for row in rows:
            camera_id = int(row[0])
            predicted = _project(
                points_final[point_id : point_id + 1],
                poses_final[camera_id],
                intrinsics,
            )[0]
            errors.append(np.linalg.norm(predicted - row[2:4]))
        point_errors[point_id] = float(np.mean(errors))
    landmark_confidence = track_lengths / (1.0 + point_errors)
    registered_paths = tuple(str(paths[index]) for index in registered_relative)
    return SfmResult(
        points=points_final,
        poses_world_to_camera=_poses_to_matrices(poses_final),
        observations=observations,
        image_paths=registered_paths,
        registered_image_indices=absolute_indices[registered_relative],
        initial_pair=(0, initialization.second_index),
        reprojection_rmse_before_px=rmse_before,
        reprojection_rmse_after_px=rmse_after,
        track_lengths=track_lengths,
        initial_landmark_count=initial_landmark_count,
        triangulation_sources=np.asarray(triangulation_sources, dtype=np.int64),
        landmark_confidence=landmark_confidence,
    )


def run_sfm(
    image_dir: str | Path, K: np.ndarray, **kwargs: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse points and world-to-camera SE(3) matrices.

    Use run_sfm_detailed when registration indices, observations, and
    reprojection diagnostics are needed.
    """
    result = run_sfm_detailed(image_dir, K, **kwargs)
    return result.points, result.poses_world_to_camera


def _demo() -> None:
    raise SystemExit("Call run_sfm_detailed with a real image directory and calibrated K")


if __name__ == "__main__":
    _demo()
