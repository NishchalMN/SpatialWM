"""Normalized sensor records for the portfolio geometry pipeline.

The adapters intentionally describe files and coordinate transforms without
loading whole sequences into memory.  Algorithms can therefore consume the
same ordered records while retaining dataset-specific provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from spatialwm.geometry.tartanair import parse_pose_to_transform


@dataclass(frozen=True)
class CalibrationEdge:
    """Rigid transform mapping points from ``source`` into ``target``."""

    source: str
    target: str
    transform: np.ndarray

    def __post_init__(self) -> None:
        _validate_se3(self.transform, "transform")


@dataclass(frozen=True)
class SensorFrame:
    """One time-indexed sensor sample with explicit provenance."""

    dataset: str
    sequence: str
    frame_id: int
    timestamp_ns: int
    modality: str
    coordinate_frame: str
    path: str
    pose_source: str | None = None


@dataclass(frozen=True)
class PoseEstimate:
    """Estimated transform plus enough metadata to audit its meaning."""

    source_frame_id: int
    target_frame_id: int
    transform_source_to_target: np.ndarray
    method: str
    scale_mode: str
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_se3(self.transform_source_to_target, "transform_source_to_target")
        if self.scale_mode not in {"metric", "monocular", "unknown"}:
            raise ValueError("scale_mode must be metric, monocular, or unknown")
        if not all(np.isfinite(value) for value in self.diagnostics.values()):
            raise ValueError("diagnostics must contain only finite values")


@dataclass(frozen=True)
class SensorSequence:
    """A validated ordered multi-modal sequence and its calibration graph."""

    dataset: str
    sequence: str
    frames: tuple[SensorFrame, ...]
    calibrations: tuple[CalibrationEdge, ...] = ()
    intrinsics: dict[str, np.ndarray] = field(default_factory=dict)
    poses_camera_to_world: dict[int, np.ndarray] = field(default_factory=dict)

    def validate(self, *, require_modalities: set[str] | None = None) -> dict[str, Any]:
        if not self.frames:
            raise ValueError("sequence must contain at least one sensor frame")
        missing = [frame.path for frame in self.frames if not Path(frame.path).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} sensor files are missing; first: {missing[0]}")

        by_modality: dict[str, list[SensorFrame]] = {}
        for frame in self.frames:
            if frame.dataset != self.dataset or frame.sequence != self.sequence:
                raise ValueError("all frames must match the sequence dataset and name")
            by_modality.setdefault(frame.modality, []).append(frame)
        for modality, records in by_modality.items():
            ids = [record.frame_id for record in records]
            timestamps = [record.timestamp_ns for record in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate frame IDs for modality {modality}")
            if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
                raise ValueError(f"timestamps must increase strictly for modality {modality}")

        modalities = set(by_modality)
        if require_modalities and not require_modalities.issubset(modalities):
            missing_modalities = sorted(require_modalities - modalities)
            raise ValueError(f"required modalities are missing: {missing_modalities}")
        for name, matrix in self.intrinsics.items():
            array = np.asarray(matrix)
            if array.shape != (3, 3) or not np.all(np.isfinite(array)):
                raise ValueError(f"intrinsics {name} must be a finite 3x3 matrix")
        for frame_id, transform in self.poses_camera_to_world.items():
            _validate_se3(transform, f"pose for frame {frame_id}")

        return {
            "dataset": self.dataset,
            "sequence": self.sequence,
            "frames": len(self.frames),
            "modalities": {key: len(value) for key, value in sorted(by_modality.items())},
            "calibration_edges": len(self.calibrations),
            "poses": len(self.poses_camera_to_world),
            "valid": True,
        }


def _validate_se3(transform: np.ndarray, name: str) -> None:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")


def _parse_timestamp_ns(value: str) -> int:
    clean = value.strip()
    try:
        parsed = datetime.fromisoformat(clean)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {clean}") from exc
    return int(parsed.timestamp() * 1_000_000_000)


def _read_kitti_calibration(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key] = np.fromstring(raw, sep=" ", dtype=np.float64)
        except ValueError:
            continue
    return values


def _rigid_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def build_tartanair_sequence(
    root: str | Path,
    *,
    start: int = 0,
    count: int = 30,
    stride: int = 1,
    focal_px: float = 320.0,
) -> SensorSequence:
    """Describe a bounded TartanAir left-camera RGB-D sequence."""
    base = Path(root)
    image_dir = base / "image_left"
    depth_dir = base / "depth_left"
    image_paths = sorted(image_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.npy"))
    poses = np.loadtxt(base / "pose_left.txt")
    end = min(len(image_paths), len(depth_paths), len(poses))
    indices = list(range(start, end, stride))[:count]
    if not indices:
        raise ValueError("requested TartanAir slice is empty")

    frames: list[SensorFrame] = []
    pose_map: dict[int, np.ndarray] = {}
    for order, index in enumerate(indices):
        timestamp_ns = order * 100_000_000  # TartanAir trajectories are sampled at 10 Hz.
        frames.extend(
            [
                SensorFrame(
                    "TartanAir",
                    base.name,
                    index,
                    timestamp_ns,
                    "rgb",
                    "camera_left",
                    str(image_paths[index]),
                    "pose_left",
                ),
                SensorFrame(
                    "TartanAir",
                    base.name,
                    index,
                    timestamp_ns,
                    "depth",
                    "camera_left",
                    str(depth_paths[index]),
                    "pose_left",
                ),
            ]
        )
        pose_map[index] = parse_pose_to_transform(np.asarray(poses[index], dtype=np.float64))

    first_image = cv2.imread(str(image_paths[indices[0]]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"could not read {image_paths[indices[0]]}")
    height, width = first_image.shape[:2]
    K = np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    sequence = SensorSequence(
        "TartanAir",
        base.name,
        tuple(frames),
        intrinsics={"camera_left": K},
        poses_camera_to_world=pose_map,
    )
    sequence.validate(require_modalities={"rgb", "depth"})
    return sequence


def build_kitti_sequence(
    root: str | Path,
    *,
    date: str = "2011_09_26",
    drive: str = "0005",
    count: int = 100,
) -> SensorSequence:
    """Describe synchronized KITTI camera, Velodyne, and OXTS records."""
    root_path = Path(root)
    date_dir = root_path / date
    drive_dir = date_dir / f"{date}_drive_{drive}_sync"
    timestamp_lines = (drive_dir / "oxts" / "timestamps.txt").read_text().splitlines()
    available = min(count, len(timestamp_lines))
    frames: list[SensorFrame] = []
    for index in range(available):
        timestamp_ns = _parse_timestamp_ns(timestamp_lines[index])
        records = [
            ("lidar", "velodyne", drive_dir / "velodyne_points" / "data" / f"{index:010d}.bin"),
            ("gps_imu", "imu", drive_dir / "oxts" / "data" / f"{index:010d}.txt"),
        ]
        image_path = drive_dir / "image_02" / "data" / f"{index:010d}.png"
        if image_path.exists():
            records.append(("rgb", "camera_02", image_path))
        for modality, coordinate_frame, path in records:
            frames.append(
                SensorFrame(
                    "KITTI",
                    f"{date}_drive_{drive}",
                    index,
                    timestamp_ns,
                    modality,
                    coordinate_frame,
                    str(path),
                    "OXTS" if modality == "gps_imu" else None,
                )
            )

    velo = _read_kitti_calibration(date_dir / "calib_velo_to_cam.txt")
    cam = _read_kitti_calibration(date_dir / "calib_cam_to_cam.txt")
    if not {"R", "T"}.issubset(velo) or "P_rect_02" not in cam:
        raise ValueError("KITTI calibration is missing R/T or P_rect_02")
    T_velo_cam0 = _rigid_from_rt(velo["R"], velo["T"])
    rectification = np.eye(4)
    rectification[:3, :3] = cam.get("R_rect_00", np.eye(3).reshape(-1)).reshape(3, 3)
    T_velo_rect0 = rectification @ T_velo_cam0
    projection = cam["P_rect_02"].reshape(3, 4)
    K = projection[:, :3]
    baseline_transform = np.eye(4)
    baseline_transform[0, 3] = projection[0, 3] / projection[0, 0]
    T_velo_cam2 = baseline_transform @ T_velo_rect0
    sequence = SensorSequence(
        "KITTI",
        f"{date}_drive_{drive}",
        tuple(frames),
        calibrations=(CalibrationEdge("velodyne", "camera_02", T_velo_cam2),),
        intrinsics={"camera_02": K},
    )
    sequence.validate(require_modalities={"lidar", "gps_imu"})
    return sequence


def project_lidar_to_camera(
    points_velodyne: np.ndarray,
    T_velodyne_to_camera: np.ndarray,
    K: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project Velodyne XYZ points to image pixels, returning pixels and depth."""
    points = np.asarray(points_velodyne, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_velodyne must have shape (N, 3)")
    _validate_se3(T_velodyne_to_camera, "T_velodyne_to_camera")
    intrinsics = np.asarray(K, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("K must have shape (3, 3)")
    height, width = image_shape
    camera = points @ T_velodyne_to_camera[:3, :3].T + T_velodyne_to_camera[:3, 3]
    positive = camera[:, 2] > 1e-6
    camera = camera[positive]
    pixels_h = camera @ intrinsics.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    inside = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    return pixels[inside], camera[inside, 2]


def write_sequence_manifest(
    sequence: SensorSequence,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write a portable Parquet manifest and human-readable validation JSON."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(frame) for frame in sequence.frames]
    manifest = output / f"{sequence.dataset.lower()}_{sequence.sequence}_manifest.parquet"
    report = output / f"{sequence.dataset.lower()}_{sequence.sequence}_validation.json"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    report.write_text(json.dumps(sequence.validate(), indent=2) + "\n")
    return manifest, report
