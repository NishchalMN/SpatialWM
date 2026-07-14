from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatialwm.data.sensors import (
    CalibrationEdge,
    PoseEstimate,
    SensorFrame,
    SensorSequence,
    project_lidar_to_camera,
    write_sequence_manifest,
)


def test_sequence_validates_and_writes_manifest(tmp_path: Path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"sample")
    sequence = SensorSequence(
        "synthetic",
        "sequence",
        (
            SensorFrame("synthetic", "sequence", 0, 1, "lidar", "sensor", str(sample)),
            SensorFrame("synthetic", "sequence", 1, 2, "lidar", "sensor", str(sample)),
        ),
        calibrations=(CalibrationEdge("sensor", "world", np.eye(4)),),
    )
    report = sequence.validate(require_modalities={"lidar"})
    assert report["modalities"] == {"lidar": 2}
    manifest, validation = write_sequence_manifest(sequence, tmp_path / "processed")
    assert manifest.is_file()
    assert validation.is_file()


def test_sequence_rejects_duplicate_ids_and_non_monotonic_time(tmp_path: Path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"sample")
    sequence = SensorSequence(
        "synthetic",
        "bad",
        (
            SensorFrame("synthetic", "bad", 0, 2, "rgb", "camera", str(sample)),
            SensorFrame("synthetic", "bad", 0, 1, "rgb", "camera", str(sample)),
        ),
    )
    with pytest.raises(ValueError, match="duplicate frame IDs"):
        sequence.validate()


def test_pose_estimate_validates_scale_and_transform():
    estimate = PoseEstimate(1, 0, np.eye(4), "icp", "metric", {"fitness": 0.9})
    assert estimate.method == "icp"
    with pytest.raises(ValueError, match="scale_mode"):
        PoseEstimate(1, 0, np.eye(4), "icp", "pixels")


def test_lidar_projection_keeps_only_positive_visible_points():
    points = np.array(
        [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.0, -2.0], [10.0, 0.0, 1.0]]
    )
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    pixels, depth = project_lidar_to_camera(points, np.eye(4), K, (80, 120))
    np.testing.assert_allclose(pixels, [[50.0, 40.0], [100.0, 40.0]])
    np.testing.assert_allclose(depth, [2.0, 2.0])
