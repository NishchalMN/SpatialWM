"""Tests for spatialwm.perception.lidar_io — fully implemented."""

from __future__ import annotations

import numpy as np

from spatialwm.perception.lidar_io import (
    load_kitti_bin,
    load_kitti_points,
    load_semantickitti_labels,
)


def test_load_kitti_bin_roundtrip(tmp_path):
    arr = np.array(
        [[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.9]], dtype=np.float32
    )
    p = tmp_path / "scan.bin"
    arr.tofile(p)
    loaded = load_kitti_bin(str(p))
    assert loaded.shape == (2, 4)
    np.testing.assert_array_equal(loaded, arr)


def test_load_kitti_points_xyz_only(tmp_path):
    arr = np.array(
        [[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.9]], dtype=np.float32
    )
    p = tmp_path / "scan.bin"
    arr.tofile(p)
    pts = load_kitti_points(str(p))
    assert pts.shape == (2, 3)
    np.testing.assert_array_equal(pts, arr[:, :3])


def test_load_semantickitti_labels_decode(tmp_path):
    # semantic in lower 16 bits, instance in upper 16 bits
    sem_in = np.array([10, 40, 252], dtype=np.uint32)
    inst_in = np.array([0, 3, 7], dtype=np.uint32)
    raw = (inst_in << 16) | sem_in
    p = tmp_path / "scan.label"
    raw.astype(np.uint32).tofile(p)

    semantic, instance = load_semantickitti_labels(str(p))
    np.testing.assert_array_equal(semantic, sem_in)
    np.testing.assert_array_equal(instance, inst_in)
