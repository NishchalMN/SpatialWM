"""Tests for spatialwm.perception.lidar_io — fully implemented."""

from __future__ import annotations

import numpy as np

from spatialwm.perception.lidar_io import load_kitti_bin, load_kitti_points


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
