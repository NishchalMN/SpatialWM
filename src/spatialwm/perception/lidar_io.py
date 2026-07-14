"""KITTI Velodyne scan I/O utilities."""

from __future__ import annotations

import numpy as np


def load_kitti_bin(path: str) -> np.ndarray:
    """Load a KITTI velodyne `.bin` scan.

    Args:
        path: path to a `*.bin` velodyne file.

    Returns:
        (N,4) float32 array of [x, y, z, reflectance].
    """
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_kitti_points(path: str) -> np.ndarray:
    """Load only the XYZ coordinates of a KITTI velodyne scan.

    Args:
        path: path to a `*.bin` velodyne file.

    Returns:
        (N,3) float32 array of [x, y, z].
    """
    return load_kitti_bin(path)[:, :3]
