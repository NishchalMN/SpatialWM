"""KITTI / SemanticKITTI I/O utilities.

KITTI velodyne scans are flat float32 binaries of [x, y, z, reflectance]
records. SemanticKITTI labels are uint32 per point; the lower 16 bits are the
semantic class, the upper 16 bits the instance id.
"""

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


def load_semantickitti_labels(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load SemanticKITTI per-point labels.

    Args:
        path: path to a `*.label` file (uint32 per point).

    Returns:
        (semantic, instance) — two (N,) uint32 arrays. `semantic` is the lower
        16 bits (class id); `instance` is the upper 16 bits.
    """
    raw = np.fromfile(path, dtype=np.uint32)
    semantic = raw & 0xFFFF
    instance = raw >> 16
    return semantic, instance
