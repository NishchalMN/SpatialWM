"""Voxelization: discretize point clouds to occupancy grids and BEV."""

from dataclasses import dataclass

import numpy as np


@dataclass
class OccGrid:
    """Occupancy grid representation."""

    origin: np.ndarray
    voxel: float
    grid: np.ndarray


def voxelize(points: np.ndarray, voxel: float = 0.05) -> OccGrid:
    """
    Discretize points to occupancy grid.

    Args:
        points: (N, 3) point cloud
        voxel: voxel size in meters

    Returns:
        OccGrid with origin, voxel size, and occupancy grid
    """
    raise NotImplementedError


def bev(points: np.ndarray, cell: float = 0.1) -> np.ndarray:
    """
    Bird's-eye-view rasterization (top-down occupancy/height).

    Args:
        points: (N, 3) point cloud
        cell: cell size in meters

    Returns:
        (H, W) BEV grid
    """
    raise NotImplementedError


def _demo():
    """Demo: voxelize random points."""
    N = 10000
    points = np.random.randn(N, 3).astype(np.float32) * 10.0

    try:
        occ_grid = voxelize(points, voxel=0.1)
        print(
            f"Voxelized: origin={occ_grid.origin}, voxel={occ_grid.voxel}, "
            f"grid shape={occ_grid.grid.shape}"
        )
    except NotImplementedError:
        print("voxelize not implemented yet")


if __name__ == "__main__":
    _demo()
