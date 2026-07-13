"""Voxelization: discretize point clouds to occupancy grids and BEV."""

from dataclasses import dataclass

import numpy as np


@dataclass
class OccGrid:
    """Occupancy grid representation."""

    origin: np.ndarray
    voxel: float
    grid: np.ndarray


def _validate_inputs(points: np.ndarray, cell: float) -> None:
    """Validate that points is a finite, non-empty (N, 3) numpy array and cell is positive."""
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points cannot be empty")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")
    if cell <= 0:
        raise ValueError("voxel/cell size must be positive")


def voxelize(points: np.ndarray, voxel: float = 0.05) -> OccGrid:
    """
    Discretize points to occupancy grid.

    Args:
        points: (N, 3) point cloud
        voxel: voxel size in meters

    Returns:
        OccGrid with origin, voxel size, and occupancy grid
    """
    _validate_inputs(points, voxel)

    # Compute a tight grid origin as the minimum coordinate of all points
    origin = np.min(points, axis=0)

    # Compute maximum coordinates to determine grid shape
    max_coords = np.max(points, axis=0)

    # Compute grid size in each dimension
    grid_size = np.floor((max_coords - origin) / voxel).astype(np.intp) + 1

    # Create the boolean grid
    grid = np.zeros(grid_size, dtype=bool)

    # Compute voxel indices for each point
    indices = np.floor((points - origin) / voxel).astype(np.intp)

    # Clip to be safe against floating point boundary issues
    indices = np.clip(indices, 0, grid_size - 1)

    # Mark occupied cells
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    return OccGrid(origin=origin, voxel=voxel, grid=grid)


def bev(points: np.ndarray, cell: float = 0.1) -> np.ndarray:
    """
    Bird's-eye-view rasterization (top-down occupancy/height).

    Semantics:
        - The 2D BEV grid has shape (H, W), mapping to the Y and X axes of the input point cloud.
        - Dimension 0 (height, H) corresponds to the Y-axis. The row index 0 corresponds to the
          maximum Y-coordinate (top-down view, matching standard image conventions).
        - Dimension 1 (width, W) corresponds to the X-axis. The column index 0 corresponds to the
          minimum X-coordinate (left-to-right view).
        - Returns a boolean grid where a cell is True if it contains at least one point.

    Args:
        points: (N, 3) point cloud
        cell: cell size in meters

    Returns:
        (H, W) BEV grid
    """
    _validate_inputs(points, cell)

    # Extract X and Y coordinates
    x = points[:, 0]
    y = points[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Compute shape of BEV grid
    H = int(np.floor((y_max - y_min) / cell)) + 1
    W = int(np.floor((x_max - x_min) / cell)) + 1

    # Create the BEV occupancy grid
    bev_grid = np.zeros((H, W), dtype=bool)

    # Map Y to row index (top-down: y_max corresponds to row 0, y_min to row H-1)
    y_idx = np.floor((y_max - y) / cell).astype(np.intp)
    # Map X to column index (left-to-right: x_min corresponds to column 0, x_max to column W-1)
    x_idx = np.floor((x - x_min) / cell).astype(np.intp)

    # Clip to be safe against floating point boundary issues
    y_idx = np.clip(y_idx, 0, H - 1)
    x_idx = np.clip(x_idx, 0, W - 1)

    # Mark cells containing points as True
    bev_grid[y_idx, x_idx] = True

    return bev_grid


def _demo():
    """Demo: voxelize random points."""
    np.random.seed(42)
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

