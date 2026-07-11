"""Elevation models: DSM, DTM, nDSM, slope for terrain analysis."""

import numpy as np


def dsm(points: np.ndarray, cell: float) -> np.ndarray:
    """
    Digital Surface Model: per-cell max-Z.

    Args:
        points: (N, 3) point cloud
        cell: cell size in meters

    Returns:
        (H, W) grid with max elevation per cell
    """
    raise NotImplementedError


def dtm(points: np.ndarray, cell: float) -> np.ndarray:
    """
    Digital Terrain Model: per-cell grid-min + median filter.

    Poor-man's ground filter; note CSF/PMF are the real methods.

    Args:
        points: (N, 3) point cloud
        cell: cell size in meters

    Returns:
        (H, W) grid with ground elevation per cell
    """
    raise NotImplementedError


def ndsm(points: np.ndarray, cell: float) -> np.ndarray:
    """
    Normalized Digital Surface Model: nDSM = DSM - DTM.

    Args:
        points: (N, 3) point cloud
        cell: cell size in meters

    Returns:
        (H, W) grid with normalized height (above ground)
    """
    raise NotImplementedError


def slope(dsm_grid: np.ndarray) -> np.ndarray:
    """
    Slope magnitude via np.gradient.

    Args:
        dsm_grid: (H, W) DSM grid

    Returns:
        (H, W) slope magnitude grid
    """
    raise NotImplementedError


def _demo():
    """Demo: compute DSM on random points."""
    N = 10000
    points = np.random.randn(N, 3).astype(np.float32) * 10.0

    try:
        dsm_grid = dsm(points, cell=0.5)
        print(f"DSM grid shape: {dsm_grid.shape}")
    except NotImplementedError:
        print("dsm not implemented yet")


if __name__ == "__main__":
    _demo()
