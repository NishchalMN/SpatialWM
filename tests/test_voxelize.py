"""
Tests for the voxel occupancy and BEV coordinate contracts.

Contracts defended:
1. voxelize() marks occupied cells correctly for known point positions.
2. bev() returns a 2-D array of the expected shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialwm.perception.voxelize import OccGrid, bev, voxelize

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVoxelize:
    def test_returns_occgrid_type(self):
        """voxelize returns an OccGrid dataclass instance."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        result = voxelize(pts, voxel=0.1)
        assert isinstance(result, OccGrid)

    def test_grid_is_3d(self):
        """OccGrid.grid is a 3-D array."""
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, (50, 3))
        occ = voxelize(pts, voxel=0.1)
        assert occ.grid.ndim == 3, f"Expected 3-D grid, got {occ.grid.ndim}-D"

    def test_known_point_marks_cell_occupied(self):
        """A single known point at the origin lands in an occupied cell."""
        # Place one point at exactly the center of a voxel
        pt = np.array([[0.05, 0.05, 0.05]])  # inside voxel [0:0.1, 0:0.1, 0:0.1]
        occ = voxelize(pt, voxel=0.1)

        # At least one cell must be occupied
        assert occ.grid.sum() >= 1, "No occupied cells despite non-empty input"

    def test_origin_and_voxel_stored(self):
        """OccGrid stores origin and voxel_size attributes."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        occ = voxelize(pts, voxel=0.5)
        assert occ.voxel == pytest.approx(0.5)
        assert occ.origin is not None
        assert len(occ.origin) == 3

    def test_more_points_at_least_as_many_occupied_cells(self):
        """
        A denser point cloud covering more space should activate >= cells
        than a sparser one (occupation can only grow or stay the same).
        """
        rng = np.random.default_rng(1)
        pts_sparse = rng.uniform(0, 0.5, (10, 3))
        pts_dense = rng.uniform(0, 2.0, (500, 3))

        occ_sparse = voxelize(pts_sparse, voxel=0.1)
        occ_dense = voxelize(pts_dense, voxel=0.1)

        n_sparse = int(occ_sparse.grid.sum())
        n_dense = int(occ_dense.grid.sum())
        assert n_dense >= n_sparse, (
            f"Dense cloud ({n_dense}) fewer occupied cells than sparse ({n_sparse})"
        )

    def test_voxelize_invalid_inputs(self):
        """voxelize raises ValueError or TypeError for invalid inputs."""
        # Not a numpy array
        with pytest.raises(TypeError):
            voxelize([[0.0, 0.0, 0.0]], voxel=0.1)

        # Invalid shape (N, 4) instead of (N, 3)
        pts_invalid_shape = np.array([[0.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            voxelize(pts_invalid_shape, voxel=0.1)

        # Empty point cloud (0, 3)
        pts_empty = np.zeros((0, 3))
        with pytest.raises(ValueError):
            voxelize(pts_empty, voxel=0.1)

        # Non-finite values
        pts_nan = np.array([[0.0, np.nan, 0.0]])
        with pytest.raises(ValueError):
            voxelize(pts_nan, voxel=0.1)

        pts_inf = np.array([[0.0, np.inf, 0.0]])
        with pytest.raises(ValueError):
            voxelize(pts_inf, voxel=0.1)

        # Non-positive voxel size
        pts = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            voxelize(pts, voxel=0.0)
        with pytest.raises(ValueError):
            voxelize(pts, voxel=-0.1)


class TestBev:
    def test_returns_2d_array(self):
        """bev returns a 2-D array."""
        rng = np.random.default_rng(2)
        pts = rng.uniform(0, 5.0, (200, 3))
        result = bev(pts, cell=0.5)
        assert result.ndim == 2, f"Expected 2-D BEV, got {result.ndim}-D"

    def test_expected_shape_for_known_extent(self):
        """
        For points in [0, N) × [0, N) × (anything), BEV with cell=1.0 has
        approximately N×N cells.
        """
        N = 10
        # Place one point per XY cell-center; z random
        rng = np.random.default_rng(3)
        xs = np.arange(N, dtype=float) + 0.5
        ys = np.arange(N, dtype=float) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel(), rng.uniform(0, 1, N * N)])

        bev_grid = bev(pts, cell=1.0)

        # Shape should be at least N×N (may have 1-2 extra border cells)
        assert bev_grid.shape[0] >= N
        assert bev_grid.shape[1] >= N

    def test_nonempty_for_nonempty_points(self):
        """BEV of a non-empty point cloud is not all-zero."""
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        result = bev(pts, cell=1.0)
        assert result.sum() > 0, "BEV of non-empty cloud should have occupied cells"

    def test_bev_invalid_inputs(self):
        """bev raises ValueError or TypeError for invalid inputs."""
        # Not a numpy array
        with pytest.raises(TypeError):
            bev([[0.0, 0.0, 0.0]], cell=0.1)

        # Invalid shape
        pts_invalid_shape = np.array([[0.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            bev(pts_invalid_shape, cell=0.1)

        # Empty point cloud
        pts_empty = np.zeros((0, 3))
        with pytest.raises(ValueError):
            bev(pts_empty, cell=0.1)

        # Non-finite values
        pts_nan = np.array([[0.0, np.nan, 0.0]])
        with pytest.raises(ValueError):
            bev(pts_nan, cell=0.1)

        pts_inf = np.array([[0.0, np.inf, 0.0]])
        with pytest.raises(ValueError):
            bev(pts_inf, cell=0.1)

        # Non-positive cell size
        pts = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            bev(pts, cell=0.0)
        with pytest.raises(ValueError):
            bev(pts, cell=-0.1)
