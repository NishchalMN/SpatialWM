"""
Tests for spatialwm.perception.elevation — raises NotImplementedError until implemented.

Contracts defended:
1. dsm, dtm, ndsm, slope all return (H, W) arrays for a given point cloud.
2. ndsm ≈ 0 on a flat plane (dsm == dtm everywhere).
3. dsm has its max at the cell containing a single peak.
4. slope ≈ 0 on a flat DSM grid; > 0 on a sloped surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialwm.perception.elevation import dsm, dtm, ndsm, slope

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_plane(cell: float = 1.0, extent: int = 5, z_val: float = 0.0) -> np.ndarray:
    """
    A flat grid of points at constant z=z_val, covering [0, extent) × [0, extent).
    10 points per cell, all at exactly z_val.
    """
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, extent, 500)
    ys = rng.uniform(0, extent, 500)
    zs = np.full(500, z_val, dtype=float)
    return np.column_stack([xs, ys, zs])


def _peak_cloud(cell: float = 1.0) -> tuple[np.ndarray, int, int]:
    """
    Flat plane at z=0 plus one peak at position (~3.5, ~3.5, 10.0).
    Returns cloud, expected peak row idx, expected peak col idx.
    """
    flat = _flat_plane(cell=cell, extent=8, z_val=0.0)
    peak_xy = np.array([[3.5, 3.5, 10.0]])
    cloud = np.vstack([flat, peak_xy])
    # Cell index = floor(coord / cell)
    peak_row = int(3.5 / cell)
    peak_col = int(3.5 / cell)
    return cloud, peak_row, peak_col


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDsm:
    def test_output_is_2d(self):
        """dsm returns a 2-D array."""
        pts = _flat_plane()
        result = dsm(pts, cell=1.0)
        assert result.ndim == 2, f"Expected 2-D, got {result.ndim}-D"

    def test_flat_plane_dsm_is_constant(self):
        """All cells of a flat plane at z=5 have DSM max-Z ≈ 5."""
        pts = _flat_plane(z_val=5.0)
        result = dsm(pts, cell=1.0)
        # Every occupied cell should have value ≈ 5.0
        # (unoccupied cells may be NaN or 0; we check the non-NaN cells)
        occupied = result[~np.isnan(result)]
        np.testing.assert_allclose(occupied, 5.0, atol=1e-6)

    def test_peak_at_correct_cell(self):
        """DSM max is at the cell containing the peak point."""
        cloud, pr, pc = _peak_cloud(cell=1.0)
        result = dsm(cloud, cell=1.0)
        max_idx = np.unravel_index(np.nanargmax(result), result.shape)
        assert max_idx == (pr, pc) or result[pr, pc] == pytest.approx(10.0, abs=1e-6), (
            f"DSM max not at peak cell ({pr},{pc}): "
            f"max at {max_idx}, value at peak={result[pr, pc]}"
        )


class TestDtm:
    def test_output_is_2d(self):
        pts = _flat_plane()
        result = dtm(pts, cell=1.0)
        assert result.ndim == 2

    def test_flat_plane_dtm_approx_z(self):
        """DTM of flat plane at z=3 returns ≈ 3 in occupied cells."""
        pts = _flat_plane(z_val=3.0)
        result = dtm(pts, cell=1.0)
        occupied = result[~np.isnan(result)]
        np.testing.assert_allclose(occupied, 3.0, atol=0.1)


class TestNdsm:
    def test_output_is_2d(self):
        pts = _flat_plane()
        result = ndsm(pts, cell=1.0)
        assert result.ndim == 2

    def test_flat_plane_ndsm_near_zero(self):
        """nDSM of a flat plane (no objects above ground) is ≈ 0 everywhere."""
        pts = _flat_plane(z_val=2.0)
        result = ndsm(pts, cell=1.0)
        occupied = result[~np.isnan(result)]
        np.testing.assert_allclose(occupied, 0.0, atol=0.5)

    def test_ndsm_equals_dsm_minus_dtm(self):
        """nDSM == DSM - DTM pointwise (definition check)."""
        pts = _flat_plane()
        d = dsm(pts, cell=1.0)
        g = dtm(pts, cell=1.0)
        n = ndsm(pts, cell=1.0)

        # Mask to cells where all three have valid values
        valid = ~(np.isnan(d) | np.isnan(g) | np.isnan(n))
        if valid.sum() > 0:
            np.testing.assert_allclose(n[valid], (d - g)[valid], atol=1e-6)


class TestSlope:
    def test_output_is_2d(self):
        """slope of any (H,W) grid returns (H,W) array."""
        grid = np.zeros((10, 10))
        result = slope(grid)
        assert result.ndim == 2
        assert result.shape == grid.shape

    def test_flat_grid_has_zero_slope(self):
        """Constant DSM -> slope is 0 everywhere (interior)."""
        grid = np.full((10, 10), 5.0)
        result = slope(grid)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_ramp_has_nonzero_slope(self):
        """A linear ramp DSM has nonzero slope."""
        xs = np.arange(10, dtype=float)
        grid = np.tile(xs, (10, 1))  # columns increase linearly
        result = slope(grid)
        assert np.any(result > 0), "Ramp DSM should have nonzero slope"

    def test_shape_preserved(self):
        """slope preserves input shape exactly."""
        grid = np.random.rand(7, 13)
        result = slope(grid)
        assert result.shape == grid.shape
