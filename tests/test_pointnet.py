"""
Tests for spatialwm.perception.pointnet — RED now, green on impl.

Contract defended:
    PointNet(num_classes=10).forward(x) returns (B, 10) logits
    when given (B, N, 3) point cloud tensor.
"""

from __future__ import annotations

import torch

from spatialwm.perception.pointnet import PointNet


class TestPointNet:
    def test_instantiates_without_error(self):
        """PointNet(num_classes=10) constructs without raising."""
        model = PointNet(num_classes=10)
        assert model.num_classes == 10

    def test_forward_output_shape_batch_4(self):
        """forward((4, 128, 3)) -> (4, 10) logits."""
        model = PointNet(num_classes=10)
        x = torch.randn(4, 128, 3)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_forward_output_shape_batch_1(self):
        """forward((1, 64, 3)) -> (1, 10) logits (batch=1 edge case)."""
        model = PointNet(num_classes=10)
        x = torch.randn(1, 64, 3)
        out = model(x)
        assert out.shape == (1, 10), f"Expected (1, 10), got {out.shape}"

    def test_forward_custom_num_classes(self):
        """num_classes=5 produces (B, 5) output."""
        model = PointNet(num_classes=5)
        x = torch.randn(2, 100, 3)
        out = model(x)
        assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"

    def test_forward_output_is_finite(self):
        """Output logits must be finite (no NaN/Inf from pathological init)."""
        model = PointNet(num_classes=10)
        x = torch.randn(3, 64, 3)
        out = model(x)
        assert torch.all(torch.isfinite(out)), "PointNet output contains NaN or Inf"

    def test_permutation_invariance(self):
        """
        Permuting the points in a cloud must not change the class logits.
        This is the core contract of PointNet.
        """
        torch.manual_seed(0)
        model = PointNet(num_classes=10)
        model.eval()

        x = torch.randn(1, 64, 3)
        perm = torch.randperm(64)
        x_perm = x[:, perm, :]

        with torch.no_grad():
            out = model(x)
            out_perm = model(x_perm)

        torch.testing.assert_close(out, out_perm, atol=1e-5, rtol=1e-5)
