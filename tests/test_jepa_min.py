"""
Tests for spatialwm.worldmodel.jepa_min — raises NotImplementedError until implemented.

Contracts defended:
1. jepa_loss(pred, target) == 0 when pred == target (L2 in representation space).
2. ema_update moves target encoder params toward ctx encoder params (momentum update).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spatialwm.worldmodel.jepa_min import ema_update, jepa_loss

# ---------------------------------------------------------------------------
# Minimal nn.Module for ema_update tests
# ---------------------------------------------------------------------------

class _TinyLinear(nn.Module):
    """Tiny single-layer module for testing EMA param movement."""

    def __init__(self, fill: float):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        nn.init.constant_(self.linear.weight, fill)


# ---------------------------------------------------------------------------
# jepa_loss
# ---------------------------------------------------------------------------

class TestJepaLoss:
    def test_zero_when_pred_equals_target(self):
        """
        jepa_loss(x, x) == 0.
        L2 distance between identical tensors is 0; stop-grad doesn't change values.
        """
        x = torch.randn(2, 8, 16)
        loss = jepa_loss(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"jepa_loss(pred, pred) should be 0, got {loss.item()}"
        )

    def test_positive_for_different_tensors(self):
        """jepa_loss > 0 when pred != target."""
        pred = torch.randn(2, 8, 16)
        target = torch.randn(2, 8, 16)
        loss = jepa_loss(pred, target)
        assert loss.item() > 0.0, (
            f"jepa_loss(pred, different_target) should be > 0, got {loss.item()}"
        )

    def test_returns_scalar(self):
        """jepa_loss returns a 0-d tensor (scalar)."""
        pred = torch.randn(3, 5, 12)
        target = torch.randn(3, 5, 12)
        loss = jepa_loss(pred, target)
        assert loss.ndim == 0, f"Expected scalar tensor, got shape {loss.shape}"

    def test_loss_is_nonnegative(self):
        """jepa_loss is always >= 0 (it's a distance/norm-based loss)."""
        pred = torch.randn(4, 6, 10)
        target = torch.randn(4, 6, 10)
        assert jepa_loss(pred, target).item() >= 0.0

    def test_loss_scales_with_difference_magnitude(self):
        """Larger pred-target gap -> larger loss."""
        pred = torch.zeros(2, 4, 8)
        target_near = torch.ones(2, 4, 8) * 0.01
        target_far = torch.ones(2, 4, 8) * 10.0

        loss_near = jepa_loss(pred, target_near).item()
        loss_far = jepa_loss(pred, target_far).item()
        assert loss_near < loss_far, (
            f"Expected loss_near < loss_far, got {loss_near:.4f} vs {loss_far:.4f}"
        )


# ---------------------------------------------------------------------------
# ema_update
# ---------------------------------------------------------------------------

class TestEmaUpdate:
    def test_params_move_toward_ctx(self):
        """
        After ema_update(target, ctx, m=0.5):
        new_target_param = 0.5 * old_target + 0.5 * ctx
        So |new - ctx| < |old - ctx|.
        """
        ctx = _TinyLinear(fill=1.0)   # ctx params all = 1.0
        target = _TinyLinear(fill=0.0)  # target params all = 0.0

        # Before update: distance from target to ctx = 1.0 per param
        dist_before = (
            target.linear.weight.data - ctx.linear.weight.data
        ).abs().mean().item()

        ema_update(target, ctx, m=0.5)

        dist_after = (
            target.linear.weight.data - ctx.linear.weight.data
        ).abs().mean().item()

        assert dist_after < dist_before, (
            f"EMA update did not move target toward ctx: "
            f"dist_before={dist_before:.4f}, dist_after={dist_after:.4f}"
        )

    def test_exact_ema_value_at_m_zero(self):
        """
        m=0 means full copy: target_param <- ctx_param.
        """
        ctx = _TinyLinear(fill=3.0)
        target = _TinyLinear(fill=0.0)

        ema_update(target, ctx, m=0.0)

        torch.testing.assert_close(
            target.linear.weight.data,
            ctx.linear.weight.data,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_exact_ema_value_at_m_one(self):
        """
        m=1 means no change: target_param stays the same.
        """
        target = _TinyLinear(fill=7.0)
        ctx = _TinyLinear(fill=0.0)
        original_weights = target.linear.weight.data.clone()

        ema_update(target, ctx, m=1.0)

        torch.testing.assert_close(
            target.linear.weight.data,
            original_weights,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_ctx_params_unchanged_after_ema(self):
        """ema_update must not modify the ctx encoder's parameters."""
        ctx = _TinyLinear(fill=2.0)
        target = _TinyLinear(fill=0.0)
        ctx_weights_before = ctx.linear.weight.data.clone()

        ema_update(target, ctx, m=0.9)

        torch.testing.assert_close(
            ctx.linear.weight.data,
            ctx_weights_before,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_update_under_no_grad(self):
        """ema_update should work even when autograd is disabled."""
        ctx = _TinyLinear(fill=1.0)
        target = _TinyLinear(fill=0.0)
        with torch.no_grad():
            ema_update(target, ctx, m=0.5)
        # Should not raise; result is the EMA blend
        expected = 0.5 * 0.0 + 0.5 * 1.0
        torch.testing.assert_close(
            target.linear.weight.data,
            torch.full_like(target.linear.weight.data, expected),
            atol=1e-6,
            rtol=1e-6,
        )
