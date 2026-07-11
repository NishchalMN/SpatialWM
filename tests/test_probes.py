"""
Tests for spatialwm.worldmodel.probes — SKIPPED (needs trained model / labeled data).

Collectible but skipped until Week 3+ resources are available.
"""

from __future__ import annotations

import pytest
import torch

from spatialwm.worldmodel.probes import LinearProbe, MLPProbe, train_probe


@pytest.mark.skip(reason="Week 3 — needs trained model and labeled data")
def test_linear_probe_output_shape():
    """LinearProbe(in_dim=128, out_dim=10) forward -> (B, 10)."""
    probe = LinearProbe(in_dim=128, out_dim=10)
    x = torch.randn(4, 128)
    out = probe(x)
    assert out.shape == (4, 10)


@pytest.mark.skip(reason="Week 3 — needs trained model and labeled data")
def test_mlp_probe_output_shape():
    """MLPProbe(128, 64, 10) forward -> (B, 10)."""
    probe = MLPProbe(in_dim=128, hidden=64, out_dim=10)
    x = torch.randn(4, 128)
    out = probe(x)
    assert out.shape == (4, 10)


@pytest.mark.skip(reason="Week 3 — needs trained model and labeled data")
def test_train_probe_returns_dict():
    """train_probe returns a dict with 'val_loss' or similar metric."""
    probe = LinearProbe(in_dim=16, out_dim=2)
    latents = torch.randn(100, 16)
    labels = torch.randint(0, 2, (100,))
    result = train_probe(probe, latents, labels)
    assert isinstance(result, dict)
