"""
Tests for spatialwm.worldmodel.predictor — SKIPPED (needs cached latents/trained model).

These tests are collectible but skipped until Week 2+ resources are available.
"""

from __future__ import annotations

import pytest
import torch

from spatialwm.worldmodel.predictor import LatentPredictor


@pytest.mark.skip(reason="Week 2 — needs cached latents/trained model")
def test_predictor_none_conditioning_output_shape():
    """LatentPredictor(conditioning='none') forward -> same shape as input latents."""
    model = LatentPredictor(dim=64, depth=2, heads=4, conditioning="none")
    latents = torch.randn(2, 10, 64)
    out = model(latents)
    assert out.shape == latents.shape


@pytest.mark.skip(reason="Week 2 — needs cached latents/trained model")
def test_predictor_token_conditioning_with_pose():
    """LatentPredictor(conditioning='token') forward with delta_pose."""
    model = LatentPredictor(dim=64, depth=2, heads=4, conditioning="token")
    latents = torch.randn(2, 10, 64)
    delta_pose = torch.randn(2, 10, 6)
    out = model(latents, delta_pose)
    assert out.shape == latents.shape


@pytest.mark.skip(reason="Week 2 — needs cached latents/trained model")
def test_predictor_film_conditioning():
    """LatentPredictor(conditioning='film') integrates pose via FiLM."""
    model = LatentPredictor(dim=64, depth=2, heads=4, conditioning="film")
    latents = torch.randn(2, 8, 64)
    delta_pose = torch.randn(2, 8, 6)
    out = model(latents, delta_pose)
    assert out.shape == latents.shape


@pytest.mark.skip(reason="Week 2 — needs cached latents/trained model")
def test_invalid_conditioning_rejected():
    """LatentPredictor with invalid conditioning raises an error."""
    with pytest.raises((AssertionError, ValueError)):
        LatentPredictor(dim=64, depth=2, heads=4, conditioning="invalid")
