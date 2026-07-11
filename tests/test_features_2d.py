"""
Tests for spatialwm.perception.features_2d — SKIPPED (needs DINOv2 download).

Collectible but skipped until DINOv2 model weights are available.
"""

from __future__ import annotations

import pytest

from spatialwm.perception.features_2d import dino_patch_features


@pytest.mark.skip(reason="Requires DINOv2 model weights via torch.hub")
def test_dino_patch_features_output_shape():
    """
    dino_patch_features returns a torch.Tensor of shape (B, N_patches, D).
    Requires torch.hub + internet access to download DINOv2.
    """
    import numpy as np

    # Synthetic RGB image batch: (1, 3, 224, 224)
    images = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)
    features = dino_patch_features(images, model_name="dinov2_vits14")
    assert features.ndim == 3, f"Expected 3-D (B, N, D), got {features.ndim}-D"
    assert features.shape[0] == 1


@pytest.mark.skip(reason="Requires DINOv2 model weights via torch.hub")
def test_dino_features_are_finite():
    """DINOv2 patch features must not contain NaN or Inf."""
    import numpy as np
    import torch

    images = np.random.randint(0, 255, (2, 3, 224, 224), dtype=np.uint8)
    features = dino_patch_features(images, model_name="dinov2_vits14")
    assert torch.all(torch.isfinite(features)), "DINOv2 features contain NaN/Inf"
