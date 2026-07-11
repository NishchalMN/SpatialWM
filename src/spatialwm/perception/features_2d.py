"""2D feature extraction with DINOv2."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def dino_patch_features(images, model_name: str = "dinov2_vits14") -> "torch.Tensor":
    """
    Load DINOv2 via torch.hub INSIDE the fn, extract patch features.

    Args:
        images: input images
        model_name: DINOv2 model name

    Returns:
        Patch features tensor
    """
    raise NotImplementedError("Requires DINOv2 model weights via torch.hub")


def _demo():
    """Demo: call dino_patch_features (raises)."""
    try:
        features = dino_patch_features(None)
        print(f"Features shape: {features.shape}")
    except NotImplementedError as e:
        print(f"dino_patch_features not implemented: {e}")


if __name__ == "__main__":
    _demo()
