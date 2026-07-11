"""
Backbone model loading and frame encoding utilities.

DINOv2-based frozen backbone for extracting patch-level features.
Requires DINOv2 model weights; functions raise NotImplementedError until weights are available.
"""

import torch
import torch.nn as nn
from torch import Tensor


def load_backbone(name: str = "dinov2_vits14", frozen: bool = True) -> nn.Module:
    """
    Load frozen DINOv2 backbone via torch.hub.
    
    Loads pre-trained DINOv2 model for visual feature extraction.
    Typically frozen for world model training.
    
    Args:
        name: Model name (e.g., "dinov2_vits14", "dinov2_vitb14")
        frozen: Whether to freeze backbone parameters
    
    Returns:
        DINOv2 model
    
    Note:
        torch.hub.load() is called INSIDE this function, not at module level,
        to avoid network calls on import.
    """
    raise NotImplementedError("Requires DINOv2 model weights via torch.hub")


def encode_frames(model: nn.Module, frames: Tensor) -> Tensor:
    """
    Per-frame patch-feature encoding using backbone.
    
    Extracts patch-level features from input frames using a frozen backbone.
    
    Args:
        model: Backbone model (e.g., DINOv2)
        frames: Input frames [B, T, C, H, W]
    
    Returns:
        Patch features [B, T, N_patches, D] where N_patches depends on
        image size and patch size (e.g., 224x224 with patch_size=14 -> 256 patches)
    """
    raise NotImplementedError("Requires DINOv2 model weights via torch.hub")


def _demo():
    """Demo backbone loading (raises NotImplementedError)."""
    print("Attempting to load DINOv2 backbone (will raise NotImplementedError)...")
    try:
        backbone = load_backbone(name="dinov2_vits14", frozen=True)
        print(f"✓ Backbone loaded: {backbone}")
    except NotImplementedError as e:
        print(f"✓ load_backbone() raises NotImplementedError: {e}")
    
    print("\nAttempting to encode frames (will raise NotImplementedError)...")
    dummy_model = nn.Identity()
    dummy_frames = torch.randn(2, 4, 3, 224, 224)
    try:
        encode_frames(dummy_model, dummy_frames)
    except NotImplementedError as e:
        print(f"✓ encode_frames() raises NotImplementedError: {e}")


if __name__ == "__main__":
    _demo()
