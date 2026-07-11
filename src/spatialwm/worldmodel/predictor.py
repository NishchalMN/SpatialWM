"""
Latent-space predictor for DINO-WM style frame-level causal world modeling.

Operates on frame-level latent tokens (not patch-level AR).
Supports pose conditioning via multiple modes: none, token, FiLM, cross-attention.
"""

import torch
import torch.nn as nn
from torch import Tensor


class LatentPredictor(nn.Module):
    """
    Causal transformer over frame-level latent tokens (DINO-WM-style).
    
    Predicts future latent representations conditioned on past frames + optional Δpose.
    Δpose is R6 representation (translation + rotation vector, normalized).
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        conditioning: str = "none"
    ):
        """
        Initialize latent predictor.
        
        Args:
            dim: Feature dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            conditioning: Pose conditioning mode, one of:
                - 'none': No pose conditioning
                - 'token': Pose as learnable token embeddings
                - 'film': FiLM modulation (scale/shift)
                - 'xattn': Cross-attention to pose embeddings
        """
        super().__init__()
        
        assert conditioning in {'none', 'token', 'film', 'xattn'}, \
            f"conditioning must be one of {{none, token, film, xattn}}, got {conditioning}"
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.conditioning = conditioning
    
    def forward(self, latents: Tensor, delta_pose: Tensor | None = None) -> Tensor:
        """
        Causal transformer over frame-level latent tokens (DINO-WM-style).
        
        Frame-level, not token-level AR: each frame is represented by a single
        latent token, and the model predicts the next frame's latent autoregressively.
        
        Args:
            latents: Frame-level latent tokens [B, T, D]
            delta_pose: Pose deltas in R6 representation (translation + rotvec, normalized)
                       [B, T, 6] or None. Injected per conditioning mode.
        
        Returns:
            Predicted latents [B, T, D]
        """
        raise NotImplementedError


def _demo():
    """Demo instantiation and forward call (raises NotImplementedError)."""
    print("Creating LatentPredictor with 'none' conditioning...")
    predictor_none = LatentPredictor(dim=384, depth=4, heads=6, conditioning="none")
    print(f"✓ LatentPredictor created: dim={predictor_none.dim}, depth={predictor_none.depth}, "
          f"heads={predictor_none.heads}, conditioning={predictor_none.conditioning}")
    
    print("\nCreating LatentPredictor with 'film' conditioning...")
    predictor_film = LatentPredictor(dim=384, depth=4, heads=6, conditioning="film")
    print(f"✓ LatentPredictor created with conditioning={predictor_film.conditioning}")
    
    print("\nCreating LatentPredictor with 'xattn' conditioning...")
    predictor_xattn = LatentPredictor(dim=384, depth=4, heads=6, conditioning="xattn")
    print(f"✓ LatentPredictor created with conditioning={predictor_xattn.conditioning}")
    
    print("\nTesting forward pass without pose (will raise NotImplementedError)...")
    dummy_latents = torch.randn(2, 10, 384)
    try:
        predictor_none(dummy_latents)
    except NotImplementedError:
        print("✓ LatentPredictor.forward() raises NotImplementedError as expected")
    
    print("\nTesting forward pass with pose (will raise NotImplementedError)...")
    dummy_pose = torch.randn(2, 10, 6)
    try:
        predictor_film(dummy_latents, dummy_pose)
    except NotImplementedError:
        print("✓ LatentPredictor.forward(with pose) raises NotImplementedError as expected")
    
    print("\nTesting invalid conditioning mode...")
    try:
        LatentPredictor(dim=384, depth=4, heads=6, conditioning="invalid")
        print("✗ Should have raised assertion error")
    except AssertionError as e:
        print(f"✓ Invalid conditioning rejected: {e}")


if __name__ == "__main__":
    _demo()
