"""
Minimal JEPA (Joint-Embedding Predictive Architecture) components.

Based on V-JEPA / I-JEPA principles: context encoder + predictor + target encoder (EMA).
"""

import torch
import torch.nn as nn
from torch import Tensor


class ContextEncoder(nn.Module):
    """Encodes context frames to representation space."""
    
    def __init__(self, dim: int = 384, patch_size: int = 16):
        """
        Initialize context encoder.
        
        Args:
            dim: Feature dimension
            patch_size: Patch size for tokenization
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode context frames to representation.
        
        Args:
            x: Input frames [B, T, C, H, W]
        
        Returns:
            Context tokens [B, T, N, D]
        """
        raise NotImplementedError


class Predictor(nn.Module):
    """Predicts target representations from context tokens + target-position queries."""
    
    def __init__(self, dim: int = 384, depth: int = 6, heads: int = 6):
        """
        Initialize predictor.
        
        Args:
            dim: Feature dimension
            depth: Transformer depth
            heads: Number of attention heads
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
    
    def forward(self, ctx_tokens: Tensor, target_pos: Tensor) -> Tensor:
        """
        Predict target representations from context tokens + target-position queries.
        
        Args:
            ctx_tokens: Context tokens [B, N_ctx, D]
            target_pos: Target position queries [B, N_target, D]
        
        Returns:
            Predicted target representations [B, N_target, D]
        """
        raise NotImplementedError


def ema_update(target_enc: nn.Module, ctx_enc: nn.Module, m: float = 0.996) -> None:
    """
    EMA target encoder update: θ_t <- m θ_t + (1-m) θ_c (stop-grad target).
    
    Args:
        target_enc: Target encoder (updated in-place)
        ctx_enc: Context encoder (momentum source)
        m: Momentum coefficient (default 0.996)
    """
    raise NotImplementedError


def jepa_loss(pred: Tensor, target_repr: Tensor) -> Tensor:
    """
    JEPA loss: || pred - stopgrad(target) ||^2 (or smooth-L1) in representation space.
    
    Args:
        pred: Predicted representations [B, N, D]
        target_repr: Target representations [B, N, D] (will be stop-grad'd)
    
    Returns:
        Scalar loss
    """
    raise NotImplementedError


def _demo():
    """Demo instantiation and forward call (raises NotImplementedError)."""
    print("Creating ContextEncoder...")
    encoder = ContextEncoder(dim=384, patch_size=16)
    print(f"✓ ContextEncoder created: dim={encoder.dim}")
    
    print("\nCreating Predictor...")
    predictor = Predictor(dim=384, depth=6, heads=6)
    print(f"✓ Predictor created: dim={predictor.dim}, depth={predictor.depth}")
    
    print("\nTesting forward pass (will raise NotImplementedError)...")
    dummy_frames = torch.randn(2, 4, 3, 224, 224)
    try:
        encoder(dummy_frames)
    except NotImplementedError:
        print("✓ ContextEncoder.forward() raises NotImplementedError as expected")
    
    dummy_ctx = torch.randn(2, 100, 384)
    dummy_target_pos = torch.randn(2, 16, 384)
    try:
        predictor(dummy_ctx, dummy_target_pos)
    except NotImplementedError:
        print("✓ Predictor.forward() raises NotImplementedError as expected")
    
    print("\nTesting ema_update (will raise NotImplementedError)...")
    target_enc = ContextEncoder(dim=384)
    ctx_enc = ContextEncoder(dim=384)
    try:
        ema_update(target_enc, ctx_enc)
    except NotImplementedError:
        print("✓ ema_update() raises NotImplementedError as expected")
    
    print("\nTesting jepa_loss (will raise NotImplementedError)...")
    pred = torch.randn(2, 16, 384)
    target = torch.randn(2, 16, 384)
    try:
        jepa_loss(pred, target)
    except NotImplementedError:
        print("✓ jepa_loss() raises NotImplementedError as expected")


if __name__ == "__main__":
    _demo()
