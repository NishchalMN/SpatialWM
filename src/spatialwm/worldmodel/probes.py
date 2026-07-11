"""
Probing modules for evaluating learned world model representations.

Linear and MLP probes for predicting downstream targets from latents:
- Future depth maps (TartanAir GT)
- Future BEV occupancy grids
"""

import torch
import torch.nn as nn
from torch import Tensor


class LinearProbe(nn.Module):
    """Linear probe: maps latent representations to target outputs."""
    
    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize linear probe.
        
        Args:
            in_dim: Input latent dimension
            out_dim: Output target dimension
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Linear map: latent -> target.
        
        Args:
            x: Input latents [B, in_dim] or [B, T, in_dim]
        
        Returns:
            Target predictions [B, out_dim] or [B, T, out_dim]
        """
        raise NotImplementedError


class MLPProbe(nn.Module):
    """2-layer MLP probe for richer capacity."""
    
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        """
        Initialize MLP probe.
        
        Args:
            in_dim: Input latent dimension
            hidden: Hidden layer size
            out_dim: Output target dimension
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
    
    def forward(self, x: Tensor) -> Tensor:
        """
        2-layer MLP: latent -> hidden -> target.
        
        Args:
            x: Input latents [B, in_dim] or [B, T, in_dim]
        
        Returns:
            Target predictions [B, out_dim] or [B, T, out_dim]
        """
        raise NotImplementedError


def train_probe(
    probe: nn.Module,
    latents: Tensor,
    targets: Tensor,
    **kwargs
) -> dict:
    """
    Fit probe on frozen latents to predict targets.
    
    Targets include:
    - Future depth maps (TartanAir ground truth)
    - Future BEV occupancy grids
    
    Args:
        probe: Probe module (LinearProbe or MLPProbe)
        latents: Frozen latent representations [B, T, D]
        targets: Ground truth targets (depth/occupancy) [B, T, ...]
        **kwargs: Training hyperparameters (lr, epochs, etc.)
    
    Returns:
        Dictionary with training metrics and results
    """
    raise NotImplementedError


def _demo():
    """Demo instantiation and forward call (raises NotImplementedError)."""
    print("Creating LinearProbe...")
    linear_probe = LinearProbe(in_dim=384, out_dim=10)
    print(f"✓ LinearProbe created: in_dim={linear_probe.in_dim}, out_dim={linear_probe.out_dim}")
    
    print("\nCreating MLPProbe...")
    mlp_probe = MLPProbe(in_dim=384, hidden=512, out_dim=10)
    print(f"✓ MLPProbe created: in_dim={mlp_probe.in_dim}, hidden={mlp_probe.hidden}, "
          f"out_dim={mlp_probe.out_dim}")
    
    print("\nTesting LinearProbe forward (will raise NotImplementedError)...")
    dummy_latents = torch.randn(4, 384)
    try:
        linear_probe(dummy_latents)
    except NotImplementedError:
        print("✓ LinearProbe.forward() raises NotImplementedError as expected")
    
    print("\nTesting MLPProbe forward (will raise NotImplementedError)...")
    try:
        mlp_probe(dummy_latents)
    except NotImplementedError:
        print("✓ MLPProbe.forward() raises NotImplementedError as expected")
    
    print("\nTesting train_probe (will raise NotImplementedError)...")
    dummy_latents_seq = torch.randn(8, 16, 384)
    dummy_targets = torch.randn(8, 16, 10)
    try:
        train_probe(linear_probe, dummy_latents_seq, dummy_targets, lr=1e-3, epochs=10)
    except NotImplementedError:
        print("✓ train_probe() raises NotImplementedError as expected")


if __name__ == "__main__":
    _demo()
