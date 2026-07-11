"""PointNet: permutation-invariant point cloud classification."""

import torch
import torch.nn as nn


class PointNet(nn.Module):
    """PointNet architecture for point cloud classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass: shared-MLP -> max-pool (permutation invariance) -> classification head.

        Args:
            x: (B, N, 3) point cloud batch

        Returns:
            (B, num_classes) logits
        """
        raise NotImplementedError


def _demo():
    """Demo: instantiate PointNet and forward pass."""
    model = PointNet(num_classes=10)
    print(f"PointNet instantiated: {model.num_classes} classes")

    B, N = 4, 1024
    x = torch.randn(B, N, 3)

    try:
        logits = model(x)
        print(f"Output logits shape: {logits.shape}")
    except NotImplementedError:
        print("PointNet.forward not implemented yet")


if __name__ == "__main__":
    _demo()
