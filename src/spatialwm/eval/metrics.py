"""Evaluation metrics for spatial world models."""

import numpy as np

from .trajectory import ate, rpe

__all__ = [
    "ate",
    "rpe",
    "chamfer",
    "miou",
    "depth_rmse",
    "depth_delta",
    "occupancy_iou",
    "reliability_curve",
]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between point sets.
    
    Computes mean of nearest-neighbor distances in both directions:
    CD(A,B) = mean(min_b ||a-b||) + mean(min_a ||b-a||)
    
    Args:
        a: First point set (N, 3)
        b: Second point set (M, 3)
    
    Returns:
        Symmetric Chamfer distance
    """
    raise NotImplementedError


def miou(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    """Mean IoU over classes.
    
    Computes Intersection-over-Union for each class and averages.
    
    Args:
        pred: Predicted class labels (N,)
        gt: Ground truth class labels (N,)
        num_classes: Total number of classes
    
    Returns:
        Mean IoU across all classes
    """
    raise NotImplementedError


def depth_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """RMSE of depth.
    
    Args:
        pred: Predicted depth map
        gt: Ground truth depth map
    
    Returns:
        Root mean squared error
    """
    raise NotImplementedError


def depth_delta(
    pred: np.ndarray,
    gt: np.ndarray,
    thresh: float = 1.25,
) -> float:
    """δ<1.25 accuracy (fraction with max(p/g,g/p)<thresh).
    
    Measures percentage of pixels where relative depth error is within threshold.
    Standard metric from depth estimation literature.
    
    Args:
        pred: Predicted depth map
        gt: Ground truth depth map
        thresh: Threshold for ratio test (default 1.25)
    
    Returns:
        Fraction of pixels passing threshold test
    """
    raise NotImplementedError


def occupancy_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU of occupancy grids.
    
    Computes Intersection-over-Union for binary occupancy grids.
    
    Args:
        pred: Predicted occupancy (binary or probability)
        gt: Ground truth occupancy (binary)
    
    Returns:
        IoU score
    """
    raise NotImplementedError


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration/reliability curve (confidence vs accuracy per bin).
    
    Bins predictions by confidence and computes empirical accuracy in each bin.
    Used to assess model calibration.
    
    Args:
        probs: Predicted probabilities (N,)
        labels: True binary labels (N,)
        bins: Number of confidence bins
    
    Returns:
        mean_confidence: Mean predicted probability per bin
        accuracy: Empirical accuracy per bin
    """
    raise NotImplementedError


def _demo():
    """Demo Chamfer distance with random point clouds."""
    a = np.random.randn(100, 3)
    b = np.random.randn(150, 3) + 0.1
    
    dist = chamfer(a, b)
    print(f"Chamfer distance: {dist:.4f}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
