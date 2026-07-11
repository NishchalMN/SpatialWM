from dataclasses import dataclass

import numpy as np


@dataclass
class RansacResult:
    """Result container for RANSAC algorithm."""
    model: np.ndarray
    inliers: np.ndarray
    n_iters: int
    inlier_ratio: float


def ransac(
    data,
    fit_fn,
    score_fn,
    min_samples: int,
    thresh: float,
    p_success: float = 0.99,
    max_iters: int = 5000,
) -> RansacResult:
    """RANSAC algorithm with adaptive iteration count.
    
    Adaptive N = log(1-p)/log(1-w^s) updating from best inlier ratio.
    
    Args:
        data: Input data for model fitting
        fit_fn: Function to fit model to samples
        score_fn: Function to score all data points against model
        min_samples: Minimum number of samples for model fitting
        thresh: Inlier threshold
        p_success: Desired probability of success (default 0.99)
        max_iters: Maximum number of iterations (default 5000)
        
    Returns:
        RansacResult containing model, inliers, iterations, and inlier ratio
    """
    raise NotImplementedError


def _demo():
    """Demo: call ransac with dummy fit/score lambdas (will raise)."""
    # Dummy data
    data = np.random.randn(100, 2)
    
    # Dummy fit and score functions
    def fit_fn(samples):
        return np.mean(samples, axis=0)

    def score_fn(model, data):
        return np.linalg.norm(data - model, axis=1)

    
    # Call ransac (will raise NotImplementedError)
    result = ransac(data, fit_fn, score_fn, min_samples=2, thresh=1.0)
    print(f"RANSAC completed with {result.n_iters} iterations")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
