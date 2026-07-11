import numpy as np


def reprojection_residuals(
    params: np.ndarray, n_cams: int, n_pts: int, K: np.ndarray, obs: np.ndarray
) -> np.ndarray:
    """Compute reprojection residuals for bundle adjustment.
    
    Pack/unpack poses (6D: rvec+t) + points.
    
    Args:
        params: Flattened parameter vector containing camera poses and 3D points
        n_cams: Number of cameras
        n_pts: Number of 3D points
        K: 3x3 camera intrinsic matrix
        obs: Observed 2D point measurements
        
    Returns:
        Residual vector
    """
    raise NotImplementedError


def bundle_adjust(
    poses0: np.ndarray, X0: np.ndarray, K: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Perform bundle adjustment using sparse least squares.
    
    Uses scipy.optimize.least_squares (trf/lm) with Jacobian sparsity pattern.
    The Schur complement exists due to the sparse structure where each 3D point
    appears in only a subset of camera views, allowing efficient marginalization.
    
    Args:
        poses0: (M,6) initial camera poses (rvec+t)
        X0: (N,3) initial 3D points
        K: 3x3 camera intrinsic matrix
        obs: Observed 2D point measurements
        
    Returns:
        poses: Optimized camera poses
        X: Optimized 3D points
    """
    raise NotImplementedError


def _demo():
    """Demo: build a tiny synthetic BA problem and call bundle_adjust."""
    # Synthetic problem
    n_cams = 3
    n_pts = 10
    
    # Random initial poses (rvec + t)
    poses0 = np.random.randn(n_cams, 6) * 0.1
    
    # Random 3D points
    X0 = np.random.randn(n_pts, 3)
    X0[:, 2] += 10.0  # Push points away from origin
    
    # Camera intrinsics
    K = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Dummy observations
    obs = np.random.randn(n_cams * n_pts, 2) * 100 + 320
    
    # Call bundle_adjust (will raise NotImplementedError)
    poses, X = bundle_adjust(poses0, X0, K, obs)
    print(f"Bundle adjustment completed for {n_cams} cameras and {n_pts} points")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
