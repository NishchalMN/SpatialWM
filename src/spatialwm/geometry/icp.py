import numpy as np


def icp_point2point(
    src: np.ndarray, dst: np.ndarray, max_iters: int = 50, tol: float = 1e-6
) -> tuple[np.ndarray, list[float]]:
    """Iterative Closest Point algorithm for point cloud registration.
    
    Algorithm:
    1. Find nearest neighbors using cKDTree
    2. Center point clouds by removing centroids
    3. Compute cross-covariance: H = Σ p q^T
    4. SVD decomposition of H
    5. Compute rotation: R = V U^T (with det(R)=+1 fix)
    6. Iterate until convergence
    
    Args:
        src: (N,3) source point cloud
        dst: (M,3) destination point cloud
        max_iters: Maximum number of iterations (default 50)
        tol: Convergence tolerance (default 1e-6)
        
    Returns:
        T: 4x4 transformation matrix
        errors: List of error values per iteration
    """
    raise NotImplementedError


def _demo():
    """Demo: call icp on two synthetic clouds."""
    # Generate synthetic point clouds
    n_points = 100
    
    # Source cloud
    src = np.random.randn(n_points, 3)
    
    # Destination: rotated and translated version of source
    angle = np.pi / 6
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t_true = np.array([1.0, 2.0, 0.5])
    dst = (R_true @ src.T).T + t_true
    
    # Add noise
    dst += np.random.randn(n_points, 3) * 0.01
    
    # Call icp (will raise NotImplementedError)
    T, errors = icp_point2point(src, dst)
    print(f"ICP completed with {len(errors)} iterations")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
