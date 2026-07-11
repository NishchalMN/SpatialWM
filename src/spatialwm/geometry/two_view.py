import numpy as np


def normalize_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize points using Hartley normalization.
    
    Hartley normalization: centroid -> 0, mean distance sqrt(2).
    
    Args:
        x: (N,2) or (N,3) points
        
    Returns:
        x_norm: normalized points
        T: transformation matrix
    """
    raise NotImplementedError


def fundamental_8pt(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute fundamental matrix using normalized 8-point algorithm.
    
    Normalized 8-point algorithm with rank-2 enforcement via SVD.
    
    Args:
        x1: (N,2) points in image 1
        x2: (N,2) points in image 2
        
    Returns:
        3x3 fundamental matrix F
    """
    raise NotImplementedError


def essential_from_F(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Compute essential matrix from fundamental matrix.
    
    Equation: E = K2^T F K1
    
    Args:
        F: 3x3 fundamental matrix
        K1: 3x3 intrinsic matrix for camera 1
        K2: 3x3 intrinsic matrix for camera 2
        
    Returns:
        3x3 essential matrix E
    """
    raise NotImplementedError


def decompose_E(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decompose essential matrix into 4 candidate (R, t) pairs.
    
    4 candidates via U W V^T, enforce det(R)=+1.
    
    Args:
        E: 3x3 essential matrix
        
    Returns:
        List of 4 (R, t) tuples
    """
    raise NotImplementedError


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Triangulate 3D points using Direct Linear Transform.
    
    Linear DLT per point.
    
    Args:
        P1: 3x4 projection matrix for camera 1
        P2: 3x4 projection matrix for camera 2
        x1: (N,2) points in image 1
        x2: (N,2) points in image 2
        
    Returns:
        (N,3) triangulated 3D points
    """
    raise NotImplementedError


def cheirality_select(
    cands: list, K1: np.ndarray, K2: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select correct (R, t) candidate using cheirality check.
    
    Positive depth in both cameras.
    
    Args:
        cands: List of (R, t) candidate tuples
        K1: 3x3 intrinsic matrix for camera 1
        K2: 3x3 intrinsic matrix for camera 2
        x1: (N,2) points in image 1
        x2: (N,2) points in image 2
        
    Returns:
        (R, t) tuple with positive depth in both cameras
    """
    raise NotImplementedError


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute Sampson distance for epipolar constraint.
    
    First-order geometric error for RANSAC scoring.
    
    Args:
        F: 3x3 fundamental matrix
        x1: (N,2) points in image 1
        x2: (N,2) points in image 2
        
    Returns:
        (N,) Sampson distances
    """
    raise NotImplementedError


def _demo():
    """Demo: call fundamental_8pt on synthetic correspondences."""
    # Generate synthetic correspondences
    n_points = 20
    x1 = np.random.randn(n_points, 2) * 100 + 320
    x2 = x1 + np.random.randn(n_points, 2) * 5  # Add small noise
    
    # Call fundamental_8pt (will raise NotImplementedError)
    fundamental_8pt(x1, x2)
    print(f"Computed fundamental matrix from {n_points} correspondences")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
