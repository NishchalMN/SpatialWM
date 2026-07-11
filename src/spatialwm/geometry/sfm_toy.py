import numpy as np


def run_sfm(image_dir: str, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Incremental Structure from Motion pipeline.
    
    Pipeline:
    1. Extract SIFT features from all images
    2. Match features with ratio test
    3. Verify matches using RANSAC + fundamental matrix
    4. Initialize with best pair
    5. Triangulate initial 3D points
    6. For each new view:
       - Run solvePnPRansac to estimate pose
       - Triangulate new points
    7. Run bundle adjustment every k views
    
    Args:
        image_dir: Directory containing input images
        K: 3x3 camera intrinsic matrix
        
    Returns:
        points: (N,3) reconstructed 3D points
        poses: (M,4,4) camera poses as SE(3) matrices
    """
    raise NotImplementedError


def _demo():
    """Demo: call run_sfm on a dummy path."""
    # Dummy camera intrinsics
    K = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Call run_sfm (will raise NotImplementedError)
    points, poses = run_sfm("/dummy/path/to/images", K)
    print(f"SfM reconstructed {len(points)} points from {len(poses)} views")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
