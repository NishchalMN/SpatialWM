import numpy as np


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D world points to 2D image pixels.
    
    Equation: x ~ K[R|t]X
    
    Derives the homogeneous projection pipeline from world coordinates to pixel coordinates.
    
    Args:
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector (world to camera)
        X: (N,3) world coordinates
        
    Returns:
        (N,2) pixel coordinates
    """
    extrinsics = np.hstack((R,t))

    # projection matrix
    P = np.dot(K, extrinsics)  # (3,4)

    # converting to homogeneous coordinates
    X_h = np.hstack((X, np.ones((X.shape[0], 1))))  # (N,4)

    x_h = X_h @ P.T

    # converting from homogeneous back to normal/cartesian coordinates
    # Using 2:3 for (N,1) instead of -1 for (N,)
    x = x_h[:,:2] / x_h[:, 2:3]

    return x


def unproject(K: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Unproject 2D pixels with depth to 3D camera frame coordinates.
    
    Equation: X = d·K^{-1}[u,v,1]^T
    
    Args:
        K: 3x3 camera intrinsic matrix
        uv: (N,2) pixel coordinates
        depth: (N,) depth values
        
    Returns:
        (N,3) points in camera frame
    """
    # this is going from image plane to camera coor system (not world coor which needs extrinsics)
    uv_h = np.hstack((uv, np.ones((uv.shape[0],1))))
    
    X_cam = uv_h @ np.linalg.inv(K).T

    # since while projecting we divide by depth, here we scale by depth
    X_cam = X_cam * depth[:, np.newaxis]

    return X_cam

def transform_points(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply SE(3) transformation to 3D points.
    
    Applies a 4x4 SE(3) transformation matrix to (N,3) points.
    
    Args:
        T: 4x4 SE(3) transformation matrix
        X: (N,3) points
        
    Returns:
        (N,3) transformed points
    """
    X_h = np.hstack((X, np.ones((X.shape[0],1))))
    
    # After applying rotation and transformation in a rigid way (hence the name SE(3))
    X_t = X_h @ T.T

    # converting back from homo to cartetian coor
    return X_t[:, :-1] / X_t[:, 3:4]


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute camera center in world coordinates.
    
    Equation: C = -R^T t
    
    Args:
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector (world to camera)
        
    Returns:
        3x1 camera center in world coordinates
    """
    return np.dot(-R.T, t)


def _demo():
    """Demo: build synthetic K, R, t, and X, then call project."""
    # Synthetic intrinsics
    K = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Identity rotation
    R = np.eye(3)
    
    # Translation
    t = np.array([[0.0], [0.0], [5.0]])
    
    # Random 3D points
    X = np.random.randn(10, 3)
    X[:, 2] += 10.0  # Push points in front of camera
    
    breakpoint()
    # Call project (will raise NotImplementedError)
    project(K, R, t, X)
    print(f"Projected {len(X)} points to pixels")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
