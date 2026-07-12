import numpy as np
import open3d as o3d


def register_point_clouds(
    src: np.ndarray,
    dst: np.ndarray,
    max_correspondence_distance: float = 0.5,
    max_iters: int = 50,
    tol: float = 1e-6,
    init: np.ndarray | None = None,
) -> o3d.pipelines.registration.RegistrationResult:
    """Register source point cloud to target using Open3D ICP point-to-point.

    Args:
        src: (N, 3) source point cloud as float array.
        dst: (M, 3) target point cloud as float array.
        max_correspondence_distance: Maximum search distance for correspondences.
        max_iters: Maximum number of ICP iterations.
        tol: Convergence tolerance.
        init: (4, 4) initial transformation matrix.

    Returns:
        RegistrationResult: The Open3D registration result.
    """
    # 1. Validation
    if not isinstance(src, np.ndarray) or not isinstance(dst, np.ndarray):
        raise TypeError("src and dst must be numpy arrays")
    if src.ndim != 2 or src.shape[1] != 3 or dst.ndim != 2 or dst.shape[1] != 3:
        raise ValueError("src and dst must be (N,3) arrays")
    if src.shape[0] < 3 or dst.shape[0] < 3:
        raise ValueError("src and dst must contain at least 3 points")
    if not np.issubdtype(src.dtype, np.floating) or not np.issubdtype(dst.dtype, np.floating):
        raise TypeError("src and dst must be of floating type")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise ValueError("src and dst must contain only finite values")

    if (
        not isinstance(max_correspondence_distance, (int, float))
        or max_correspondence_distance <= 0
    ):
        raise ValueError("max_correspondence_distance must be positive")
    if not isinstance(max_iters, (int, np.integer)) or max_iters <= 0:
        raise ValueError("max_iters must be a positive integer")
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError("tol must be positive")

    if init is not None:
        if not isinstance(init, np.ndarray):
            raise TypeError("init must be a numpy array")
        if init.shape != (4, 4):
            raise ValueError("init must be of shape (4,4)")
        if not np.issubdtype(init.dtype, np.floating):
            raise TypeError("init must be of floating type")
        if not np.all(np.isfinite(init)):
            raise ValueError("init must contain only finite values")

    # 2. Conversion to Open3D PointCloud
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src)
    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(dst)

    # 3. Call Open3D ICP
    init_trans = init if init is not None else np.identity(4)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=tol,
        relative_rmse=tol,
        max_iteration=max_iters,
    )

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, dst_pcd, max_correspondence_distance, init_trans, estimation, criteria
    )
    return result


def icp_point2point(
    src: np.ndarray, dst: np.ndarray, max_iters: int = 50, tol: float = 1e-6
) -> tuple[np.ndarray, list[float]]:
    """Iterative Closest Point algorithm wrapper around Open3D point-to-point registration.

    Open3D does not return a per-iteration error/RMSE curve, so the returned errors
    list is a single-element list containing the final inlier RMSE summary.

    Args:
        src: (N,3) source point cloud
        dst: (M,3) destination point cloud
        max_iters: Maximum number of iterations (default 50)
        tol: Convergence tolerance (default 1e-6)

    Returns:
        T: 4x4 transformation matrix
        errors: List containing the single final float inlier RMSE value
    """
    res = register_point_clouds(
        src,
        dst,
        max_correspondence_distance=9999.0,
        max_iters=max_iters,
        tol=tol,
    )
    return res.transformation, [float(res.inlier_rmse)]


def _demo():
    """Demo: call icp on two synthetic clouds."""
    # Generate synthetic point clouds
    n_points = 100
    rng = np.random.default_rng(42)

    # Source cloud
    src = rng.standard_normal((n_points, 3))

    # Destination: rotated and translated version of source
    angle = np.pi / 6
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    t_true = np.array([1.0, 2.0, 0.5])
    dst = (R_true @ src.T).T + t_true

    # Add noise
    dst += rng.standard_normal((n_points, 3)) * 0.01

    # Call icp
    T, errors = icp_point2point(src, dst)
    print(f"ICP completed. Final RMSE: {errors[0]:.6f}")
    print("Estimated transformation:\n", T)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
