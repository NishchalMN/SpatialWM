"""Point cloud visualization utilities."""

import numpy as np


def show_pointcloud(points: np.ndarray, colors=None) -> None:
    """Display a point cloud using Open3D.
    
    Args:
        points: Point cloud coordinates (N, 3)
        colors: Optional RGB colors (N, 3) in range [0, 1]
    """
    import open3d as o3d
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])
