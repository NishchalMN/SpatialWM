"""Trajectory evaluation metrics: ATE, RPE, Umeyama alignment."""

import numpy as np


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama Sim(3) alignment via SVD.
    
    Computes optimal similarity transformation (rotation, translation, scale)
    that aligns source points to destination points.
    
    Args:
        src: Source points (N, 3)
        dst: Destination points (N, 3)
        with_scale: Whether to compute scale factor
    
    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        s: Scale factor (float)
    """
    raise NotImplementedError


def ate(traj_est: np.ndarray, traj_gt: np.ndarray) -> float:
    """Absolute Trajectory Error (RMSE of translation after Umeyama alignment).
    
    Aligns estimated trajectory to ground truth using Sim(3) transformation,
    then computes RMSE of residual position errors.
    
    Args:
        traj_est: Estimated trajectory (N, 3) or (N, 4, 4) poses
        traj_gt: Ground truth trajectory (N, 3) or (N, 4, 4) poses
    
    Returns:
        RMSE error in meters
    """
    raise NotImplementedError


def rpe(traj_est: np.ndarray, traj_gt: np.ndarray, delta: int = 1) -> float:
    """Relative Pose Error over frame gap delta.
    
    Computes error in relative motion between frames separated by delta timesteps.
    Measures local consistency rather than global drift.
    
    Args:
        traj_est: Estimated trajectory (N, 4, 4) poses
        traj_gt: Ground truth trajectory (N, 4, 4) poses
        delta: Frame gap for computing relative motion
    
    Returns:
        Mean RPE translation error
    """
    raise NotImplementedError


def _demo():
    """Demo ATE with synthetic trajectories."""
    # Create two synthetic trajectories
    n_frames = 100
    traj_gt = np.cumsum(np.random.randn(n_frames, 3) * 0.1, axis=0)
    traj_est = traj_gt + np.random.randn(n_frames, 3) * 0.05
    
    error = ate(traj_est, traj_gt)
    print(f"ATE: {error:.4f}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
