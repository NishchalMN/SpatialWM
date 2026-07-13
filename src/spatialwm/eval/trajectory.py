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
    src = np.asarray(src)
    dst = np.asarray(dst)
    
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src must have shape (N, 3), got {src.shape}")
    if dst.ndim != 2 or dst.shape[1] != 3:
        raise ValueError(f"dst must have shape (N, 3), got {dst.shape}")
    if src.shape != dst.shape:
        raise ValueError(f"src shape {src.shape} and dst shape {dst.shape} must match")
        
    if not np.isfinite(src).all():
        raise ValueError("src contains non-finite values (NaN or Inf).")
    if not np.isfinite(dst).all():
        raise ValueError("dst contains non-finite values (NaN or Inf).")
        
    N = src.shape[0]
    if N < 3:
        raise ValueError(f"At least 3 points are required for Umeyama alignment, got N = {N}.")
        
    mu_x = src.mean(axis=0)
    mu_y = dst.mean(axis=0)
    
    src_c = src - mu_x
    dst_c = dst - mu_y
    
    if np.linalg.matrix_rank(src_c) < 2:
        raise ValueError("Source point cloud is degenerate (collinear or coincident).")
    if np.linalg.matrix_rank(dst_c) < 2:
        raise ValueError("Destination point cloud is degenerate (collinear or coincident).")
        
    H = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(H)
    
    det_UVt = np.linalg.det(U) * np.linalg.det(Vt)
    d = 1.0 if det_UVt >= 0.0 else -1.0
    S = np.eye(3)
    S[2, 2] = d
    
    R = U @ S @ Vt
    
    if with_scale:
        var_x = np.mean(np.sum(src_c ** 2, axis=1))
        if var_x < 1e-9:
            raise ValueError("Variance of source points is too close to zero (degenerate).")
        s = float(np.sum(D * np.diagonal(S)) / var_x)
    else:
        s = 1.0
        
    t = mu_y - s * R @ mu_x
    
    return R, t, s


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
    traj_est = np.asarray(traj_est)
    traj_gt = np.asarray(traj_gt)
    
    if traj_est.ndim == 2 and traj_est.shape[1] == 3:
        pos_est = traj_est
    elif traj_est.ndim == 3 and traj_est.shape[1:] == (4, 4):
        pos_est = traj_est[:, :3, 3]
    else:
        raise ValueError(
            f"traj_est must have shape (N, 3) or (N, 4, 4), got {traj_est.shape}"
        )
        
    if traj_gt.ndim == 2 and traj_gt.shape[1] == 3:
        pos_gt = traj_gt
    elif traj_gt.ndim == 3 and traj_gt.shape[1:] == (4, 4):
        pos_gt = traj_gt[:, :3, 3]
    else:
        raise ValueError(
            f"traj_gt must have shape (N, 3) or (N, 4, 4), got {traj_gt.shape}"
        )
        
    if pos_est.shape[0] != pos_gt.shape[0]:
        raise ValueError(
            f"traj_est and traj_gt must have the same number of frames, "
            f"got {pos_est.shape[0]} and {pos_gt.shape[0]}"
        )
        
    if not np.isfinite(pos_est).all():
        raise ValueError("traj_est contains non-finite values (NaN or Inf).")
    if not np.isfinite(pos_gt).all():
        raise ValueError("traj_gt contains non-finite values (NaN or Inf).")
        
    R, t, s = umeyama(pos_est, pos_gt, with_scale=True)
    
    pos_est_aligned = s * (R @ pos_est.T).T + t
    
    rmse = np.sqrt(np.mean(np.sum((pos_est_aligned - pos_gt) ** 2, axis=1)))
    return float(rmse)


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
    traj_est = np.asarray(traj_est)
    traj_gt = np.asarray(traj_gt)
    
    if traj_est.ndim != 3 or traj_est.shape[1:] != (4, 4):
        raise ValueError(
            f"traj_est must have shape (N, 4, 4), got {traj_est.shape}"
        )
    if traj_gt.ndim != 3 or traj_gt.shape[1:] != (4, 4):
        raise ValueError(
            f"traj_gt must have shape (N, 4, 4), got {traj_gt.shape}"
        )
    if traj_est.shape[0] != traj_gt.shape[0]:
        raise ValueError(
            f"traj_est and traj_gt must have the same number of frames, "
            f"got {traj_est.shape[0]} and {traj_gt.shape[0]}"
        )
        
    if not np.isfinite(traj_est).all():
        raise ValueError("traj_est contains non-finite values (NaN or Inf).")
    if not np.isfinite(traj_gt).all():
        raise ValueError("traj_gt contains non-finite values (NaN or Inf).")
        
    N = traj_est.shape[0]
    if not isinstance(delta, (int, np.integer)):
        raise TypeError(f"delta must be an integer, got {type(delta)}")
    if delta < 1:
        raise ValueError(f"delta must be at least 1, got {delta}")
    if delta >= N:
        raise ValueError(
            f"delta ({delta}) must be strictly less than trajectory length N ({N})"
        )
        
    T_gt_i = traj_gt[:-delta]
    T_gt_id = traj_gt[delta:]
    inv_T_gt_i = np.linalg.inv(T_gt_i)
    rel_gt = inv_T_gt_i @ T_gt_id
    
    T_est_i = traj_est[:-delta]
    T_est_id = traj_est[delta:]
    inv_T_est_i = np.linalg.inv(T_est_i)
    rel_est = inv_T_est_i @ T_est_id
    
    inv_rel_gt = np.linalg.inv(rel_gt)
    E = inv_rel_gt @ rel_est
    
    t_errs = np.linalg.norm(E[:, :3, 3], axis=1)
    return float(np.mean(t_errs))


def _demo():
    """Demo ATE and RPE with synthetic trajectories."""
    rng = np.random.default_rng(42)
    n_frames = 100
    traj_gt_pos = np.cumsum(rng.standard_normal((n_frames, 3)) * 0.1, axis=0)
    traj_est_pos = traj_gt_pos + rng.standard_normal((n_frames, 3)) * 0.05
    
    error_ate = ate(traj_est_pos, traj_gt_pos)
    print(f"ATE (positions): {error_ate:.4f}")
    
    traj_gt_poses = np.zeros((n_frames, 4, 4))
    traj_est_poses = np.zeros((n_frames, 4, 4))
    for i in range(n_frames):
        traj_gt_poses[i] = np.eye(4)
        traj_gt_poses[i, :3, 3] = traj_gt_pos[i]
        
        traj_est_poses[i] = np.eye(4)
        traj_est_poses[i, :3, 3] = traj_est_pos[i]
        
    error_ate_poses = ate(traj_est_poses, traj_gt_poses)
    print(f"ATE (poses): {error_ate_poses:.4f}")
    
    error_rpe = rpe(traj_est_poses, traj_gt_poses, delta=1)
    print(f"RPE (poses, delta=1): {error_rpe:.4f}")

if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
