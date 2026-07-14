#!/usr/bin/env python3
"""Evaluate KITTI LiDAR odometry against OXTS ground truth trajectory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pykitti

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatialwm.eval.trajectory import umeyama
from spatialwm.geometry.lidar_odometry import lidar_odometry
from spatialwm.perception.lidar_io import load_kitti_points
from spatialwm.perception.voxelize import bev


def validate_kitti_data(kitti_root: str, date: str, drive: str, num_frames: int) -> None:
    """Validate that all required KITTI files for the given config are present on disk."""
    root = Path(kitti_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"KITTI root directory does not exist or is not a directory: {kitti_root}"
        )

    date_dir = root / date
    if not date_dir.is_dir():
        raise FileNotFoundError(f"Date directory does not exist: {date_dir}")

    drive_dir = date_dir / f"{date}_drive_{drive}_sync"
    if not drive_dir.is_dir():
        raise FileNotFoundError(f"Drive sync directory does not exist: {drive_dir}")

    # Check calibration files
    calib_files = ["calib_imu_to_velo.txt", "calib_velo_to_cam.txt", "calib_cam_to_cam.txt"]
    for f in calib_files:
        p = date_dir / f
        if not p.is_file():
            raise FileNotFoundError(f"Required KITTI calibration file is missing: {p}")

    # Check timestamps file
    timestamps_path = drive_dir / "oxts" / "timestamps.txt"
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"Drive timestamps file is missing: {timestamps_path}")

    # Check velodyne files and oxts files
    for frame in range(num_frames):
        velo_file = drive_dir / "velodyne_points" / "data" / f"{frame:010d}.bin"
        oxts_file = drive_dir / "oxts" / "data" / f"{frame:010d}.txt"
        if not velo_file.is_file():
            raise FileNotFoundError(f"Velodyne scan file missing: {velo_file}")
        if not oxts_file.is_file():
            raise FileNotFoundError(f"OXTS packet file missing: {oxts_file}")


def normalize_poses(T_w_velo: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Normalize a sequence of poses such that the first pose is the identity.

    Given T_w_velo[k], returns P_gt[k] = inv(T_w_velo[0]) @ T_w_velo[k].
    """
    T_w_velo = np.asarray(T_w_velo)
    if T_w_velo.ndim != 3 or T_w_velo.shape[1:] != (4, 4):
        raise ValueError(f"T_w_velo must have shape (N, 4, 4), got {T_w_velo.shape}")

    T_w_velo_0_inv = np.linalg.inv(T_w_velo[0])
    normalized = []
    for T in T_w_velo:
        normalized.append(T_w_velo_0_inv @ T)
    return np.stack(normalized)


def make_json_safe(data):
    """Recursively convert NumPy objects to standard Python JSON-serializable types."""
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_safe(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    else:
        return data


def filter_and_downsample(
    points: np.ndarray,
    x_range: tuple[float, float] = (-25.0, 25.0),
    y_range: tuple[float, float] = (-25.0, 25.0),
    z_range: tuple[float, float] = (-3.0, 5.0),
    target_count: int = 1500,
) -> np.ndarray:
    """Filter and deterministically thin points for the legacy pair diagnostic."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    mask = (
        (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= y_range[0])
        & (points[:, 1] <= y_range[1])
        & (points[:, 2] >= z_range[0])
        & (points[:, 2] <= z_range[1])
    )
    filtered = points[mask]
    if len(filtered) == 0:
        filtered = points
    if len(filtered) <= target_count:
        return filtered
    indices = np.random.default_rng(42).choice(len(filtered), target_count, replace=False)
    return filtered[indices]


def compute_metrics(poses_est: np.ndarray, poses_gt: np.ndarray) -> dict[str, float]:
    """Compute metric-aware trajectory errors and state the alignment policy.

    LiDAR already has metric scale, so rigid-aligned ATE is the primary summary.
    Sim(3)-aligned ATE is retained only as a diagnostic showing how much an
    unconstrained scale fit would hide.
    """
    if len(poses_est) < 3:
        raise ValueError(
            f"At least 3 frames are required for trajectory evaluation. Got {len(poses_est)}"
        )

    estimated_centres = poses_est[:, :3, 3]
    ground_truth_centres = poses_gt[:, :3, 3]
    rigid_rotation, rigid_translation, _ = umeyama(
        estimated_centres, ground_truth_centres, with_scale=False
    )
    rigid_aligned = estimated_centres @ rigid_rotation.T + rigid_translation
    sim3_rotation, sim3_translation, sim3_scale = umeyama(
        estimated_centres, ground_truth_centres, with_scale=True
    )
    sim3_aligned = (
        sim3_scale * (estimated_centres @ sim3_rotation.T) + sim3_translation
    )

    raw_errors = np.linalg.norm(estimated_centres - ground_truth_centres, axis=1)
    rigid_errors = np.linalg.norm(rigid_aligned - ground_truth_centres, axis=1)
    sim3_errors = np.linalg.norm(sim3_aligned - ground_truth_centres, axis=1)

    relative_estimated = np.linalg.inv(poses_est[:-1]) @ poses_est[1:]
    relative_ground_truth = np.linalg.inv(poses_gt[:-1]) @ poses_gt[1:]
    relative_error = np.linalg.inv(relative_ground_truth) @ relative_estimated
    step_translation_errors = np.linalg.norm(relative_error[:, :3, 3], axis=1)
    cos_angles = np.clip(
        (np.trace(relative_error[:, :3, :3], axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    step_rotation_errors = np.degrees(np.arccos(cos_angles))

    rigid_ate = float(np.sqrt(np.mean(rigid_errors**2)))
    mean_rpe = float(np.mean(step_translation_errors))
    return {
        "ate": rigid_ate,
        "rpe": mean_rpe,
        "raw_position_rmse_m": float(np.sqrt(np.mean(raw_errors**2))),
        "rigid_aligned_ate_rmse_m": rigid_ate,
        "sim3_aligned_ate_rmse_m": float(np.sqrt(np.mean(sim3_errors**2))),
        "sim3_scale": float(sim3_scale),
        "final_raw_position_error_m": float(raw_errors[-1]),
        "mean_step_translation_error_m": mean_rpe,
        "max_step_translation_error_m": float(np.max(step_translation_errors)),
        "mean_step_rotation_error_deg": float(np.mean(step_rotation_errors)),
        "max_step_rotation_error_deg": float(np.max(step_rotation_errors)),
    }


def generate_diagnostic_plot(
    points_source: np.ndarray,
    points_target: np.ndarray,
    T_pair: np.ndarray,
    poses_est: np.ndarray,
    poses_gt: np.ndarray,
    frame_ids: list[int],
    voxel: float,
    max_corr: float,
    pair_source_idx: int,
    pair_target_idx: int,
    output_path: str,
) -> matplotlib.figure.Figure:
    """Generate and save the 3-panel scientific diagnostic figure."""
    # 1. Filter and downsample points
    target_plot = filter_and_downsample(points_target, target_count=1500)
    source_plot = filter_and_downsample(points_source, target_count=1500)

    # 2. Transform the source points
    N = len(source_plot)
    pts_hom = np.hstack([source_plot, np.ones((N, 1))])
    source_trans_plot = (T_pair @ pts_hom.T).T[:, :3]

    # 3. Create the 3-panel figure
    fig = plt.figure(figsize=(18, 5))

    # Shared 3D camera viewpoint parameters
    elev = 30
    azim = -120

    # Panel (a): 3D points before ICP
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(
        target_plot[:, 0],
        target_plot[:, 1],
        target_plot[:, 2],
        s=2,
        c="orange",
        label=f"Target (Scan {pair_target_idx})",
        alpha=0.6,
    )
    ax1.scatter(
        source_plot[:, 0],
        source_plot[:, 1],
        source_plot[:, 2],
        s=2,
        c="blue",
        label=f"Source (Scan {pair_source_idx})",
        alpha=0.6,
    )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_box_aspect(None, zoom=0.85)
    ax1.set_title("Before ICP (Source & Target)", pad=12)
    ax1.view_init(elev=elev, azim=azim)
    ax1.legend()

    # Panel (b): 3D points after ICP
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(
        target_plot[:, 0],
        target_plot[:, 1],
        target_plot[:, 2],
        s=2,
        c="orange",
        label=f"Target (Scan {pair_target_idx})",
        alpha=0.6,
    )
    ax2.scatter(
        source_trans_plot[:, 0],
        source_trans_plot[:, 1],
        source_trans_plot[:, 2],
        s=2,
        c="blue",
        label="Source Transformed",
        alpha=0.6,
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_box_aspect(None, zoom=0.85)
    ax2.set_title("After ICP (Source Transformed & Target)", pad=12)
    ax2.view_init(elev=elev, azim=azim)
    ax2.legend()

    # Panel (c): XY Trajectory
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(
        poses_est[:, 0, 3],
        poses_est[:, 1, 3],
        label="Estimated (ICP)",
        marker="o",
        markersize=4,
        color="blue",
    )
    ax3.plot(
        poses_gt[:, 0, 3],
        poses_gt[:, 1, 3],
        label="Ground Truth (OXTS)",
        marker="x",
        markersize=4,
        color="orange",
        linestyle="--",
    )
    ax3.scatter(
        0.0,
        0.0,
        color="red",
        marker="*",
        s=150,
        zorder=5,
        label="Start (Scan 0)",
    )
    ax3.set_xlabel("X (meters)")
    ax3.set_ylabel("Y (meters)")
    ax3.set_title("Trajectory (XY Plane)", pad=12)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax3.set_aspect("equal")

    # Overall title indicating it's a diagnostic
    fig.suptitle(
        "KITTI LiDAR Odometry Evaluation (Diagnostic - Not a Benchmark)",
        fontsize=14,
        fontweight="bold",
    )

    # Info footer
    info_text = (
        f"Frame IDs: {frame_ids} | Voxel Size: {voxel}m | "
        f"Max Correspondence Distance: {max_corr}m\n"
        "Point clouds downsampled to 1500 points and cropped to "
        "[-25, 25]m in X/Y, [-3, 5]m in Z for visualization clarity.\n"
        "Notice: This is a diagnostic evaluation on a short raw slice "
        "and is not a general benchmark."
    )
    fig.text(
        0.5,
        0.01,
        info_text,
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
    )

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fig


def _align_centres_rigid(
    poses_est: np.ndarray, poses_gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly align estimated centres to GT without changing metric scale."""
    estimated = poses_est[:, :3, 3]
    ground_truth = poses_gt[:, :3, 3]
    rotation, translation, _ = umeyama(estimated, ground_truth, with_scale=False)
    return estimated @ rotation.T + translation, ground_truth


def _step_pose_errors(
    poses_est: np.ndarray, poses_gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step translation and rotation errors."""
    relative_estimated = np.linalg.inv(poses_est[:-1]) @ poses_est[1:]
    relative_ground_truth = np.linalg.inv(poses_gt[:-1]) @ poses_gt[1:]
    error = np.linalg.inv(relative_ground_truth) @ relative_estimated
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    cos_angles = np.clip(
        (np.trace(error[:, :3, :3], axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return translation, np.degrees(np.arccos(cos_angles))


def generate_trajectory_figure(
    poses_est: np.ndarray,
    poses_gt: np.ndarray,
    metrics: dict[str, float],
    frame_ids: list[int],
    output_path: str,
) -> matplotlib.figure.Figure:
    """Render an alignment-explicit trajectory and drift diagnostic."""
    aligned_estimated, ground_truth = _align_centres_rigid(poses_est, poses_gt)
    position_errors = np.linalg.norm(aligned_estimated - ground_truth, axis=1)
    step_translation, step_rotation = _step_pose_errors(poses_est, poses_gt)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    trajectory_axis, position_axis, step_axis = axes

    trajectory_axis.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        "o-",
        color="black",
        linewidth=2.8,
        markersize=6,
        label="OXTS ground truth",
    )
    trajectory_axis.plot(
        aligned_estimated[:, 0],
        aligned_estimated[:, 1],
        "x--",
        color="#ef476f",
        linewidth=2.2,
        markersize=7,
        label="ICP odometry (rigid aligned)",
    )
    for order, frame_id in enumerate(frame_ids):
        trajectory_axis.annotate(
            str(frame_id),
            (aligned_estimated[order, 0], aligned_estimated[order, 1]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    trajectory_axis.scatter(
        ground_truth[0, 0],
        ground_truth[0, 1],
        marker="*",
        s=150,
        color="#4361ee",
        zorder=5,
        label="Start",
    )
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.set_xlabel("X forward [m]")
    trajectory_axis.set_ylabel("Y left [m]")
    trajectory_axis.set_title(
        "Metric trajectory (SE(3) alignment, no scale)\n"
        f"ATE RMSE={metrics['rigid_aligned_ate_rmse_m']:.3f} m",
        fontweight="bold",
    )
    trajectory_axis.grid(True, linestyle=":", alpha=0.4)
    trajectory_axis.legend()

    position_axis.plot(
        frame_ids,
        position_errors,
        "o-",
        color="#d62728",
        linewidth=2.2,
    )
    position_axis.axhline(
        metrics["rigid_aligned_ate_rmse_m"],
        color="black",
        linestyle="--",
        label="ATE RMSE",
    )
    position_axis.set_xlabel("Frame")
    position_axis.set_ylabel("Rigid-aligned position error [m]")
    position_axis.set_title(
        "Position disagreement over the slice\n"
        f"raw final error={metrics['final_raw_position_error_m']:.3f} m",
        fontweight="bold",
    )
    position_axis.set_xticks(frame_ids)
    position_axis.set_ylim(bottom=0.0)
    position_axis.grid(True, linestyle=":", alpha=0.4)
    position_axis.legend()

    steps = np.asarray(frame_ids[1:])
    step_axis.bar(
        steps - 0.12,
        step_translation,
        width=0.24,
        color="#2a9d8f",
        label="Translation error",
    )
    step_axis.set_xlabel("Target frame of step")
    step_axis.set_ylabel("Relative translation error [m]", color="#2a9d8f")
    step_axis.tick_params(axis="y", labelcolor="#2a9d8f")
    step_axis.set_xticks(steps)
    step_axis.set_ylim(bottom=0.0)
    step_axis.grid(True, axis="y", linestyle=":", alpha=0.35)
    rotation_axis = step_axis.twinx()
    rotation_axis.plot(
        steps,
        step_rotation,
        "o-",
        color="#f4a261",
        linewidth=2.0,
        label="Rotation error",
    )
    rotation_axis.set_ylabel("Relative rotation error [deg]", color="#f4a261")
    rotation_axis.tick_params(axis="y", labelcolor="#f4a261")
    rotation_axis.set_ylim(bottom=0.0)
    step_axis.set_title(
        "One-step relative-pose error\n"
        f"mean={metrics['mean_step_translation_error_m']:.3f} m / "
        f"{metrics['mean_step_rotation_error_deg']:.3f} deg",
        fontweight="bold",
    )
    handles1, labels1 = step_axis.get_legend_handles_labels()
    handles2, labels2 = rotation_axis.get_legend_handles_labels()
    step_axis.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.suptitle(
        "KITTI Scan-to-Scan LiDAR Odometry: Local Accuracy and Accumulated Drift\n"
        "2011-09-26 drive 0005, frames 0-9 | metric sensor, rigid alignment only | "
        "bounded diagnostic, not a benchmark",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        f"A free Sim(3) fit would report {metrics['sim3_aligned_ate_rmse_m']:.3f} m "
        f"but rescales the metric trajectory by {metrics['sim3_scale']:.3f}; "
        "the primary ATE therefore forbids scale adjustment.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.88])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return fig


def _crop_lidar_points(
    points: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    mask = (
        (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= y_range[0])
        & (points[:, 1] <= y_range[1])
        & (points[:, 2] >= z_range[0])
        & (points[:, 2] <= z_range[1])
    )
    return points[mask]


def _plot_bev_grid(
    axis: plt.Axes,
    points: np.ndarray,
    *,
    cell: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    title: str,
) -> int:
    """Rasterize and display boolean BEV occupancy with metric axes."""
    occupancy = bev(points, cell=cell)
    extent = [
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    ]
    axis.imshow(
        occupancy,
        cmap="binary",
        origin="upper",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_xlim(*x_range)
    axis.set_ylim(*y_range)
    axis.set_xlabel("X forward [m]")
    axis.set_ylabel("Y left [m]")
    axis.set_title(title, fontweight="bold")
    axis.grid(True, linestyle=":", alpha=0.25)
    return int(np.count_nonzero(occupancy))


def generate_bev_figure(
    scans: list[np.ndarray],
    poses_est: np.ndarray,
    output_path: str,
    *,
    cell: float = 0.10,
) -> tuple[matplotlib.figure.Figure, dict[str, int | float]]:
    """Create single-scan and accumulated-map BEV occupancy evidence."""
    x_range = (-5.0, 50.0)
    y_range = (-22.0, 22.0)
    z_range = (-2.5, 1.5)
    cropped_single = _crop_lidar_points(scans[0], x_range, y_range, z_range)

    accumulated = []
    for scan, pose in zip(scans, poses_est):
        homogeneous = np.column_stack((scan, np.ones(len(scan))))
        transformed = (pose @ homogeneous.T).T[:, :3]
        cropped = _crop_lidar_points(transformed, x_range, y_range, z_range)
        accumulated.append(cropped)
    accumulated_points = np.vstack(accumulated)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), sharex=True, sharey=True)
    single_cells = _plot_bev_grid(
        axes[0],
        cropped_single,
        cell=cell,
        x_range=x_range,
        y_range=y_range,
        title=(
            f"Single Velodyne scan (frame 0)\n"
            f"{len(cropped_single):,} points after spatial crop"
        ),
    )
    accumulated_cells = _plot_bev_grid(
        axes[1],
        accumulated_points,
        cell=cell,
        x_range=x_range,
        y_range=y_range,
        title=(
            f"Ten scans accumulated with estimated poses\n"
            f"{len(accumulated_points):,} transformed points"
        ),
    )
    centres = poses_est[:, :3, 3]
    axes[1].plot(
        centres[:, 0],
        centres[:, 1],
        "o-",
        color="#ef476f",
        linewidth=2,
        markersize=4,
        label="Estimated sensor path",
    )
    axes[1].scatter(
        centres[0, 0], centres[0, 1], marker="*", s=150, color="#4361ee", label="Start"
    )
    axes[1].legend(loc="upper right")

    fig.suptitle(
        "KITTI LiDAR to Bird's-Eye-View Occupancy\n"
        f"Boolean {cell:.2f} m cells, X/Y fixed to identical metric limits, "
        "Z crop [-2.5, 1.5] m",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "The accumulated map becomes denser but also inherits every scan-to-scan pose error; "
        "no loop closure, scan-to-map refinement, or semantic filtering is used.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.91])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return fig, {
        "bev_cell_m": cell,
        "single_scan_cropped_points": len(cropped_single),
        "single_scan_occupied_cells": single_cells,
        "accumulated_cropped_points": len(accumulated_points),
        "accumulated_occupied_cells": accumulated_cells,
    }


def positive_integer(value):
    try:
        val = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be a positive integer, got {value}")
    if val <= 0:
        raise argparse.ArgumentTypeError(f"Must be a positive integer, got {val}")
    return val


def positive_float(value):
    try:
        val = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be a positive float, got {value}")
    if val <= 0.0:
        raise argparse.ArgumentTypeError(f"Must be a positive float, got {val}")
    return val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI LiDAR odometry against OXTS ground truth trajectory."
    )
    parser.add_argument(
        "--kitti-root",
        type=str,
        default="data/raw/kitti",
        help="Path to the directory containing the KITTI dataset.",
    )
    parser.add_argument(
        "--frames",
        type=positive_integer,
        default=10,
        help="Number of consecutive frames to evaluate (must be at least 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save the diagnostic figure and JSON report.",
    )
    parser.add_argument("--date", type=str, default="2011_09_26", help="KITTI recording date.")
    parser.add_argument("--drive", type=str, default="0005", help="KITTI drive number.")
    parser.add_argument(
        "--voxel", type=positive_float, default=0.2, help="Voxel downsampling size in meters."
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=positive_float,
        default=1.0,
        help="Maximum correspondence distance for ICP.",
    )
    parser.add_argument(
        "--max-iters",
        type=positive_integer,
        default=50,
        help="Maximum number of iterations for ICP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Validation of parameters
    if args.frames < 3:
        print(
            f"Error: The number of frames (--frames) must be at least 3, got {args.frames}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Validation of files on disk
    try:
        validate_kitti_data(args.kitti_root, args.date, args.drive, args.frames)
    except FileNotFoundError as e:
        print(f"Error: Missing data files: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Load dataset
    print(f"Loading {args.frames} frames from KITTI sequence {args.date} drive {args.drive}...")
    try:
        dataset = pykitti.raw(args.kitti_root, args.date, args.drive, frames=range(args.frames))
    except Exception as e:
        print(f"Error: Failed to load dataset via pykitti: {e}", file=sys.stderr)
        sys.exit(1)

    # Explicit validation of loaded structure
    if not hasattr(dataset, "calib") or dataset.calib is None or dataset.calib.T_velo_imu is None:
        print("Error: Calibration T_velo_imu missing from loaded dataset", file=sys.stderr)
        sys.exit(1)
    if not hasattr(dataset, "oxts") or len(dataset.oxts) != args.frames:
        print("Error: OXTS ground truth missing or size mismatch", file=sys.stderr)
        sys.exit(1)
    if not hasattr(dataset, "timestamps") or len(dataset.timestamps) != args.frames:
        print("Error: Timestamps missing or size mismatch", file=sys.stderr)
        sys.exit(1)

    # 4. Construct ground truth poses
    print("Constructing OXTS ground truth poses...")
    T_velo_imu = dataset.calib.T_velo_imu
    T_imu_velo = np.linalg.inv(T_velo_imu)
    T_w_velo = []
    for oxts_packet in dataset.oxts:
        T_w_velo.append(oxts_packet.T_w_imu @ T_imu_velo)
    poses_gt = normalize_poses(T_w_velo)

    # 5. Run LiDAR odometry (estimating trajectories)
    print("Running LiDAR odometry estimation...")
    scans = [load_kitti_points(p) for p in dataset.velo_files]
    poses_est = lidar_odometry(
        scans,
        voxel=args.voxel,
        max_iters=args.max_iters,
        max_correspondence_distance=args.max_correspondence_distance,
    )

    # 6. Compute metrics
    print("Computing ATE and RPE trajectory errors...")
    try:
        metrics = compute_metrics(poses_est, poses_gt)
    except ValueError as e:
        print(f"Error computing trajectory metrics: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Rigid-aligned ATE (RMSE, no scale): {metrics['ate']:.6f} meters")
    print(f"Mean one-step translation error: {metrics['rpe']:.6f} meters")
    print(f"Mean one-step rotation error: {metrics['mean_step_rotation_error_deg']:.6f} degrees")

    # 7. Generate the curated trajectory and BEV figures
    os.makedirs(args.output_dir, exist_ok=True)
    trajectory_path = os.path.join(args.output_dir, "kitti_lidar_odometry.png")
    bev_path = os.path.join(args.output_dir, "kitti_lidar_bev.png")
    print(f"Generating trajectory figure at: {trajectory_path}")
    generate_trajectory_figure(
        poses_est=poses_est,
        poses_gt=poses_gt,
        metrics=metrics,
        frame_ids=list(range(args.frames)),
        output_path=trajectory_path,
    )
    print(f"Generating BEV figure at: {bev_path}")
    _, bev_metrics = generate_bev_figure(
        scans=scans,
        poses_est=poses_est,
        output_path=bev_path,
        cell=0.10,
    )

    # 8. Persist machine-readable trajectory and metrics
    npz_path = os.path.join(args.output_dir, "kitti_lidar_trajectory.npz")
    print(f"Generating trajectory arrays at: {npz_path}")
    np.savez_compressed(
        npz_path,
        poses_estimated=poses_est,
        poses_ground_truth=poses_gt,
        frame_ids=np.arange(args.frames, dtype=np.int64),
    )
    json_path = os.path.join(args.output_dir, "kitti_lidar_metrics.json")
    print(f"Generating JSON report at: {json_path}")

    limitations = (
        "Diagnostic-only evaluation on a short raw slice; does not represent a full "
        "benchmark. Drift accumulates because frame-to-frame registrations compose "
        "multiplicatively without loop closure, scan-to-map registration, or global "
        "pose-graph optimization."
    )

    report_data = {
        "dataset_identity": (f"KITTI raw sequence {args.date} drive {args.drive} sync"),
        "source_url": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
        "frame_ids": list(range(args.frames)),
        "config": {
            "voxel": args.voxel,
            "max_iters": args.max_iters,
            "max_correspondence_distance": args.max_correspondence_distance,
            "bev_cell_m": 0.10,
        },
        "trajectory_convention": {
            "poses_estimated": "scan k to scan 0",
            "poses_ground_truth": "Velodyne k to normalized Velodyne-0 world frame",
            "primary_ate_alignment": "rigid SE(3) position alignment; scale fixed to 1",
            "rpe_delta_frames": 1,
        },
        "metrics": metrics,
        "bev": bev_metrics,
        "limitations": limitations,
    }

    report_data_safe = make_json_safe(report_data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data_safe, f, indent=2)

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
