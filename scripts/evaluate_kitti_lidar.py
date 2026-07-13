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

from spatialwm.eval.trajectory import ate, rpe
from spatialwm.geometry.icp import register_point_clouds
from spatialwm.geometry.lidar_odometry import lidar_odometry, voxel_downsample
from spatialwm.perception.lidar_io import load_kitti_points


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


def validate_pair_indices(source_idx: int, target_idx: int, num_frames: int) -> None:
    """Validate that source and target indices are distinct and within valid range."""
    if source_idx == target_idx:
        raise ValueError(
            f"Source index and target index must be distinct, got both as {source_idx}."
        )
    if not (0 <= source_idx < num_frames):
        raise ValueError(f"Source index {source_idx} out of range [0, {num_frames - 1}].")
    if not (0 <= target_idx < num_frames):
        raise ValueError(f"Target index {target_idx} out of range [0, {num_frames - 1}].")


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
    """Filter points within a bounding box and randomly downsample for plotting."""
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

    n = len(filtered)
    if n <= target_count:
        return filtered

    indices = np.random.default_rng(42).choice(n, target_count, replace=False)
    return filtered[indices]


def compute_metrics(poses_est: np.ndarray, poses_gt: np.ndarray) -> dict[str, float]:
    """Compute trajectory error metrics: ATE and RPE."""
    if len(poses_est) < 3:
        raise ValueError(
            f"At least 3 frames are required for trajectory evaluation. Got {len(poses_est)}"
        )

    ate_val = ate(poses_est, poses_gt)
    rpe_val = rpe(poses_est, poses_gt, delta=1)
    return {"ate": ate_val, "rpe": rpe_val}


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
    parser.add_argument(
        "--pair-source-index",
        type=int,
        default=1,
        help="Index of the source frame for the pairwise registration plot.",
    )
    parser.add_argument(
        "--pair-target-index",
        type=int,
        default=0,
        help="Index of the target frame for the pairwise registration plot.",
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

    try:
        validate_pair_indices(args.pair_source_index, args.pair_target_index, args.frames)
    except ValueError as e:
        print(f"Error: Invalid pair indices: {e}", file=sys.stderr)
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

    print(f"ATE (RMSE): {metrics['ate']:.6f} meters")
    print(f"RPE (Mean, delta=1): {metrics['rpe']:.6f} meters")

    # 7. Perform pairwise registration for visualization
    print(
        f"Performing pairwise registration between "
        f"source={args.pair_source_index} and target={args.pair_target_index}..."
    )
    scan_source = scans[args.pair_source_index]
    scan_target = scans[args.pair_target_index]
    src_down = voxel_downsample(scan_source, args.voxel)
    dst_down = voxel_downsample(scan_target, args.voxel)
    reg_result = register_point_clouds(
        src=src_down,
        dst=dst_down,
        max_correspondence_distance=args.max_correspondence_distance,
        max_iters=args.max_iters,
    )
    T_pair = reg_result.transformation

    # 8. Generate and save diagnostic plot
    os.makedirs(args.output_dir, exist_ok=True)
    png_path = os.path.join(args.output_dir, "evaluate_kitti_lidar.png")
    print(f"Generating diagnostic figure at: {png_path}")
    generate_diagnostic_plot(
        points_source=scan_source,
        points_target=scan_target,
        T_pair=T_pair,
        poses_est=poses_est,
        poses_gt=poses_gt,
        frame_ids=list(range(args.frames)),
        voxel=args.voxel,
        max_corr=args.max_correspondence_distance,
        pair_source_idx=args.pair_source_index,
        pair_target_idx=args.pair_target_index,
        output_path=png_path,
    )

    # 9. Create JSON beside the figure
    json_path = os.path.join(args.output_dir, "evaluate_kitti_lidar.json")
    print(f"Generating JSON report at: {json_path}")

    limitations = (
        "Diagnostic-only evaluation on a short 10-frame slice; "
        "does not represent a full benchmark. Drift accumulates because each "
        "frame-to-frame registration compiles multiplicatively without loop "
        "closure or global bundle adjustment."
    )

    report_data = {
        "dataset_identity": (f"KITTI raw sequence {args.date} drive {args.drive} sync"),
        "source_url": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
        "frame_ids": list(range(args.frames)),
        "config": {
            "voxel": args.voxel,
            "max_iters": args.max_iters,
            "max_correspondence_distance": args.max_correspondence_distance,
            "pair_source_index": args.pair_source_index,
            "pair_target_index": args.pair_target_index,
        },
        "transforms": {
            "T_pair": T_pair,
            "poses_estimated": poses_est,
            "poses_ground_truth": poses_gt,
        },
        "metrics": metrics,
        "limitations": limitations,
    }

    report_data_safe = make_json_safe(report_data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data_safe, f, indent=2)

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
