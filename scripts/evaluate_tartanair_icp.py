#!/usr/bin/env python3
"""Evaluate Open3D ICP registration against TartanAir ground truth.

This script parses TartanAir poses, selects source and target frames matching
baseline targets (easy/medium/hard), backprojects RGB-D frames into colored
point clouds, runs point-to-point ICP registration, computes registration
errors against ground truth, and renders side-by-side point clouds of the
medium baseline pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from spatialwm.geometry.icp import register_point_clouds
from spatialwm.geometry.tartanair import (
    compute_se3_error,
    derive_relative_transform,
    parse_pose_to_transform,
    select_target_frames,
)


def backproject_rgbd(
    rgb_path: str,
    depth_path: str,
    fx: float = 320.0,
    fy: float = 320.0,
    cx: float = 320.0,
    cy: float = 240.0,
    max_points: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject RGB-D image into a colored 3D point cloud.

    Args:
        rgb_path: Path to the RGB image PNG.
        depth_path: Path to the depth map NPY.
        fx, fy, cx, cy: Camera intrinsic parameters.
        max_points: Maximum number of points to subsample.
        seed: Random seed for deterministic subsampling.

    Returns:
        pts_3d: (N, 3) float array of 3D points.
        colors: (N, 3) float array of normalized RGB colors [0, 1].
    """
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"Could not load RGB image from {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = np.load(depth_path)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth map from {depth_path}")

    h, w = depth.shape
    v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u = u_grid.flatten()
    v = v_grid.flatten()
    z = depth.flatten()

    # Valid depth mask: finite, positive, and bounded to 1000m (sky filter)
    valid = np.isfinite(z) & (z < 1000.0) & (z > 0.0)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors = rgb[v, u].astype(float) / 255.0

    # Deterministic subsample
    if len(z) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(z), max_points, replace=False)
        u = u[indices]
        v = v[indices]
        z = z[indices]
        colors = colors[indices]

    # Unproject camera RDF coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_3d = np.stack([x, y, z], axis=1)

    return pts_3d, colors


def render_medium_pair(
    pts_src: np.ndarray,
    colors_src: np.ndarray,
    pts_dst: np.ndarray,
    colors_dst: np.ndarray,
    T_est: np.ndarray,
    out_png_path: str,
    baseline: float,
    t_err: float,
    r_err: float,
) -> None:
    """Render the medium-baseline pair with Open3D and save a side-by-side comparison.

    Args:
        pts_src: (N, 3) points of the source cloud.
        colors_src: (N, 3) colors of the source cloud.
        pts_dst: (M, 3) points of the target cloud.
        colors_dst: (M, 3) colors of the target cloud.
        T_est: 4x4 estimated transform matrix.
        out_png_path: File path to save the generated image.
        baseline: The measured baseline translation distance in meters.
        t_err: The translation registration error in meters.
        r_err: The rotation registration error in degrees.
    """
    pcd_src_unaligned = o3d.geometry.PointCloud()
    pcd_src_unaligned.points = o3d.utility.Vector3dVector(pts_src)
    pcd_src_unaligned.colors = o3d.utility.Vector3dVector(colors_src)

    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(pts_dst)
    pcd_dst.colors = o3d.utility.Vector3dVector(colors_dst)

    pcd_src_aligned = o3d.geometry.PointCloud()
    pcd_src_aligned.points = o3d.utility.Vector3dVector(pts_src)
    pcd_src_aligned.colors = o3d.utility.Vector3dVector(colors_src)
    pcd_src_aligned.transform(T_est)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # 1. Render Before ICP
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(width=640, height=480, visible=False)
    vis1.add_geometry(pcd_src_unaligned)
    vis1.add_geometry(pcd_dst)
    vis1.add_geometry(coord_frame)
    vis1.poll_events()
    vis1.update_renderer()

    view_ctl1 = vis1.get_view_control()
    cam_params = view_ctl1.convert_to_pinhole_camera_parameters()
    img_before = np.asarray(vis1.capture_screen_float_buffer(do_render=True))
    vis1.destroy_window()

    # 2. Render After ICP (using exact same camera view parameters)
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(width=640, height=480, visible=False)
    vis2.add_geometry(pcd_src_aligned)
    vis2.add_geometry(pcd_dst)
    vis2.add_geometry(coord_frame)
    vis2.poll_events()
    vis2.update_renderer()

    view_ctl2 = vis2.get_view_control()
    view_ctl2.convert_from_pinhole_camera_parameters(cam_params)
    vis2.poll_events()
    vis2.update_renderer()
    img_after = np.asarray(vis2.capture_screen_float_buffer(do_render=True))
    vis2.destroy_window()

    # 3. Plot side-by-side using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(img_before)
    axes[0].set_title("Before ICP (Unaligned RDF)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_after)
    axes[1].set_title("After ICP (Aligned RDF)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Add subtitle metadata
    text_str = (
        f"TartanAir Registration (Medium Baseline)\n"
        f"Measured Baseline: {baseline:.4f} m\n"
        f"GT-vs-ICP Translation Error: {t_err:.4f} m\n"
        f"GT-vs-ICP Rotation Error: {r_err:.4f} deg"
    )
    fig.suptitle(
        text_str,
        fontsize=11,
        y=0.98,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D render visualization to: {out_png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Open3D ICP registration on TartanAir sequence data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/tartanair/abandonedfactory/Easy/P000",
        help="Path to the TartanAir trajectory directory.",
    )
    parser.add_argument(
        "--source-idx",
        type=int,
        default=1750,
        help="Index of the source frame.",
    )
    parser.add_argument(
        "--baselines",
        type=float,
        nargs="+",
        default=[0.07, 0.33, 0.73],
        help="Target baseline values in meters.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/spatialwm-registration",
        help="Directory to save output reports and rendered images.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10000,
        help="Number of points to subsample per point cloud.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subsampling.",
    )
    args = parser.parse_args()

    # Mark visualization intrinsic assumptions clearly
    print("=====================================================================")
    print("Assumption: Camera intrinsics HFOV 90 degrees (fx=fy=320, cx=320, cy=240)")
    print("=====================================================================")

    # Verify input directories
    image_dir = os.path.join(args.data_dir, "image_left")
    depth_dir = os.path.join(args.data_dir, "depth_left")
    pose_path = os.path.join(args.data_dir, "pose_left.txt")

    if (
        not os.path.isdir(image_dir)
        or not os.path.isdir(depth_dir)
        or not os.path.isfile(pose_path)
    ):
        print(
            f"Error: Missing data under {args.data_dir}. "
            "Check image_left/, depth_left/, and pose_left.txt.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load poses
    try:
        poses = np.loadtxt(pose_path)
    except Exception as e:
        print(f"Error: Malformed pose file '{pose_path}': {e}", file=sys.stderr)
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])

    # Select target frames
    try:
        selected_pairs = select_target_frames(
            poses=poses,
            source_idx=args.source_idx,
            target_baselines=args.baselines,
        )
    except Exception as e:
        print(f"Error selecting target frames: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Source Frame Index: {args.source_idx}")
    print("Selected forward pairs:")
    for pair in selected_pairs:
        print(
            f"  Target Baseline: {pair['requested_baseline']:.2f} m -> "
            f"Target Index: {pair['target_idx']} "
            f"(Actual: {pair['actual_baseline']:.4f} m)"
        )

    # Run ICP evaluation
    os.makedirs(args.output_dir, exist_ok=True)
    report_data = []

    # Table header
    print("\nEvaluation Table:")
    header = (
        f"{'Target (m)':<12}{'Selected Idx':<14}{'Actual Base (m)':<18}"
        f"{'Trans Err (m)':<15}{'Rot Err (deg)':<15}{'Fitness':<10}{'RMSE':<10}"
    )
    print(header)
    print("-" * 94)

    for pair in selected_pairs:
        target_idx = pair["target_idx"]
        requested_b = pair["requested_baseline"]
        actual_b = pair["actual_baseline"]

        # Find matching paths
        src_img_name = image_files[args.source_idx]
        src_frame_id = src_img_name.split("_")[0]
        src_depth_name = [f for f in depth_files if f.startswith(src_frame_id)][0]

        dst_img_name = image_files[target_idx]
        dst_frame_id = dst_img_name.split("_")[0]
        dst_depth_name = [f for f in depth_files if f.startswith(dst_frame_id)][0]

        src_rgb_path = os.path.join(image_dir, src_img_name)
        src_depth_path = os.path.join(depth_dir, src_depth_name)
        dst_rgb_path = os.path.join(image_dir, dst_img_name)
        dst_depth_path = os.path.join(depth_dir, dst_depth_name)

        # Backproject
        try:
            pts_src, colors_src = backproject_rgbd(
                rgb_path=src_rgb_path,
                depth_path=src_depth_path,
                max_points=args.max_points,
                seed=args.seed,
            )
            pts_dst, colors_dst = backproject_rgbd(
                rgb_path=dst_rgb_path,
                depth_path=dst_depth_path,
                max_points=args.max_points,
                seed=args.seed,
            )
        except Exception as e:
            print(f"Error backprojecting RGB-D images: {e}", file=sys.stderr)
            sys.exit(1)

        # Run ICP (using register_point_clouds wrapper, init = Identity)
        try:
            res = register_point_clouds(
                src=pts_src,
                dst=pts_dst,
                max_correspondence_distance=1.0,
            )
            T_est = res.transformation
        except Exception as e:
            print(f"Error running ICP registration: {e}", file=sys.stderr)
            sys.exit(1)

        # Compute GT relative transform
        T_source_gt = parse_pose_to_transform(poses[args.source_idx])
        T_target_gt = parse_pose_to_transform(poses[target_idx])
        T_gt = derive_relative_transform(T_source_gt, T_target_gt)

        # Compute SE(3) error
        t_err, r_err = compute_se3_error(T_est, T_gt)

        # Print table row
        row_str = (
            f"{requested_b:<12.2f}{target_idx:<14}{actual_b:<18.4f}"
            f"{t_err:<15.4f}{r_err:<15.4f}{res.fitness:<10.4f}{res.inlier_rmse:<10.4f}"
        )
        print(row_str)

        # Store report data
        row_dict = {
            "source_idx": args.source_idx,
            "target_idx": target_idx,
            "requested_baseline": requested_b,
            "actual_baseline": actual_b,
            "translation_error_m": t_err,
            "rotation_error_deg": r_err,
            "fitness": res.fitness,
            "inlier_rmse": res.inlier_rmse,
        }
        report_data.append(row_dict)

        # Render the medium baseline pair (requested 0.33 m)
        if abs(requested_b - 0.33) < 1e-4:
            render_png_path = os.path.join(args.output_dir, "tartanair_icp_alignment.png")
            try:
                render_medium_pair(
                    pts_src=pts_src,
                    colors_src=colors_src,
                    pts_dst=pts_dst,
                    colors_dst=colors_dst,
                    T_est=T_est,
                    out_png_path=render_png_path,
                    baseline=actual_b,
                    t_err=t_err,
                    r_err=r_err,
                )
            except Exception as e:
                print(f"Warning: Failed to render 3D point clouds: {e}", file=sys.stderr)

    # Persist report
    json_path = os.path.join(args.output_dir, "registration_report.json")
    csv_path = os.path.join(args.output_dir, "registration_report.csv")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=report_data[0].keys())
            writer.writeheader()
            writer.writerows(report_data)
        print(f"\nSaved CSV report to: {csv_path}")
        print(f"Saved JSON report to: {json_path}")
    except Exception as e:
        print(f"Error persisting reports: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
