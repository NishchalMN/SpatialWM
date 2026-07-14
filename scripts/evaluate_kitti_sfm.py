#!/usr/bin/env python3
"""Run incremental monocular SfM on real KITTI imagery and render portfolio evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pykitti

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatialwm.eval.trajectory import umeyama
from spatialwm.geometry.sfm_toy import SfmResult, run_sfm_detailed


def camera_centres_from_world_to_camera(poses: np.ndarray) -> np.ndarray:
    """Return world-frame camera centres from world-to-camera SE(3) matrices."""
    matrices = np.asarray(poses, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    return np.asarray(
        [-pose[:3, :3].T @ pose[:3, 3] for pose in matrices], dtype=np.float64
    )


def load_kitti_camera_ground_truth(
    kitti_root: str,
    date: str,
    drive: str,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load camera-02-to-world transforms and calibrated intrinsics for selected frames."""
    ids = np.asarray(frame_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 2 or np.any(ids < 0):
        raise ValueError("frame_ids must be a one-dimensional nonnegative array")
    dataset = pykitti.raw(kitti_root, date, drive, frames=ids.tolist())
    if len(dataset.oxts) != len(ids):
        raise ValueError("KITTI OXTS count does not match the requested frames")
    T_imu_cam2 = np.linalg.inv(dataset.calib.T_cam2_imu)
    poses = np.asarray(
        [packet.T_w_imu @ T_imu_cam2 for packet in dataset.oxts], dtype=np.float64
    )
    poses = np.asarray([np.linalg.inv(poses[0]) @ pose for pose in poses])
    return poses, np.asarray(dataset.calib.K_cam2, dtype=np.float64)


def similarity_align_reconstruction(
    estimated_centres: np.ndarray,
    ground_truth_centres: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Sim(3)-align a monocular reconstruction for diagnostic visualization."""
    rotation, translation, scale = umeyama(
        estimated_centres, ground_truth_centres, with_scale=True
    )

    def align(values: np.ndarray) -> np.ndarray:
        return scale * (values @ rotation.T) + translation

    aligned_centres = align(estimated_centres)
    aligned_points = align(points)
    errors = np.linalg.norm(aligned_centres - ground_truth_centres, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors**2)))
    return aligned_centres, aligned_points, float(scale), ate_rmse


def landmark_colours(result: SfmResult) -> np.ndarray:
    """Sample each landmark's colour from its earliest available image observation."""
    colours = np.full((len(result.points), 3), 0.55, dtype=np.float64)
    assigned = np.zeros(len(result.points), dtype=bool)
    for camera_id, image_path in enumerate(result.image_paths):
        rows = result.observations[result.observations[:, 0] == camera_id]
        point_ids = rows[:, 1].astype(np.int64)
        keep = ~assigned[point_ids]
        if not np.any(keep):
            continue
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"could not decode registered image {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        point_ids = point_ids[keep]
        pixels = np.rint(rows[keep, 2:4]).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, image.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, image.shape[0] - 1)
        colours[point_ids] = image[pixels[:, 1], pixels[:, 0]] / 255.0
        assigned[point_ids] = True
    return colours


def generate_kitti_sfm_figure(
    result: SfmResult,
    ground_truth_centres: np.ndarray,
    aligned_centres: np.ndarray,
    aligned_points: np.ndarray,
    point_colours: np.ndarray,
    ate_rmse: float,
    output_path: Path,
) -> dict[str, float | int]:
    """Render the real-world images-to-map-to-trajectory story in one figure."""
    reference_bgr = cv2.imread(result.image_paths[0], cv2.IMREAD_COLOR)
    if reference_bgr is None:
        raise ValueError("failed to decode the first registered KITTI image")
    reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    first_rows = result.observations[result.observations[:, 0] == 0]
    initial_rows = first_rows[first_rows[:, 1] < result.initial_landmark_count]
    expanded_rows = first_rows[first_rows[:, 1] >= result.initial_landmark_count]

    distance = np.linalg.norm(aligned_points - aligned_centres[0], axis=1)
    display_mask = (
        (result.track_lengths >= 3)
        & np.all(np.isfinite(aligned_points), axis=1)
        & (distance <= 60.0)
    )
    if np.count_nonzero(display_mask) < 100:
        display_mask = np.all(np.isfinite(aligned_points), axis=1) & (distance <= 80.0)

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.16)
    image_axis = fig.add_subplot(grid[0, 0])
    map_axis = fig.add_subplot(grid[0, 1], projection="3d")
    trajectory_axis = fig.add_subplot(grid[1, 0])
    diagnostic_axis = fig.add_subplot(grid[1, 1])

    image_axis.imshow(reference)
    for rows, colour, label in [
        (initial_rows, "#ffd166", "Initial-pair landmark"),
        (expanded_rows, "#00e5ff", "Later map expansion"),
    ]:
        if len(rows):
            sample = rows[
                np.linspace(0, len(rows) - 1, min(160, len(rows)), dtype=np.int64)
            ]
            image_axis.scatter(
                sample[:, 2],
                sample[:, 3],
                s=18,
                facecolors="none",
                edgecolors=colour,
                linewidths=0.8,
                label=label,
            )
    image_axis.set_title(
        f"Real KITTI image and persistent landmark observations\n"
        f"frame {result.registered_image_indices[0]} | "
        f"{len(first_rows)} visible mapped points",
        fontweight="bold",
    )
    image_axis.legend(loc="lower right", fontsize=9)
    image_axis.axis("off")

    map_axis.scatter(
        aligned_points[display_mask, 0],
        aligned_points[display_mask, 2],
        -aligned_points[display_mask, 1],
        c=point_colours[display_mask],
        s=7,
        alpha=0.75,
        depthshade=False,
    )
    map_axis.plot(
        aligned_centres[:, 0],
        aligned_centres[:, 2],
        -aligned_centres[:, 1],
        "o-",
        color="#ef476f",
        linewidth=2.2,
        markersize=4,
        label="Estimated camera path",
    )
    map_axis.set_xlabel("Camera-0 X right [m]")
    map_axis.set_ylabel("Camera-0 Z forward [m]")
    map_axis.set_zlabel("Camera-0 -Y up [m]")
    map_axis.set_title(
        f"Similarity-aligned outdoor sparse map\n"
        f"{np.count_nonzero(display_mask):,} high-support points shown / "
        f"{len(result.points):,} reconstructed",
        fontweight="bold",
    )
    map_axis.view_init(elev=24, azim=-64)
    map_axis.legend(loc="upper left")

    trajectory_axis.plot(
        ground_truth_centres[:, 0],
        ground_truth_centres[:, 2],
        "o-",
        color="black",
        linewidth=3,
        markersize=5,
        label="KITTI OXTS camera GT",
    )
    trajectory_axis.plot(
        aligned_centres[:, 0],
        aligned_centres[:, 2],
        "x--",
        color="#ef476f",
        linewidth=2,
        markersize=6,
        label="Monocular SfM + Sim(3)",
    )
    for order in range(0, len(aligned_centres), 4):
        trajectory_axis.annotate(
            str(result.registered_image_indices[order]),
            (aligned_centres[order, 0], aligned_centres[order, 2]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.set_xlabel("Camera-0 X right [m]")
    trajectory_axis.set_ylabel("Camera-0 Z forward [m]")
    trajectory_axis.set_title(
        f"Camera trajectory over {len(aligned_centres)} real frames\n"
        f"Sim(3)-aligned ATE={ate_rmse:.3f} m (diagnostic only)",
        fontweight="bold",
    )
    trajectory_axis.grid(True, linestyle=":", alpha=0.4)
    trajectory_axis.legend()

    diagnostic_axis.bar(
        ["Initial\nlandmarks", "Expanded\nmap"],
        [result.initial_landmark_count, len(result.points)],
        color=["#ffd166", "#4361ee"],
        width=0.55,
        label="Landmarks",
    )
    diagnostic_axis.set_ylabel("Landmark count")
    diagnostic_axis.set_title(
        f"Map growth and global consistency\n{len(result.observations):,} observations; "
        f"median support={np.median(result.track_lengths):.0f} views",
        fontweight="bold",
    )
    diagnostic_axis.grid(True, axis="y", linestyle=":", alpha=0.35)
    reprojection_axis = diagnostic_axis.twinx()
    reprojection_axis.plot(
        [0, 1],
        [result.reprojection_rmse_before_px, result.reprojection_rmse_after_px],
        "o-",
        color="#d62728",
        linewidth=2.5,
        markersize=7,
        label="Reprojection RMSE",
    )
    reprojection_axis.set_ylabel("Reprojection RMSE [px]", color="#d62728")
    reprojection_axis.tick_params(axis="y", labelcolor="#d62728")
    handles1, labels1 = diagnostic_axis.get_legend_handles_labels()
    handles2, labels2 = reprojection_axis.get_legend_handles_labels()
    diagnostic_axis.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    diagnostic_axis.text(
        0.98,
        0.05,
        f"BA: {result.reprojection_rmse_before_px:.3f} → "
        f"{result.reprojection_rmse_after_px:.3f} px",
        transform=diagnostic_axis.transAxes,
        ha="right",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        "Real-World Incremental SfM on KITTI: Calibrated Images to Sparse 3D Map\n"
        f"2011-09-26 drive 0005 | frames {result.registered_image_indices[0]}–"
        f"{result.registered_image_indices[-1]} (stride 2) | "
        f"{len(result.poses_world_to_camera)} registered cameras",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Monocular scale is arbitrary; Sim(3) is used only to compare the camera path with "
        "OXTS. The reprojection and map-growth metrics are the portfolio claims.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.88, bottom=0.07)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    path_length = float(
        np.sum(np.linalg.norm(np.diff(ground_truth_centres, axis=0), axis=1))
    )
    return {
        "displayed_high_support_landmarks": int(np.count_nonzero(display_mask)),
        "ground_truth_path_length_m": path_length,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", default="data/raw/kitti")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--drive", default="0005")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output-dir", default="figures/curated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0 or args.stride < 1 or args.frames < 3:
        raise ValueError("start must be nonnegative, stride positive, and frames >= 3")
    drive_dir = (
        Path(args.kitti_root)
        / args.date
        / f"{args.date}_drive_{args.drive}_sync"
    )
    image_dir = drive_dir / "image_02" / "data"
    requested_ids = np.arange(
        args.start, args.start + args.stride * args.frames, args.stride, dtype=np.int64
    )
    _, K = load_kitti_camera_ground_truth(
        args.kitti_root, args.date, args.drive, requested_ids
    )
    result = run_sfm_detailed(
        image_dir,
        K,
        start=args.start,
        stride=args.stride,
        max_images=args.frames,
        detector="sift",
        ratio=0.75,
        max_features=5000,
        ransac_threshold_px=1.0,
        min_initial_points=30,
        min_pnp_points=12,
        seed=0,
        refine=True,
    )
    ground_truth_poses, _ = load_kitti_camera_ground_truth(
        args.kitti_root,
        args.date,
        args.drive,
        result.registered_image_indices,
    )
    ground_truth_centres = ground_truth_poses[:, :3, 3]
    estimated_centres = camera_centres_from_world_to_camera(
        result.poses_world_to_camera
    )
    aligned_centres, aligned_points, scale, ate_rmse = similarity_align_reconstruction(
        estimated_centres, ground_truth_centres, result.points
    )
    colours = landmark_colours(result)

    output_dir = Path(args.output_dir)
    figure_path = output_dir / "kitti_sparse_sfm.png"
    visual_metrics = generate_kitti_sfm_figure(
        result,
        ground_truth_centres,
        aligned_centres,
        aligned_points,
        colours,
        ate_rmse,
        figure_path,
    )
    np.savez_compressed(
        output_dir / "kitti_sparse_sfm_reconstruction.npz",
        points=result.points,
        poses_world_to_camera=result.poses_world_to_camera,
        poses_camera_to_world_ground_truth=ground_truth_poses,
        observations=result.observations,
        registered_image_indices=result.registered_image_indices,
        track_lengths=result.track_lengths,
        triangulation_sources=result.triangulation_sources,
        landmark_confidence=result.landmark_confidence,
        intrinsics=K,
    )
    metrics = {
        "dataset": f"KITTI raw {args.date} drive {args.drive} sync image_02",
        "requested_start_frame": args.start,
        "requested_stride": args.stride,
        "requested_frame_count": args.frames,
        "registered_image_indices": result.registered_image_indices.tolist(),
        "n_registered_cameras": len(result.poses_world_to_camera),
        "initial_pair_relative_indices": list(result.initial_pair),
        "initial_landmark_count": result.initial_landmark_count,
        "new_landmark_count": len(result.points) - result.initial_landmark_count,
        "n_points": len(result.points),
        "n_observations": len(result.observations),
        "median_track_length": float(np.median(result.track_lengths)),
        "reprojection_rmse_before_px": result.reprojection_rmse_before_px,
        "reprojection_rmse_after_px": result.reprojection_rmse_after_px,
        "pose_convention": "world-to-camera estimated; camera-to-world KITTI GT",
        "scale_convention": "monocular arbitrary scale; Sim(3) for trajectory diagnostic only",
        "sim3_scale_to_ground_truth": scale,
        "sim3_aligned_ate_rmse_m": ate_rmse,
        **visual_metrics,
        "interpretation": (
            "real-world bounded integration diagnostic; reprojection and map growth are the "
            "portfolio evidence, not the similarity-aligned ATE"
        ),
    }
    metrics_path = output_dir / "kitti_sparse_sfm_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Figure: {figure_path}")
    print(f"Metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
