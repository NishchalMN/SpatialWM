"""Generate reproducible geometry, real-data, and later experiment figures."""

from __future__ import annotations

import argparse
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from spatialwm.eval.trajectory import umeyama
from spatialwm.geometry.bundle_adjust import bundle_adjust, reprojection_residuals
from spatialwm.geometry.camera import transform_points, unproject
from spatialwm.geometry.features import match_features
from spatialwm.geometry.icp import register_point_clouds
from spatialwm.geometry.ransac import fundamental_ransac
from spatialwm.geometry.sfm_toy import run_sfm_detailed
from spatialwm.geometry.tartanair import (
    compute_se3_error,
    derive_relative_transform,
    parse_pose_to_transform,
)
from spatialwm.geometry.two_view import sampson_distance


def generate_bundle_adjust(output_dir: str) -> str:
    """Generate a deterministic before/after bundle-adjustment diagnostic."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(99)
    n_cams = 5
    n_points = 100

    K = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points_gt = rng.uniform(-1.5, 1.5, (n_points, 3))
    points_gt[:, 2] += 5.0

    poses_gt = np.zeros((n_cams, 6), dtype=np.float64)
    for camera_id in range(n_cams):
        rotation = Rotation.from_euler("y", camera_id * 8.0, degrees=True)
        poses_gt[camera_id, :3] = rotation.as_rotvec()
        poses_gt[camera_id, 3] = camera_id * 0.3

    observation_rows = []
    for camera_id in range(n_cams):
        rotation = Rotation.from_rotvec(poses_gt[camera_id, :3])
        points_camera = rotation.apply(points_gt) + poses_gt[camera_id, 3:]
        homogeneous = points_camera @ K.T
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        pixels += rng.normal(0.0, 0.5, pixels.shape)
        observation_rows.append(
            np.column_stack(
                [
                    np.full(n_points, camera_id),
                    np.arange(n_points),
                    pixels,
                ]
            )
        )
    observations = np.vstack(observation_rows)

    poses_initial = poses_gt + rng.normal(0.0, 0.05, poses_gt.shape)
    points_initial = points_gt + rng.normal(0.0, 0.2, points_gt.shape)
    poses_optimized, points_optimized = bundle_adjust(
        poses_initial,
        points_initial,
        K,
        observations,
    )

    def residual_matrix(poses: np.ndarray, points: np.ndarray) -> np.ndarray:
        params = np.concatenate([poses.ravel(), points.ravel()])
        return reprojection_residuals(
            params,
            n_cams,
            n_points,
            K,
            observations,
        ).reshape(-1, 2)

    residuals_before = residual_matrix(poses_initial, points_initial)
    residuals_after = residual_matrix(poses_optimized, points_optimized)
    errors_before = np.linalg.norm(residuals_before, axis=1)
    errors_after = np.linalg.norm(residuals_after, axis=1)
    mean_before = float(np.mean(errors_before))
    mean_after = float(np.mean(errors_after))
    median_before = float(np.median(errors_before))
    median_after = float(np.median(errors_after))
    improvement = mean_before / mean_after

    representative_camera = 2
    camera_mask = observations[:, 0].astype(int) == representative_camera
    observed = observations[camera_mask, 2:4]
    projected_before = observed + residuals_before[camera_mask]
    projected_after = observed + residuals_after[camera_mask]
    selected = np.linspace(0, len(observed) - 1, 35, dtype=int)

    all_pixels = np.vstack([observed, projected_before, projected_after])
    x_margin = 0.05 * np.ptp(all_pixels[:, 0])
    y_margin = 0.05 * np.ptp(all_pixels[:, 1])
    x_limits = (
        float(np.min(all_pixels[:, 0]) - x_margin),
        float(np.max(all_pixels[:, 0]) + x_margin),
    )
    y_limits = (
        float(np.max(all_pixels[:, 1]) + y_margin),
        float(np.min(all_pixels[:, 1]) - y_margin),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Sparse Bundle Adjustment: Joint Camera and 3D-Point Refinement\n"
        f"Mean reprojection error {mean_before:.2f}px -> {mean_after:.2f}px "
        f"({improvement:.1f}x improvement, 5 cameras / 100 points / 500 observations)",
        fontsize=14,
        fontweight="bold",
    )

    panels = [
        (
            axes[0],
            projected_before,
            "#d62728",
            f"Before optimization\nmean={mean_before:.2f}px, median={median_before:.2f}px",
        ),
        (
            axes[1],
            projected_after,
            "#2ca02c",
            f"After optimization\nmean={mean_after:.2f}px, median={median_after:.2f}px",
        ),
    ]
    for axis, projected, color, title in panels:
        axis.scatter(
            observed[:, 0],
            observed[:, 1],
            c="black",
            s=20,
            alpha=0.8,
            label="Observed pixels",
            zorder=3,
        )
        axis.scatter(
            projected[:, 0],
            projected[:, 1],
            c=color,
            marker="x",
            s=28,
            alpha=0.85,
            label="Projected 3D points",
            zorder=4,
        )
        for point_index in selected:
            axis.plot(
                [observed[point_index, 0], projected[point_index, 0]],
                [observed[point_index, 1], projected[point_index, 1]],
                color=color,
                linewidth=0.8,
                alpha=0.55,
                zorder=2,
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("u [pixels]")
        axis.set_ylabel("v [pixels]")
        axis.grid(True, linestyle=":", alpha=0.35)
        axis.legend(loc="upper right")

    positive_errors = np.concatenate([errors_before, errors_after])
    lower = max(float(np.min(positive_errors[positive_errors > 0])) * 0.8, 1e-3)
    upper = float(np.max(positive_errors)) * 1.2
    bins = np.geomspace(lower, upper, 35)
    axes[2].hist(errors_before, bins=bins, alpha=0.65, color="#d62728", label="Before")
    axes[2].hist(errors_after, bins=bins, alpha=0.65, color="#2ca02c", label="After")
    axes[2].axvline(mean_before, color="#d62728", linestyle="--", linewidth=2)
    axes[2].axvline(mean_after, color="#2ca02c", linestyle="--", linewidth=2)
    axes[2].set_xscale("log")
    axes[2].set_title("All-observation error distribution", fontweight="bold")
    axes[2].set_xlabel("Euclidean reprojection error [pixels, log scale]")
    axes[2].set_ylabel("Observation count")
    axes[2].grid(True, which="both", linestyle=":", alpha=0.35)
    axes[2].legend()

    fig.text(
        0.5,
        0.02,
        "Same camera, axes, and pixel scale in both image-plane panels. "
        "Lines connect observed pixels to current projections. Seed=99.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.88])

    figure_path = os.path.join(output_dir, "bundle_adjust_reprojection.png")
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    metrics = {
        "seed": 99,
        "n_cameras": n_cams,
        "n_points": n_points,
        "n_observations": len(observations),
        "mean_reprojection_error_before_px": mean_before,
        "mean_reprojection_error_after_px": mean_after,
        "median_reprojection_error_before_px": median_before,
        "median_reprojection_error_after_px": median_after,
        "mean_error_improvement_factor": improvement,
        "gauge": "first camera pose and first point world-z fixed",
    }
    metrics_path = os.path.join(output_dir, "bundle_adjust_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    return figure_path


def generate_geometry_icp(output_dir: str) -> str:
    """Generate a known-transform ICP success case with fixed visual axes."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(21)
    main_cloud = rng.uniform([-2.0, -0.8, 3.0], [2.0, 0.8, 6.5], (900, 3))
    cluster = rng.normal([1.4, 0.5, 4.2], [0.25, 0.15, 0.35], (250, 3))
    source = np.vstack([main_cloud, cluster]).astype(np.float64)

    rotation_gt = Rotation.from_euler("xyz", [2.0, -4.0, 3.0], degrees=True).as_matrix()
    translation_gt = np.array([0.25, -0.12, 0.10])
    target = source @ rotation_gt.T + translation_gt
    target += rng.normal(0.0, 0.002, target.shape)

    transform_gt = np.eye(4)
    transform_gt[:3, :3] = rotation_gt
    transform_gt[:3, 3] = translation_gt
    registration = register_point_clouds(
        source,
        target,
        max_correspondence_distance=0.8,
        max_iters=100,
        tol=1e-8,
    )
    aligned = transform_points(registration.transformation, source)
    translation_error, rotation_error = compute_se3_error(
        registration.transformation,
        transform_gt,
    )

    tree = cKDTree(target)
    residual_before, _ = tree.query(source, k=1)
    residual_after, _ = tree.query(aligned, k=1)
    median_before = float(np.median(residual_before))
    median_after = float(np.median(residual_after))

    combined_x = np.concatenate([source[:, 0], target[:, 0], aligned[:, 0]])
    combined_z = np.concatenate([source[:, 2], target[:, 2], aligned[:, 2]])
    x_limits = (float(np.min(combined_x) - 0.1), float(np.max(combined_x) + 0.1))
    z_limits = (float(np.min(combined_z) - 0.1), float(np.max(combined_z) + 0.1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def alignment_panel(axis, moving, title):
        axis.scatter(
            target[:, 0],
            target[:, 2],
            s=7,
            facecolors="none",
            edgecolors="#377eb8",
            linewidths=0.55,
            alpha=0.6,
            label="Target",
        )
        axis.scatter(
            moving[:, 0],
            moving[:, 2],
            s=6,
            c="#e41a1c",
            alpha=0.5,
            label="Source",
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Z [m]")
        axis.set_title(title, fontweight="bold")
        axis.grid(True, linestyle=":", alpha=0.35)
        axis.legend(markerscale=3)

    alignment_panel(
        axes[0],
        source,
        f"Before ICP\nmedian NN residual={median_before:.3f} m",
    )
    alignment_panel(
        axes[1],
        aligned,
        f"After ICP — identical view\nmedian NN residual={median_after:.4f} m",
    )

    positive = np.concatenate([residual_before, residual_after])
    positive = positive[positive > 0]
    bins = np.geomspace(
        max(float(np.min(positive)) * 0.8, 1e-6),
        float(np.max(positive)) * 1.2,
        35,
    )
    axes[2].hist(
        residual_before,
        bins=bins,
        color="#d62728",
        alpha=0.7,
        label="Before",
    )
    axes[2].hist(
        residual_after,
        bins=bins,
        color="#2ca02c",
        alpha=0.7,
        label="After",
    )
    axes[2].set_xscale("log")
    axes[2].set_title("Nearest-target residual distribution", fontweight="bold")
    axes[2].set_xlabel("Distance [m, log scale]")
    axes[2].set_ylabel("Point count")
    axes[2].grid(True, which="both", linestyle=":", alpha=0.35)
    axes[2].legend()

    fig.suptitle(
        "ICP Known-Transform Success Case\n"
        f"GT translation={np.linalg.norm(translation_gt):.3f} m, "
        "GT rotation=5.4 deg | "
        f"recovered error={translation_error:.4f} m, {rotation_error:.3f} deg",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Asymmetric synthetic cloud, fixed axes/view, 2 mm target noise, identity initialization.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])

    figure_path = os.path.join(output_dir, "geometry_icp_known_transform.png")
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    metrics_path = os.path.join(output_dir, "geometry_icp_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "seed": 21,
                "n_points": len(source),
                "target_noise_std_m": 0.002,
                "gt_translation_norm_m": float(np.linalg.norm(translation_gt)),
                "gt_rotation_deg": float(
                    np.degrees(
                        np.arccos(np.clip((np.trace(rotation_gt) - 1.0) / 2.0, -1.0, 1.0))
                    )
                ),
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "median_nearest_neighbour_before_m": median_before,
                "median_nearest_neighbour_after_m": median_after,
            },
            metrics_file,
            indent=2,
        )
        metrics_file.write("\n")

    return figure_path


def generate_geometry_ransac(output_dir: str) -> str:
    """Generate the geometry RANSAC and epipolar geometry sanity visualization.

    Synthesizes a calibrated two-view scene, runs OpenCV-backed RANSAC estimation,
    and plots the correspondences and epipolar lines.

    Args:
        output_dir: Directory where the output image should be saved.

    Returns:
        The file path to the generated figure.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Fixed NumPy RNG
    rng = np.random.default_rng(42)

    # Intrinsic camera matrix (VGA resolution equivalent)
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # True camera pose: translation along x, small rotation around y
    R_true = Rotation.from_euler("y", 10.0, degrees=True).as_matrix()
    t_true = np.array([[0.5], [0.0], [0.0]])

    def project(R: np.ndarray, t: np.ndarray, pts_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates."""
        Xc = (R @ pts_3d.T + t).T
        xh = (K @ Xc.T).T
        return xh[:, :2] / xh[:, 2:3]

    # Generate inlier correspondences within image bounds (640x480) in both views
    inlier_x1_list: list[np.ndarray] = []
    inlier_x2_list: list[np.ndarray] = []
    target_inliers = 120

    while len(inlier_x1_list) < target_inliers:
        n_batch = target_inliers - len(inlier_x1_list)
        # Generate random 3D points in a volume in front of the camera
        X = rng.uniform(-1.5, 1.5, (n_batch * 2, 3))
        X[:, 2] = rng.uniform(3.0, 7.0, n_batch * 2)

        pts1 = project(np.eye(3), np.zeros((3, 1)), X)
        pts2 = project(R_true, t_true, X)

        # Filter points within camera image plane boundaries
        valid = (
            (pts1[:, 0] >= 0)
            & (pts1[:, 0] < 640)
            & (pts1[:, 1] >= 0)
            & (pts1[:, 1] < 480)
            & (pts2[:, 0] >= 0)
            & (pts2[:, 0] < 640)
            & (pts2[:, 1] >= 0)
            & (pts2[:, 1] < 480)
        )

        for p1, p2 in zip(pts1[valid], pts2[valid]):
            if len(inlier_x1_list) < target_inliers:
                inlier_x1_list.append(p1)
                inlier_x2_list.append(p2)
            else:
                break

    inlier_x1 = np.array(inlier_x1_list)
    inlier_x2 = np.array(inlier_x2_list)

    # Generate outlier correspondences (uniformly random in image space)
    n_outliers = 40
    outlier_x1 = rng.uniform([0.0, 0.0], [640.0, 480.0], (n_outliers, 2))
    outlier_x2 = rng.uniform([0.0, 0.0], [640.0, 480.0], (n_outliers, 2))

    x1 = np.vstack([inlier_x1, outlier_x1])
    x2 = np.vstack([inlier_x2, outlier_x2])

    # Run OpenCV-backed fundamental matrix RANSAC
    result = fundamental_ransac(
        x1, x2, thresh=1.0, p_success=0.99, max_iters=5000, method="usac_magsac"
    )

    true_inliers = np.zeros(len(x1), dtype=bool)
    true_inliers[:target_inliers] = True
    true_positive = np.count_nonzero(result.inliers & true_inliers)
    false_positive = np.count_nonzero(result.inliers & ~true_inliers)
    false_negative = np.count_nonzero(~result.inliers & true_inliers)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    errors = sampson_distance(result.model, x1, x2)

    # Plotting setup
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    # Super title detailing deterministic synthetic parameters & results
    fig.suptitle(
        "Robust Fundamental-Matrix Recovery with 25% Injected Outliers\n"
        f"Estimated inliers {np.sum(result.inliers)}/{len(x1)} | "
        f"precision={precision:.1%}, recall={recall:.1%} | "
        f"configured iteration cap={result.n_iters}",
        fontsize=14,
        fontweight="bold",
    )

    inliers_mask = result.inliers
    outliers_mask = ~inliers_mask

    # Select a subset of 8 inliers for drawing epipolar lines
    inlier_indices = np.where(inliers_mask)[0]
    selected_indices = inlier_indices[np.linspace(0, len(inlier_indices) - 1, 8, dtype=int)]
    other_inliers = np.setdiff1d(inlier_indices, selected_indices)

    colors = plt.cm.tab10(np.arange(8))

    # Plot base points (inliers/outliers) in both views
    ax1.scatter(
        x1[other_inliers, 0],
        x1[other_inliers, 1],
        c="green",
        marker="o",
        s=20,
        alpha=0.6,
        label="RANSAC Inliers",
    )
    ax1.scatter(
        x1[outliers_mask, 0],
        x1[outliers_mask, 1],
        c="red",
        marker="x",
        s=25,
        alpha=0.6,
        label="RANSAC Outliers",
    )

    ax2.scatter(
        x2[other_inliers, 0],
        x2[other_inliers, 1],
        c="green",
        marker="o",
        s=20,
        alpha=0.6,
        label="RANSAC Inliers",
    )
    ax2.scatter(
        x2[outliers_mask, 0],
        x2[outliers_mask, 1],
        c="red",
        marker="x",
        s=25,
        alpha=0.6,
        label="RANSAC Outliers",
    )

    # Draw selected correspondences and their epipolar lines
    for idx, (k, col) in enumerate(zip(selected_indices, colors)):
        p1 = x1[k]
        p2 = x2[k]

        # Plot corresponding points with distinct colors and black borders
        ax1.scatter(p1[0], p1[1], color=col, marker="o", s=80, edgecolors="black", zorder=5)
        ax2.scatter(p2[0], p2[1], color=col, marker="o", s=80, edgecolors="black", zorder=5)

        # Compute epipolar line in View 2 (whichImage = 1)
        line = cv2.computeCorrespondEpilines(p1.reshape(1, 1, 2), 1, result.model)
        a, b, c = line[0, 0]

        # Draw line segment robustly intersecting the boundaries
        if abs(b) < 1e-5:
            lx = [-c / a, -c / a]
            ly = [0.0, 480.0]
        else:
            lx = [0.0, 640.0]
            ly0 = -c / b
            ly1 = -(a * 640.0 + c) / b
            ly = [ly0, ly1]

        ax2.plot(lx, ly, color=col, linestyle="--", linewidth=2, zorder=4)

    # Format both axes
    axes_info = [
        (ax1, "Image Plane 1 (View 1)"),
        (ax2, "Image Plane 2 (View 2) with Epipolar Lines"),
    ]
    for ax, title in axes_info:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # Inverted for image coordinates
        ax.set_xlabel("u (pixels)", fontsize=10)
        ax.set_ylabel("v (pixels)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right")

    positive_errors = errors[errors > 0]
    bins = np.geomspace(
        max(float(np.min(positive_errors)) * 0.8, 1e-9),
        float(np.max(errors)) * 1.2,
        35,
    )
    ax3.hist(
        errors[true_inliers],
        bins=bins,
        color="#2ca02c",
        alpha=0.7,
        label="Injected inliers",
    )
    ax3.hist(
        errors[~true_inliers],
        bins=bins,
        color="#d62728",
        alpha=0.7,
        label="Injected outliers",
    )
    ax3.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="1 px threshold")
    ax3.set_xscale("log")
    ax3.set_title("Sampson error separates consensus", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Sampson distance [pixel², log scale]")
    ax3.set_ylabel("Correspondence count")
    ax3.grid(True, which="both", linestyle=":", alpha=0.35)
    ax3.legend()

    fig.text(
        0.5,
        0.02,
        "Synthetic unit test: green/red classifications come from the estimated consensus; "
        "histogram colours use known injected labels. OpenCV does not report actual iterations.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.9])
    out_path = os.path.join(output_dir, "geometry_ransac_epipolar.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    metrics_path = os.path.join(output_dir, "geometry_ransac_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "seed": 42,
                "injected_inliers": target_inliers,
                "injected_outliers": n_outliers,
                "estimated_inliers": int(np.sum(result.inliers)),
                "precision": precision,
                "recall": recall,
                "inlier_ratio": result.inlier_ratio,
                "configured_iteration_cap": result.n_iters,
                "median_sampson_injected_inliers": float(np.median(errors[true_inliers])),
                "median_sampson_injected_outliers": float(np.median(errors[~true_inliers])),
            },
            metrics_file,
            indent=2,
        )
        metrics_file.write("\n")

    return out_path


def generate_tartanair_matches(
    tartanair_root: str,
    frame_idx: int,
    stride: int,
    output_dir: str,
) -> str:
    """Generate a real-image SIFT, mutual-match, and RANSAC diagnostic."""
    image_dir = os.path.join(tartanair_root, "image_left")
    pose_path = os.path.join(tartanair_root, "pose_left.txt")
    if not os.path.isdir(image_dir) or not os.path.isfile(pose_path):
        raise FileNotFoundError("TartanAir image_left and pose_left.txt are required")

    image_files = sorted(file for file in os.listdir(image_dir) if file.endswith(".png"))
    target_idx = frame_idx + stride
    if frame_idx < 0 or target_idx < 0 or target_idx >= len(image_files):
        raise IndexError("requested TartanAir feature-matching frame pair is out of bounds")

    source_bgr = cv2.imread(os.path.join(image_dir, image_files[frame_idx]))
    target_bgr = cv2.imread(os.path.join(image_dir, image_files[target_idx]))
    if source_bgr is None or target_bgr is None:
        raise ValueError("failed to decode a TartanAir RGB image")

    matches = match_features(
        source_bgr,
        target_bgr,
        detector="sift",
        ratio=0.75,
        mutual=True,
        max_features=4000,
        seed=0,
    )
    if len(matches.points1) < 8:
        raise RuntimeError("fewer than eight filtered TartanAir correspondences")
    geometry = fundamental_ransac(
        matches.points1,
        matches.points2,
        thresh=1.0,
        p_success=0.999,
        max_iters=5000,
    )
    errors = sampson_distance(geometry.model, matches.points1, matches.points2)

    poses = np.loadtxt(pose_path)
    source_pose = parse_pose_to_transform(poses[frame_idx])
    target_pose = parse_pose_to_transform(poses[target_idx])
    relative_pose = derive_relative_transform(source_pose, target_pose)
    baseline = float(np.linalg.norm(relative_pose[:3, 3]))
    rotation_deg = float(
        np.degrees(
            np.arccos(
                np.clip((np.trace(relative_pose[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            )
        )
    )

    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    height, width = source_rgb.shape[:2]
    canvas = np.hstack([source_rgb, target_rgb])

    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    match_axis = fig.add_subplot(grid[0, :])
    epipolar_axis = fig.add_subplot(grid[1, 0])
    error_axis = fig.add_subplot(grid[1, 1])

    match_axis.imshow(canvas)
    inlier_indices = np.flatnonzero(geometry.inliers)
    outlier_indices = np.flatnonzero(~geometry.inliers)
    show_inliers = inlier_indices[
        np.linspace(0, len(inlier_indices) - 1, min(55, len(inlier_indices)), dtype=int)
    ]
    show_outliers = outlier_indices[: min(15, len(outlier_indices))]
    for index in show_inliers:
        p1 = matches.points1[index]
        p2 = matches.points2[index] + np.array([width, 0.0])
        match_axis.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="#76c893",
            linewidth=0.9,
            alpha=0.72,
        )
    for index in show_outliers:
        p1 = matches.points1[index]
        p2 = matches.points2[index] + np.array([width, 0.0])
        match_axis.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="#ef476f",
            linewidth=1.2,
            alpha=0.85,
        )
    match_axis.axvline(width, color="white", linewidth=2)
    match_axis.text(12, 28, f"Source frame {frame_idx}", color="white", weight="bold")
    match_axis.text(width + 12, 28, f"Target frame {target_idx}", color="white", weight="bold")
    match_axis.set_title(
        f"Geometrically verified matches: {len(inlier_indices)} inliers / "
        f"{len(matches.points1)} filtered ({geometry.inlier_ratio:.1%})",
        fontweight="bold",
    )
    match_axis.axis("off")

    epipolar_axis.imshow(target_rgb)
    line_indices = show_inliers[
        np.linspace(0, len(show_inliers) - 1, min(8, len(show_inliers)), dtype=int)
    ]
    colours = plt.cm.tab10(np.arange(len(line_indices)))
    for index, colour in zip(line_indices, colours):
        point1 = matches.points1[index]
        point2 = matches.points2[index]
        line = cv2.computeCorrespondEpilines(point1.reshape(1, 1, 2), 1, geometry.model)[0, 0]
        a, b, c = line
        if abs(b) > 1e-8:
            xs = np.array([0.0, float(width)])
            ys = -(a * xs + c) / b
        else:
            xs = np.full(2, -c / a)
            ys = np.array([0.0, float(height)])
        epipolar_axis.plot(xs, ys, color=colour, linewidth=1.5)
        epipolar_axis.scatter(
            point2[0],
            point2[1],
            color=colour,
            edgecolor="black",
            s=45,
            zorder=3,
        )
    epipolar_axis.set_xlim(0, width)
    epipolar_axis.set_ylim(height, 0)
    epipolar_axis.set_title("Target points lie on estimated epipolar lines", fontweight="bold")
    epipolar_axis.axis("off")

    inlier_errors = errors[geometry.inliers]
    outlier_errors = errors[~geometry.inliers]
    positive = errors[errors > 0]
    bins = np.geomspace(max(float(np.min(positive)) * 0.8, 1e-9), float(np.max(errors)) * 1.2, 35)
    error_axis.hist(inlier_errors, bins=bins, alpha=0.72, color="#2ca02c", label="RANSAC inliers")
    if len(outlier_errors):
        error_axis.hist(
            outlier_errors,
            bins=bins,
            alpha=0.72,
            color="#d62728",
            label="RANSAC outliers",
        )
    error_axis.set_xscale("log")
    error_axis.set_title("Real-data Sampson error", fontweight="bold")
    error_axis.set_xlabel("Sampson distance [pixel², log scale]")
    error_axis.set_ylabel("Correspondence count")
    error_axis.grid(True, which="both", linestyle=":", alpha=0.35)
    error_axis.legend()

    fig.suptitle(
        "TartanAir Classical Correspondence Pipeline\n"
        f"SIFT keypoints {matches.n_keypoints1}/{matches.n_keypoints2} | "
        f"raw {matches.raw_match_count} -> ratio {matches.ratio_match_count} -> "
        f"mutual {matches.mutual_match_count} -> RANSAC {len(inlier_indices)} | "
        f"GT motion {baseline:.3f} m, {rotation_deg:.2f} deg",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Green lines are estimated geometric inliers; red lines are rejected filtered matches. "
        "Single-sequence diagnostic, not a benchmark.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.91])

    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, "tartanair_feature_matches.png")
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    metrics_path = os.path.join(output_dir, "tartanair_feature_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "source_frame": frame_idx,
                "target_frame": target_idx,
                "detector": matches.detector,
                "keypoints_source": matches.n_keypoints1,
                "keypoints_target": matches.n_keypoints2,
                "raw_matches": matches.raw_match_count,
                "ratio_matches": matches.ratio_match_count,
                "mutual_matches": matches.mutual_match_count,
                "ransac_inliers": int(np.sum(geometry.inliers)),
                "ransac_inlier_ratio": geometry.inlier_ratio,
                "median_sampson_inliers": float(np.median(inlier_errors)),
                "gt_baseline_m": baseline,
                "gt_rotation_deg": rotation_deg,
            },
            metrics_file,
            indent=2,
        )
        metrics_file.write("\n")

    return figure_path


def generate_tartanair_sfm(
    tartanair_root: str,
    frame_idx: int,
    frame_stride: int,
    n_frames: int,
    output_dir: str,
) -> str:
    """Generate and persist a bounded real-image sparse-SfM reconstruction."""
    image_dir = os.path.join(tartanair_root, "image_left")
    pose_path = os.path.join(tartanair_root, "pose_left.txt")
    if not os.path.isdir(image_dir) or not os.path.isfile(pose_path):
        raise FileNotFoundError("TartanAir image_left and pose_left.txt are required")
    if n_frames < 3 or frame_stride < 1:
        raise ValueError("n_frames must be >= 3 and frame_stride must be positive")

    K = np.array(
        [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    result = run_sfm_detailed(
        image_dir,
        K,
        start=frame_idx,
        stride=frame_stride,
        max_images=n_frames,
        detector="sift",
        ratio=0.75,
        max_features=4000,
        ransac_threshold_px=1.0,
        min_initial_points=30,
        min_pnp_points=12,
        seed=0,
        refine=True,
    )
    if len(result.poses_world_to_camera) < 3:
        raise RuntimeError(
            "at least three registered cameras are required for trajectory alignment"
        )

    estimated_centres = np.array(
        [
            -pose[:3, :3].T @ pose[:3, 3]
            for pose in result.poses_world_to_camera
        ]
    )
    pose_rows = np.loadtxt(pose_path)
    ground_truth_centres = np.array(
        [
            parse_pose_to_transform(pose_rows[index])[:3, 3]
            for index in result.registered_image_indices
        ]
    )
    alignment_rotation, alignment_translation, alignment_scale = umeyama(
        estimated_centres,
        ground_truth_centres,
        with_scale=True,
    )

    def align(points: np.ndarray) -> np.ndarray:
        return alignment_scale * (points @ alignment_rotation.T) + alignment_translation

    aligned_centres = align(estimated_centres)
    aligned_points = align(result.points)
    trajectory_errors = np.linalg.norm(aligned_centres - ground_truth_centres, axis=1)
    ate_rmse = float(np.sqrt(np.mean(trajectory_errors**2)))

    source_bgr = cv2.imread(result.image_paths[0], cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise ValueError("failed to decode the first registered SfM image")
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    point_colours = np.full((len(result.points), 3), 0.45, dtype=np.float64)
    first_rows = result.observations[result.observations[:, 0] == 0]
    first_ids = first_rows[:, 1].astype(np.int64)
    first_pixels = np.rint(first_rows[:, 2:4]).astype(np.int64)
    first_pixels[:, 0] = np.clip(first_pixels[:, 0], 0, source_rgb.shape[1] - 1)
    first_pixels[:, 1] = np.clip(first_pixels[:, 1], 0, source_rgb.shape[0] - 1)
    point_colours[first_ids] = (
        source_rgb[first_pixels[:, 1], first_pixels[:, 0]].astype(np.float64) / 255.0
    )

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    observation_axis = fig.add_subplot(grid[0, 0])
    cloud_axis = fig.add_subplot(grid[0, 1], projection="3d")
    trajectory_axis = fig.add_subplot(grid[1, 0])
    diagnostic_axis = fig.add_subplot(grid[1, 1])

    observation_axis.imshow(source_rgb)
    display_rows = first_rows[
        np.linspace(0, len(first_rows) - 1, min(180, len(first_rows)), dtype=int)
    ]
    observation_axis.scatter(
        display_rows[:, 2],
        display_rows[:, 3],
        s=18,
        facecolors="none",
        edgecolors="#00e5ff",
        linewidths=0.7,
    )
    observation_axis.set_title(
        f"Reference observations (frame {result.registered_image_indices[0]})\n"
        f"{len(first_rows)} visible expanded-map landmarks",
        fontweight="bold",
    )
    observation_axis.axis("off")

    cloud_axis.scatter(
        aligned_points[:, 0],
        aligned_points[:, 1],
        aligned_points[:, 2],
        c=point_colours,
        s=10,
        alpha=0.75,
        depthshade=False,
    )
    cloud_axis.plot(
        aligned_centres[:, 0],
        aligned_centres[:, 1],
        aligned_centres[:, 2],
        "o-",
        color="#ef476f",
        linewidth=2.0,
        markersize=5,
        label="Estimated cameras",
    )
    cloud_axis.set_xlabel("World X [m]")
    cloud_axis.set_ylabel("World Y [m]")
    cloud_axis.set_zlabel("World Z [m]")
    cloud_axis.set_title(
        f"Similarity-aligned sparse cloud\n{len(result.points)} points / "
        f"{len(result.poses_world_to_camera)} cameras",
        fontweight="bold",
    )
    cloud_axis.view_init(elev=22, azim=-58)
    cloud_axis.legend(loc="upper left")

    trajectory_axis.plot(
        ground_truth_centres[:, 0],
        ground_truth_centres[:, 2],
        "o-",
        color="black",
        linewidth=3,
        markersize=7,
        label="TartanAir GT",
    )
    trajectory_axis.plot(
        aligned_centres[:, 0],
        aligned_centres[:, 2],
        "x--",
        color="#ef476f",
        linewidth=2,
        markersize=8,
        label="Monocular SfM + Sim(3)",
    )
    for order, frame in enumerate(result.registered_image_indices):
        trajectory_axis.annotate(
            str(frame),
            (aligned_centres[order, 0], aligned_centres[order, 2]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.set_xlabel("World X [m]")
    trajectory_axis.set_ylabel("World Z [m]")
    trajectory_axis.set_title(
        f"Camera path after monocular scale alignment\nATE RMSE={ate_rmse:.4f} m "
        f"over {len(aligned_centres)} nearby frames",
        fontweight="bold",
    )
    trajectory_axis.grid(True, linestyle=":", alpha=0.4)
    trajectory_axis.legend()

    diagnostic_axis.bar(
        ["Before BA", "After BA"],
        [result.reprojection_rmse_before_px, result.reprojection_rmse_after_px],
        color=["#d62728", "#2ca02c"],
        width=0.55,
        label="Reprojection RMSE",
    )
    diagnostic_axis.axhline(2.0, color="#555555", linestyle="--", label="2 px target")
    diagnostic_axis.set_ylabel("Reprojection RMSE [pixels]")
    diagnostic_axis.set_title(
        "Global refinement and track support\n"
        f"{result.initial_landmark_count} initial → {len(result.points)} landmarks; "
        f"{len(result.observations)} observations; "
        f"median track length={np.median(result.track_lengths):.0f}",
        fontweight="bold",
    )
    diagnostic_axis.grid(True, axis="y", linestyle=":", alpha=0.4)
    diagnostic_axis.legend(loc="upper right")
    diagnostic_axis.set_xlim(-0.7, 1.7)
    track_axis = diagnostic_axis.inset_axes([0.55, 0.15, 0.40, 0.48])
    track_values, track_counts = np.unique(result.track_lengths, return_counts=True)
    track_axis.bar(
        track_values,
        track_counts,
        color="#4361ee",
        alpha=0.8,
        width=0.65,
    )
    track_axis.set_title("Landmark support", fontsize=9, fontweight="bold")
    track_axis.set_xlabel("Track length [views]", fontsize=8)
    track_axis.set_ylabel("Landmarks", fontsize=8)
    track_axis.set_xticks(track_values)
    track_axis.tick_params(labelsize=8)
    track_axis.grid(True, axis="y", linestyle=":", alpha=0.3)

    fig.suptitle(
        "Incremental Sparse SfM on TartanAir: Images to Expanded 3D Structure\n"
        f"P000 frames {result.registered_image_indices[0]}-"
        f"{result.registered_image_indices[-1]} | initial pair relative indices "
        f"{result.initial_pair} | reprojection RMSE "
        f"{result.reprojection_rmse_before_px:.3f} -> "
        f"{result.reprojection_rmse_after_px:.3f} px",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "World-to-camera poses; arbitrary monocular scale aligned to GT only for trajectory "
        f"evaluation. {n_frames}-frame controlled diagnostic, not a benchmark.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, "tartanair_sparse_sfm.png")
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    np.savez_compressed(
        os.path.join(output_dir, "tartanair_sparse_sfm_reconstruction.npz"),
        points=result.points,
        poses_world_to_camera=result.poses_world_to_camera,
        observations=result.observations,
        registered_image_indices=result.registered_image_indices,
        track_lengths=result.track_lengths,
        initial_landmark_count=result.initial_landmark_count,
        triangulation_sources=result.triangulation_sources,
        landmark_confidence=result.landmark_confidence,
        intrinsics=K,
    )
    metrics_path = os.path.join(output_dir, "tartanair_sparse_sfm_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "sequence": "abandonedfactory/Easy/P000",
                "requested_start_frame": frame_idx,
                "requested_frame_stride": frame_stride,
                "requested_frame_count": n_frames,
                "registered_image_indices": result.registered_image_indices.tolist(),
                "initial_pair_relative_indices": list(result.initial_pair),
                "pose_convention": "world-to-camera",
                "scale_convention": "initial translation unit; monocular scale arbitrary",
                "n_registered_cameras": len(result.poses_world_to_camera),
                "initial_landmark_count": result.initial_landmark_count,
                "new_landmark_count": len(result.points) - result.initial_landmark_count,
                "n_points": len(result.points),
                "n_observations": len(result.observations),
                "median_track_length": float(np.median(result.track_lengths)),
                "reprojection_rmse_before_px": result.reprojection_rmse_before_px,
                "reprojection_rmse_after_px": result.reprojection_rmse_after_px,
                "sim3_alignment_scale_to_gt": alignment_scale,
                "sim3_aligned_ate_rmse_m": ate_rmse,
                "interpretation": (
                    f"bounded {n_frames}-frame integration diagnostic; the short aligned "
                    "trajectory is not a benchmark result"
                ),
            },
            metrics_file,
            indent=2,
        )
        metrics_file.write("\n")
    return figure_path


def generate_tartanair_rgbd(tartanair_root: str, frame_idx: int, output_dir: str) -> str:
    """Generate the TartanAir RGB-D and top-down trajectory preview.

    Args:
        tartanair_root: Root directory of the TartanAir sequence.
        frame_idx: Index of the frame to preview.
        output_dir: Directory where the output image should be saved.

    Returns:
        The file path to the generated figure.
    """
    if not os.path.exists(tartanair_root):
        raise FileNotFoundError(f"TartanAir root directory '{tartanair_root}' does not exist.")
    if not os.path.isdir(tartanair_root):
        raise NotADirectoryError(f"TartanAir root path '{tartanair_root}' is not a directory.")

    image_dir = os.path.join(tartanair_root, "image_left")
    depth_dir = os.path.join(tartanair_root, "depth_left")
    pose_path = os.path.join(tartanair_root, "pose_left.txt")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' not found.")
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Depth directory '{depth_dir}' not found.")
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"Pose file '{pose_path}' not found.")

    try:
        poses = np.loadtxt(pose_path)
    except Exception as e:
        raise ValueError(f"Malformed pose file '{pose_path}': {e}")

    if poses.size == 0:
        raise ValueError(f"Pose file '{pose_path}' is empty.")
    if len(poses.shape) != 2 or poses.shape[1] < 3:
        raise ValueError(
            f"Malformed pose file '{pose_path}': expected at least 3 columns (x, y, z), "
            f"but got shape {poses.shape}."
        )

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in '{image_dir}'.")

    n_images = len(image_files)
    n_poses = len(poses)
    if n_images != n_poses:
        raise ValueError(
            f"Mismatched dataset bounds: found {n_images} images in '{image_dir}' "
            f"but {n_poses} poses in '{pose_path}'."
        )

    if frame_idx < 0 or frame_idx >= n_images:
        raise IndexError(
            f"Frame index {frame_idx} is out of bounds for the {n_images} available frames."
        )

    selected_image_name = image_files[frame_idx]
    import re
    match = re.match(r"^(\d+)", selected_image_name)
    if not match:
        raise ValueError(
            f"Could not extract frame identifier from image filename '{selected_image_name}'. "
            f"Expected filename to start with digits."
        )
    frame_id = match.group(1)

    depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
    matching_depth_files = [f for f in depth_files if f.startswith(frame_id)]
    if not matching_depth_files:
        raise FileNotFoundError(
            f"No depth file found matching frame identifier '{frame_id}' in '{depth_dir}'."
        )
    if len(matching_depth_files) > 1:
        matching_filename = next(
            (f for f in matching_depth_files if "left" in f),
            matching_depth_files[0]
        )
    else:
        matching_filename = matching_depth_files[0]

    depth_path = os.path.join(depth_dir, matching_filename)

    try:
        depth_image = np.load(depth_path)
    except Exception as e:
        raise ValueError(f"Malformed depth file '{depth_path}': {e}")

    rgb_path = os.path.join(image_dir, selected_image_name)
    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        raise ValueError(f"Malformed or missing RGB image file '{rgb_path}'.")
    rgb_image = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    norm_path = os.path.normpath(tartanair_root)
    parts = norm_path.split(os.sep)
    if len(parts) >= 3:
        seq = parts[-1]
        diff = parts[-2]
        env = parts[-3]
        if env.lower() == "abandonedfactory":
            env_display = "AbandonedFactory"
        else:
            env_display = env.capitalize() if env.islower() else env
        title_str = f"TartanAir {env_display} / {diff} / {seq}"
    else:
        title_str = "TartanAir AbandonedFactory / Easy / P000"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    # RGB Plot
    ax1.imshow(rgb_image)
    ax1.set_title(f"RGB Image (Frame {frame_id})", fontsize=12)
    ax1.axis("off")

    # Depth Plot
    invalid_mask = ~np.isfinite(depth_image) | (depth_image >= 1e3)
    masked_depth = np.ma.masked_where(invalid_mask, depth_image)

    valid_depth = depth_image[~invalid_mask]
    if len(valid_depth) > 0:
        vmin = np.percentile(valid_depth, 1)
        vmax = np.percentile(valid_depth, 99)
    else:
        vmin, vmax = 0.1, 10.0

    cmap = plt.colormaps.get_cmap("viridis").copy()
    cmap.set_bad(color="gray")
    im = ax2.imshow(masked_depth, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title("Depth Map", fontsize=12)
    ax2.axis("off")
    cbar = fig.colorbar(im, ax=ax2, orientation="vertical", pad=0.05, shrink=0.8)
    cbar.set_label("Depth [m]")

    # Trajectory Plot
    ax3.plot(poses[:, 0], poses[:, 1], "b-", alpha=0.7, label="Trajectory")
    ax3.plot(poses[frame_idx, 0], poses[frame_idx, 1], "ro", markersize=8, label="Current Frame")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_title("Top-down Trajectory", fontsize=12)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend()
    ax3.set_aspect("equal", "box")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tartanair_rgbd_preview.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return out_path


def _generate_tartanair_icp_legacy(
    tartanair_root: str,
    frame_idx: int,
    stride: int,
    output_dir: str,
) -> str:
    """Generate TartanAir real-data ICP point-cloud alignment visualization.

    Args:
        tartanair_root: Root directory of the TartanAir sequence.
        frame_idx: Index of the first frame to load.
        stride: Frame stride/step to the second frame.
        output_dir: Directory where the output image should be saved.

    Returns:
        The file path to the generated figure.
    """
    if not os.path.exists(tartanair_root):
        raise FileNotFoundError(f"TartanAir root directory '{tartanair_root}' does not exist.")
    if not os.path.isdir(tartanair_root):
        raise NotADirectoryError(f"TartanAir root path '{tartanair_root}' is not a directory.")

    image_dir = os.path.join(tartanair_root, "image_left")
    depth_dir = os.path.join(tartanair_root, "depth_left")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' not found.")
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Depth directory '{depth_dir}' not found.")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in '{image_dir}'.")

    n_images = len(image_files)
    if frame_idx < 0 or frame_idx >= n_images:
        raise IndexError(
            f"Frame index {frame_idx} is out of bounds for the {n_images} available frames."
        )

    frame_idx2 = frame_idx + stride
    if frame_idx2 < 0 or frame_idx2 >= n_images:
        raise IndexError(
            f"Second frame index {frame_idx2} (frame_idx={frame_idx} + stride={stride}) "
            f"is out of bounds for the {n_images} available frames."
        )

    # Local helper to load and project a frame's depth map
    def load_and_project(idx: int) -> tuple[np.ndarray, str]:
        selected_image_name = image_files[idx]
        import re
        match = re.match(r"^(\d+)", selected_image_name)
        if not match:
            raise ValueError(
                f"Could not extract frame identifier from image filename '{selected_image_name}'."
            )
        frame_id = match.group(1)

        depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
        matching_depth_files = [f for f in depth_files if f.startswith(frame_id)]
        if not matching_depth_files:
            raise FileNotFoundError(
                f"No depth file found matching frame identifier '{frame_id}' in '{depth_dir}'."
            )

        if len(matching_depth_files) > 1:
            matching_filename = next(
                (f for f in matching_depth_files if "left" in f),
                matching_depth_files[0]
            )
        else:
            matching_filename = matching_depth_files[0]

        depth_path = os.path.join(depth_dir, matching_filename)
        try:
            depth_image = np.load(depth_path)
        except Exception as e:
            raise ValueError(f"Malformed depth file '{depth_path}': {e}")

        # Mask non-finite and >= 1e3
        h, w = depth_image.shape
        v_grid, u_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        u = u_grid.flatten()
        v = v_grid.flatten()
        z = depth_image.flatten()

        valid_mask = np.isfinite(z) & (z < 1e3) & (z > 0.0)
        u = u[valid_mask]
        v = v[valid_mask]
        z = z[valid_mask]

        # Bounded subsample: 5000 points
        max_points = 5000
        if len(z) > max_points:
            rng = np.random.default_rng(42)
            sel_idx = rng.choice(len(z), max_points, replace=False)
            u = u[sel_idx]
            v = v[sel_idx]
            z = z[sel_idx]

        # Calibration assumption: fx=fy=320, cx=320, cy=240
        K = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)

        uv = np.stack([u, v], axis=1)
        pts_3d = unproject(K, uv, z)
        return pts_3d, frame_id

    pts1, id1 = load_and_project(frame_idx)
    pts2, id2 = load_and_project(frame_idx2)

    # Run Open3D ICP registration via the spatialwm wrapper
    res = register_point_clouds(pts1, pts2, max_correspondence_distance=1.0)
    pts1_trans = transform_points(res.transformation, pts1)

    fig = plt.figure(figsize=(18, 6))

    def setup_3d_scatter(ax, p1: np.ndarray, p2: np.ndarray, title: str):
        # Map camera coords (x,y,z) to matplotlib 3D coords:
        # x_plt = x, y_plt = z, z_plt = -y
        ax.scatter(p1[:, 0], p1[:, 2], -p1[:, 1], c="red", s=2, alpha=0.5, label=f"Source ({id1})")
        ax.scatter(p2[:, 0], p2[:, 2], -p2[:, 1], c="blue", s=2, alpha=0.5, label=f"Target ({id2})")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X (Right) [m]", fontsize=9)
        ax.set_ylabel("Z (Depth) [m]", fontsize=9)
        ax.set_zlabel("-Y (Up) [m]", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", markerscale=5)

    # Panel 1: Before alignment
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    setup_3d_scatter(ax1, pts1, pts2, "Before Alignment (Source vs Target)")

    # Panel 2: After alignment
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    setup_3d_scatter(ax2, pts1_trans, pts2, "After Alignment (Aligned Source vs Target)")

    # Panel 3: Top-down alignment view (2D X-vs-Z scatter)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(
        pts1_trans[:, 0],
        pts1_trans[:, 2],
        c="red",
        s=2,
        alpha=0.5,
        label=f"Source ({id1})",
    )
    ax3.scatter(
        pts2[:, 0],
        pts2[:, 2],
        c="blue",
        s=2,
        alpha=0.5,
        label=f"Target ({id2})",
    )
    ax3.set_title("Top-down Alignment View", fontsize=12, fontweight="bold")
    ax3.set_xlabel("X (Right) [m]", fontsize=9)
    ax3.set_ylabel("Z (Depth) [m]", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", markerscale=5)
    ax3.set_aspect("equal", "box")

    title_str = (
        f"TartanAir Real-Data ICP Alignment (P000: {id1} -> {id2}, stride={stride})\n"
        f"Sampled Points: {len(pts1)} | Open3D Fitness: {res.fitness:.4f} | "
        f"Open3D Inlier RMSE: {res.inlier_rmse:.4f}m\n"
        "Assumed Camera Calibration: fx=fy=320, cx=320, cy=240 (90° HFOV) | "
        "*Diagnostic Visualization Only - Not a Benchmark*"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tartanair_icp_alignment.png")
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return out_path


def generate_tartanair_icp(
    tartanair_root: str,
    frame_idx: int,
    stride: int,
    output_dir: str,
) -> str:
    """Generate a fixed-view, ground-truth-validated RGB-D ICP diagnostic."""
    image_dir = os.path.join(tartanair_root, "image_left")
    depth_dir = os.path.join(tartanair_root, "depth_left")
    pose_path = os.path.join(tartanair_root, "pose_left.txt")
    if (
        not os.path.isdir(image_dir)
        or not os.path.isdir(depth_dir)
        or not os.path.isfile(pose_path)
    ):
        raise FileNotFoundError("TartanAir image_left, depth_left, and pose_left.txt are required")

    image_files = sorted(file for file in os.listdir(image_dir) if file.endswith(".png"))
    depth_files = sorted(file for file in os.listdir(depth_dir) if file.endswith(".npy"))
    target_idx = frame_idx + stride
    if frame_idx < 0 or target_idx < 0 or target_idx >= len(image_files):
        raise IndexError("requested TartanAir ICP frame pair is out of bounds")

    K = np.array(
        [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    def load_frame(index: int) -> tuple[np.ndarray, np.ndarray, str]:
        image_name = image_files[index]
        frame_id = image_name.split("_")[0]
        depth_candidates = [file for file in depth_files if file.startswith(frame_id)]
        if not depth_candidates:
            raise FileNotFoundError(f"no depth map found for TartanAir frame {frame_id}")

        bgr = cv2.imread(os.path.join(image_dir, image_name))
        if bgr is None:
            raise ValueError(f"failed to decode TartanAir frame {frame_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(os.path.join(depth_dir, depth_candidates[0]))

        height, width = depth.shape
        rows, columns = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )
        z = depth.ravel()
        u = columns.ravel()
        v = rows.ravel()
        valid = np.isfinite(z) & (z > 0.0) & (z < 50.0)
        z = z[valid]
        u = u[valid]
        v = v[valid]

        rng = np.random.default_rng(42)
        if len(z) > 8000:
            selected = rng.choice(len(z), 8000, replace=False)
            z = z[selected]
            u = u[selected]
            v = v[selected]
        points = unproject(K, np.column_stack([u, v]), z)
        return points, rgb, frame_id

    source_points, source_rgb, source_id = load_frame(frame_idx)
    target_points, target_rgb, target_id = load_frame(target_idx)
    registration = register_point_clouds(
        source_points,
        target_points,
        max_correspondence_distance=1.0,
        max_iters=80,
        tol=1e-7,
    )
    aligned_source = transform_points(registration.transformation, source_points)

    poses = np.loadtxt(pose_path)
    source_pose = parse_pose_to_transform(poses[frame_idx])
    target_pose = parse_pose_to_transform(poses[target_idx])
    ground_truth = derive_relative_transform(source_pose, target_pose)
    translation_error, rotation_error = compute_se3_error(
        registration.transformation,
        ground_truth,
    )
    baseline = float(np.linalg.norm(ground_truth[:3, 3]))
    ground_truth_rotation = float(
        np.degrees(
            np.arccos(
                np.clip((np.trace(ground_truth[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            )
        )
    )

    target_tree = cKDTree(target_points)
    residual_before, _ = target_tree.query(source_points, k=1)
    residual_after, _ = target_tree.query(aligned_source, k=1)
    median_before = float(np.median(residual_before))
    median_after = float(np.median(residual_after))
    translation_error_ratio = translation_error / max(baseline, 1e-12)

    display_rng = np.random.default_rng(5)
    source_display = display_rng.choice(
        len(source_points),
        min(1800, len(source_points)),
        replace=False,
    )
    target_display = display_rng.choice(
        len(target_points),
        min(1800, len(target_points)),
        replace=False,
    )

    combined_x = np.concatenate(
        [source_points[:, 0], target_points[:, 0], aligned_source[:, 0]]
    )
    combined_z = np.concatenate(
        [source_points[:, 2], target_points[:, 2], aligned_source[:, 2]]
    )
    x_limits = tuple(np.percentile(combined_x, [1, 99]))
    z_limits = tuple(np.percentile(combined_z, [1, 99]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].imshow(source_rgb)
    axes[0, 0].set_title(f"Source RGB: frame {source_id}", fontweight="bold")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(target_rgb)
    axes[0, 1].set_title(f"Target RGB: frame {target_id}", fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.02,
        0.97,
        "Ground-truth validation\n\n"
        f"GT motion: {baseline:.3f} m, {ground_truth_rotation:.2f} deg\n"
        f"ICP translation error: {translation_error:.3f} m\n"
        f"ICP rotation error: {rotation_error:.2f} deg\n\n"
        "Internal alignment diagnostics\n\n"
        f"Open3D fitness: {registration.fitness:.3f}\n"
        f"Open3D inlier RMSE: {registration.inlier_rmse:.3f} m\n"
        f"Median NN residual: {median_before:.3f} -> {median_after:.3f} m",
        va="top",
        fontsize=12,
        bbox={"facecolor": "#f5f5f5", "edgecolor": "#888888", "boxstyle": "round,pad=0.7"},
    )

    def plot_alignment(axis, source, title):
        axis.scatter(
            target_points[target_display, 0],
            target_points[target_display, 2],
            s=5,
            facecolors="none",
            edgecolors="#377eb8",
            linewidths=0.45,
            alpha=0.55,
            label="Target",
        )
        axis.scatter(
            source[source_display, 0],
            source[source_display, 2],
            s=5,
            c="#e41a1c",
            alpha=0.5,
            label="Source",
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("X right [m]")
        axis.set_ylabel("Z forward [m]")
        axis.set_title(title, fontweight="bold")
        axis.grid(True, linestyle=":", alpha=0.35)
        axis.legend(markerscale=3)

    plot_alignment(
        axes[1, 0],
        source_points,
        f"Before ICP\nmedian nearest-neighbour residual={median_before:.3f} m",
    )
    plot_alignment(
        axes[1, 1],
        aligned_source,
        f"After ICP — identical view\nmedian nearest-neighbour residual={median_after:.3f} m",
    )

    colour_max = max(float(np.percentile(residual_after, 95)), 1e-6)
    residual_plot = axes[1, 2].scatter(
        aligned_source[source_display, 0],
        aligned_source[source_display, 2],
        c=residual_after[source_display],
        cmap="magma",
        vmin=0.0,
        vmax=colour_max,
        s=7,
        alpha=0.8,
    )
    axes[1, 2].scatter(
        target_points[target_display, 0],
        target_points[target_display, 2],
        c="#999999",
        s=2,
        alpha=0.15,
    )
    axes[1, 2].set_xlim(*x_limits)
    axes[1, 2].set_ylim(*z_limits)
    axes[1, 2].set_aspect("equal", adjustable="box")
    axes[1, 2].set_xlabel("X right [m]")
    axes[1, 2].set_ylabel("Z forward [m]")
    axes[1, 2].set_title("After-ICP residual map", fontweight="bold")
    axes[1, 2].grid(True, linestyle=":", alpha=0.35)
    colour_bar = fig.colorbar(residual_plot, ax=axes[1, 2], fraction=0.046, pad=0.04)
    colour_bar.set_label("Nearest-target distance [m]")

    fig.suptitle(
        "ICP Failure Case: Local Alignment Improves, but Ground-Truth Translation Is Wrong\n"
        f"P000 frame {frame_idx} -> {target_idx} (stride={stride}) | "
        f"translation error is {translation_error_ratio:.1f}x the {baseline:.3f} m baseline | "
        "single-pair diagnostic, not a benchmark",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, "tartanair_icp_alignment.png")
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    metrics_path = os.path.join(output_dir, "tartanair_icp_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "source_frame": frame_idx,
                "target_frame": target_idx,
                "stride": stride,
                "gt_baseline_m": baseline,
                "gt_rotation_deg": ground_truth_rotation,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "translation_error_over_baseline": translation_error_ratio,
                "open3d_fitness": registration.fitness,
                "open3d_inlier_rmse_m": registration.inlier_rmse,
                "median_nearest_neighbour_before_m": median_before,
                "median_nearest_neighbour_after_m": median_after,
                "estimated_source_to_target": registration.transformation.tolist(),
                "ground_truth_source_to_target": ground_truth.tolist(),
                "interpretation": (
                    "failure case: nearest-neighbour alignment improves while "
                    "ground-truth translation remains incorrect"
                ),
            },
            metrics_file,
            indent=2,
        )
        metrics_file.write("\n")

    return figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from experimental results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experimental results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for generated figures",
    )
    choices = [
        "bundle-adjust",
        "collapse",
        "motion-binned",
        "horizon",
        "geometry-ransac",
        "geometry-icp",
        "tartanair-rgbd",
        "tartanair-matches",
        "tartanair-sfm",
        "tartanair-icp",
    ]
    parser.add_argument(
        "--figures",
        type=str,
        nargs="+",
        choices=choices,
        help="Specific figures to generate (default: all)",
    )
    parser.add_argument(
        "--tartanair-root",
        type=str,
        default="data/raw/tartanair/abandonedfactory/Easy/P000",
        help="Root directory of TartanAir slice",
    )
    parser.add_argument(
        "--tartanair-frame",
        type=int,
        default=1750,
        help="Frame index to preview in tartanair-rgbd",
    )
    parser.add_argument(
        "--tartanair-stride",
        type=int,
        default=5,
        help="Frame stride/step to the second frame for tartanair-icp figure",
    )
    parser.add_argument(
        "--tartanair-sfm-stride",
        type=int,
        default=1,
        help="Frame step within the bounded TartanAir SfM sequence",
    )
    parser.add_argument(
        "--tartanair-sfm-frames",
        type=int,
        default=6,
        help="Number of requested frames in the bounded TartanAir SfM sequence",
    )
    args = parser.parse_args()

    # Determine which figures to run
    figures_to_generate = args.figures
    if figures_to_generate is None:
        figures_to_generate = ["geometry-ransac", "geometry-icp", "bundle-adjust"]

    # Figures requiring experimental results
    results_needed_figs = ["collapse", "motion-binned", "horizon"]
    requested_results_figs = [fig for fig in figures_to_generate if fig in results_needed_figs]

    # If any requested figure requires results files, check results-dir and fail clearly
    if requested_results_figs:
        raise FileNotFoundError(
            f"Experimental results directory '{args.results_dir}' is missing or does not contain "
            f"the required files to generate the following figures: {requested_results_figs}."
        )

    # Generate chosen figures
    for fig in figures_to_generate:
        if fig == "bundle-adjust":
            out_path = generate_bundle_adjust(args.output_dir)
            print(f"Generated figure '{fig}' saved to: {out_path}")
        elif fig == "geometry-icp":
            out_path = generate_geometry_icp(args.output_dir)
            print(f"Generated figure '{fig}' saved to: {out_path}")
        elif fig == "geometry-ransac":
            out_path = generate_geometry_ransac(args.output_dir)
            print(f"Generated figure '{fig}' saved to: {out_path}")
        elif fig == "tartanair-rgbd":
            out_path = generate_tartanair_rgbd(
                args.tartanair_root, args.tartanair_frame, args.output_dir
            )
            print(f"Generated figure '{fig}' saved to: {out_path}")
        elif fig == "tartanair-matches":
            out_path = generate_tartanair_matches(
                args.tartanair_root,
                args.tartanair_frame,
                args.tartanair_stride,
                args.output_dir,
            )
            print(f"Generated figure '{fig}' saved to: {out_path}")
        elif fig == "tartanair-sfm":
            out_path = generate_tartanair_sfm(
                args.tartanair_root,
                args.tartanair_frame,
                args.tartanair_sfm_stride,
                args.tartanair_sfm_frames,
                args.output_dir,
            )
            print(f"Generated figure '{fig}' saved to: {out_path}")
        elif fig == "tartanair-icp":
            out_path = generate_tartanair_icp(
                args.tartanair_root, args.tartanair_frame, args.tartanair_stride, args.output_dir
            )
            print(f"Generated figure '{fig}' saved to: {out_path}")


if __name__ == "__main__":
    main()
