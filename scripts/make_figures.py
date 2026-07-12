"""
Regenerate paper figures (collapse.png, motion-binned, horizon curves) from results,
or generate deterministic synthetic geometry visualizations.
"""

from __future__ import annotations

import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from spatialwm.geometry.ransac import fundamental_ransac


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
    outlier_x1 = rng.uniform(0.0, 640.0, (n_outliers, 2))
    outlier_x2 = rng.uniform(0.0, 480.0, (n_outliers, 2))

    x1 = np.vstack([inlier_x1, outlier_x1])
    x2 = np.vstack([inlier_x2, outlier_x2])

    # Run OpenCV-backed fundamental matrix RANSAC
    result = fundamental_ransac(
        x1, x2, thresh=1.0, p_success=0.99, max_iters=5000, method="usac_magsac"
    )

    # Plotting setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Super title detailing deterministic synthetic parameters & results
    fig.suptitle(
        "Geometry RANSAC & Epipolar Geometry Sanity (Deterministic Synthetic Scene)\n"
        f"Estimated Inliers: {np.sum(result.inliers)} / {len(x1)} ({result.inlier_ratio:.1%}), "
        f"RANSAC Iterations: {result.n_iters}",
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

    plt.tight_layout()
    out_path = os.path.join(output_dir, "geometry_ransac_epipolar.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return out_path


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
    parser.add_argument(
        "--figures",
        type=str,
        nargs="+",
        choices=["collapse", "motion-binned", "horizon", "geometry-ransac"],
        help="Specific figures to generate (default: all)",
    )
    args = parser.parse_args()

    # Determine which figures to run
    figures_to_generate = args.figures
    if figures_to_generate is None:
        figures_to_generate = ["collapse", "motion-binned", "horizon", "geometry-ransac"]

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
        if fig == "geometry-ransac":
            out_path = generate_geometry_ransac(args.output_dir)
            print(f"Generated figure '{fig}' saved to: {out_path}")


if __name__ == "__main__":
    main()
