#!/usr/bin/env python3
"""Validate local sensor slices and render the KITTI LiDAR-camera overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatialwm.data import (
    build_kitti_sequence,
    build_tartanair_sequence,
    project_lidar_to_camera,
    write_sequence_manifest,
)
from spatialwm.perception.lidar_io import load_kitti_points


def render_kitti_projection(sequence, frame_id: int, output_path: Path) -> dict[str, float]:
    """Render depth-coloured Velodyne returns over the rectified camera image."""
    frame_lookup = {(item.frame_id, item.modality): item for item in sequence.frames}
    image_frame = frame_lookup.get((frame_id, "rgb"))
    lidar_frame = frame_lookup.get((frame_id, "lidar"))
    if image_frame is None or lidar_frame is None:
        raise ValueError(f"frame {frame_id} must contain both RGB and LiDAR")
    image_bgr = cv2.imread(image_frame.path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"could not read {image_frame.path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    points = load_kitti_points(lidar_frame.path)
    transform = sequence.calibrations[0].transform
    pixels, depth = project_lidar_to_camera(
        points, transform, sequence.intrinsics["camera_02"], image.shape[:2]
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(image)
    scatter = ax.scatter(
        pixels[:, 0], pixels[:, 1], c=depth, s=2.0, cmap="turbo", vmin=2.0, vmax=60.0
    )
    fig.colorbar(scatter, ax=ax, pad=0.01, label="Forward camera depth (m)")
    ax.set_title(
        f"KITTI frame {frame_id}: calibrated Velodyne → rectified camera_02\n"
        f"{len(pixels):,}/{len(points):,} returns have positive depth and land in-frame"
    )
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "frame_id": frame_id,
        "lidar_points": len(points),
        "visible_projected_points": len(pixels),
        "visible_fraction": len(pixels) / len(points),
        "median_visible_depth_m": float(np.median(depth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", default="data/raw/kitti")
    parser.add_argument(
        "--tartanair-root", default="data/raw/tartanair/abandonedfactory/Easy/P000"
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--projection-frame", type=int, default=0)
    parser.add_argument("--output-dir", default="data/processed/manifests")
    parser.add_argument(
        "--figure", default="figures/curated/kitti_lidar_camera_projection.png"
    )
    args = parser.parse_args()

    kitti = build_kitti_sequence(args.kitti_root, count=args.frames)
    tartanair = build_tartanair_sequence(args.tartanair_root, count=min(args.frames, 30))
    kitti_paths = write_sequence_manifest(kitti, args.output_dir)
    tartan_paths = write_sequence_manifest(tartanair, args.output_dir)
    projection_metrics = render_kitti_projection(
        kitti, args.projection_frame, Path(args.figure)
    )
    metrics_path = Path(args.figure).with_suffix(".json")
    metrics_path.write_text(json.dumps(projection_metrics, indent=2) + "\n")
    print(f"KITTI manifest: {kitti_paths[0]}")
    print(f"TartanAir manifest: {tartan_paths[0]}")
    print(f"Projection: {args.figure}")
    print(json.dumps(projection_metrics, indent=2))


if __name__ == "__main__":
    main()
