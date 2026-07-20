#!/usr/bin/env python3
"""Run and summarize frozen SfM/LiDAR settings across curated KITTI raw drives."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURATED_DRIVES = ("0001", "0005", "0011")


def _run_logged(command: list[str], log_path: Path) -> tuple[bool, str | None, float]:
    """Run one evaluator, retaining its complete output outside the committed artifacts."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.perf_counter() - start
    if completed.returncode == 0:
        return True, None, elapsed
    return False, f"exit code {completed.returncode}; inspect {log_path}", elapsed


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_sequence(
    date: str,
    drive: str,
    sfm_metrics: dict | None,
    lidar_metrics: dict | None,
    sfm_error: str | None = None,
    lidar_error: str | None = None,
    sfm_runtime_s: float | None = None,
    lidar_runtime_s: float | None = None,
) -> dict:
    """Reduce verbose per-estimator reports to stable cross-sequence evidence."""
    summary: dict[str, object] = {"date": date, "drive": drive}
    if sfm_metrics is None:
        summary["sfm"] = {"status": "failed", "error": sfm_error or "metrics missing"}
    else:
        requested = int(sfm_metrics["requested_frame_count"])
        registered = int(sfm_metrics["n_registered_cameras"])
        summary["sfm"] = {
            "status": "ok",
            "requested_views": requested,
            "registered_views": registered,
            "registration_rate": registered / requested,
            "landmarks": int(sfm_metrics["n_points"]),
            "observations": int(sfm_metrics["n_observations"]),
            "reprojection_rmse_px": float(sfm_metrics["reprojection_rmse_after_px"]),
            "sim3_aligned_ate_rmse_m_diagnostic_only": float(
                sfm_metrics["sim3_aligned_ate_rmse_m"]
            ),
            "runtime_s": sfm_runtime_s,
            "runtime_per_requested_view_s": (
                sfm_runtime_s / requested if sfm_runtime_s is not None else None
            ),
        }

    if lidar_metrics is None:
        summary["lidar"] = {
            "status": "failed",
            "error": lidar_error or "metrics missing",
        }
    else:
        scan = lidar_metrics["method_comparison"]["scan_to_scan"]
        submap = lidar_metrics["method_comparison"]["scan_to_submap"]
        diagnostics = lidar_metrics["registration_diagnostics"]["scan_to_scan"]
        summary["lidar"] = {
            "status": "ok",
            "frames": len(lidar_metrics["frame_ids"]),
            "scan_to_scan_ate_rmse_m": float(scan["rigid_aligned_ate_rmse_m"]),
            "scan_to_submap_ate_rmse_m": float(submap["rigid_aligned_ate_rmse_m"]),
            "raw_endpoint_error_m": float(scan["final_raw_position_error_m"]),
            "mean_step_translation_error_m": float(
                scan["mean_step_translation_error_m"]
            ),
            "mean_step_rotation_error_deg": float(scan["mean_step_rotation_error_deg"]),
            "mean_icp_fitness": float(np.mean([item["fitness"] for item in diagnostics])),
            "minimum_icp_fitness": float(np.min([item["fitness"] for item in diagnostics])),
            "submap_improves_ate": bool(
                submap["rigid_aligned_ate_rmse_m"]
                < scan["rigid_aligned_ate_rmse_m"]
            ),
            "runtime_s": lidar_runtime_s,
            "runtime_per_frame_s": (
                lidar_runtime_s / len(lidar_metrics["frame_ids"])
                if lidar_runtime_s is not None
                else None
            ),
        }
    return summary


def aggregate_summaries(sequences: list[dict], config: dict) -> dict:
    """Build aggregate statistics without hiding failed sequences."""
    sfm_ok = [item["sfm"] for item in sequences if item["sfm"]["status"] == "ok"]
    lidar_ok = [item["lidar"] for item in sequences if item["lidar"]["status"] == "ok"]
    return {
        "suite": "KITTI raw bounded multi-sequence perception robustness",
        "frozen_config": config,
        "sequence_count": len(sequences),
        "successful_sfm_sequences": len(sfm_ok),
        "successful_lidar_sequences": len(lidar_ok),
        "aggregate": {
            "mean_sfm_registration_rate": (
                float(np.mean([item["registration_rate"] for item in sfm_ok]))
                if sfm_ok
                else None
            ),
            "median_sfm_reprojection_rmse_px": (
                float(np.median([item["reprojection_rmse_px"] for item in sfm_ok]))
                if sfm_ok
                else None
            ),
            "median_lidar_scan_to_scan_ate_rmse_m": (
                float(np.median([item["scan_to_scan_ate_rmse_m"] for item in lidar_ok]))
                if lidar_ok
                else None
            ),
            "worst_lidar_scan_to_scan_ate_rmse_m": (
                float(np.max([item["scan_to_scan_ate_rmse_m"] for item in lidar_ok]))
                if lidar_ok
                else None
            ),
            "submap_ate_improvement_count": int(
                sum(item["submap_improves_ate"] for item in lidar_ok)
            ),
            "median_sfm_runtime_per_view_s": (
                float(
                    np.median(
                        [
                            item["runtime_per_requested_view_s"]
                            for item in sfm_ok
                            if item["runtime_per_requested_view_s"] is not None
                        ]
                    )
                )
                if any(item["runtime_per_requested_view_s"] is not None for item in sfm_ok)
                else None
            ),
            "median_lidar_runtime_per_frame_s": (
                float(
                    np.median(
                        [
                            item["runtime_per_frame_s"]
                            for item in lidar_ok
                            if item["runtime_per_frame_s"] is not None
                        ]
                    )
                )
                if any(item["runtime_per_frame_s"] is not None for item in lidar_ok)
                else None
            ),
        },
        "sequences": sequences,
        "claim_boundary": (
            "Bounded raw-drive diagnostics with one frozen configuration; not a KITTI "
            "odometry or reconstruction benchmark submission. Monocular Sim(3) ATE remains "
            "diagnostic only."
        ),
    }


def render_summary(report: dict, output_path: Path) -> None:
    """Render a compact success/failure-aware comparison across drives."""
    sequences = report["sequences"]
    labels = [f"drive {item['drive']}" for item in sequences]
    x = np.arange(len(labels))

    sfm_points = np.array(
        [item["sfm"].get("landmarks", np.nan) for item in sequences], dtype=float
    )
    sfm_rmse = np.array(
        [item["sfm"].get("reprojection_rmse_px", np.nan) for item in sequences],
        dtype=float,
    )
    scan_ate = np.array(
        [item["lidar"].get("scan_to_scan_ate_rmse_m", np.nan) for item in sequences],
        dtype=float,
    )
    submap_ate = np.array(
        [item["lidar"].get("scan_to_submap_ate_rmse_m", np.nan) for item in sequences],
        dtype=float,
    )
    fitness = np.array(
        [item["lidar"].get("mean_icp_fitness", np.nan) for item in sequences], dtype=float
    )
    step_error = np.array(
        [
            item["lidar"].get("mean_step_translation_error_m", np.nan)
            for item in sequences
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].bar(x, sfm_points, color="#4361ee")
    axes[0, 0].set_ylabel("Reconstructed landmarks")
    axes[0, 0].set_title("Real-image sparse map size", fontweight="bold")

    axes[0, 1].bar(x, sfm_rmse, color="#2a9d8f")
    axes[0, 1].set_ylabel("BA reprojection RMSE [px]")
    axes[0, 1].set_title("Global image consistency", fontweight="bold")

    width = 0.34
    axes[1, 0].bar(x - width / 2, scan_ate, width, label="Scan-to-scan", color="#4361ee")
    axes[1, 0].bar(x + width / 2, submap_ate, width, label="5-scan submap", color="#ef476f")
    axes[1, 0].set_ylabel("Rigid-aligned ATE RMSE [m]")
    axes[1, 0].set_title("Metric LiDAR trajectory error", fontweight="bold")
    axes[1, 0].legend()

    axes[1, 1].bar(x, step_error, color="#f4a261", label="Step translation error")
    axes[1, 1].set_ylabel("Mean step translation error [m]", color="#f4a261")
    axes[1, 1].tick_params(axis="y", labelcolor="#f4a261")
    fitness_axis = axes[1, 1].twinx()
    fitness_axis.plot(x, fitness, "ko--", label="Mean ICP fitness")
    fitness_axis.set_ylim(0.0, 1.05)
    fitness_axis.set_ylabel("Mean ICP fitness")
    axes[1, 1].set_title("Local motion error vs internal fit", fontweight="bold")

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.suptitle(
        "KITTI Multi-Sequence Robustness — One Frozen Configuration\n"
        f"{report['frozen_config']['sfm_views']} SfM views and "
        f"{report['frozen_config']['lidar_frames']} LiDAR frames per drive",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Failures remain visible in the JSON; SfM Sim(3) alignment is diagnostic, while "
        "LiDAR ATE fixes metric scale. These are bounded diagnostics, not benchmark scores.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.91])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", default="data/raw/kitti")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--drives", nargs="+", default=list(CURATED_DRIVES))
    parser.add_argument("--sfm-views", type=int, default=16)
    parser.add_argument("--sfm-stride", type=int, default=2)
    parser.add_argument("--lidar-frames", type=int, default=80)
    parser.add_argument("--runs-dir", default="data/processed/kitti_multisequence")
    parser.add_argument("--output-dir", default="figures/curated")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Aggregate existing per-drive reports instead of rerunning successful evaluators.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unsupported = sorted(set(args.drives) - set(CURATED_DRIVES))
    if unsupported:
        raise ValueError(f"unsupported drives: {unsupported}; curated={list(CURATED_DRIVES)}")
    if args.sfm_views < 3 or args.sfm_stride < 1 or args.lidar_frames < 3:
        raise ValueError("sfm views/lidar frames must be >=3 and stride must be positive")

    runs_dir = Path(args.runs_dir)
    sequence_summaries = []
    for drive in args.drives:
        run_dir = runs_dir / f"{args.date}_drive_{drive}"
        sfm_dir = run_dir / "sfm"
        lidar_dir = run_dir / "lidar"
        sfm_path = sfm_dir / "kitti_sparse_sfm_metrics.json"
        lidar_path = lidar_dir / "kitti_lidar_metrics.json"
        sfm_error = None
        lidar_error = None
        sfm_runtime_s = None
        lidar_runtime_s = None

        sfm_ok = args.reuse_existing and sfm_path.exists()
        if not sfm_ok:
            sfm_ok, sfm_error, sfm_runtime_s = _run_logged(
                [
                    sys.executable,
                    "scripts/evaluate_kitti_sfm.py",
                    "--kitti-root",
                    args.kitti_root,
                    "--date",
                    args.date,
                    "--drive",
                    drive,
                    "--stride",
                    str(args.sfm_stride),
                    "--frames",
                    str(args.sfm_views),
                    "--output-dir",
                    str(sfm_dir),
                ],
                run_dir / "sfm.log",
            )
        lidar_ok = args.reuse_existing and lidar_path.exists()
        if not lidar_ok:
            lidar_ok, lidar_error, lidar_runtime_s = _run_logged(
                [
                    sys.executable,
                    "scripts/evaluate_kitti_lidar.py",
                    "--kitti-root",
                    args.kitti_root,
                    "--date",
                    args.date,
                    "--drive",
                    drive,
                    "--frames",
                    str(args.lidar_frames),
                    "--output-dir",
                    str(lidar_dir),
                ],
                run_dir / "lidar.log",
            )

        sequence_summaries.append(
            summarize_sequence(
                args.date,
                drive,
                _load_json(sfm_path) if sfm_ok else None,
                _load_json(lidar_path) if lidar_ok else None,
                sfm_error,
                lidar_error,
                sfm_runtime_s,
                lidar_runtime_s,
            )
        )

    config = {
        "drives": args.drives,
        "sfm_views": args.sfm_views,
        "sfm_stride": args.sfm_stride,
        "lidar_frames": args.lidar_frames,
        "sfm_detector": "SIFT",
        "lidar_voxel_m": 0.2,
        "lidar_max_correspondence_distance_m": 1.0,
        "per_drive_tuning": False,
    }
    report = aggregate_summaries(sequence_summaries, config)
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "kitti_multisequence_metrics.json"
    figure_path = output_dir / "kitti_multisequence_summary.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    render_summary(report, figure_path)
    print(f"Metrics: {metrics_path}")
    print(f"Figure: {figure_path}")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
