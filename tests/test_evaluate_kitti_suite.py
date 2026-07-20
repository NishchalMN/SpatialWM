from __future__ import annotations

import sys
from pathlib import Path

import pytest

scripts_path = str(Path(__file__).parent.parent / "scripts")
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from evaluate_kitti_suite import aggregate_summaries, summarize_sequence  # noqa: E402


def _sfm_metrics():
    return {
        "requested_frame_count": 10,
        "n_registered_cameras": 9,
        "n_points": 1234,
        "n_observations": 3456,
        "reprojection_rmse_after_px": 0.42,
        "sim3_aligned_ate_rmse_m": 0.08,
    }


def _lidar_metrics():
    scan = {
        "rigid_aligned_ate_rmse_m": 0.3,
        "final_raw_position_error_m": 0.8,
        "mean_step_translation_error_m": 0.04,
        "mean_step_rotation_error_deg": 0.1,
    }
    submap = {**scan, "rigid_aligned_ate_rmse_m": 0.25}
    return {
        "frame_ids": list(range(20)),
        "method_comparison": {"scan_to_scan": scan, "scan_to_submap": submap},
        "registration_diagnostics": {
            "scan_to_scan": [{"fitness": 0.9}, {"fitness": 0.8}]
        },
    }


def test_summarize_sequence_preserves_claim_boundaries():
    summary = summarize_sequence("2011_09_26", "0001", _sfm_metrics(), _lidar_metrics())
    assert summary["sfm"]["registration_rate"] == pytest.approx(0.9)
    assert summary["sfm"]["reprojection_rmse_px"] == pytest.approx(0.42)
    assert summary["lidar"]["mean_icp_fitness"] == pytest.approx(0.85)
    assert summary["lidar"]["submap_improves_ate"] is True


def test_aggregate_counts_failures_instead_of_dropping_sequences():
    success = summarize_sequence("2011_09_26", "0001", _sfm_metrics(), _lidar_metrics())
    failure = summarize_sequence("2011_09_26", "0011", None, None, "sfm failed", "lidar failed")
    report = aggregate_summaries([success, failure], {"sfm_views": 10, "lidar_frames": 20})
    assert report["sequence_count"] == 2
    assert report["successful_sfm_sequences"] == 1
    assert report["successful_lidar_sequences"] == 1
    assert report["sequences"][1]["sfm"]["status"] == "failed"
