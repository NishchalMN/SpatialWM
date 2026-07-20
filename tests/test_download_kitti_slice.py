from __future__ import annotations

import sys
from pathlib import Path

import pytest

scripts_path = str(Path(__file__).parent.parent / "scripts")
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from download_kitti_slice import archive_urls  # noqa: E402


def test_archive_urls_are_drive_specific_and_pinned():
    calibration_url, calibration_size, drive_url, drive_size = archive_urls(
        "2011_09_26", "0001"
    )
    assert calibration_url.endswith("2011_09_26_calib.zip")
    assert calibration_size == 4068
    assert "2011_09_26_drive_0001" in drive_url
    assert drive_size == 458643963


def test_archive_urls_reject_unreviewed_drive():
    with pytest.raises(ValueError, match="unsupported KITTI raw drive"):
        archive_urls("2011_09_26", "9999")
