"""Download a pinned slice from one of the curated KITTI raw drives.

Lays out data under `<output-dir>/<date>/` to match pykitti structure.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

KITTI_RAW_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

# Archive sizes are pinned so a changed or truncated remote object fails before extraction.
# These three drives provide bounded, fixed inputs for the portfolio robustness suite.
CURATED_DRIVE_SIZES = {
    ("2011_09_26", "0001"): 458643963,
    ("2011_09_26", "0005"): 645940900,
    ("2011_09_26", "0011"): 939815328,
}
CALIBRATION_SIZES = {"2011_09_26": 4068}


def archive_urls(date: str, drive: str) -> tuple[str, int, str, int]:
    """Return pinned calibration/drive URLs and byte sizes for a curated raw drive."""
    key = (date, drive)
    if key not in CURATED_DRIVE_SIZES or date not in CALIBRATION_SIZES:
        supported = ", ".join(
            f"{item_date}/{item_drive}"
            for item_date, item_drive in sorted(CURATED_DRIVE_SIZES)
        )
        raise ValueError(f"unsupported KITTI raw drive {date}/{drive}; choose one of: {supported}")
    stem = f"{date}_drive_{drive}"
    drive_url = f"{KITTI_RAW_BASE_URL}/{stem}/{stem}_sync.zip"
    calibration_url = f"{KITTI_RAW_BASE_URL}/{date}_calib.zip"
    return (
        calibration_url,
        CALIBRATION_SIZES[date],
        drive_url,
        CURATED_DRIVE_SIZES[key],
    )


def positive_int(val: str) -> int:
    """Validate that the argument is a positive integer."""
    try:
        ival = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{val}' is not an integer")
    if ival <= 0:
        raise argparse.ArgumentTypeError(f"'{val}' must be positive")
    return ival


def get_remote_size(url: str) -> int:
    """Perform a HEAD request to verify size of remote archive."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            cl = resp.getheader("Content-Length")
            if cl is None:
                raise ValueError(f"Content-Length header is missing for {url}")
            return int(cl)
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve remote archive size from {url}: {e}") from e


def download_file(url: str, dest_path: Path, expected_size: int, desc: str) -> None:
    """Download a remote file with a progress bar and check downloaded size."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            cl = response.getheader("Content-Length")
            size = int(cl) if cl is not None else expected_size
            block_size = 1024 * 64
            with open(dest_path, "wb") as f:
                with tqdm(total=size, unit="B", unit_scale=True, desc=desc) as pbar:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        pbar.update(len(buffer))
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {e}") from e

    downloaded_size = dest_path.stat().st_size
    if downloaded_size != expected_size:
        raise ValueError(
            f"Downloaded file size ({downloaded_size} B) does not match "
            f"expected size ({expected_size} B) for {url}"
        )


def main() -> None:
    """Download and extract a bounded KITTI raw dataset slice."""
    ap = argparse.ArgumentParser(description="Download KITTI raw drive slice.")
    ap.add_argument(
        "--frames",
        type=positive_int,
        default=10,
        help="Number of frames to extract (positive integer, default 10)",
    )
    ap.add_argument(
        "--date",
        default="2011_09_26",
        help="KITTI raw recording date (default: 2011_09_26)",
    )
    ap.add_argument(
        "--drive",
        default="0005",
        help="Pinned KITTI raw drive number: 0001, 0005, or 0011 (default: 0005)",
    )
    ap.add_argument(
        "--output-dir",
        default="data/raw/kitti",
        help="Output directory layout base (default: data/raw/kitti)",
    )
    ap.add_argument(
        "--max-gb",
        type=float,
        default=1.0,
        help="Max allowed download size in GB (default: 1.0)",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Perform actual network transfer and extraction (default is dry run)",
    )
    ap.add_argument(
        "--list-files",
        action="store_true",
        help="List every selected archive member during a dry run.",
    )

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    frames_count = args.frames
    calib_url, calib_size, drive_url, drive_size = archive_urls(args.date, args.drive)

    # Determine files to extract
    calib_files = [
        f"{args.date}/calib_cam_to_cam.txt",
        f"{args.date}/calib_imu_to_velo.txt",
        f"{args.date}/calib_velo_to_cam.txt",
    ]

    drive_prefix = f"{args.date}/{args.date}_drive_{args.drive}_sync"
    drive_files = [
        f"{drive_prefix}/oxts/timestamps.txt",
        f"{drive_prefix}/velodyne_points/timestamps.txt",
        f"{drive_prefix}/image_02/timestamps.txt",
    ]
    for i in range(frames_count):
        drive_files.append(f"{drive_prefix}/oxts/data/{i:010d}.txt")
        drive_files.append(f"{drive_prefix}/velodyne_points/data/{i:010d}.bin")
        drive_files.append(f"{drive_prefix}/image_02/data/{i:010d}.png")

    if not args.download:
        print("[Dry Run] Dry-run mode active. No network request will be made.")
        print(f"[Dry Run] Target base directory: {out_dir.resolve()}")
        print(f"[Dry Run] Requested frames to extract: {frames_count}")
        print("[Dry Run] Archives that would be validated and downloaded:")
        print(f"  - Calibration: {calib_url} (expected size: {calib_size} bytes)")
        print(f"  - Drive Sync: {drive_url} (expected size: {drive_size} bytes)")
        print(f"[Dry Run] Total expected download size: {calib_size + drive_size} bytes")

        max_bytes = int(args.max_gb * 1024 * 1024 * 1024)
        if calib_size + drive_size > max_bytes:
            print(
                f"[Dry Run] WARNING: Expected download size exceeds "
                f"--max-gb limit of {args.max_gb} GB ({max_bytes} bytes)."
            )

        print("[Dry Run] Creating target output directory structure...")
        os.makedirs(
            out_dir / args.date / f"{args.date}_drive_{args.drive}_sync" / "oxts" / "data",
            exist_ok=True,
        )
        os.makedirs(
            out_dir
            / args.date
            / f"{args.date}_drive_{args.drive}_sync"
            / "velodyne_points"
            / "data",
            exist_ok=True,
        )
        os.makedirs(
            out_dir / args.date / f"{args.date}_drive_{args.drive}_sync" / "image_02" / "data",
            exist_ok=True,
        )

        total_files = len(calib_files) + len(drive_files)
        print(
            f"[Dry Run] Selection: {total_files} archive members = 3 calibration files, "
            f"3 timestamp files, and {frames_count} synchronized RGB/LiDAR/OXTS samples."
        )
        if args.list_files:
            for member in calib_files + drive_files:
                print(f"  - {member}")
        else:
            print("[Dry Run] Pass --list-files to print every selected archive member.")
        print("[Dry Run] Dry run complete.")
        return

    # Check sizes before downloading
    print("Performing pre-download archive validation...")
    remote_calib_size = get_remote_size(calib_url)
    remote_drive_size = get_remote_size(drive_url)

    if remote_calib_size != calib_size:
        raise ValueError(
            f"Remote calibration archive size ({remote_calib_size}) does not match "
            f"expected size ({calib_size})"
        )
    if remote_drive_size != drive_size:
        raise ValueError(
            f"Remote drive archive size ({remote_drive_size}) does not match "
            f"expected size ({drive_size})"
        )

    total_bytes = remote_calib_size + remote_drive_size
    max_bytes = int(args.max_gb * 1024 * 1024 * 1024)
    if total_bytes > max_bytes:
        raise ValueError(
            f"Total download size ({total_bytes} bytes) exceeds the "
            f"allowed --max-gb limit of {args.max_gb} GB ({max_bytes} bytes)"
        )

    # Use a temporary directory for downloading zip archives
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        calib_zip = tmp_path / "calib.zip"
        drive_zip = tmp_path / "drive.zip"

        print(f"Downloading calibration archive to {calib_zip}...")
        download_file(calib_url, calib_zip, calib_size, "Calibration")

        print(f"Downloading drive sync archive to {drive_zip}...")
        download_file(drive_url, drive_zip, drive_size, "Drive Sync")

        # Selective extraction of calibration data
        print(f"Extracting calibration files to {out_dir}...")
        with zipfile.ZipFile(calib_zip) as z:
            for member in calib_files:
                if member not in z.namelist():
                    raise ValueError(f"Required file {member} not found in calibration archive")
                z.extract(member, path=out_dir)

        # Selective extraction of drive data
        print(f"Extracting drive sync files to {out_dir}...")
        with zipfile.ZipFile(drive_zip) as z:
            for member in drive_files:
                if member not in z.namelist():
                    raise ValueError(f"Required file {member} not found in drive sync archive")
                z.extract(member, path=out_dir)

    print("Verifying extracted file count...")
    velo_dir = out_dir / drive_prefix / "velodyne_points" / "data"
    extracted_bins = list(velo_dir.glob("*.bin"))
    if len(extracted_bins) != frames_count:
        raise ValueError(
            f"Extracted velodyne scan count {len(extracted_bins)} "
            f"does not match requested frames {frames_count}"
        )

    oxts_dir = out_dir / drive_prefix / "oxts" / "data"
    extracted_txts = list(oxts_dir.glob("*.txt"))
    if len(extracted_txts) != frames_count:
        raise ValueError(
            f"Extracted OXTS data count {len(extracted_txts)} "
            f"does not match requested frames {frames_count}"
        )

    image_dir = out_dir / drive_prefix / "image_02" / "data"
    extracted_images = list(image_dir.glob("*.png"))
    if len(extracted_images) != frames_count:
        raise ValueError(
            f"Extracted camera image count {len(extracted_images)} "
            f"does not match requested frames {frames_count}"
        )

    print(f"Successfully downloaded and verified KITTI slice ({frames_count} frames) in {out_dir}")


if __name__ == "__main__":
    main()
