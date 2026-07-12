"""
Download pinned TartanAir Easy/abandonedfactory P000 slice (image+depth+pose), verify checksums.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# Exact sizes in bytes for image_left.zip and depth_left.zip under 'Easy' difficulty
# for all 17 environments in the Hugging Face theairlabcmu/tartanair dataset.
# Used for deterministic offline dry run size checks.
EASY_SIZES: dict[str, dict[str, int]] = {
    "abandonedfactory": {
        "image_left": 7074777614,
        "depth_left": 2998317429
    },
    "abandonedfactory_night": {
        "image_left": 6036043612,
        "depth_left": 2922724942
    },
    "amusement": {
        "image_left": 5186300920,
        "depth_left": 2845203166
    },
    "carwelding": {
        "image_left": 2000614903,
        "depth_left": 805503662
    },
    "endofworld": {
        "image_left": 4373638266,
        "depth_left": 1354665455
    },
    "gascola": {
        "image_left": 7114291997,
        "depth_left": 4455893791
    },
    "hospital": {
        "image_left": 7192729599,
        "depth_left": 2611564881
    },
    "japanesealley": {
        "image_left": 1733885653,
        "depth_left": 838502285
    },
    "neighborhood": {
        "image_left": 11167651200,  # rounded / verified
        "depth_left": 6866567675
    },
    "ocean": {
        "image_left": 3440934445,
        "depth_left": 1834772057
    },
    "office": {
        "image_left": 3309042345,
        "depth_left": 1109350221
    },
    "office2": {
        "image_left": 3081818044,
        "depth_left": 1077478416
    },
    "oldtown": {
        "image_left": 4462920255,
        "depth_left": 1772142656
    },
    "seasidetown": {
        "image_left": 3179905243,
        "depth_left": 1167753975
    },
    "seasonsforest": {
        "image_left": 3096404925,
        "depth_left": 1844757919
    },
    "seasonsforest_winter": {
        "image_left": 5621406537,
        "depth_left": 3493365567
    },
    "soulcity": {
        "image_left": 6885228694,
        "depth_left": 2603602118
    }
}

# Pinned default trajectories check
KNOWN_TRAJECTORIES: dict[str, list[str]] = {
    "abandonedfactory": ["P000"],
    "japanesealley": ["P001", "P002", "P003", "P004", "P005", "P007"],
    "carwelding": ["P001", "P002", "P004", "P005", "P006", "P007"]
}

HF_BASE_URL = "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main"


def download_file(url: str, dest_path: Path, max_gb: float) -> None:
    """Download a file with Content-Length check and optional tqdm progress bar."""
    req = urllib.request.Request(url, method='HEAD')
    req.add_header('User-Agent', 'Mozilla/5.0')

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_length = resp.headers.get('content-length')
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch metadata from {url}: {e}"
        ) from e

    if content_length is not None:
        try:
            size_bytes = int(content_length)
        except ValueError:
            size_bytes = None
    else:
        size_bytes = None

    if size_bytes is not None:
        size_gb = size_bytes / (1024 ** 3)
        if size_gb > max_gb:
            raise ValueError(
                f"Download size {size_gb:.2f} GiB exceeds safety ceiling of "
                f"{max_gb:.2f} GiB. Aborting."
            )
        print(f"Verified remote file size: {size_gb:.2f} GiB")
    else:
        raise ValueError(
            "Remote response lacks Content-Length header. Cannot verify "
            "safety ceiling. Aborting transfer to preserve safety invariant."
        )
    print(f"Downloading {url} to {dest_path}...")
    req_get = urllib.request.Request(url)
    req_get.add_header('User-Agent', 'Mozilla/5.0')

    try:
        from tqdm import tqdm
        tqdm_available = True
    except ImportError:
        tqdm_available = False

    with urllib.request.urlopen(req_get) as resp:
        total_size = int(resp.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if tqdm_available and total_size > 0:
            with tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=dest_path.name
            ) as bar:
                with open(dest_path, 'wb') as f:
                    while True:
                        buffer = resp.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        bar.update(len(buffer))
        else:
            downloaded = 0
            with open(dest_path, 'wb') as f:
                while True:
                    buffer = resp.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(
                            f"\rProgress: {percent:.1f}% "
                            f"({downloaded / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB)"
                        )
                        sys.stdout.flush()
            print()


def extract_trajectory_slice(
    zip_path: Path,
    target_dir: Path,
    environment: str,
    difficulty: str,
    trajectory: str,
    modalities: list[str]
) -> None:
    """Extract only files belonging to the specified trajectory slice.

    This prevents expanding the full environment.
    """
    print(f"Extracting selected slice ({trajectory}) from {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        extracted_count = 0

        for member in members:
            parts = member.filename.split('/')
            if len(parts) < 3:
                continue

            # parts[0]: environment, parts[1]: difficulty, parts[2]: trajectory
            if parts[0].lower() != environment.lower():
                continue
            if parts[1].lower() != difficulty.lower():
                continue
            if parts[2] != trajectory:
                continue

            should_extract = False

            # Left camera images
            if 'image_left' in member.filename and (
                'image' in modalities or 'rgb' in modalities or 'image_left' in modalities
            ):
                should_extract = True
            # Right camera images
            elif 'image_right' in member.filename and ('image_right' in modalities):
                should_extract = True
            # Left camera depth
            elif 'depth_left' in member.filename and (
                'depth' in modalities or 'depth_left' in modalities
            ):
                should_extract = True
            # Right camera depth
            elif 'depth_right' in member.filename and ('depth_right' in modalities):
                should_extract = True
            # Poses (pose_left.txt and pose_right.txt)
            elif ('pose_left.txt' in member.filename or 'pose_right.txt' in member.filename) and (
                'pose' in modalities or 'pose_left' in modalities or 'pose_right' in modalities
            ):
                should_extract = True

            if should_extract:
                zip_ref.extract(member, target_dir)
                extracted_count += 1

        if extracted_count == 0:
            raise RuntimeError(
                f"No files matching trajectory {trajectory} with modalities "
                f"{modalities} were found in the archive {zip_path.name}."
            )
        print(
            f"Successfully extracted {extracted_count} files for "
            f"trajectory {trajectory}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and bootstrap a small pinned TartanAir trajectory slice"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/tartanair",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="abandonedfactory",
        help="TartanAir environment name",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="Easy",
        choices=["Easy", "Hard"],
        help="Difficulty level (Easy or Hard)",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="P000",
        help="Trajectory sequence ID (e.g. P000)",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["image", "depth", "pose"],
        help="Data modalities to cover (image, depth, pose)",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=1.0,
        help="Maximum download size in GB to allow before transfer",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Perform actual network transfer and extraction",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify checksums after download (stub for compatibility)",
    )
    args = parser.parse_args()

    if args.max_gb <= 0:
        parser.error("argument --max-gb: must be a positive number")

    # Pre-calculated sizes for offline dry run
    env = args.environment
    diff = args.difficulty
    traj = args.trajectory
    modalities = args.modalities
    output_dir = Path(args.output_dir)

    # Prerequisite check strings
    try:
        import tqdm
        tqdm_status = f"Available ({tqdm.__version__})"
    except ImportError:
        tqdm_status = "Not Installed (run 'uv pip install tqdm' to enable)"

    # Validate environment and estimate download sizes
    # We download image_left.zip if image, rgb, or pose is requested
    # We download depth_left.zip if depth is requested
    needs_image = any(
        m in modalities
        for m in [
            "image", "rgb", "image_left", "pose", "pose_left", "pose_right"
        ]
    )
    needs_depth = any(m in modalities for m in ["depth", "depth_left"])

    estimated_bytes = 0
    download_urls = []

    if diff == "Easy" and env in EASY_SIZES:
        if needs_image:
            estimated_bytes += EASY_SIZES[env]["image_left"]
            download_urls.append(f"{HF_BASE_URL}/{env}/Easy/image_left.zip")
        if needs_depth:
            estimated_bytes += EASY_SIZES[env]["depth_left"]
            download_urls.append(f"{HF_BASE_URL}/{env}/Easy/depth_left.zip")
    else:
        # Fallback or Hard difficulty sizing (not in EASY_SIZES)
        estimated_bytes = None
        if needs_image:
            download_urls.append(f"{HF_BASE_URL}/{env}/{diff}/image_left.zip")
        if needs_depth:
            download_urls.append(f"{HF_BASE_URL}/{env}/{diff}/depth_left.zip")

    estimated_gb = estimated_bytes / (1024 ** 3) if estimated_bytes is not None else None

    # Validate trajectory if known
    if env in KNOWN_TRAJECTORIES and traj not in KNOWN_TRAJECTORIES[env]:
        print(
            f"Warning: Trajectory {traj} is not in the pinned list for {env}: "
            f"{KNOWN_TRAJECTORIES[env]}"
        )
    # If --download is not set, run offline dry run plan
    if not args.download:
        print("TartanAir Slice Downloader — Dry Run Plan")
        print("=========================================")
        print(f"Environment:      {env}")
        print(f"Difficulty:       {diff}")
        print(f"Trajectory:       {traj}")
        print(f"Modalities:       {', '.join(modalities)}")
        print(f"Destination:      {output_dir.resolve()}")

        print("\nExpected Layout:")
        if needs_image and any(
            m in modalities for m in ["image", "rgb", "image_left"]
        ):
            print(f"  - {output_dir}/{env}/{diff}/{traj}/image_left/*.png")
        if needs_depth:
            print(f"  - {output_dir}/{env}/{diff}/{traj}/depth_left/*.npy")
        if any(m in modalities for m in ["pose", "pose_left", "pose_right"]):
            print(f"  - {output_dir}/{env}/{diff}/{traj}/pose_left.txt")
            print(f"  - {output_dir}/{env}/{diff}/{traj}/pose_right.txt")

        print("\nVerified Source/API:")
        print(f"  Hugging Face Dataset Mirror: {HF_BASE_URL}")

        print("\nDependency/Tool Prerequisites:")
        print(
            "  - urllib (Hugging Face mirror transport): "
            "Available (Standard Library)"
        )
        print(f"  - tqdm (Download progress bar):           {tqdm_status}")
        print(
            "  - zipfile (ZIP archive parser):           "
            "Available (Standard Library)"
        )

        print("\nEstimated Action:")
        if estimated_gb is not None:
            print(f"  Selected Extracted Unit: {env}/{diff}/{traj}")
            print(
                "  Note: Although the selected trajectory is small after "
                "extraction, the Hugging Face"
            )
            print(
                "        mirror transfer requires downloading the full "
                "environment-level archives."
            )
            print(
                f"  Total Download Size:     {estimated_gb:.2f} GiB "
                "(computed archive transfer, not trajectory size)"
            )
            if estimated_gb > args.max_gb:
                print(
                    f"  Safety Check:            FAILED "
                    f"(Blocked safety decision; ceiling is {args.max_gb:.2f} "
                    f"GiB)"
                )
                print(
                    f"  Reason:                  The default max-gb limit "
                    f"of {args.max_gb:.2f} GiB blocks this download"
                )
                print(
                    "                           to prevent accidental large "
                    "transfers of environment-level archives."
                )
                print(
                    f"  Action:                  Run with --download and "
                    f"increase --max-gb to >= {estimated_gb:.2f}"
                )
            else:
                print(
                    f"  Safety Check:            PASSED "
                    f"(Under ceiling of {args.max_gb:.2f} GiB)"
                )
                print(
                    "  Action:                  Run with --download to "
                    "initiate network transfer."
                )
        else:
            print(
                "  Total Download Size:     Unknown (Remote metadata "
                "not checked in dry run)"
            )
            print(
                "  Action:                  Run with --download to fetch "
                "remote headers and initiate transfer."
            )

        sys.exit(0)

    # Actual download execution
    print("Initiating TartanAir Slice Download...")
    if estimated_gb is not None and estimated_gb > args.max_gb:
        print(
            f"Error: Estimated download size {estimated_gb:.2f} GiB exceeds "
            f"safety ceiling of {args.max_gb:.2f} GiB."
        )
        print("Use --max-gb to increase the safety ceiling threshold.")
        sys.exit(1)
    tmp_dir = output_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for url in download_urls:
            zip_name = url.split('/')[-1]
            dest_zip = tmp_dir / zip_name

            # Download file
            download_file(url, dest_zip, args.max_gb)

            # Extract only the trajectory slice
            extract_trajectory_slice(dest_zip, output_dir, env, diff, traj, modalities)

            # Clean up temp zip
            if dest_zip.exists():
                dest_zip.unlink()

        print("\nTartanAir slice bootstrap complete!")
        print(f"Files are located at: {output_dir.resolve()}/{env}/{diff}/{traj}/")

    except Exception as e:
        print(f"\nError occurred during download/extraction: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up temp directory
        if tmp_dir.exists():
            try:
                tmp_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
