"""Download a pinned KITTI odometry + SemanticKITTI slice (🟢, deferred).

Fetches 1-2 KITTI odometry sequences (velodyne + GT poses + calib) and the
matching SemanticKITTI label archives, verifies checksums, and lays them out
under data/raw/kitti/. Sequence IDs are pinned here; never expand the slice
without a note in docs/decisions.md.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Download KITTI/SemanticKITTI slice.")
    ap.add_argument("--out", default="data/raw/kitti", help="output directory")
    ap.add_argument(
        "--sequences",
        nargs="+",
        default=["00", "05"],
        help="KITTI odometry sequence ids to fetch",
    )
    ap.parse_args()
    raise NotImplementedError("Week 1 — KITTI/SemanticKITTI download")


if __name__ == "__main__":
    main()
