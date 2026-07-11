"""
Download pinned TartanAir 2-env/~8-seq slice (image+depth+pose), verify checksums.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TartanAir slice with image, depth, and pose data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/tartanair",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify checksums after download",
    )
    parser.parse_args()

    raise NotImplementedError("Week 1 — TartanAir download")


if __name__ == "__main__":
    main()
