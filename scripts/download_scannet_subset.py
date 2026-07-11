"""
Download 10-20 ScanNet scenes after agreement approval.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ScanNet subset after agreement approval"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/scannet",
        help="Output directory for downloaded scenes",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=10,
        help="Number of scenes to download (10-20)",
    )
    parser.add_argument(
        "--agreement-token",
        type=str,
        required=True,
        help="ScanNet agreement approval token",
    )
    parser.parse_args()

    raise NotImplementedError("Week 7 — ScanNet agreement")


if __name__ == "__main__":
    main()
