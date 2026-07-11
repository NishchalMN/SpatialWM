"""
Encode-once pipeline: TartanAir slice -> 224², 5fps, clips of 8 context + targets
Δ∈{1,4,8,16} -> shards + manifest parquet with GT pose deltas.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache latent encodings from TartanAir slice"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/tartanair",
        help="Input directory with TartanAir data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cache/latents",
        help="Output directory for cached latents",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="data/manifests/tartanair_slice.parquet",
        help="Output manifest parquet path",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=8,
        help="Number of context frames",
    )
    parser.add_argument(
        "--deltas",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Target frame deltas",
    )
    parser.parse_args()

    raise NotImplementedError("Week 4 — needs DINOv2 backbone")


if __name__ == "__main__":
    main()
