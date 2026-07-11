"""
Regenerate paper figures (collapse.png, motion-binned, horizon curves) from results.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from experimental results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experimental results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--figures",
        type=str,
        nargs="+",
        choices=["collapse", "motion-binned", "horizon"],
        help="Specific figures to generate (default: all)",
    )
    parser.parse_args()

    raise NotImplementedError("Week 8 — needs results")


if __name__ == "__main__":
    main()
