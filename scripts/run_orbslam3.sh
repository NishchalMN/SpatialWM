#!/usr/bin/env bash
#
# Run ORB-SLAM3 on TartanAir sequences.
#
# Usage:
#   ./run_orbslam3.sh <sequence_dir> <output_dir>
#
# Arguments:
#   sequence_dir  Directory containing TartanAir sequence (image_left/, poses/)
#   output_dir    Directory for ORB-SLAM3 output (trajectory, map)
#
# Expected workflow:
#   1. Load calibration parameters for TartanAir
#   2. Run monocular SLAM on left camera images
#   3. Save estimated trajectory for comparison with ground truth
#

echo "TODO Week 2"
exit 1
