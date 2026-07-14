# KITTI LiDAR Odometry, Local Submaps, and BEV

**Story position:** Stage 7. This closes the classical sensor path after image-based SfM and
RGB-D registration. See [the complete 3D vision story](3d_vision_story.md).

## Data and transform contract

The diagnostic uses KITTI raw `2011_09_26_drive_0005_sync`, frames 0–99.

- Velodyne points are metric `X forward, Y left, Z up`.
- Estimated `P_k` maps scan `k` into the scan-0 world frame; `P_0 = I`.
- OXTS IMU poses are calibrated into Velodyne coordinates and normalized to frame 0.
- OXTS is evaluation ground truth, not an ICP input.

For scan-to-scan ICP, `T_(k+1→k)` maps the newer source scan to the previous target scan:

`P_(k+1) = P_k @ T_(k+1→k)`.

The estimator uses the previous accepted relative transform as a constant-velocity
initialization after the first pair.

## Two odometry variants

### Scan-to-scan baseline

Every 0.20 m voxel-downsampled scan registers directly to its predecessor with point-to-point
Open3D ICP, a 1.0 m correspondence threshold, and 50 iterations. It is simple and local, so
small errors compound through pose composition.

### Scan-to-submap experiment

The current scan is transformed using a constant-velocity global prediction. A target submap
is built from the five most recent accepted scans transformed into frame 0, downsampled at
0.30 m, and bounded in size. ICP estimates a correction in the world frame:

`P_(k+1) = correction @ predicted_P_(k+1)`.

No loop closure, global pose graph, semantic filtering, or OXTS prior is used.

## 100-frame result

LiDAR is already metric, so primary ATE uses rigid alignment with scale fixed to one.

| Measurement | Scan-to-scan | Scan-to-submap |
|---|---:|---:|
| Rigid-aligned ATE RMSE | **0.318 m** | 0.485 m |
| Raw final position error | **1.264 m** | 1.588 m |
| Mean one-step translation error | 0.049 m | **0.036 m** |
| Mean one-step rotation error | 0.098 deg | **0.085 deg** |
| Mean ICP fitness | 0.988 | **0.997** |
| Mean ICP inlier RMSE | 0.197 m | **0.158 m** |

![KITTI scan-to-scan versus scan-to-submap odometry](../figures/curated/kitti_lidar_odometry.png)

The result is deliberately not simplified into “submaps are better.” The local submap gives
better one-step errors and internal ICP diagnostics, but worse global ATE and endpoint drift.
Repeatedly fitting a locally self-consistent map can reinforce small systematic bias. It also
shows why fitness and inlier RMSE are confidence features, not substitutes for ground truth.

The portfolio trajectory therefore remains scan-to-scan. The submap path is a documented
sensitivity experiment and a useful geometry-quality case for the research branch.

## BEV return-density map

All 100 scans are transformed by the primary scan-to-scan trajectory, cropped to
`X=[-5,50] m`, `Y=[-22,22] m`, `Z=[-2.5,1.5] m`, and rasterized into 0.10 m cells. The figure
shows `log(1 + return count)` so both sparse scan rings and dense accumulated surfaces remain
visible.

![KITTI single-scan and accumulated BEV](../figures/curated/kitti_lidar_bev.png)

The accumulated map contains 9,660,016 cropped/transformed points. Increased density and
coverage are visually useful, but they are not a map-accuracy metric because every pose error
is written into the raster.

## Reproduce

```bash
uv run python scripts/download_kitti_slice.py \
  --frames 100 --output-dir data/raw/kitti --max-gb 1.0 --download
uv run python scripts/evaluate_kitti_lidar.py \
  --kitti-root data/raw/kitti \
  --frames 100 \
  --voxel 0.2 \
  --max-correspondence-distance 1.0 \
  --max-iters 50 \
  --output-dir figures/curated
```

The NPZ stores GT, scan-to-scan, and scan-to-submap trajectories. The JSON records both metric
sets and every registration's fitness, RMSE, point counts, and correction magnitude.

## Limitations and next improvements

- Point-to-point ICP does not model surface normals.
- Dynamic objects, ground, vegetation, and distant returns are not filtered or weighted.
- A five-scan submap can preserve systematic local bias.
- There is no loop detection or global optimization.
- This bounded raw slice is not a KITTI odometry benchmark submission.
- BEV represents return density, not height, semantics, free space, or uncertainty.

A stronger mapping release would add point-to-plane/generalized ICP, motion-aware filtering,
keyframes, loop closure, and pose-graph optimization, then evaluate a longer standard split.

## Interview Q&A

**Why register `k+1` to `k`?**  It produces the source-to-target transform needed to express
the new scan in the already established scan-0 map frame.

**Why can local RPE improve while global ATE worsens?**  A method can make each local fit
smoother while introducing a small directional bias. Composition accumulates that bias, and
rigid alignment cannot remove its changing shape.

**Why must LiDAR ATE forbid free scale?**  LiDAR directly measures metres. Sim(3) scaling
would hide an estimator error the sensor should preserve.

**What is the constant-velocity initialization?**  The last accepted relative transform is
used as the next ICP starting guess. It widens the practical convergence basin without using
ground truth.

**Why bound the submap?**  Runtime and memory otherwise grow with the sequence, and distant
old geometry is less relevant to current overlap.

**Why is high fitness not proof of correct motion?**  Fitness measures overlap under the
estimated local alignment. A biased local map can overlap well while the global trajectory
drifts.
