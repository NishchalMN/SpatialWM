# SpatialWM

> A geometry-first multi-sensor 3D perception project that recovers motion and structure
> from images, RGB-D, and LiDAR, then exposes geometry quality for controlled world-model
> research.

`main` is the portfolio-grade classical perception pipeline. Experimental predictors live on
`research/world-model`; they are not presented as completed research.

## What the project is trying to answer

The engineering question is how calibrated sensor samples become defensible 3D structure,
motion, and maps. The research question is:

> When does explicit ego-motion improve visual prediction, and when does noisy or
> overconfident geometry make it worse?

That second question is meaningful only if pose estimates, transform directions, scale,
confidence, and failure modes are measured first.

## The complete 3D story

```text
                       ┌─ matches → F/E RANSAC → pose → triangulation
sensor manifests ──────┤                                ↓
timestamps + frames    │                         incremental SfM → BA
calibration + poses    │                                ↓
                       ├─ RGB + depth → metric cloud → ICP → GT SE(3) error
                       │
                       └─ camera + Velodyne + OXTS
                              ├─ LiDAR→image calibration check
                              └─ scan-to-scan / submap odometry → ATE/RPE → BEV

all estimators → PoseEstimate + confidence → B1 / T-GT / T-EST / T-NOISE / T-GATED
```

Read [the stage-by-stage 3D vision story](docs/3d_vision_story.md) for the intuition,
coordinate contracts, failure modes, and completion checks.

## Verified evidence

### 1. Synchronized sensor ingestion and calibration

TartanAir RGB/depth/poses and KITTI camera/Velodyne/OXTS are normalized into validated sensor
records with timestamps, named coordinate frames, calibration edges, and provenance. KITTI
frame 0 projects 20,468 of 123,397 Velodyne returns into the rectified camera image; the
depth-coloured points visibly follow scene structure.

![KITTI calibrated LiDAR-camera projection](figures/curated/kitti_lidar_camera_projection.png)

See [sensor ingestion and calibration](docs/sensor_ingestion.md).

### 2. Robust image geometry

The classical SIFT pipeline applies ratio and mutual filtering before fundamental-matrix
RANSAC. On TartanAir frames 1750 and 1755, 474 of 481 symmetric matches are geometric inliers
(98.5%). A separate deterministic diagnostic with 25% injected outliers recovers the known
consensus at 99.2% precision and 100% recall.

![TartanAir feature correspondence pipeline](figures/curated/tartanair_feature_matches.png)

### 3. Incremental sparse SfM

The transparent SfM integration connects matching, F/E recovery, cheirality, triangulation,
PnP registration, landmark expansion, and bundle adjustment. On TartanAir P000 frames
1750–1769 it registers all 20 cameras, grows from 416 initial landmarks to 2,963 landmarks,
records 12,457 observations, and reduces reprojection RMSE from 0.661 px to **0.177 px**.

![TartanAir incremental sparse SfM](figures/curated/tartanair_sparse_sfm.png)

The monocular trajectory is Sim(3)-aligned only for a short diagnostic. Its metric-scale ATE
is not a portfolio claim. See [incremental sparse SfM](docs/sparse_sfm.md).

### 4. Bundle adjustment and RGB-D registration

On a deterministic 5-camera/100-point problem, sparse gauge-fixed BA reduces mean
reprojection error from 60.59 px to 0.51 px (118.5x).

![Bundle-adjustment reprojection diagnostic](figures/curated/bundle_adjust_reprojection.png)

Synthetic ICP recovers a known 0.295 m / 5.4° transform with 0.0004 m translation and 0.006°
rotation error. The real TartanAir case is intentionally retained as a failure: Open3D reports
0.991 fitness, yet translation error is 0.690 m. Local overlap is not ground-truth accuracy.

![TartanAir ICP ground-truth failure](figures/curated/tartanair_icp_alignment.png)

### 5. KITTI LiDAR odometry and BEV mapping

On 100 synchronized KITTI raw frames, the primary scan-to-scan trajectory achieves:

- **0.318 m** rigid-aligned ATE RMSE with scale fixed;
- **1.264 m** raw endpoint error;
- **0.049 m / 0.098°** mean one-step translation/rotation error.

A five-scan submap improves local translation error to 0.036 m and ICP inlier RMSE from 0.197
m to 0.158 m, but worsens global ATE to 0.485 m. It is documented as a sensitivity result,
not advertised as an improvement.

![KITTI odometry comparison](figures/curated/kitti_lidar_odometry.png)

The same primary poses transform 9.66 million cropped returns into a fixed-axis 0.10 m
bird's-eye-view return-density map.

![KITTI LiDAR BEV return density](figures/curated/kitti_lidar_bev.png)

See [LiDAR odometry, local submaps, and BEV](docs/lidar_odometry_bev.md).

## Reproduce

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests scripts
```

Generate the TartanAir reconstruction:

```bash
uv run python scripts/make_figures.py \
  --figures tartanair-sfm \
  --output-dir figures/curated \
  --tartanair-frame 1750 \
  --tartanair-sfm-stride 1 \
  --tartanair-sfm-frames 20
```

Download and evaluate the bounded KITTI slice:

```bash
uv run python scripts/download_kitti_slice.py \
  --frames 100 --output-dir data/raw/kitti --max-gb 1.0 --download
uv run python scripts/build_sensor_manifest.py \
  --frames 100 --output-dir data/processed/manifests
uv run python scripts/evaluate_kitti_lidar.py \
  --kitti-root data/raw/kitti --frames 100 --output-dir figures/curated
```

Raw data and generated manifests are not committed. Curated figures, JSON metrics, and compact
NPZ reconstructions are versioned.

## Research branch and claim boundary

The geometry-quality experiment proceeds in a fixed order:

| Condition | Motion input | Decision question |
|---|---|---|
| B1 | none | How predictable is the future from visual context alone? |
| T-GT | ground-truth pose | Does ideal ego-motion add signal? |
| T-EST | TartanAir SfM/RGB-D pose | Does the signal survive real error? |
| T-NOISE | controlled pose corruption | Where does geometry stop helping? |
| T-GATED | pose plus confidence | Can reliability-aware conditioning recover value? |

B1 versus T-GT is the first GO/NO-GO experiment. KITTI LiDAR and TartanAir visual clips are
separate validation tracks—not falsely described as one fused dataset. Read the
[geometry-quality research bridge](docs/geometry_quality_bridge.md).

## Honest limitations

- This is an offline, bounded pipeline, not production streaming infrastructure.
- SfM uses classical local tracks without loop closure, relocalization, or COLMAP-scale track
  management.
- ICP is point-to-point and does not remove dynamic objects.
- LiDAR odometry has no global pose graph or loop closure.
- GPS/IMU is validation ground truth, not a fused navigation estimator.
- No novelty, state-of-the-art, benchmark-leader, or completed-paper claim is made.

The implementation uses mature OpenCV, Open3D, SciPy, NumPy, and pykitti components with
repository-specific orchestration, transform contracts, tests, metrics, and inspected
visuals. It is not described as “from scratch.”

## Documentation

- [Complete 3D vision story](docs/3d_vision_story.md)
- [Sensor ingestion and calibration](docs/sensor_ingestion.md)
- [Feature matching](docs/feature_matching.md)
- [Two-view geometry](docs/two_view_geometry.md)
- [RANSAC](docs/ransac.md)
- [Incremental sparse SfM](docs/sparse_sfm.md)
- [Bundle adjustment](docs/bundle_adjust.md)
- [ICP](docs/icp.md)
- [TartanAir registration](docs/tartanair_registration.md)
- [KITTI LiDAR odometry and BEV](docs/lidar_odometry_bev.md)
- [Geometry-quality research bridge](docs/geometry_quality_bridge.md)
