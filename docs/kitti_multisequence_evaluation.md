# KITTI Multi-Sequence Robustness Evaluation

## Why this evaluation exists

The original portfolio figures deliberately used one bounded KITTI raw drive so every image,
trajectory, and map could be inspected. That proves integration, but it does not show whether
the same configuration survives a change of scene and motion. This suite freezes the estimator
settings and evaluates three drives without per-drive tuning.

It is a robustness diagnostic, not a KITTI odometry or reconstruction benchmark submission.

## Dataset and frozen protocol

| Setting | Value |
|---|---|
| KITTI date | `2011_09_26` |
| Drives | `0001`, `0005`, `0011` |
| SfM input | 16 rectified `image_02` views, start 0, stride 2 |
| SfM features | SIFT, 5,000-feature cap, ratio 0.75 |
| SfM robust geometry | 1.0 px RANSAC threshold, fixed seed 0 |
| LiDAR input | First 80 Velodyne scans |
| LiDAR methods | Scan-to-scan and five-scan local submap |
| LiDAR downsampling | 0.20 m voxel |
| ICP correspondence limit | 1.0 m |
| Ground truth | KITTI OXTS, transformed into the normalized Velodyne frame |
| Per-drive tuning | None |

The downloader pins archive byte sizes and extracts only the requested synchronized RGB,
Velodyne, and OXTS records. Raw data and verbose per-drive artifacts remain ignored. The
aggregate JSON and summary figure are versioned.

## Results

| Drive | SfM views | Landmarks | BA RMSE | Scan ATE | Submap ATE | Step translation error | Mean ICP fitness |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0001 | 16/16 | 3,520 | 0.215 px | 0.713 m | 0.528 m | 0.105 m | 0.981 |
| 0005 | 16/16 | 2,357 | 0.265 px | 0.283 m | 0.349 m | 0.050 m | 0.988 |
| 0011 | 16/16 | 4,723 | 0.221 px | 0.162 m | 0.160 m | 0.057 m | 0.972 |

Aggregate evidence:

- 3/3 SfM sequences and 3/3 LiDAR sequences complete;
- 48/48 requested SfM views register;
- median final SfM reprojection RMSE: 0.221 px;
- median scan-to-scan rigid-aligned ATE: 0.283 m;
- worst scan-to-scan rigid-aligned ATE: 0.713 m on drive 0001;
- the local submap improves ATE on two of three drives;
- median measured runtime is 1.07 s per requested SfM view and 0.32 s per LiDAR frame on the
  development machine, including figure/report generation in each per-drive process.

![Frozen KITTI multi-sequence comparison](../figures/curated/kitti_multisequence_summary.png)

## How to interpret the result

### SfM generalizes across these bounded slices

All requested views register, while landmark counts vary substantially with scene texture,
visibility, and track overlap. Low reprojection error says the retained observations are
globally self-consistent after bundle adjustment. It does not prove metric scale, dense map
completeness, or long-horizon drift performance.

The saved monocular trajectory ATE uses Sim(3) only as a path-shape diagnostic and is excluded
from the cross-drive headline.

### LiDAR error is sequence dependent

One frozen ICP configuration produces a 4.4x range between the best and worst scan-to-scan
ATE. Drive 0001 also has the largest mean step translation error even though its mean ICP
fitness remains high. This is direct evidence that internal overlap is not a substitute for
ground-truth trajectory evaluation.

### The submap is a sensitivity result

The five-scan submap improves global ATE on drives 0001 and 0011 but worsens it on drive 0005.
It should therefore be described as a local-context tradeoff, not as a universal improvement.
The next technical milestone is keyframe/scan-to-map design with motion-aware filtering and
multi-sequence acceptance criteria.

## Reproduce

From the repository root:

```bash
uv sync --extra dev
uv run python scripts/download_kitti_slice.py \
  --drive 0001 --frames 80 --output-dir data/raw/kitti --max-gb 0.5 --download
uv run python scripts/download_kitti_slice.py \
  --drive 0005 --frames 80 --output-dir data/raw/kitti --max-gb 1.0 --download
uv run python scripts/download_kitti_slice.py \
  --drive 0011 --frames 80 --output-dir data/raw/kitti --max-gb 1.0 --download
uv run python scripts/evaluate_kitti_suite.py \
  --drives 0001 0005 0011 \
  --sfm-views 16 --sfm-stride 2 --lidar-frames 80 \
  --runs-dir data/processed/kitti_multisequence \
  --output-dir figures/curated
```

Use `--reuse-existing` only when regenerating the aggregate report from already completed
per-drive runs. Runtime fields will be `null` in that mode because no evaluator runtime was
measured.

Committed outputs:

- `figures/curated/kitti_multisequence_metrics.json`;
- `figures/curated/kitti_multisequence_summary.png`.

Ignored diagnostic outputs:

- per-drive SfM and LiDAR figures;
- full per-drive JSON and NPZ files;
- stdout/stderr logs;
- raw KITTI samples.

## Claim boundary

The sequence count, frame counts, frozen configuration, alignment policy, and failure spread
must accompany these numbers. Do not call them leaderboard results, survey-grade mapping, or
sensor fusion. OXTS is used for evaluation, not supplied as an ICP prior.
