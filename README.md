# SpatialWM

> A geometry-conditioned latent-world-model study built on a practical 3D-perception stack for multi-view RGB, RGB-D, and LiDAR data.

**Status:** Public scaffold — geometry, LiDAR, perception, and world-model experiments are in active development. This repository makes no completed-results claim yet. It will distinguish components implemented from scratch from AI-assisted, human-reviewed components as work lands.

## Objective

SpatialWM investigates a focused question: **when and where does explicit spatial information improve future prediction in a latent world model?**

The project pairs classical 3D reconstruction and point-cloud processing with a geometry-conditioned latent predictor. It is designed as a controlled reproduction and analysis informed by NWM, DINO-WM, and V-JEPA 2-AC—not as a claim of novel foundation-model research.

## Scope

- **Multi-view geometry:** calibrated projection, two-view estimation, robust matching, triangulation, bundle adjustment, and trajectory evaluation.
- **LiDAR and 3D perception:** scan-to-scan odometry, point-cloud registration, occupancy/BEV grids, elevation products, and 3D semantic segmentation.
- **Sensor data:** multi-view RGB/RGB-D simulation data plus real KITTI and SemanticKITTI LiDAR sequences; optional multi-sensor evaluation on nuScenes.
- **Latent world modelling:** frozen visual representations, JEPA-style representation prediction, motion-conditioned forecasting, and depth/occupancy probes.
- **Evaluation:** pose error, point-cloud/occupancy/segmentation metrics, horizon curves, and performance binned by ego-motion magnitude.

## Intended Experiments

The central comparison uses three predictors:

1. **B0:** a pixel-space baseline.
2. **B1:** an action-free latent predictor.
3. **T:** a latent predictor conditioned on relative camera motion.

The target result is not merely lower aggregate loss: it is an analysis of whether geometry helps most at large ego-motion and long prediction horizons. A complementary LiDAR track evaluates scan registration, odometry drift, and BEV representations on real sensor data.

## Data and Tools

Planned data sources include TartanAir, KITTI Odometry, and SemanticKITTI. The project uses established tools and models where appropriate—for example DINOv2, COLMAP, ORB-SLAM3, Open3D, and Habitat—and will clearly distinguish those dependencies from project-specific implementations and analysis.

## Repository State

This initial public commit establishes the Python package, reproducible environment, configuration schema, sensor-data interfaces, evaluation contracts, and test suite. Implementations and experimental artifacts will be added incrementally with corresponding tests, figures, and reproducible commands.

## Visual & Data Bootstrap

### Current Visual Baseline

![Geometry RANSAC Epipolar Sanity](figures/geometry_ransac_epipolar.png)

*Note: The figure linked above is a deterministic synthetic geometry/RANSAC sanity visualization, not a benchmark or real-data result.*

To generate the geometry visualization:
```bash
uv run python scripts/make_figures.py --figures geometry-ransac
```

### Data Pipeline and Pinned TartanAir Slice

The initial real-data unit consists of RGB, depth, and pose data from one pinned TartanAir trajectory, stored under `data/raw/tartanair/` (which is gitignored; no data is committed to the repository).

To plan or fetch this slice, use the download helper. The command below performs a dry-run transport plan, outputting details of the pinned source, slice, and modalities without transferring data:
```bash
uv run python scripts/download_tartanair_slice.py --output-dir data/raw/tartanair --max-gb 1
```
*Note: While the selected trajectory is small after extraction, the current source transport requires transferring ~2.40 GiB of environment-level archives for RGB+depth+pose. To initiate the actual download, run the command with the explicit `--download` and `--max-gb 2.5` options:*
```bash
uv run python scripts/download_tartanair_slice.py --output-dir data/raw/tartanair --download --max-gb 2.5
```

### Visual Artifact Roadmap

Progress and deliverables follow this ordered sequence:
1. **Geometry RANSAC:** Synthetic epipolar sanity visualization (current baseline).
2. **ICP Alignment:** Point-cloud registration alignment and transformation visuals.
3. **LiDAR BEV / Odometry:** Scan registration, drift plots, and bird's-eye view occupancy grids.
4. **JEPA Collapse:** Diagnostic plots for checking representation collapse.
5. **Evaluation Curves:** Motion-binned geometry-conditioning and latent-prediction horizon curves.
6. **Habitat Demo:** Latent-MPC trajectories within the simulator.

*(No claims are made that any downstream visual artifact past the current baseline already exists.)*

## Local Setup

```bash
uv sync --extra dev
uv run pytest --co -q
uv run ruff check src tests scripts
```

## Planned Outputs

- Geometry and LiDAR-odometry evaluation tables.
- LiDAR BEV occupancy and 3D semantic-segmentation visualizations.
- A JEPA collapse diagnostic and latent-prediction horizon curves.
- A motion-binned geometry-conditioning analysis.
- A small latent-MPC Habitat demonstration.
