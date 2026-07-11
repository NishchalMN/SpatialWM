# SpatialWM

> A geometry-conditioned latent-world-model study built on a practical 3D-perception stack for multi-view RGB, RGB-D, and LiDAR data.

**Status:** Public scaffold — geometry, LiDAR, perception, and world-model experiments are in active development. This repository makes no completed-results claim yet.

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
