# Sensor Ingestion, Synchronization, and Calibration

**Story position:** Stage 0. Geometry is only meaningful after every sample has a time,
coordinate frame, calibration path, and provenance.

## Shared contract

`spatialwm.data` provides four small records:

- `SensorFrame`: dataset, sequence, frame ID, timestamp, modality, coordinate frame,
  file path, and optional pose source;
- `CalibrationEdge`: an SE(3) transform that maps points from one named sensor frame to
  another;
- `SensorSequence`: ordered multi-modal samples, intrinsics, calibrations, and optional
  camera-to-world poses;
- `PoseEstimate`: an estimated source-to-target transform with method, scale mode, and
  confidence diagnostics.

Adapters describe files without loading an entire sequence into memory. Validation rejects
missing samples, duplicate IDs, non-increasing timestamps, invalid rotations, incomplete
modalities, and malformed intrinsics. A Parquet manifest preserves the sample table; a JSON
report records what was validated.

## Dataset adapters

### TartanAir

The P000 adapter pairs left RGB and depth frames with `pose_left.txt`. Both modalities share
the camera frame and synthetic 10 Hz frame time. Poses are converted from native NED into
OpenCV RDF camera coordinates. This path supplies monocular images, metric depth, and
camera-pose ground truth.

### KITTI

The raw-drive adapter pairs rectified `image_02`, Velodyne, and OXTS samples by frame ID and
the synchronized timestamps distributed with the drive. Calibration is composed as:

`Velodyne → camera_00 → rectified camera_00 → camera_02`.

OXTS is retained as evaluation ground truth. It is not silently supplied as an odometry
prior. That distinction prevents ground-truth leakage into the estimated trajectory.

## Visual calibration check

The strongest quick check of a LiDAR-camera calibration is to transform each Velodyne point
into the rectified camera frame, keep positive depth, project with `K`, and keep pixels inside
the image. On KITTI frame 0, 20,468 of 123,397 returns land in the image; their depth-coloured
structure follows roads, vehicles, pedestrians, and building boundaries.

![KITTI LiDAR-camera calibration overlay](../figures/curated/kitti_lidar_camera_projection.png)

This does not prove perfect temporal calibration, but a wrong transform direction, axis
convention, or camera projection would produce an obvious spatial mismatch.

## Reproduce

```bash
uv run python scripts/download_kitti_slice.py \
  --frames 100 --output-dir data/raw/kitti --max-gb 1.0 --download
uv run python scripts/build_sensor_manifest.py \
  --frames 100 \
  --output-dir data/processed/manifests \
  --figure figures/curated/kitti_lidar_camera_projection.png
uv run pytest -q tests/test_sensor_data.py
```

Raw samples and generated manifests remain untracked. The curated projection and its JSON
summary are publishable evidence.

## Interview checks

**Why is a file loader not enough for sensor integration?**  A production-facing ingestion
layer must also preserve timestamp, sensor identity, coordinate frame, calibration, missing
data behavior, and pose provenance.

**Why not fuse OXTS into ICP?**  This project evaluates geometry-only odometry against OXTS.
Using OXTS as a prior would change the estimator and make the comparison circular unless it
were explicitly presented as sensor fusion.

**Why use a calibration graph?**  Sensor rigs contain transforms between multiple frames.
Named directed edges make composition and inversion auditable instead of hiding them in one
anonymous matrix.

**What remains for production ingestion?**  Hardware-specific packet decoding, clock-drift
estimation, interpolation, schema/version management, corrupt-packet quarantine, streaming,
and operational monitoring are not claimed here.
