# From Classical Geometry to World-Model Experiments

## The research question

The future research branch asks whether explicit motion improves visual prediction, and how
that benefit changes when the motion estimate is imperfect. The classical pipeline exists to
make “imperfect geometry” measured and interpretable rather than an arbitrary noise vector.

## Geometry-quality ladder

| Condition | Motion source | What it isolates |
|---|---|---|
| B1 | no pose | action-free visual prediction baseline |
| T-GT | dataset camera pose | value of ideal ego-motion |
| T-EST | TartanAir SfM or RGB-D estimate | value under real estimator error |
| T-NOISE | controlled perturbation of GT | sensitivity boundary in translation/rotation |
| T-GATED | estimated pose plus confidence | whether reliability-aware conditioning helps |

`PoseEstimate` is the shared hand-off: source/target IDs, transform direction, method,
metric-versus-monocular scale, and diagnostics. SfM can export reprojection residual, track
support, triangulation source, and landmark confidence. ICP can export fitness, inlier RMSE,
overlap proxy, and correction magnitude.

## What the classical extensions contribute

- **SfM landmark expansion** changes map completeness, track support, and pose observability.
  It creates naturally varying image-geometry quality instead of keeping a fixed initial map.
- **Scan-to-submap odometry** tests whether stronger local geometric consistency improves
  global motion. On the current 100-frame KITTI slice it improves local RPE and ICP residuals
  but worsens ATE and endpoint drift—a useful example that internal confidence is not the
  same as global correctness.
- **LiDAR-camera projection** validates that sensor frames and calibration are meaningful
  before any learned cross-modal experiment is proposed.

## Dataset boundary

TartanAir and KITTI are parallel validation tracks, not one fused sequence. The first
world-model experiment uses TartanAir visual clips, so T-EST must come from TartanAir SfM or
RGB-D motion. KITTI LiDAR cannot condition that model unless a genuinely paired image-LiDAR
dataset is introduced. KITTI currently informs estimator design and confidence analysis.

## Decision gates

1. Train matched-capacity B1 and T-GT models on the same seeded clip split.
2. Continue only if T-GT gives a stable improvement, particularly at larger motion or longer
   horizon.
3. Add T-EST and T-NOISE while keeping architecture, data, and compute matched.
4. Add T-GATED only if confidence predicts actual pose error better than a trivial baseline.
5. Repeat seeds and report uncertainty before discussing a paper claim.

A negative result is acceptable: geometry may help only beyond a motion threshold, fail when
pose error exceeds a boundary, or add no value once the visual context already identifies the
future. Those are stronger conclusions than claiming novelty from a pose token alone.
