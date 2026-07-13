# Bundle Adjustment: Making the Reconstruction Agree Globally

**Story position:** Stage 4, after two-view pose and triangulation and before end-to-end sparse SfM. See [the complete 3D vision story](3d_vision_story.md).

## What problem it solves

Two-view geometry produces an initial estimate of cameras and 3D points, but every match and pose contains noise. A point observed in five images should project near its measured pixel in all five images. Bundle adjustment jointly changes the camera poses and 3D points until those projections agree as well as possible.

For observation j, the residual is:

    r_j = project(K, camera_i, point_k) - observed_pixel_j

The optimizer minimizes the collection of 2D residuals over all observations. It is called bundle adjustment because the rays joining camera centres to scene points form bundles that are adjusted together.

## Representation used here

- A camera pose is a six-vector: three rotation-vector values followed by three translation values.
- The pose maps a world point into the camera frame: X_camera = R X_world + t.
- A point is a three-vector in the shared world frame.
- One observation is [camera_index, point_index, u, v].
- Intrinsics K are shared and fixed in the current calibrated implementation.
- Residual order is [du_0, dv_0, du_1, dv_1, ...].

## Main calls and API shape

### reprojection_residuals

~~~python
reprojection_residuals(
    params: np.ndarray,
    n_cams: int,
    n_pts: int,
    K: np.ndarray,
    obs: np.ndarray,
) -> np.ndarray
~~~

- params has shape (6M + 3N,): all camera poses followed by all points.
- K has shape (3, 3).
- obs has shape (P, 4).
- output has shape (2P,).
- residual sign is predicted minus observed.

### bundle_adjust

~~~python
bundle_adjust(
    poses0: np.ndarray,
    X0: np.ndarray,
    K: np.ndarray,
    obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]
~~~

- poses0 has shape (M, 6).
- X0 has shape (N, 3).
- returned poses and points preserve those shapes.
- SciPy least_squares uses the trust-region reflective method and the sparse LSMR linear solver.
- soft-L1 loss reduces the influence of observations with unusually large residuals.

The closest mature-library equivalent is SciPy least_squares with jac_sparsity. Large production SfM systems commonly use Ceres Solver or GTSAM and exploit a Schur-complement structure.

## Why the Jacobian is sparse

A residual from camera i observing point k depends only on:

- the six variables of camera i;
- the three variables of point k.

It does not depend directly on every other camera or point. Therefore each two-row residual block touches at most nine parameter columns. The implementation supplies this dependency pattern to SciPy so it does not treat the problem as one dense numerical system.

The Schur-complement idea takes this one step further: because points couple cameras only through observations, point increments can be eliminated first, leaving a smaller camera system, then recovered afterward. This repository relies on SciPy's sparse trust-region machinery rather than implementing a custom Schur solver.

## Gauge freedom

Images do not define an absolute world origin, orientation, or monocular scale. If every point and camera is transformed by the same similarity transform, the image projections can remain unchanged. The objective therefore has seven unobservable degrees of freedom:

- three global rotations;
- three global translations;
- one global scale.

This implementation chooses a gauge by holding the first camera pose and the first point's world-z coordinate fixed. These constraints select a coordinate system and scale; they do not add visual information. The gauge choice is why optimized coordinates should not be compared directly with ground truth before an appropriate alignment.

## Robust loss and outliers

Ordinary squared loss gives a large residual disproportionate influence. A wrong feature track can then drag both a camera and a 3D point away from the consistent reconstruction. The soft-L1 loss behaves quadratically around small residuals but grows more gently for large residuals.

Robust loss is not a replacement for geometric verification. A scene with many wrong tracks, a weak initialization, or a degenerate camera path can still converge to a bad result. RANSAC and track filtering must remove gross data-association errors before BA.

## Current deterministic checkpoint

Generate the artifact:

~~~bash
uv run python scripts/make_figures.py \
  --figures bundle-adjust \
  --output-dir figures/curated
~~~

Seed-99 synthetic problem:

- 5 cameras;
- 100 points;
- 500 observations;
- 0.5-pixel observation noise;
- noisy initial cameras and points.

Measured result:

- mean reprojection error: 60.59 px to 0.51 px;
- median reprojection error: 57.03 px to 0.47 px;
- mean-error improvement: 118.5x.

The two image-plane panels use the same camera, axes, and pixel scale. Black dots are observations, coloured crosses are projections, and connecting lines show residual magnitude.

![Bundle-adjustment reprojection error before and after](../figures/curated/bundle_adjust_reprojection.png)

The matching machine-readable values are stored beside the figure in figures/curated/bundle_adjust_metrics.json.

## What the result proves—and does not

It proves that the implementation can jointly refine a well-initialized, calibrated, fully observed synthetic reconstruction and satisfy the greater-than-5x numerical gate.

It does not yet prove:

- robustness on real feature tracks;
- successful initialization from an image sequence;
- handling of changing intrinsics;
- large-scale performance;
- correctness under severe outliers or weak baseline.

Those questions belong to the controlled sparse-SfM integration stage.

## Failure modes and diagnostics

### Unfixed gauge

Symptom: singular or poorly conditioned optimization, coordinate drift, or arbitrary scale.

Check: confirm the first camera and one scale value are held fixed.

### Bad initialization

Symptom: error decreases but cameras or points settle into an incorrect local solution.

Check: visualize cameras and positive-depth points; inspect initial-pair baseline and PnP inliers.

### Wrong observations

Symptom: a few very long residual lines or a camera pulled away from the rest.

Check: plot per-observation residuals and trace large errors back to feature tracks.

### Points behind cameras

Symptom: unstable projection values or apparently low error with physically invalid geometry.

Check: count positive-depth observations before and after optimization.

### Weak camera motion

Symptom: reprojection can improve while depth remains uncertain.

Check: inspect parallax and triangulation angle, not only pixel error.

### Incorrect coordinate convention

Symptom: every projection is systematically displaced or poses appear inverted.

Check: restate whether each pose is world-to-camera and validate one known transform manually.

## Interview Q&A

**What does bundle adjustment optimize?**  
Camera parameters and 3D scene points jointly, usually by minimizing image reprojection error.

**Why is it nonlinear?**  
Rotation parameterization and perspective division by depth make the projection function nonlinear.

**Why is the Jacobian sparse?**  
Each observation depends on one camera and one point, not on every variable in the reconstruction.

**What is gauge freedom?**  
The reconstruction's global frame and monocular scale are not observable from reprojection alone. Constraints choose a coordinate system without adding scene evidence.

**Why not compare optimized points directly with ground truth?**  
Two equivalent monocular reconstructions can differ by a global similarity transform, so they must first be aligned.

**Why use robust loss if RANSAC already ran?**  
RANSAC removes gross outliers, while robust loss limits the influence of smaller remaining track errors. They address different stages of contamination.

**Can low reprojection error still be misleading?**  
Yes. Weak baseline, wrong scale, degenerate motion, or physically invalid point placement can produce low image error. Inspect cheirality, camera geometry, and real-data structure too.

**What is the Schur complement intuition?**  
Eliminate the many point variables using their local blocks, solve a smaller coupled camera system, then recover point updates.
