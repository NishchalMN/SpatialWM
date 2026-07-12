# Two-View Geometry: Intuition and Practical Use

Two-view geometry explains what can be learned when the same scene is observed by two cameras. A pixel is not a 3D point by itself: with camera calibration it defines a **ray** leaving the camera center. A matching pixel in a second view defines another ray. The relative camera motion and scene structure are the geometry that makes those rays consistent. With noisy image matches, the goal is usually a useful estimate—not an exact reconstruction of every point.

## F versus E

- **Fundamental matrix (`F`)** describes the epipolar relationship directly in pixel coordinates. It works with uncalibrated images and maps a pixel in one image to the epipolar line where its match should lie in the other image.
- **Essential matrix (`E`)** describes the same relationship after camera intrinsics have been accounted for. It represents calibrated relative camera geometry: rotation and translation direction between the two camera frames. Translation scale is not observable from two views alone.

## Functions and when to use them

The `spatialwm.geometry.two_view` functions form a small, NumPy-only reference pipeline. Inputs are correspondence arrays with one row per match, and outputs retain that row-batch convention.

| Function | Meaning / return value | Use it when |
|---|---|---|
| `normalize_points` | Hartley-normalizes 2D points and returns normalized points plus the similarity transform used. | Conditioning the eight-point fit, especially when pixel coordinates are large or unevenly distributed. |
| `fundamental_8pt` | Estimates `F` from point correspondences, normalizes the fit, and enforces rank 2. | Starting an epipolar model from putative matches (typically inside RANSAC). |
| `essential_from_F` | Converts `F` to calibrated `E` using camera intrinsics. | Moving from pixel-coordinate constraints to relative calibrated camera geometry. |
| `decompose_E` | Returns exactly four `(R, t)` pose candidates, each with a proper rotation and translation direction. | Enumerating the pose ambiguity before checking which candidate agrees with the observed scene. |
| `triangulate_dlt` | Estimates one 3D point per correspondence from two projection matrices and image points. | Recovering provisional structure after choosing a relative pose. |
| `cheirality_select` | Selects the candidate for which triangulated points lie in front of both cameras. | Resolving the four essential-matrix pose candidates on a scene with enough valid points. |
| `sampson_distance` | Computes a first-order geometric error for each correspondence under `F`. | Ranking matches, inspecting residuals, or comparing an estimated model with a reference. It is a diagnostic approximation, not a full reprojection optimization. |

## Why the pipeline has these steps

- **Normalization** translates points to their centroid and scales them to a comparable spread. This reduces numerical conditioning problems in the linear eight-point solve; it is undone afterward.
- **Rank-2 enforcement** reflects the geometry of a valid fundamental matrix. The raw least-squares estimate usually has a small third singular value from noise; setting that singular value to zero projects it back onto the valid model family.
- **Four pose candidates** are unavoidable when an essential matrix is decomposed: signs and the two SVD rotation choices produce four mathematically compatible relative poses.
- **Cheirality** means positive depth. The physically meaningful candidate puts reconstructed points in front of both cameras, rather than behind one camera because of an unresolved sign ambiguity.
- **Triangulation** intersects (in a noisy, least-squares sense) the two viewing rays. Poor baseline, mismatches, and near-parallel rays make depth unreliable.
- **Sampson distance** is a cheap first-order approximation to geometric reprojection error. It is useful for robust filtering and diagnostics without repeatedly solving a nonlinear correction problem.

## Production guidance

For production reconstruction, normally use a mature implementation such as OpenCV's `findFundamentalMat`, `recoverPose`, and `triangulatePoints`, or a tested SfM/SLAM library that also handles feature matching, robust estimation, scale conventions, degeneracy checks, and refinement. This module is intentionally small: use it for controlled analysis, teaching, regression tests, and debugging the meaning of each stage—not as a replacement for a production SfM/SLAM stack.

## Diagnostics

| Symptom | Likely source of error | What to inspect first |
|---|---|---|
| `F` is visibly wrong or epipolar lines miss matches | Bad correspondences, too few or poorly distributed points, planar/pure-rotation degeneracy, or unstable normalization | Match quality and spatial coverage; finite inputs; residuals before trusting the matrix. |
| Rotation or translation direction is wrong | Wrong intrinsics, incorrect point ordering, pose candidate not selected by cheirality, or an implementation convention mismatch | Camera calibration, whether rows are `(x, y)`, and which image is camera 1 versus camera 2. |
| Many triangulated depths are negative | The wrong one of four pose candidates, reversed camera convention, or a degenerate/too-small baseline | Projection-matrix convention and positive-depth counts in both views. |
| Sampson residuals are high | Incorrect `F`, outliers, mismatched point order, or points measured in a different coordinate system | Coordinate units, correspondence pairing, outlier rejection, and the epipolar residual distribution. |

## Minimal NumPy reminder

Points are row-batched: an array of `N` pixels has shape `(N, 2)`, homogeneous pixels `(N, 3)`, and 3D points `(N, 3)`. A mathematical column-vector operation `M @ x` is written for a row batch as `X_rows @ M.T`. Keep that transpose rule explicit when translating equations into NumPy. Homogeneous results must be divided by their last coordinate only after checking it is finite and safely away from zero.

## Interview Q&A

**Q: What is the difference between `F` and `E`?**
**A:** `F` expresses epipolar geometry in pixel coordinates and does not require calibration. `E` expresses the calibrated relationship after intrinsics are removed, so it exposes relative rotation and translation direction.

**Q: Why can’t two views recover the scale of translation?**
**A:** Image measurements constrain ray directions and relative motion, but multiplying the baseline and scene depth by the same factor leaves the projections unchanged. Only translation direction is observable without an external scale cue.

**Q: Why normalize image points before the eight-point fit?**
**A:** Normalization centers and scales coordinates so the linear system is better conditioned. The estimate is transformed back afterward; it improves numerical stability without changing the intended pixel-coordinate model.

**Q: Why enforce rank 2 on `F`?**
**A:** A physically valid fundamental matrix has rank 2. Noise makes the raw linear estimate full rank, so removing its smallest singular value projects the estimate back onto the valid geometry.

**Q: Why does decomposing `E` produce four poses?**
**A:** SVD decomposition has two rotation choices, and translation has a sign ambiguity. Their combinations give four mathematically compatible candidates before scene depth resolves the ambiguity.

**Q: What is cheirality?**
**A:** Cheirality is the positive-depth test: triangulated points should lie in front of both cameras. Counting such points selects the physically meaningful pose candidate.

**Q: When would you use Sampson distance rather than reprojection error?**
**A:** Sampson distance is a cheap first-order approximation that is useful for ranking and robust filtering many matches. Reprojection error is a more direct geometric measure but costs more and is better suited to final refinement.

**Q: What tooling would you use in production?**
**A:** I would use tested OpenCV or SfM/SLAM components for matching, robust `F` estimation, pose recovery, triangulation, degeneracy checks, and refinement, while keeping this small pipeline for controlled analysis and regression tests.

**Q: What is the row-batch transpose rule in NumPy?**
**A:** Equations often use column vectors, `M @ x`; for rows stored as `X` with shape `(N, d)`, write `X @ M.T`. Making that transpose explicit prevents silently applying transforms in the wrong direction.
