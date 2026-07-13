# TartanAir RGB-D Registration Validation: Diagnostic Reference

This document serves as a technical reference and revision guide for the ground-truth-validated RGB-D registration milestone. This diagnostic evaluation checks point-to-point ICP (Iterative Closest Point) registration performance under varying controlled baseline distances on a TartanAir sequence.

> **Important Note:** This diagnostic uses a single-sequence trajectory (e.g. from the Abandoned Factory sequence) as a diagnostic test case. It is not designed to be, nor should it be claimed to be, a comprehensive registration benchmark.

---

## 1. Input Specifications & Coordinate Frames

### Input Data
- **RGB Images:** 8-bit standard 3-channel color images (`image_left/` subdirectory).
- **Depth Maps:** 32-bit single-channel floating-point depth maps stored in NumPy format (`depth_left/` subdirectory). Values represent metric distance in meters. Point clouds are filtered to remove infinite depth values and sky regions by keeping only points where $0.0 < z < 1000.0$.
- **Camera Intrinsics:** The sequence assumes a horizontal field-of-view (HFOV) of 90 degrees. Under this projection model, the parameters are:
  - Focal lengths: $f_x = 320.0$, $f_y = 320.0$
  - Principal point: $c_x = 320.0$, $c_y = 240.0$

### Pose Row Layout
Trajectory poses are loaded from `pose_left.txt`. Each line represents a camera pose formatted as a 7-element vector:
$$\text{pose\_row} = (t_x, t_y, t_z, q_x, q_y, q_z, q_w)$$
- $(t_x, t_y, t_z)$: Translation vector representing the camera center position in the world NED (North-East-Down) frame.
- $(q_x, q_y, q_z, q_w)$: Unit quaternion representing the rotation of the camera's local NED frame relative to the world NED frame.

### NED $\leftrightarrow$ Camera RDF Transformation
TartanAir ground truth is specified in the NED coordinate system. However, standard computer vision tools (e.g., OpenCV, Open3D) operate in the RDF (Right-Down-Forward) camera coordinate system:
- **NED Frame:** $X$-axis points Forward/North, $Y$-axis points Right/East, $Z$-axis points Down.
- **RDF Frame:** $X$-axis points Right (East), $Y$-axis points Down, $Z$-axis points Forward (North).

To transform poses from the world NED frame to standard OpenCV RDF frame, we perform a change-of-basis transformation. This aligns with the official transform defined in the `tartanair_tools` repository:
$$\text{Source URL: } \text{https://github.com/castacks/tartanair\_tools/blob/master/evaluation/trajectory\_transform.py}$$

Let $T_{\text{ned2cam}}$ be the 4x4 matrix mapping NED coordinates to RDF coordinates, and $T_{\text{cam2ned}}$ be its inverse (which is also its transpose):

$$T_{\text{ned2cam}} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad T_{\text{cam2ned}} = T_{\text{ned2cam}}^{-1} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

Given a 4x4 camera-to-world transformation matrix in the NED frame $T_{\text{ned}}$, the corresponding camera pose in the RDF frame $T_{\text{cam}}$ is derived using the change-of-basis conjugation:
$$T_{\text{cam}} = T_{\text{ned2cam}} \cdot T_{\text{ned}} \cdot T_{\text{cam2ned}}$$

This conjugation applies the transformation consistently to both the rotation and translation components, converting the camera trajectory into a representation suitable for visual processing.

---

## 2. Controlled Frame Pair Selection

To analyze registration quality across different spatial displacements, frame pairs are selected based on controlled translation baselines:
- **Source Frame Index:** Fixed to $1750$ (default).
- **Target Baselines:** $0.07\text{ m}$ (short), $0.33\text{ m}$ (medium), and $0.73\text{ m}$ (long).

### Search & Stride Strategy
The algorithm searches forward along the trajectory (where index $i > 1750$) and computes the Euclidean distance between the source camera translation vector $t_{\text{source}}$ and the target translation vector $t_{\text{target}}$. It selects the target index that minimizes the baseline difference:

$$\text{target\_idx} = \arg\min_{i} \left| \| t_{i} - t_{\text{source}} \|_{2} - b \right|$$

Where $b \in \{0.07, 0.33, 0.73\}$. In the case of exact metric distance ties, the search resolves deterministically by selecting the smaller index.

**Crucial Distinction:** The frame index stride (e.g. $+1$ frame, $+12$ frames, $+25$ frames) is a **result** of the search, not an input experiment parameter. Because the camera's physical velocity varies, indexing by a constant stride would result in varying metric baselines. Selecting frames based on physical translation distance isolates baseline distance as the independent variable.

---

## 3. Main Calls & API Shape

### Evaluation Execution
The evaluation script unprojects the selected RGB-D pairs into 3D space, registers them using point-to-point ICP initialized with the identity matrix, computes errors against ground truth, and writes reports.

```bash
python scripts/evaluate_tartanair_icp.py --output-dir /tmp/spatialwm-registration
```

### Output Artifacts
The script writes three output files to the target directory:
1. `registration_report.json`: JSON file detailing the metrics for all tested baselines.
2. `registration_report.csv`: CSV table containing the same metrics for spreadsheet integration.
3. `tartanair_icp_alignment.png`: A side-by-side 3D render visualization of the medium-baseline (0.33 m) registration.

#### CSV & JSON Fields
Each row or entry in the reports contains the following fields:
- `source_idx`: The index of the source frame ($1750$).
- `target_idx`: The selected target frame index matching the baseline criteria.
- `requested_baseline`: The target baseline in meters ($0.07$, $0.33$, or $0.73$).
- `actual_baseline`: The actual ground truth Euclidean baseline distance in meters.
- `translation_error_m`: Absolute translation error between the estimated relative transform and ground truth relative transform in meters (computed as $\|t_{\text{diff}}\|_{2}$).
- `rotation_error_deg`: Absolute rotation angle error in degrees (derived from the trace of $R_{\text{diff}}$).
- `fitness`: The ratio of overlapping inlier points to the total number of points in the source cloud (computed by Open3D).
- `inlier_rmse`: The root mean square error of all point-to-point inlier correspondences.

---

## 4. Repository APIs

The core coordinate math and selection helpers reside in the module `spatialwm.geometry.tartanair`.

### `parse_pose_to_transform`
- **Signature:** `parse_pose_to_transform(pose_row: np.ndarray) -> np.ndarray`
- **Accepted Inputs:** A 1D NumPy float array of shape `(7,)` representing `(tx, ty, tz, qx, qy, qz, qw)`.
- **Returned Value:** A 2D NumPy float array of shape `(4, 4)` representing the camera pose $T_{\text{cam}}$ in RDF camera-to-world frame.
- **Internal Operation:** normalizes the quaternion to avoid numerical drift, constructs the $T_{\text{ned}}$ matrix, and applies the change-of-basis conjugation $T_{\text{ned2cam}} T_{\text{ned}} T_{\text{cam2ned}}$.

### `derive_relative_transform`
- **Signature:** `derive_relative_transform(T_source: np.ndarray, T_target: np.ndarray) -> np.ndarray`
- **Accepted Inputs:** Two 2D NumPy float arrays of shape `(4, 4)` representing camera-to-world poses.
- **Returned Value:** A 2D NumPy float array of shape `(4, 4)` representing the relative transform $T_{\text{source\_to\_target}}$ mapping coordinates from the source camera frame to the target camera frame.
- **Internal Operation:** Computes $T_{\text{target}}^{-1} T_{\text{source}}$.

### `select_target_frames`
- **Signature:** `select_target_frames(poses: np.ndarray, source_idx: int, target_baselines: list[float], max_search_frames: int | None = None) -> list[dict]`
- **Accepted Inputs:**
  - `poses`: 2D NumPy array of shape `(N, 7)`.
  - `source_idx`: `int` indicating the starting frame.
  - `target_baselines`: A non-empty list of positive floats representing target baselines in meters.
  - `max_search_frames`: An optional positive integer bounding the forward search window.
- **Returned Value:** A list of dictionaries, where each dictionary corresponds to a baseline target and contains the keys: `'target_idx'`, `'requested_baseline'`, `'actual_baseline'`, and `'rotation_deg'`.
- **Internal Operation:** Searches forward from the source frame, computes Euclidean distances, resolves ties, and outputs the closest matches.

### `compute_se3_error`
- **Signature:** `compute_se3_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]`
- **Accepted Inputs:** Two 2D NumPy float arrays of shape `(4, 4)` representing the estimated and ground truth transformations.
- **Returned Value:** A tuple of two floats: `(translation_error_meters, rotation_error_degrees)`.
- **Internal Operation:** Computes the relative difference matrix $T_{\text{diff}} = T_{\text{gt}}^{-1} T_{\text{est}}$. Extracts translation norm $\|t_{\text{diff}}\|_2$, and computes the angle $\theta$ in degrees from the trace of the rotation submatrix $R_{\text{diff}}$ using:
  $$\theta = \arccos\left(\text{clip}\left(\frac{\text{trace}(R_{\text{diff}}) - 1.0}{2.0}, -1.0, 1.0\right)\right) \times \frac{180.0}{\pi}$$

---

## 5. Local Metrics vs. Global Pose Errors

A primary observation from this diagnostic is that ICP registration can return **high fitness** and **low local RMSE** while simultaneously yielding **poor ground truth translation errors**. This decoupling happens due to the following structural and algorithmic limits:

1. **Identity Initialization & Local Optimization:** ICP is a local registration algorithm that uses local search (gradient descent-like updates). When initialized with the identity matrix $I_{4}$ (assuming zero motion), the optimization will fail if the actual relative rotation or translation lies outside the local basin of convergence. ICP will quickly settle into a nearby local minimum.
2. **Geometric Ambiguity (Aperture Problem):** In environments dominated by planar structures (such as walls, flat floors, or long corridors) or symmetric structures (such as pipes), there is a geometric degeneracy. The point cloud can slide along these planes without changing the point-to-point correspondence distance. ICP will converge with a low RMSE and high fitness because the points lie flat against each other, but the translation along the degenerate direction is unconstrained and will be incorrect.
3. **Partial Overlap & FoV Mismatches:** For medium and long baselines, parts of the scene enter or exit the camera's field of view. The points in these non-overlapping regions do not have true physical correspondences in the target cloud. If the correspondence threshold is set too wide and no outlier filtering (like RANSAC) is applied, ICP will force these points to align with unrelated target surfaces. This produces a false alignment with high point proximity (low RMSE) but incorrect physical camera alignment (high SE(3) error).
4. **Incorrect Calibration Assumptions:** Point unprojection depends on the camera intrinsic parameters ($f_x, f_y, c_x, c_y$). If the assumed intrinsics differ from the actual calibration used to generate the depth maps, the backprojected 3D geometries will be distorted (e.g., stretched, bent). ICP will align these distorted point clouds, yielding low RMSE in the distorted space, but the computed rigid transform will be physically incorrect, resulting in high translation and rotation errors against the ground-truth trajectory poses.

---

## 6. Practical Debugging & Render Interpretation

### Registration Debugging Checklist
1. **Initial Alignment Visual Check:** Render both point clouds *before* running ICP. If they do not visually overlap or have a large rotational offset, provide a better initialization matrix (e.g., from an odometry model, IMU, or 2D feature-based relative pose estimation) instead of the identity matrix.
2. **Coordinate Axes Verification:** Ensure both point clouds are unprojected into the same camera frame (RDF) and that the intrinsic parameters are applied correctly.
3. **Pre-Filter Sky and Noise:** Filter out infinite depth values and points representing the sky (depth $\ge 1000$ m) to prevent spurious correspondences at the margins of the scene.
4. **Tune Correspondence Threshold ($d_{\text{max}}$):** If $d_{\text{max}}$ is too small, ICP cannot capture larger movements. If it is too large, non-overlapping points will drag the alignment. Use a multi-scale approach (coarse-to-fine) starting with a larger threshold and decreasing it.
5. **Analyze Scene Geometry:** Inspect if the scene contains geometric features (corners, curves) to constrain all 6 Degrees of Freedom. If the scene is planar, consider incorporating photometric constraints (Color ICP).
6. **Subsampling Seed:** Verify that the point cloud subsampling uses a fixed random seed to ensure deterministic and reproducible behavior.

### Interpreting the 3D Render
The Matplotlib side-by-side plot (`tartanair_icp_alignment.png`) provides visual confirmation of the registration quality:
- **Structural Alignment:** Looking at geometric boundaries (corners, wall intersections, edges) in the "After ICP" panel reveals if they line up. Double-images or blurry overlaps indicate registration drift.
- **Texture Consistency:** Since the point clouds are colored, the visual overlap of texture patterns (e.g., floor textures, rust on factory structures) indicates if rotation and translation are correctly recovered.
- **Convergence Failure Mode:** The render helps distinguish between global divergence (clouds completely separated) and local slide errors (clouds appear aligned on planes but slid along a degenerate axis).
- **Coordinate Reference:** The coordinate frame visualizer at the origin shows the RDF axes ($X$=Right, $Y$=Down, $Z$=Forward), verifying the camera orientation.

---

## Interview Q&A

### Q: Explain the mathematical mapping and conjugation used for NED to RDF camera pose conversion.
**A:** TartanAir trajectory poses are defined in the world NED (North-East-Down) frame. Standard computer vision libraries expect RDF (Right-Down-Forward) coordinates. We define a change-of-basis matrix $T_{\text{ned2cam}}$ that permutes the axes such that $X_{\text{rdf}} = Y_{\text{ned}}$ (Right), $Y_{\text{rdf}} = Z_{\text{ned}}$ (Down), and $Z_{\text{rdf}} = X_{\text{ned}}$ (Forward). To convert the camera NED pose $T_{\text{ned}}$ to RDF, we apply the change-of-basis conjugation:
$$T_{\text{cam}} = T_{\text{ned2cam}} \cdot T_{\text{ned}} \cdot T_{\text{cam2ned}}$$
where $T_{\text{cam2ned}} = T_{\text{ned2cam}}^{-1}$. This conjugation transforms both the rotation matrix and translation components into the RDF world frame.

### Q: How does the frame pair selection algorithm work, and why is the frame index stride a result rather than an input parameter?
**A:** The algorithm searches forward from a fixed source frame index and computes the Euclidean distance between the source camera center and subsequent camera centers in 3D space. It matches each requested metric baseline (e.g., 0.33 m) to the forward frame index that minimizes the baseline error. Because the camera velocity and frame rate are variable, the frame index step (stride) required to reach a physical distance varies. Treating the stride as an output ensures that the registration evaluation is controlled directly by physical metric baseline distances.

### Q: Why can an ICP registration yield high fitness and low RMSE while suffering from a large ground truth translation error?
**A:** Local metrics like fitness and inlier RMSE only measure the proximity of matched point pairs, not absolute trajectory accuracy. This discrepancy occurs because:
1. **Identity Initialization:** ICP gets stuck in local minima if the displacement exceeds the convergence basin.
2. **Aperture Problem:** In planar or symmetric environments (e.g., walls, corridors), points can slide along the surface without affecting the point-to-point error, yielding low RMSE but high translation error along the slide direction.
3. **FoV Mismatch:** Points in non-overlapping regions are forced into incorrect correspondences, biasing the transform.
4. **Calibration Errors:** Mismatched camera intrinsics distort the 3D geometry, causing ICP to find a transformation that aligns distorted shapes well locally but is physically incorrect.

### Q: How can a developer visually diagnose registration errors using the generated side-by-side 3D render?
**A:** By comparing the "Before ICP" and "After ICP" panels, a developer can evaluate:
1. **Structural Overlap:** Check if distinct geometric structures (corners, walls, pillars) overlap without double-images or ghosting.
2. **Texture and Color Alignment:** Check if colored textures align consistently across the registered point clouds.
3. **Failure Modality:** Determine if the algorithm completely diverged (separated clouds) or slid along a degenerate axis (well-aligned plane but wrong translation).
4. **RDF Orientation:** Inspect the origin coordinate frame to ensure that the unprojected coordinates match the RDF convention.

### Q: Derive the formula for the relative transformation $T_{\text{source} \to \text{target}}$ between two camera poses.
**A:** Let $T_{\text{source}}$ and $T_{\text{target}}$ be the camera-to-world transformations mapping coordinates from the local source and target camera frames to the world coordinate system:
$$P_{\text{world}} = T_{\text{source}} P_{\text{source}}$$
$$P_{\text{world}} = T_{\text{target}} P_{\text{target}}$$
Equating the two expressions for the world point $P_{\text{world}}$:
$$T_{\text{target}} P_{\text{target}} = T_{\text{source}} P_{\text{source}}$$
Multiplying both sides by the inverse matrix $T_{\text{target}}^{-1}$ from the left yields:
$$P_{\text{target}} = T_{\text{target}}^{-1} T_{\text{source}} P_{\text{source}}$$
Thus, the relative transformation matrix $T_{\text{source} \to \text{target}}$ mapping points from the source camera coordinate system to the target camera coordinate system is:
$$T_{\text{source} \to \text{target}} = T_{\text{target}}^{-1} T_{\text{source}}$$
