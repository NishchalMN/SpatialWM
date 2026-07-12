# ICP: Iterative Closest Point Registration

Iterative Closest Point (ICP) is a local optimization algorithm used to estimate the relative rigid transformation $T \in \text{SE}(3)$ (a $4 \times 4$ rotation and translation matrix) that best aligns a source point cloud $\mathcal{P}_s$ to a target (or destination) point cloud $\mathcal{P}_d$.

Given a set of point correspondences, the algorithm iteratively solves for the transformation that minimizes the alignment error and updates the coordinates of the source point cloud. Because the correspondence matching step relies on local search (e.g., finding the nearest neighbor in the target cloud for each source point), ICP is a **local optimization method**. If the initial alignment (initialization $T_0$) is too far from the ground truth, the nearest-neighbor search will associate incorrect points, causing the algorithm to converge to a poor local minimum.

---

## Point-to-Point vs. Point-to-Plane

The registration objective varies depending on the surface geometry and the availability of normal vectors:

### 1. Point-to-Point ICP
This formulation minimizes the sum of squared Euclidean distances between matching points:
$$E(T) = \sum_{(i,j) \in \mathcal{C}} \| T p_i - q_j \|^2$$
where $p_i \in \mathcal{P}_s$, $q_j \in \mathcal{P}_d$, and $\mathcal{C}$ is the set of active correspondences. Once correspondences are fixed, the optimal rigid transformation is solved in closed form using the Singular Value Decomposition (SVD) of the cross-covariance matrix (the Kabsch algorithm).

### 2. Point-to-Plane ICP
This formulation minimizes the orthogonal distance from the transformed source points to the tangent planes at the corresponding target points:
$$E(T) = \sum_{(i,j) \in \mathcal{C}} \left( (T p_i - q_j) \cdot n_j \right)^2$$
where $n_j$ is the unit normal vector at the target point $q_j$.
* **Advantage:** Point-to-plane allows points to slide along flat regions (e.g., walls, floors) without penalty. It typically converges significantly faster and is far more robust to geometric degeneracies than point-to-point.
* **Disadvantage:** It requires reliable target surface normals. Noisy or misoriented normals degrade or ruin the optimization.

---

## Practical Knobs & Challenges

### 1. Initialization ($T_0$)
ICP requires a starting pose. For consecutive frames in a sequence (e.g., video or LiDAR odometry), the identity matrix is a common baseline (assuming small motion). In high-dynamics settings, a motion model (e.g., constant velocity) or global alignment (e.g., FPFH features with RANSAC) must warm-start ICP to place the source cloud within the convergence basin.

### 2. Correspondence Distance Threshold
Matches are only kept if their distance is below a threshold.
* **Too tight:** Rejects true matches when the initial offset is large, preventing convergence.
* **Too loose:** Admits outlier matches from distant surfaces, introducing bias and leading to incorrect poses.

### 3. Partial Overlap
When registering scans with only partial field-of-view overlap, unmatched points will be erroneously paired with boundary points of the target cloud. ICP must reject these false pairings. Common strategies include trimmed ICP (retaining only the top $k\%$ best matches) or distance/angle-based rejection thresholds.

### 4. Dynamic Content
Moving objects (pedestrians, vehicles) generate transient points that do not represent static scene structure. These act as structured outliers and pull the estimated pose away from the ground truth. M-estimators (e.g., Huber or Tukey loss) or semantic/temporal segmentation masks are required to downweight or filter these regions.

### 5. Odometry Drift
Chaining relative transforms ($P_{k+1} = P_k T_k$) compounds registration errors multiplicatively. Without loop closure, global map registration (scan-to-map), or bundle adjustment, local tracking errors accumulate into significant global translation and rotation drift.

---

## Main Calls & API Shape

### Repository Interface

* **Rich Wrapper:** `spatialwm.geometry.icp.register_point_clouds(src, dst, max_correspondence_distance=0.5, max_iters=50, tol=1e-6, init=None) -> RegistrationResult`
  * *Inputs:*
    * `src`: `(N, 3)` source point cloud NumPy array.
    * `dst`: `(M, 3)` target point cloud NumPy array.
    * `max_correspondence_distance`: `float` maximum distance for a match to be considered an inlier (default `0.5`).
    * `max_iters`: `int` maximum iterations before stopping (default `50`).
    * `tol`: `float` convergence tolerance on transformation delta (default `1e-6`).
    * `init`: `(4, 4)` initial guess SE(3) transformation matrix (optional, default `None`).
  * *Configuration:*
    * Note that this wrapper adapter supports only Open3D point-to-point registration. Point-to-plane registration is a production Open3D alternative that is not exposed by this library-backed adapter yet.
  * *Returns:* An object containing:
    * `transformation`: `(4, 4)` final SE(3) transformation matrix.
    * `fitness`: `float` ratio of inlier correspondences to total source points (range $[0.0, 1.0]$).
    * `inlier_rmse`: `float` root-mean-squared distance error of the matched inlier pairs.
    * `correspondence_set`: `(C, 2)` array of paired point indices.

* **Compatibility function:** `spatialwm.geometry.icp.icp_point2point(src, dst, max_iters=50, tol=1e-6) -> tuple[np.ndarray, list[float]]`
  * *Inputs:* `src` `(N, 3)`, `dst` `(M, 3)` arrays, `max_iters` `int`, `tol` `float`.
  * *Returns:*
    * `T`: `(4, 4)` final transformation matrix.
    * `errors`: A **one-item list** containing only the final inlier RMSE as a float: `[final_rmse]`. It does not construct or fabricate an iterative error trace since Open3D does not expose per-iteration RMSE through standard calls.

### Open3D Backend Solver

The production-grade backend relies on Open3D's optimized C++ registration module:

* **Function:** `open3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance, init, estimation_method, criteria)`
  * *Key Inputs:*
    * `source`, `target`: `open3d.geometry.PointCloud` objects.
    * `max_correspondence_distance`: `float`.
    * `init`: `(4, 4)` float64 NumPy array.
    * `estimation_method`: Open3D estimation object:
      * `open3d.pipelines.registration.TransformationEstimationPointToPoint()`
      * `open3d.pipelines.registration.TransformationEstimationPointToPlane()` (requires `target` to have computed normals).
    * `criteria`: `open3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness, relative_rmse, max_iteration)`.
  * *Returns:* `open3d.pipelines.registration.RegistrationResult` exposing:
    * `transformation`: `(4, 4)` float64 array.
    * `fitness`: `float` (overlap fraction).
    * `inlier_rmse`: `float` (RMS of matching distances).
    * `correspondence_set`: `open3d.utility.Vector2iVector` array of shape `(C, 2)`.

---

## Production & Debugging Checklist

- [ ] **Warm-Start Check:** Confirm that the initialization pose ($T_0$) aligns the clouds closely enough for the correspondence threshold to capture initial inliers.
- [ ] **Target Normal Computation:** If using point-to-plane, verify that `dst` point cloud normals are computed, normalized, and consistently oriented (e.g., pointing toward the sensor center).
- [ ] **Scale Calibration:** Match the correspondence threshold (e.g. 0.05m or 5.0cm) to the physical scale of the sensor data and expected initial error.
- [ ] **Voxel Downsampling:** Downsample point clouds (e.g., using `voxel_downsample`) to establish uniform point density, preventing high-density regions from biasing the objective.
- [ ] **Reflection Detection:** Verify that the rotation component $R = T[:3, :3]$ is orthonormal ($\text{det}(R) = +1$) and lacks reflection artifacts.
- [ ] **Diagnostics vs. Benchmarks:** Keep in mind that `tartanair-icp` is a diagnostic visualization tool for qualitative analysis on a single real daylight sequence (`abandonedfactory/Easy/P000`) and does not constitute a quantitative trajectory benchmark.

---

## Troubleshooting Guide

| Symptom | Likely Source | First Inspection |
| :--- | :--- | :--- |
| Converges to incorrect pose / local minimum | Bad initialization or too narrow correspondence threshold | Check the initial alignment of the clouds relative to the search threshold. |
| High `inlier_rmse` and low `fitness` | Outlier points, dynamic motion, or very small spatial overlap | Verify scan overlap visually; inspect distance filtering parameters. |
| Registration fails or runs slowly | Missing surface normals (point-to-plane) or excessively dense clouds | Check target normals; apply voxel downsampling to reduce point count. |
| Rapid drift in LiDAR odometry | Compounding error in scan-to-scan matching without mapping | Plot accumulated trajectory; evaluate scan-to-map or loop closure options. |
| Inlier count is zero | Threshold is too tight or clouds are completely disjoint | Scale of threshold; verify the coordinate frames of both point clouds. |

---

## Interview Q&A

**Q: What does ICP estimate?**
**A:** It estimates the relative 3D rigid body transformation (rotation $R \in \text{SO}(3)$ and translation $t \in \mathbb{R}^3$, combined into $T \in \text{SE}(3)$) that aligns a source point cloud to a target point cloud.

**Q: Why is ICP considered a local optimization algorithm?**
**A:** Because it alternates between matching nearest neighbors and updating the transformation. If the initial alignment is poor, the nearest-neighbor search pairs incorrect points, causing the optimizer to converge to a local minimum.

**Q: When does point-to-plane ICP outperform point-to-point?**
**A:** Point-to-plane performs best in structured environments with flat surfaces (e.g., walls, floors). By projecting residuals onto surface normals, points can slide along flat surfaces without increasing the cost. This speeds up convergence and avoids local minima caused by planar slide geometry.

**Q: What is a major failure mode of point-to-plane ICP?**
**A:** It is highly dependent on target normals. If normals are not computed, are extremely noisy, or are flipped in random directions, the optimization direction becomes corrupted, causing the registration to diverge.

**Q: How do you handle partial overlap between point clouds?**
**A:** We apply a strict correspondence distance cutoff, use trimmed ICP (which only optimizes the top $k\%$ closest matches), or employ robust loss functions (like Huber or Tukey M-estimators) to discard or downweight matches that lack true correspondence.

**Q: Why is voxel-downsampling crucial before running ICP?**
**A:** Voxel-downsampling ensures a uniform density of points. Without it, areas close to the scanner (which contain many dense points) will dominate the loss function, biasing the pose estimate toward local sensor geometry. It also reduces data size, which accelerates nearest-neighbor search.

**Q: What causes drift in scan-to-scan LiDAR odometry, and how is it resolved?**
**A:** Scan-to-scan matching accumulates small registration errors over time, compounding translation and rotation errors multiplicatively. It is mitigated by aligning scans to a growing keyframe map (scan-to-map), registering to a global submap, or running backend loop closure.

**Q: What do "fitness" and "inlier_rmse" measure in registration?**
**A:** `fitness` measures the ratio of source points that found a target match within the correspondence distance threshold (indicating overlap quality). `inlier_rmse` is the root-mean-squared distance of these active matches (indicating alignment precision).

**Q: How does the Kabsch algorithm solve the transformation once correspondences are fixed?**
**A:** It centers both clouds by subtracting their centroids, computes the cross-covariance matrix $H = \Sigma p_i q_j^T$, performs Singular Value Decomposition (SVD) to get $H = U \Sigma V^T$, and computes rotation as $R = V U^T$. If $\det(R) = -1$, it flips the sign of the last column of $V$ to resolve reflection.

**Q: How does the choice of the maximum correspondence distance threshold affect ICP?**
**A:** A threshold that is too large admits outliers, biasing the pose. A threshold that is too small limits the convergence basin; if the initial displacement is larger than the threshold, no inlier correspondences are found, and the algorithm fails to converge.
