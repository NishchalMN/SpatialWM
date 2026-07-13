# RANSAC: Robust Model Fitting by Consensus

**Story position:** Stage 2, turning noisy candidate matches into a geometrically consistent inlier set. See [the complete 3D vision story](3d_vision_story.md).

RANSAC (Random Sample Consensus) is a way to fit a model when some observations are wrong. Ordinary least squares gives every row influence, so a sufficiently large outlier group can pull the estimate away from the structure shared by the good data. RANSAC instead asks a simpler question repeatedly: **does a model fitted from a tiny sample agree with many of the other rows?**

## The consensus loop

A generic RANSAC estimator receives data, a `fit_fn(sample)` that builds a model from a minimal sample, and a `score_fn(model, data)` that returns one residual or score per row. Each trial:

1. Draw a minimal sample, with size `s`, that is just sufficient to fit the model.
2. Fit a candidate model from that sample.
3. Score every observation under the candidate.
4. Mark rows below the inlier threshold as inliers.
5. Keep the candidate with the strongest consensus (and, when useful, the best total inlier score).
6. Optionally refit the model using the winning inlier set.

The minimal sample is deliberately small: it gives outliers a chance to be excluded entirely. The consensus check then tests whether the candidate explains a much larger population. A result should include both the model and its inlier mask so downstream code can inspect what was trusted.

## Adaptive iteration count

The required number of trials depends on the chance of drawing an all-inlier minimal sample. If `w` is the estimated inlier ratio, `s` is the minimal sample size, and `p` is the desired probability of seeing at least one all-inlier sample, a common bound is:

`N = ceil(log(1 - p) / log(1 - w^s))`

As `w` rises, `w^s` rises and the required `N` falls sharply. With many outliers, all-inlier samples are rare, so the estimator needs more trials. A practical implementation starts with a configured cap, updates the estimate of `w` whenever it finds a larger consensus, and lowers the remaining target accordingly. It must also guard the `w = 0` and `w = 1` limits rather than evaluating a singular logarithm.

This is an efficiency bound, not a guarantee that the model is correct: it assumes the inlier estimate is meaningful, samples are independent enough, and at least one non-degenerate all-inlier sample exists.

## Choosing the threshold

The threshold converts a continuous residual into a consensus decision. A threshold that is too tight rejects genuine measurements because of pixel noise, calibration error, or an imperfect model. A threshold that is too loose admits outliers and can make a wrong model look popular. Set it in the units and scale of the score function; for calibrated residuals, relate it to expected measurement noise. Inspect the residual distribution and inlier ratio rather than treating one universal number as portable across datasets.

## When RANSAC fails

RANSAC cannot rescue a model that is unidentifiable from its minimal sample or whose assumptions do not match the data. It can fail when the inlier ratio is too low, the minimal sample is degenerate, the threshold is badly scaled, correspondences are correlated, or the scene contains multiple valid structures. Pure rotation, near-planar configurations, and weak baselines can make two-view geometry ambiguous even when matches are clean. A consensus mask is evidence, not proof: inspect spatial coverage, residuals, and downstream pose/cheirality.

## Relationship to fundamental-matrix estimation

For two-view matching, robust estimation is performed by wrapping OpenCV's optimized fundamental matrix estimation. Rather than maintaining a custom random-sampling loop in Python, the system delegates the solver loop to OpenCV's `findFundamentalMat` with methods such as `cv2.USAC_MAGSAC` or `cv2.FM_RANSAC`. This keeps the production path realistic while preserving the conceptual separation between model fitting and consensus evaluation.

## Main Calls & API Shape

### Repository Interface

* **Primary function:** `spatialwm.geometry.ransac.fundamental_ransac(x1, x2, thresh=1.0, p_success=0.99, max_iters=5000, method="usac_magsac") -> RansacResult`
* **Compatibility function:** `spatialwm.geometry.ransac.ransac(data, ..., thresh=1.0, p_success=0.99, max_iters=5000) -> RansacResult`
  * Only supports `data` of shape `(N, 4)` with rows `[u1, v1, u2, v2]`.
  * It intentionally does **not** implement generic scratch RANSAC anymore.
* **Input shapes/types:**
  * `x1`, `x2`: `np.ndarray`-like arrays of shape `(N, 2)` containing matched pixel coordinates.
  * `thresh`: positive `float`, OpenCV inlier reprojection/epipolar-line threshold in pixels.
  * `p_success`: `float` in `(0, 1)`, passed as OpenCV `confidence`.
  * `max_iters`: positive `int`, passed as OpenCV `maxIters`.
  * `method`: `"usac_magsac"` by default, with `"ransac"`/`"fm_ransac"` and `"lmeds"`/`"fm_lmeds"` also accepted.
* **Returned values:**
  * `RansacResult.model`: `(3, 3)` `float64` fundamental matrix.
  * `RansacResult.inliers`: `(N,)` boolean inlier mask.
  * `RansacResult.n_iters`: the configured `max_iters`, because OpenCV does not expose the actual performed iteration count through this API.
  * `RansacResult.inlier_ratio`: `inliers.sum() / N`.

### OpenCV Backend Solver

For production two-view geometry, the robust F-estimation wraps OpenCV's C++ solver directly:

* **Function:** `cv2.findFundamentalMat(points1, points2, method, ransacReprojThreshold, confidence, maxIters)`
* **Input shapes/types:**
  * `points1`, `points2`: `np.ndarray` of shape `(N, 2)` (or `(N, 1, 2)`) containing pixel coordinates.
  * `method`: Solver method flags:
    * `cv2.FM_RANSAC`: Standard random sample consensus.
    * `cv2.FM_LMEDS`: Least Median of Squares.
    * `cv2.USAC_DEFAULT` / `cv2.USAC_ACCURATE`: Modern, highly optimized solvers.
    * `cv2.USAC_MAGSAC`: State-of-the-art threshold-free robust estimator (highly recommended).
  * `ransacReprojThreshold`: `float` maximum distance from point to epipolar line in pixels (e.g., `1.0` to `3.0`).
  * `confidence`: `float` in `(0, 1)`, success confidence level (equivalent to `p_success`, typically `0.99`).
  * `maxIters`: `int`, max iterations (typically `1000` to `5000`).
* **Returned values:**
  * `F`: `np.ndarray` of shape `(3, 3)`, the estimated fundamental matrix (or `None` on failure).
  * `mask`: `np.ndarray` of shape `(N, 1)` and type `uint8` (`1` for inlier, `0` for outlier).

### What to Remember for Interviews

* **Outlier Insensitivity:** RANSAC bounds outlier influence by counting consensus (inliers matching a model within noise threshold) rather than minimizing average error (least squares), which is arbitrarily skewed by large outlier residuals.
* **Minimal Sample Size ($s$):** Must be as small as possible (e.g., 8 points for linear F-estimation, 5 points for relative pose E-estimation) because the probability of drawing an all-inlier sample decays exponentially: $P(\text{all inliers}) = w^s$, where $w$ is the inlier fraction.
* **Adaptive Loop Termination:** Instead of running for a fixed budget, we compute the required iteration count dynamically: $N = \lceil \frac{\ln(1-p)}{\ln(1-w^s)} \rceil$. Every time a larger inlier set is found, we update our estimate of $w$ and reduce the remaining iteration target.
* **USAC/MAGSAC vs. Vanilla RANSAC:** Modern implementations (like `USAC_MAGSAC`) combine guided sampling (PROSAC), local optimization (LO-RANSAC) to refine models, degeneracy checks (DEGENSAC), and marginalization over noise distributions (MAGSAC) to be dramatically faster and more robust.

## Production guidance

For production pipelines in this repository, we construct thin wrappers around mature libraries like OpenCV rather than maintaining custom Python solvers. Deployed estimation must log the solver method (e.g. `USAC_MAGSAC`), the outlier threshold, final inlier ratio, and total iteration count to monitor tracking health. Deterministic seeds are preserved in testing for reproducible validation.

| Symptom | Likely source | First inspection |
|---|---|---|
| Very few inliers | Threshold too tight, low-quality matches, or a wrong score scale | Residual units, threshold, and residual histogram |
| Almost every row is an inlier | Threshold too loose or score is not measuring model error | Score implementation and a visual/known-good residual check |
| Inlier ratio changes wildly between runs | Random sampling, marginal consensus, or degenerate samples | Seed, sample validity, spatial distribution, and best-model history |
| Good count but poor geometry | Clustered inliers, multiple structures, or degeneracy | Image coverage, rank/conditioning checks, and downstream pose/cheirality |
| Iteration budget is unexpectedly large | Current inlier estimate is low or minimal sample is large | `w`, `s`, adaptive bound, and remaining-trial calculation |

## Interview Q&A

**Q: What problem does RANSAC solve?**
**A:** It estimates a model when a minority of observations are outliers by fitting from minimal samples and selecting the model supported by the largest consensus set.

**Q: Why not just use least squares?**
**A:** Least squares gives outliers unbounded influence; a few large residuals can move the fit, while RANSAC uses an inlier decision to limit their influence.

**Q: What are the basic RANSAC steps?**
**A:** Sample minimally, fit a candidate, score every row, threshold the scores into an inlier mask, keep the best consensus, and usually refit on its inliers.

**Q: How does the threshold affect the result?**
**A:** Too tight rejects noisy inliers; too loose accepts outliers. It must match the score's units and expected measurement noise.

**Q: What does the adaptive iteration bound mean?**
**A:** `N = ceil(log(1-p)/log(1-w^s))` chooses enough trials for probability `p` of drawing at least one all-inlier sample, using inlier ratio `w` and minimal size `s`.

**Q: Why does a better inlier ratio reduce iterations?**
**A:** An all-inlier sample has probability `w^s`; increasing `w` makes that event much more likely, so fewer trials are needed.

**Q: What is degeneracy in RANSAC?**
**A:** A sample can be technically inlier-only but unable to determine a stable model, such as geometrically special or poorly distributed points; reject or diagnose those samples.

**Q: What alternatives exist?**
**A:** M-estimators, least-median-of-squares, PROSAC/USAC-style guided sampling, and modern robust estimators trade assumptions, speed, and accuracy differently.

**Q: How is RANSAC used with F or E geometry?**
**A:** Fit a candidate `F` from minimal matches, score all matches with an epipolar residual such as Sampson distance, retain inliers, then refit and continue to `E`, pose, and triangulation.

**Q: What should production diagnostics report?**
**A:** The seed or sampling policy, threshold, trial count, inlier ratio, residual statistics, spatial coverage, degeneracy checks, and whether downstream geometry remains physically valid.
