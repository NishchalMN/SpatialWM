# RANSAC: Robust Model Fitting by Consensus

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

RANSAC cannot rescue a model that is unidentifiable from its minimal sample or whose assumptions do not match the data. It can fail when the inlier ratio is too low, the minimal sample is degenerate (for example, poorly distributed or geometrically special points), the threshold is badly scaled, correspondences are correlated, or the scene contains multiple valid structures. Pure rotation, near-planar configurations, and weak baselines can make two-view geometry ambiguous even when matches are clean. A consensus mask is evidence, not proof: always inspect spatial coverage, residuals, and degeneracy diagnostics.

## Relationship to fundamental-matrix estimation

For two-view matching, RANSAC wraps a fundamental-matrix estimator. A minimal correspondence sample is passed to the eight-point-style `fit_fn`; the candidate `F` is scored on all matches, commonly with Sampson distance; and matches below the threshold form the consensus set. The winning inlier mask can then support a final refit and the downstream essential-matrix, pose, and triangulation pipeline. Keeping RANSAC generic means geometry-specific choices—sample size, fitting, and residual definition—are supplied by callers rather than embedded in the robust-estimation loop.

## Production guidance

For production, use a mature implementation such as OpenCV's USAC/RANSAC variants or a well-tested SfM/SLAM library when its model, score, and diagnostics match the application. Production systems should record the threshold, trial budget, achieved inlier ratio, residual statistics, spatial coverage, and whether refitting succeeded. Deterministic sampling is useful for reproducible tests and debugging; deployed systems may choose a controlled seed or a stronger sampler when repeatability and adversarial robustness requirements are understood.

## Diagnostics

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
