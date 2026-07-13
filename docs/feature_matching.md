# Classical Feature Matching: From Appearance to Geometry

**Story position:** Stage 1, creating candidate correspondences before robust two-view geometry. See [the complete 3D vision story](3d_vision_story.md).

## What the stage computes

Feature matching tries to identify pixels in two images that depict the same physical scene location. The result is not yet trusted geometry. It is a set of candidate point pairs that RANSAC can verify against one fundamental matrix.

The repository pipeline performs:

1. SIFT or ORB keypoint detection and description.
2. Nearest and second-nearest descriptor search.
3. Lowe ratio filtering to remove ambiguous appearance matches.
4. Optional symmetric filtering: image 1 to image 2 and back must select the same pair.
5. Fundamental-matrix RANSAC to identify a geometrically consistent subset.

## Main call and API shape

~~~python
match_features(
    image1: np.ndarray,
    image2: np.ndarray,
    detector: str = "sift",
    ratio: float = 0.75,
    mutual: bool = True,
    max_features: int = 4000,
    seed: int = 0,
) -> FeatureMatchResult
~~~

Accepted images are grayscale, BGR, or BGRA NumPy arrays. Color order follows OpenCV.

FeatureMatchResult contains:

- points1 and points2 with shape (N, 2);
- descriptor distances with shape (N,);
- keypoint counts for both images;
- raw query, ratio-filtered, and mutual-filtered counts;
- the detector name.

The points can be passed directly into fundamental_ransac.

## Why ratio filtering works

A nearest descriptor is not automatically distinctive. In repeated texture, the best and second-best candidates may be almost equally good. Lowe's test keeps a match only when:

    best_distance < ratio * second_best_distance

A lower ratio is stricter. It generally improves precision at the cost of fewer correspondences.

## Why mutual filtering helps

If keypoint A chooses B, but B chooses a different keypoint when matching in reverse, the pairing is unstable. Requiring the reverse filtered match to return to A removes many-to-one and ambiguous matches. It still cannot enforce scene geometry; RANSAC does that next.

## Current real-data checkpoint

Generate:

~~~bash
uv run python scripts/make_figures.py \
  --figures tartanair-matches \
  --output-dir figures/curated \
  --tartanair-frame 1750 \
  --tartanair-stride 5
~~~

For TartanAir P000 frames 1750 to 1755:

- ground-truth motion: 0.329 m and 5.35 degrees;
- SIFT keypoints: 965 and 1100;
- raw descriptor queries: 965;
- ratio-filtered matches: 508;
- mutual matches: 481;
- RANSAC inliers: 474;
- geometric inlier ratio: 98.5%.

![TartanAir classical feature matching](../figures/curated/tartanair_feature_matches.png)

The green lines are estimated geometric inliers. Red lines are filtered descriptor matches rejected by the fundamental-matrix consensus. The lower panels show epipolar constraints and the real-data Sampson-error distribution.

## Failure modes

- Repeated textures produce several similar descriptors.
- Motion blur and large illumination change reduce repeatability.
- Tiny baseline gives many matches but weak depth.
- Very large baseline makes local appearance less similar.
- Moving objects may form a strong but incorrect motion consensus.
- A planar scene can make two-view estimation degenerate.
- A high inlier ratio does not prove metric translation scale.

## Interview Q&A

**What is the difference between detection and description?**  
Detection selects repeatable image locations. Description converts the local appearance into a vector used for comparison.

**Why compare the best and second-best descriptor matches?**  
Their ratio measures ambiguity. Two similarly good candidates indicate that the nearest match is not distinctive.

**What does mutual matching remove?**  
Asymmetric and many-to-one pairings that are not stable when the search direction is reversed.

**Why is RANSAC still needed after descriptor filtering?**  
Appearance filters do not enforce a single camera geometry. RANSAC selects matches consistent with one fundamental matrix.

**What does a high RANSAC inlier ratio prove?**  
It shows that many retained correspondences agree with one epipolar model. It does not prove correct metric scale, non-degenerate motion, or static-scene validity.

**SIFT versus ORB?**  
SIFT uses floating-point gradient descriptors and is usually more robust to scale and appearance changes. ORB uses fast binary descriptors and Hamming distance, trading some robustness for speed.
