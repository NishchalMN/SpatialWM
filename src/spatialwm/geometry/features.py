"""Classical local-feature extraction and two-image matching."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureMatchResult:
    """Filtered point correspondences and match-stage diagnostics."""

    points1: np.ndarray
    points2: np.ndarray
    distances: np.ndarray
    n_keypoints1: int
    n_keypoints2: int
    raw_match_count: int
    ratio_match_count: int
    mutual_match_count: int
    detector: str


def _to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Validate and convert an image to the OpenCV grayscale contract."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W), (H, W, 3), or (H, W, 4)")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("a color image must have 3 or 4 channels")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image must contain numeric values")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values")

    values = image.astype(np.float64, copy=False)
    if np.issubdtype(image.dtype, np.floating) and np.max(values) <= 1.0:
        values = values * 255.0
    image_uint8 = np.clip(values, 0.0, 255.0).astype(np.uint8)

    if image_uint8.ndim == 2:
        return image_uint8
    if image_uint8.shape[2] == 3:
        return cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image_uint8, cv2.COLOR_BGRA2GRAY)


def _make_detector(name: str, max_features: int):
    """Create an OpenCV detector and its descriptor-distance norm."""
    detector = name.lower().strip()
    if detector == "sift":
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2, detector
    if detector == "orb":
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING, detector
    raise ValueError("detector must be 'sift' or 'orb'")


def _ratio_matches(
    descriptor_query: np.ndarray,
    descriptor_train: np.ndarray,
    norm: int,
    ratio: float,
) -> tuple[list[cv2.DMatch], int]:
    """Apply Lowe's nearest/second-nearest ambiguity test."""
    matcher = cv2.BFMatcher(normType=norm, crossCheck=False)
    neighbours = matcher.knnMatch(descriptor_query, descriptor_train, k=2)
    accepted = []
    for candidates in neighbours:
        if len(candidates) == 2 and candidates[0].distance < ratio * candidates[1].distance:
            accepted.append(candidates[0])
    return accepted, len(neighbours)


def match_features(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    detector: str = "sift",
    ratio: float = 0.75,
    mutual: bool = True,
    max_features: int = 4000,
    seed: int = 0,
) -> FeatureMatchResult:
    """Extract and filter classical correspondences between two images.

    The images follow OpenCV channel order when supplied as color arrays.
    Matching uses Lowe's ratio test in both directions. With mutual=True, a
    forward match survives only when the reverse filtered match returns to the
    same keypoint.

    Args:
        image1: First grayscale, BGR, or BGRA image.
        image2: Second image with the same accepted layouts.
        detector: Either sift or orb.
        ratio: Nearest/second-nearest distance ratio in the open interval (0, 1).
        mutual: Whether to require symmetric filtered matches.
        max_features: Positive detector feature cap.
        seed: OpenCV RNG seed for repeatable detector internals.

    Returns:
        FeatureMatchResult with one row per retained correspondence.
    """
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("ratio must lie strictly between 0 and 1")
    if not isinstance(max_features, (int, np.integer)) or max_features < 1:
        raise ValueError("max_features must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    gray1 = _to_grayscale_uint8(image1)
    gray2 = _to_grayscale_uint8(image2)
    feature_detector, norm, detector_name = _make_detector(detector, int(max_features))
    cv2.setRNGSeed(int(seed))

    keypoints1, descriptors1 = feature_detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = feature_detector.detectAndCompute(gray2, None)
    n_keypoints1 = len(keypoints1)
    n_keypoints2 = len(keypoints2)

    if descriptors1 is None or descriptors2 is None:
        empty_points = np.empty((0, 2), dtype=np.float64)
        return FeatureMatchResult(
            points1=empty_points,
            points2=empty_points.copy(),
            distances=np.empty(0, dtype=np.float64),
            n_keypoints1=n_keypoints1,
            n_keypoints2=n_keypoints2,
            raw_match_count=0,
            ratio_match_count=0,
            mutual_match_count=0,
            detector=detector_name,
        )

    forward, raw_match_count = _ratio_matches(descriptors1, descriptors2, norm, ratio)
    ratio_match_count = len(forward)

    if mutual:
        reverse, _ = _ratio_matches(descriptors2, descriptors1, norm, ratio)
        reverse_pairs = {(match.trainIdx, match.queryIdx) for match in reverse}
        filtered = [
            match
            for match in forward
            if (match.queryIdx, match.trainIdx) in reverse_pairs
        ]
    else:
        filtered = forward

    filtered.sort(key=lambda match: (match.distance, match.queryIdx, match.trainIdx))
    points1 = np.array([keypoints1[m.queryIdx].pt for m in filtered], dtype=np.float64)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in filtered], dtype=np.float64)
    distances = np.array([m.distance for m in filtered], dtype=np.float64)
    if not filtered:
        points1 = np.empty((0, 2), dtype=np.float64)
        points2 = np.empty((0, 2), dtype=np.float64)

    return FeatureMatchResult(
        points1=points1,
        points2=points2,
        distances=distances,
        n_keypoints1=n_keypoints1,
        n_keypoints2=n_keypoints2,
        raw_match_count=raw_match_count,
        ratio_match_count=ratio_match_count,
        mutual_match_count=len(filtered),
        detector=detector_name,
    )
