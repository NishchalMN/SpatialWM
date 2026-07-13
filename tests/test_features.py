"""Tests for deterministic classical feature matching."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spatialwm.geometry.features import FeatureMatchResult, match_features
from spatialwm.geometry.ransac import fundamental_ransac


def _textured_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a repeatable textured image and a translated second view."""
    rng = np.random.default_rng(12)
    image = np.zeros((360, 480), dtype=np.uint8)
    for _ in range(90):
        center = tuple(rng.integers([20, 20], [460, 340]))
        radius = int(rng.integers(3, 10))
        intensity = int(rng.integers(80, 256))
        cv2.circle(image, center, radius, intensity, -1)
    for _ in range(35):
        top_left = rng.integers([10, 10], [430, 310])
        size = rng.integers([8, 8], [35, 35])
        bottom_right = top_left + size
        intensity = int(rng.integers(80, 256))
        cv2.rectangle(image, tuple(top_left), tuple(bottom_right), intensity, 2)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)

    transform = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 6.0]], dtype=np.float32)
    shifted = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]))
    return image, shifted, np.array([12.0, 6.0])


def test_sift_matching_recovers_known_image_translation():
    image1, image2, expected_shift = _textured_pair()

    result = match_features(image1, image2, detector="sift", ratio=0.75, mutual=True)

    assert isinstance(result, FeatureMatchResult)
    assert len(result.points1) >= 40
    displacement = result.points2 - result.points1
    np.testing.assert_allclose(np.median(displacement, axis=0), expected_shift, atol=0.5)
    assert result.raw_match_count >= result.ratio_match_count >= result.mutual_match_count


def test_correspondences_feed_robust_fundamental_estimation():
    image1, image2, _ = _textured_pair()
    result = match_features(image1, image2, detector="sift", ratio=0.8, mutual=True)

    geometry = fundamental_ransac(result.points1, result.points2, thresh=1.0)

    assert geometry.inlier_ratio > 0.85
    assert geometry.inliers.shape == (len(result.points1),)


def test_orb_returns_consistent_array_contract():
    image1, image2, _ = _textured_pair()
    result = match_features(image1, image2, detector="orb", ratio=0.85, mutual=False)

    assert result.detector == "orb"
    assert result.points1.shape == result.points2.shape
    assert result.points1.ndim == 2 and result.points1.shape[1] == 2
    assert result.distances.shape == (len(result.points1),)
    assert np.all(np.isfinite(result.points1))
    assert np.all(np.isfinite(result.points2))


def test_blank_images_return_an_explicit_empty_result():
    blank = np.zeros((100, 120), dtype=np.uint8)
    result = match_features(blank, blank)

    assert result.points1.shape == (0, 2)
    assert result.points2.shape == (0, 2)
    assert result.distances.shape == (0,)
    assert result.n_keypoints1 == 0
    assert result.n_keypoints2 == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"detector": "surf"}, "detector"),
        ({"ratio": 1.0}, "ratio"),
        ({"max_features": 0}, "max_features"),
    ],
)
def test_invalid_matching_configuration_fails_clearly(kwargs, message):
    image1, image2, _ = _textured_pair()
    with pytest.raises(ValueError, match=message):
        match_features(image1, image2, **kwargs)


def test_invalid_image_shape_fails_clearly():
    bad = np.zeros((10, 10, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="channels"):
        match_features(bad, bad)
