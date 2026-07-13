from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RansacResult:
    """Result container for RANSAC estimators.

    Fields:
        model (np.ndarray): The estimated model parameter, e.g. (3, 3) float64 matrix for F.
        inliers (np.ndarray): Boolean mask of shape (N,) indicating consensus set members.
        n_iters (int): Configured iteration cap. OpenCV does not report the
            number of iterations actually performed through this API.
        inlier_ratio (float): Ratio of consensus inliers to total data points (N_inliers / N).
    """
    model: np.ndarray
    inliers: np.ndarray
    n_iters: int
    inlier_ratio: float


def fundamental_ransac(
    x1,
    x2,
    thresh: float = 1.0,
    p_success: float = 0.99,
    max_iters: int = 5000,
    method: str = "usac_magsac",
) -> RansacResult:
    """Fit a fundamental matrix robustly using OpenCV's `findFundamentalMat`.

    Parameters:
        x1 (array_like): Points in the first image, shape (N, 2).
            Converted to finite float64.
        x2 (array_like): Corresponding points in the second image, shape (N, 2).
            Converted to finite float64.
        thresh (float): OpenCV inlier reprojection/epipolar-line threshold in pixels.
        p_success (float): Confidence level/probability of success, in (0, 1). Defaults to 0.99.
        max_iters (int): Maximum number of iterations, >= 1. Defaults to 5000.
        method (str or int): Robust estimator method. Default is 'usac_magsac'
            (uses cv2.USAC_MAGSAC, falling back to cv2.FM_RANSAC if unavailable).
            Also accepts 'ransac'/'fm_ransac' (cv2.FM_RANSAC), 'lmeds'/'fm_lmeds'
            (cv2.FM_LMEDS), or direct integer OpenCV method enums.

    Returns:
        RansacResult: The robust estimation result containing:
            - model: (3, 3) float64 fundamental matrix.
            - inliers: (N,) boolean mask of inliers.
            - n_iters: Returns max_iters (since OpenCV does not report actual iterations).
            - inlier_ratio: Inlier fraction.

    Raises:
        ValueError: For shape mismatches, N < 8, non-finite values, or invalid arguments.
        RuntimeError: If OpenCV fails to return a valid fundamental matrix and mask.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.ndim != 2 or x1.shape[1] != 2:
        raise ValueError(f"x1 must have shape (N, 2), got {x1.shape}")
    if x2.ndim != 2 or x2.shape[1] != 2:
        raise ValueError(f"x2 must have shape (N, 2), got {x2.shape}")
    if x1.shape[0] != x2.shape[0]:
        raise ValueError(
            f"x1 and x2 must have the same length, got {x1.shape[0]} and {x2.shape[0]}"
        )

    n_data = x1.shape[0]
    if n_data < 8:
        raise ValueError(f"At least 8 correspondences are required, got {n_data}")
    if not (np.isfinite(x1).all() and np.isfinite(x2).all()):
        raise ValueError("All point coordinates must be finite (not NaN or Inf).")

    if not (np.isfinite(thresh) and thresh > 0):
        raise ValueError("Threshold must be finite and positive.")
    if not (0 < p_success < 1):
        raise ValueError("p_success must lie strictly between 0 and 1.")
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")

    # Parse method parameter
    if isinstance(method, str):
        method_key = method.lower().strip()
        if method_key == "usac_magsac":
            if hasattr(cv2, "USAC_MAGSAC"):
                cv2_method = cv2.USAC_MAGSAC
            else:
                cv2_method = cv2.FM_RANSAC
        elif method_key in ("ransac", "fm_ransac"):
            cv2_method = cv2.FM_RANSAC
        elif method_key in ("lmeds", "fm_lmeds"):
            cv2_method = cv2.FM_LMEDS
        else:
            attr_name = method.upper()
            if not attr_name.startswith("FM_") and not attr_name.startswith("USAC_"):
                attr_name = "FM_" + attr_name
            if hasattr(cv2, attr_name):
                cv2_method = getattr(cv2, attr_name)
            else:
                raise ValueError(f"Unknown RANSAC method: {method}")
    elif isinstance(method, (int, np.integer)):
        cv2_method = int(method)
    else:
        raise TypeError("method must be a string or integer")

    # Call OpenCV's findFundamentalMat
    F, mask = cv2.findFundamentalMat(
        points1=x1,
        points2=x2,
        method=cv2_method,
        ransacReprojThreshold=thresh,
        confidence=p_success,
        maxIters=max_iters,
    )

    if F is None or mask is None:
        raise RuntimeError("OpenCV findFundamentalMat failed to find a valid model.")

    if F.shape != (3, 3):
        raise RuntimeError(f"OpenCV findFundamentalMat returned unexpected F shape: {F.shape}")

    inliers = mask.ravel().astype(bool)
    inlier_ratio = float(np.count_nonzero(inliers) / n_data)

    return RansacResult(
        model=F.astype(np.float64),
        inliers=inliers,
        n_iters=int(max_iters),  # OpenCV doesn't return iteration count
        inlier_ratio=inlier_ratio,
    )


def ransac(
    data,
    fit_fn=None,
    score_fn=None,
    min_samples: int = 8,
    thresh: float = 1.0,
    p_success: float = 0.99,
    max_iters: int = 5000,
) -> RansacResult:
    """Compatibility wrapper for fundamental matrix RANSAC.

    Generic scratch RANSAC has been removed. This function now only supports Nx4 point
    correspondences (columns 0-1 as x1, columns 2-3 as x2) and dispatches them to
    `fundamental_ransac`.

    Parameters:
        data (array_like): Correspondences of shape (N, 4).
        fit_fn (callable, optional): Ignored.
        score_fn (callable, optional): Ignored.
        min_samples (int): Minimum samples (ignored, OpenCV defaults to 8).
        thresh (float): OpenCV inlier reprojection/epipolar-line threshold in pixels.
        p_success (float): Confidence level in (0, 1). Defaults to 0.99.
        max_iters (int): Maximum iterations. Defaults to 5000.

    Returns:
        RansacResult: Result containing the model, inliers mask, n_iters, and inlier_ratio.

    Raises:
        NotImplementedError: If data shape is not (N, 4), as generic RANSAC is no longer supported.
    """
    try:
        arr = np.asarray(data)
    except Exception as exc:
        raise TypeError("data must be indexable as a numpy array") from exc

    if arr.ndim == 2 and arr.shape[1] == 4:
        return fundamental_ransac(
            arr[:, :2],
            arr[:, 2:],
            thresh=thresh,
            p_success=p_success,
            max_iters=max_iters,
        )

    raise NotImplementedError(
        "Generic scratch RANSAC has been intentionally removed in favor of "
        "library-specific estimators. The compatibility `ransac` symbol "
        "only supports Nx4 point correspondences."
    )


def _demo():
    """Run a small fundamental matrix estimation demo using OpenCV."""
    print("Running OpenCV-backed RANSAC demo...")
    rng = np.random.default_rng(0)

    # Generate points satisfying x2^T F x1 = 0
    # In this case: y2 = y1, x1 and x2 can be anything.
    n_pts = 100
    x1 = rng.uniform(10, 100, (n_pts, 2))
    x2 = x1.copy()

    # Add outliers
    n_outliers = 30
    x2[:n_outliers] = rng.uniform(10, 100, (n_outliers, 2))

    result = fundamental_ransac(
        x1,
        x2,
        thresh=1.0,
        p_success=0.99,
        max_iters=1000,
        method="usac_magsac"
    )
    print("Demo complete!")
    print(f"Estimated F:\n{result.model}")
    print(f"Inliers count: {np.sum(result.inliers)}/{n_pts} ({result.inlier_ratio:.1%} inliers)")
    print(f"Configured iteration cap: {result.n_iters}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
