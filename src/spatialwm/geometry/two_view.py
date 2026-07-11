import numpy as np


def normalize_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley-normalize 2-D points.

    The returned points have zero centroid and mean radial distance
    :math:`\\sqrt{2}`.  Homogeneous ``(N, 3)`` input is dehomogenized first.
    A degenerate point cloud (or point at infinity) is rejected rather than
    producing an unusable transform.
    """
    pts = np.asarray(x, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3) or pts.shape[0] == 0:
        raise ValueError("points must have shape (N,2) or (N,3), N > 0")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite values")
    if pts.shape[1] == 3:
        w = pts[:, 2]
        if np.any(np.abs(w) <= np.finfo(float).eps):
            raise ValueError("homogeneous points must have nonzero scale")
        pts = pts[:, :2] / w[:, None]
        if not np.all(np.isfinite(pts)):
            raise ValueError("dehomogenized points must be finite")
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    mean_radius = float(np.mean(np.linalg.norm(centered, axis=1)))
    if not np.isfinite(mean_radius) or mean_radius <= np.finfo(float).eps:
        raise ValueError("point cloud has degenerate spread")
    scale = np.sqrt(2.0) / mean_radius
    T = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return centered * scale, T


def fundamental_8pt(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Estimate a rank-2 fundamental matrix with the normalized 8-point method."""
    p1, T1 = normalize_points(x1)
    p2, T2 = normalize_points(x2)
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("point sets must contain the same number of correspondences")
    if p1.shape[0] < 8:
        raise ValueError("at least 8 correspondences are required")
    u1, v1 = p1[:, 0], p1[:, 1]
    u2, v2 = p2[:, 0], p2[:, 1]
    A = np.column_stack(
        (u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, np.ones_like(u1))
    )
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    F_norm = Vt[-1].reshape(3, 3)
    Uf, sf, Vtf = np.linalg.svd(F_norm)
    sf[-1] = 0.0
    F_norm = (Uf * sf) @ Vtf
    F = T2.T @ F_norm @ T1
    scale = np.linalg.norm(F)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("fundamental matrix is numerically degenerate")
    return F / scale


def essential_from_F(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Convert a fundamental matrix to an essential matrix, ``K2.T @ F @ K1``."""
    return np.asarray(K2, dtype=float).T @ np.asarray(F, dtype=float) @ np.asarray(K1, dtype=float)


def decompose_E(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the four valid ``(R, t)`` candidates from an essential matrix."""
    U, _, Vt = np.linalg.svd(np.asarray(E, dtype=float))
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1, R2 = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2]
    t /= np.linalg.norm(t)
    return [(R1, t.copy()), (R1, -t.copy()), (R2, t.copy()), (R2, -t.copy())]


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Triangulate each correspondence by the linear DLT null-space solve."""
    p1, p2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    if p1.ndim != 2 or p2.ndim != 2 or p1.shape != p2.shape or p1.shape[1] not in (2, 3):
        raise ValueError("point arrays must have matching shape (N,2) or (N,3)")
    if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
        raise ValueError("points must contain only finite values")
    h1 = np.column_stack((p1, np.ones(len(p1)))) if p1.shape[1] == 2 else p1
    h2 = np.column_stack((p2, np.ones(len(p2)))) if p2.shape[1] == 2 else p2
    P1, P2 = np.asarray(P1, dtype=float), np.asarray(P2, dtype=float)
    out = np.full((len(p1), 3), np.nan)
    for i, (a, b) in enumerate(zip(h1, h2)):
        A = np.vstack((a[0] * P1[2] - a[2] * P1[0],
                       a[1] * P1[2] - a[2] * P1[1],
                       b[0] * P2[2] - b[2] * P2[0],
                       b[1] * P2[2] - b[2] * P2[1]))
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) > np.finfo(float).eps:
            out[i] = X[:3] / X[3]
    return out


def cheirality_select(
    cands: list, K1: np.ndarray, K2: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select the candidate whose triangulated points have greatest positive depth."""
    p1 = np.asarray(K1, dtype=float)
    p2 = np.asarray(K2, dtype=float)
    best_count, best = -1, None
    for R, t in cands:
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float).reshape(3)
        P1 = p1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = p2 @ np.hstack((R, t[:, None]))
        X = triangulate_dlt(P1, P2, x1, x2)
        z1 = X[:, 2]
        z2 = (R @ X.T + t[:, None]).T[:, 2]
        count = int(np.count_nonzero(np.isfinite(z1) & np.isfinite(z2) & (z1 > 0) & (z2 > 0)))
        if count > best_count:
            best_count, best = count, (R.copy(), t.copy())
    if best is None:
        raise ValueError("no pose candidates supplied")
    return best


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute the first-order Sampson distance for corresponding image points."""
    p1, p2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    if p1.ndim != 2 or p2.shape != p1.shape or p1.shape[1] not in (2, 3):
        raise ValueError("point arrays must have matching shape (N,2) or (N,3)")
    h1 = np.column_stack((p1, np.ones(len(p1)))) if p1.shape[1] == 2 else p1
    h2 = np.column_stack((p2, np.ones(len(p2)))) if p2.shape[1] == 2 else p2
    F = np.asarray(F, dtype=float)
    Fx1, Ftx2 = h1 @ F.T, h2 @ F
    numerator = np.sum(h2 * Fx1, axis=1) ** 2
    denominator = np.sum(Fx1[:, :2] ** 2, axis=1) + np.sum(Ftx2[:, :2] ** 2, axis=1)
    return numerator / np.maximum(denominator, np.finfo(float).eps)




def _demo():
    """Demo: call fundamental_8pt on synthetic correspondences."""
    # Generate synthetic correspondences
    n_points = 20
    x1 = np.random.randn(n_points, 2) * 100 + 320
    x2 = x1 + np.random.randn(n_points, 2) * 5
    fundamental_8pt(x1, x2)
    print(f"Computed fundamental matrix from {n_points} correspondences")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
