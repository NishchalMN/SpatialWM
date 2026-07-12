from dataclasses import dataclass

import numpy as np


@dataclass
class RansacResult:
    """Result container for RANSAC algorithm."""
    model: np.ndarray
    inliers: np.ndarray
    n_iters: int
    inlier_ratio: float


def ransac(
    data,
    fit_fn,
    score_fn,
    min_samples: int,
    thresh: float,
    p_success: float = 0.99,
    max_iters: int = 5000,
) -> RansacResult:
    """Fit a model robustly with deterministic, adaptive RANSAC sampling.

    ``fit_fn`` receives a random subset of rows and ``score_fn`` must return
    one residual (lower is better) for every row in ``data``. Sampling uses a
    local generator seeded with zero, making repeated calls reproducible. The
    iteration limit is reduced as the best observed inlier ratio improves,
    using ``ceil(log(1-p_success) / log(1-w**min_samples))``.
    """
    try:
        n_data = len(data)
    except TypeError as exc:
        raise TypeError("data must support len()") from exc
    try:
        valid_thresh = bool(np.isfinite(thresh) and thresh > 0)
    except (TypeError, ValueError):
        valid_thresh = False
    if not valid_thresh:
        raise ValueError("thresh must be finite and positive")
    try:
        valid_probability = bool(0 < p_success < 1)
    except (TypeError, ValueError):
        valid_probability = False
    if not valid_probability:
        raise ValueError("p_success must lie strictly between 0 and 1")
    try:
        valid_counts = bool(min_samples >= 1 and n_data >= min_samples)
    except (TypeError, ValueError):
        valid_counts = False
    if not valid_counts:
        raise ValueError("data length must satisfy n_data >= min_samples >= 1")
    try:
        valid_max_iters = bool(max_iters >= 1)
    except (TypeError, ValueError):
        valid_max_iters = False
    if not valid_max_iters:
        raise ValueError("max_iters must be at least 1")

    try:
        rows = np.asarray(data)
    except Exception as exc:
        raise TypeError("data must be indexable as rows") from exc
    rng = np.random.default_rng(0)
    best_model = None
    best_mask = None
    best_count = -1
    best_score = np.inf
    target_iters = int(max_iters)
    n_iters = 0

    def evaluate(model):
        scores = np.asarray(score_fn(model, data))
        if scores.shape != (n_data,):
            raise ValueError(
                f"score_fn must return shape ({n_data},), got {scores.shape}"
            )
        try:
            finite = np.isfinite(scores)
        except (TypeError, ValueError) as exc:
            raise ValueError("score_fn must return numeric scores") from exc
        mask = finite & (scores <= thresh)
        return scores, mask

    while n_iters < target_iters and n_iters < max_iters:
        n_iters += 1
        sample_indices = rng.choice(n_data, size=min_samples, replace=False)
        try:
            model = fit_fn(rows[sample_indices])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

        scores, mask = evaluate(model)
        count = int(np.count_nonzero(mask))
        total = float(np.sum(scores[mask], dtype=float))
        if count < best_count or (count == best_count and total >= best_score):
            continue

        # Refitting on the provisional consensus generally improves the model,
        # but a failed or invalid refit must not discard a usable sample model.
        chosen_model, chosen_mask, chosen_count, chosen_score = (
            model,
            mask,
            count,
            total,
        )
        # bounded so a pathological fit/score pair cannot consume the budget.
        for _ in range(5):
            fit_failure = object()
            try:
                refined_model = fit_fn(rows[chosen_mask])
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                refined_model = fit_failure
            if refined_model is fit_failure:
                break
            refined_scores, refined_mask = evaluate(refined_model)
            refined_count = int(np.count_nonzero(refined_mask))
            refined_score = float(np.sum(refined_scores[refined_mask], dtype=float))
            if refined_count < chosen_count or (
                refined_count == chosen_count and refined_score >= chosen_score
            ):
                break
            chosen_model, chosen_mask, chosen_count, chosen_score = (
                refined_model,
                refined_mask,
                refined_count,
                refined_score,
            )
        if chosen_count > best_count or (
            chosen_count == best_count and chosen_score < best_score
        ):
            best_model, best_mask = chosen_model, chosen_mask
            best_count, best_score = chosen_count, chosen_score
            w = best_count / n_data
            if w >= 1.0:
                required = 1
            elif w <= 0.0:
                required = max_iters
            else:
                denominator = np.log1p(-(w**min_samples))
                required = int(np.ceil(np.log1p(-p_success) / denominator))
                required = max(1, required)
            target_iters = max(n_iters, min(int(max_iters), required))

    if best_model is None:
        raise RuntimeError(
            "RANSAC could not fit a valid model; check fit_fn and data degeneracy"
        )
    return RansacResult(
        model=best_model,
        inliers=np.asarray(best_mask, dtype=bool),
        n_iters=n_iters,
        inlier_ratio=float(best_count / n_data),
    )


def _demo():
    """Run a small generic mean-distance RANSAC example."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2))

    def fit_fn(samples):
        return np.mean(samples, axis=0)

    def score_fn(model, points):
        return np.linalg.norm(points - model, axis=1)

    result = ransac(data, fit_fn, score_fn, min_samples=2, thresh=1.0)
    print(
        f"RANSAC completed with {result.n_iters} iterations "
        f"({result.inlier_ratio:.1%} inliers)"
    )


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
