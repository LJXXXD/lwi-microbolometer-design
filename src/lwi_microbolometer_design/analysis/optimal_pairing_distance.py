"""
Optimal pairing distance calculation.

This module provides optimal pairing (Hungarian algorithm) preprocessing for
distance calculations between parameter sets. Optimal pairing reorders groups
within each array to find the best matching before computing distances, ensuring
robust comparison regardless of group ordering.

**Key Insight:**
Optimal pairing is a preprocessing step for distance calculation, not a separate
distance metric. Each pair comparison requires its own optimal matching (cannot
be vectorized across full distance matrices).

**Primary API:**
- calculate_optimal_pairing_distance(): Compute distance between two lists
  using optimal pairing. Supports vectorized metrics and custom distance functions.

**Note:**
This module is typically used internally by `distance_matrix.py`. Most users
should use `compute_distance_matrix()` with `use_optimal_pairing=True` rather
than importing this module directly.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def calculate_optimal_pairing_distance(
    items_a: list[Any],
    items_b: list[Any],
    metric: str | None = "euclidean",
    distance_func: Callable[[Any, Any], float] | None = None,
    metric_params: dict[str, Any] | None = None,
) -> float:
    """
    Compute distance between two lists using optimal pairing.

    Uses the Hungarian algorithm to find the best matching pairs before computing
    distances, ensuring robust comparison regardless of item ordering.

    Parameters
    ----------
    items_a : List[Any]
        First list of items to compare.
    items_b : List[Any]
        Second list of items to compare.
    metric : Optional[str], default="euclidean"
        Vectorized distance metric. Options: "euclidean", "manhattan",
        "chebyshev", "minkowski", "cosine", "hamming", "canberra",
        "bray_curtis", "jaccard", "correlation", "mahalanobis".
        Ignored if distance_func provided.
    distance_func : Optional[Callable[[Any, Any], float]]
        Custom distance function. When provided, uses slower non-vectorized path
        and takes precedence over metric.
    metric_params : Optional[Dict[str, Any]]
        Parameters for metric (e.g., {"p": 3} for Minkowski,
        {"cov_matrix": matrix} for Mahalanobis).

    Returns
    -------
    float
        Sum of distances between optimally paired items.

    Examples
    --------
    >>> # Vectorized Euclidean distance
    >>> items_a = [(8.0, 1.0), (12.0, 1.5), (16.0, 2.0), (20.0, 1.0)]
    >>> items_b = [(8.5, 1.2), (12.5, 1.7), (16.5, 2.2), (20.5, 1.2)]
    >>> distance = calculate_optimal_pairing_distance(items_a, items_b, metric='euclidean')

    >>> # Custom distance function
    >>> def manhattan_distance(t1, t2):
    ...     return sum(abs(a - b) for a, b in zip(t1, t2))
    >>> distance = calculate_optimal_pairing_distance(
    ...     items_a, items_b, distance_func=manhattan_distance
    ... )
    """
    # Custom distance function path (non-vectorized, warns about performance)
    if distance_func is not None:
        warnings.warn(
            "Custom distance_func provided; using non-vectorized pairwise computation. "
            "Expect slower performance for large n.",
            UserWarning,
            stacklevel=2,
        )

        if len(items_a) != len(items_b):
            raise ValueError(
                f"Both lists must have the same length. Got {len(items_a)} and {len(items_b)}"
            )

        n = len(items_a)
        pairwise_distances = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                pairwise_distances[i, j] = float(distance_func(items_a[i], items_b[j]))

    else:
        # Vectorized path for supported metrics
        if metric is None:
            raise ValueError(
                "metric must be provided when distance_func is None. Supported metrics: "
                "'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine', "
                "'hamming', 'canberra', 'bray_curtis', 'jaccard', 'correlation', 'mahalanobis'"
            )

        a, b = _ensure_2d_numeric_arrays(items_a, items_b)
        try:
            pairwise_distances = _pairwise_distances_vectorized(
                a=a, b=b, metric=metric, metric_params=metric_params
            )
        except Exception as exc:
            # Vectorization failed - user needs to provide custom function
            warnings.warn(
                f"Vectorized metric computation failed ({exc!s}). "
                f"Use distance_func parameter for custom distance functions.",
                UserWarning,
                stacklevel=2,
            )
            raise

    # Find optimal pairing using Hungarian algorithm and return total distance
    row_indices, col_indices = linear_sum_assignment(pairwise_distances)
    return float(pairwise_distances[row_indices, col_indices].sum())


def _ensure_2d_numeric_arrays(
    items_a: list[Any], items_b: list[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 2D numeric arrays for vectorized distance computation.

    Converts scalars to (n, 1) and 1D vectors to (n, d) shape.
    """
    if len(items_a) != len(items_b):
        raise ValueError(
            f"Both lists must have the same length. Got {len(items_a)} and {len(items_b)}"
        )

    # Convert to arrays (items_a and items_b are lists, convert to arrays)
    a = np.array(items_a)
    b = np.array(items_b)

    # Ensure 2D shape for vectorized operations
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    return a, b


def _cdist_arrays_and_kwargs(
    metric: str,
    metric_params: dict[str, Any],
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Map public metric names to :func:`scipy.spatial.distance.cdist` inputs."""
    kwargs: dict[str, Any] = {}
    aa, bb = a, b

    if metric == "manhattan":
        return aa, bb, "cityblock", kwargs
    if metric == "bray_curtis":
        return aa, bb, "braycurtis", kwargs
    if metric == "minkowski":
        p = float(metric_params.get("p", 2))
        if p <= 0:
            raise ValueError("Minkowski p must be > 0")
        kwargs["p"] = p
        return aa, bb, "minkowski", kwargs
    if metric == "mahalanobis":
        cov_matrix = metric_params.get("cov_matrix")
        if cov_matrix is None:
            combined = np.vstack([a, b])
            cov_matrix = np.cov(combined.T)
        try:
            kwargs["VI"] = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            kwargs["VI"] = np.linalg.pinv(cov_matrix)
        return aa, bb, "mahalanobis", kwargs
    if metric == "jaccard":
        aa = np.asarray(a > 0, dtype=bool)
        bb = np.asarray(b > 0, dtype=bool)
        return aa, bb, "jaccard", kwargs

    if metric in {
        "euclidean",
        "chebyshev",
        "cosine",
        "hamming",
        "canberra",
        "correlation",
    }:
        return aa, bb, metric, kwargs

    raise ValueError(
        "Unsupported metric. Choose one of: 'euclidean', 'manhattan', "
        "'chebyshev', 'minkowski', 'cosine', 'hamming', 'canberra', "
        "'bray_curtis', 'jaccard', 'correlation', 'mahalanobis'."
    )


def _pairwise_distances_vectorized(
    a: np.ndarray,
    b: np.ndarray,
    metric: str,
    metric_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute pairwise distances between rows of ``a`` and ``b`` via SciPy ``cdist``."""
    if metric_params is None:
        metric_params = {}

    aa, bb, scipy_metric, kwargs = _cdist_arrays_and_kwargs(metric, metric_params, a, b)
    return np.asarray(cdist(aa, bb, metric=scipy_metric, **kwargs), dtype=np.float64)
