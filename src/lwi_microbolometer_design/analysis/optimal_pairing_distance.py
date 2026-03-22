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


def _compute_euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance."""
    diff = a[:, None, :] - b[None, :, :]
    result = np.linalg.norm(diff, axis=2)
    return np.asarray(result, dtype=np.float64)


def _compute_manhattan_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance."""
    diff = a[:, None, :] - b[None, :, :]
    result = np.abs(diff).sum(axis=2)
    return np.asarray(result, dtype=np.float64)


def _compute_chebyshev_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Chebyshev distance."""
    diff = a[:, None, :] - b[None, :, :]
    result = np.abs(diff).max(axis=2)
    return np.asarray(result, dtype=np.float64)


def _compute_minkowski_distance(
    a: np.ndarray, b: np.ndarray, metric_params: dict[str, Any]
) -> np.ndarray:
    """Compute Minkowski distance."""
    p = metric_params.get("p", 2)
    if p <= 0:
        raise ValueError("Minkowski p must be > 0")
    diff = np.abs(a[:, None, :] - b[None, :, :]) ** p
    result = diff.sum(axis=2) ** (1.0 / p)
    return np.asarray(result, dtype=np.float64)


def _compute_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine distance."""
    # Cosine distance = 1 - cosine similarity
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    # Handle zero vectors to avoid division by zero
    a_safe = np.where(a_norm == 0.0, 1.0, a_norm)
    b_safe = np.where(b_norm == 0.0, 1.0, b_norm)
    a_unit = a / a_safe
    b_unit = b / b_safe
    sim = a_unit @ b_unit.T
    result = 1.0 - sim
    return np.asarray(result, dtype=np.float64)


def _compute_hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Hamming distance."""
    return np.sum(a[:, None, :] != b[None, :, :], axis=2)


def _compute_canberra_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Canberra distance."""
    diff = np.abs(a[:, None, :] - b[None, :, :])
    denom = np.abs(a[:, None, :]) + np.abs(b[None, :, :])
    # Avoid division by zero
    denom = np.where(denom == 0.0, 1.0, denom)
    result = np.sum(diff / denom, axis=2)
    return np.asarray(result, dtype=np.float64)


def _compute_bray_curtis_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Bray-Curtis distance."""
    diff = np.abs(a[:, None, :] - b[None, :, :])
    sum_diff = np.sum(diff, axis=2)
    sum_total = np.sum(a[:, None, :] + b[None, :, :], axis=2)
    # Avoid division by zero
    sum_total = np.where(sum_total == 0.0, 1.0, sum_total)
    result = sum_diff / sum_total
    return np.asarray(result, dtype=np.float64)


def _compute_jaccard_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Jaccard distance."""
    # Convert to binary if not already
    a_binary = (a > 0).astype(float)
    b_binary = (b > 0).astype(float)
    intersection = a_binary @ b_binary.T
    union = (
        np.sum(a_binary, axis=1, keepdims=True)
        + np.sum(b_binary, axis=1, keepdims=True).T
        - intersection
    )
    # Avoid division by zero
    union = np.where(union == 0.0, 1.0, union)
    result = 1.0 - (intersection / union)
    return np.asarray(result, dtype=np.float64)


def _compute_correlation_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute correlation distance."""
    # Center the data
    a_centered = a - np.mean(a, axis=1, keepdims=True)
    b_centered = b - np.mean(b, axis=1, keepdims=True)
    # Compute correlation
    numerator = a_centered @ b_centered.T
    a_std = np.sqrt(np.sum(a_centered**2, axis=1, keepdims=True))
    b_std = np.sqrt(np.sum(b_centered**2, axis=1, keepdims=True))
    denominator = a_std @ b_std.T
    # Avoid division by zero
    denominator = np.where(denominator == 0.0, 1.0, denominator)
    correlation = numerator / denominator
    result = 1.0 - correlation
    return np.asarray(result, dtype=np.float64)


def _compute_mahalanobis_distance(
    a: np.ndarray, b: np.ndarray, metric_params: dict[str, Any]
) -> np.ndarray:
    """Compute Mahalanobis distance."""
    cov_matrix = metric_params.get("cov_matrix")
    if cov_matrix is None:
        # Use sample covariance if not provided
        combined = np.vstack([a, b])
        cov_matrix = np.cov(combined.T)

    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if singular
        cov_inv = np.linalg.pinv(cov_matrix)

    diff = a[:, None, :] - b[None, :, :]  # Shape: (n, n, d)
    # Compute Mahalanobis distance for each pair
    distances = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            d = diff[i, j]
            distances[i, j] = np.sqrt(d.T @ cov_inv @ d)
    return distances


def _pairwise_distances_vectorized(
    a: np.ndarray,
    b: np.ndarray,
    metric: str,
    metric_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute pairwise distances using vectorized operations for supported metrics.

    Supported metrics: euclidean, manhattan, chebyshev, minkowski, cosine,
    hamming, canberra, bray_curtis, jaccard, correlation, mahalanobis.

    Uses dictionary dispatch pattern (Python's equivalent of switch statement)
    to avoid long if/elif chains and multiple return statements.
    """
    if metric_params is None:
        metric_params = {}

    # Dictionary dispatch: map metric names to computation functions
    # This is Python's equivalent of a switch statement
    # Metrics that don't require metric_params
    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "euclidean": _compute_euclidean_distance,
        "manhattan": _compute_manhattan_distance,
        "chebyshev": _compute_chebyshev_distance,
        "cosine": _compute_cosine_distance,
        "hamming": _compute_hamming_distance,
        "canberra": _compute_canberra_distance,
        "bray_curtis": _compute_bray_curtis_distance,
        "jaccard": _compute_jaccard_distance,
        "correlation": _compute_correlation_distance,
    }

    # Metrics that require metric_params
    metric_functions_with_params: dict[
        str, Callable[[np.ndarray, np.ndarray, dict[str, Any]], np.ndarray]
    ] = {
        "minkowski": _compute_minkowski_distance,
        "mahalanobis": _compute_mahalanobis_distance,
    }

    # Single dispatch lookup and return statement
    if metric in metric_functions:
        return metric_functions[metric](a, b)
    if metric in metric_functions_with_params:
        return metric_functions_with_params[metric](a, b, metric_params)

    # Unsupported metric
    raise ValueError(
        "Unsupported metric. Choose one of: 'euclidean', 'manhattan', "
        "'chebyshev', 'minkowski', 'cosine', 'hamming', 'canberra', "
        "'bray_curtis', 'jaccard', 'correlation', 'mahalanobis'."
    )
