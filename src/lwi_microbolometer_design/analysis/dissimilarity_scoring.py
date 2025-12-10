"""Dissimilarity scoring functions for evaluating discrimination performance.

This module provides scoring functions that evaluate how well a sensor (or any
distance-based system) can discriminate between different substances or items.

**Common Parameters (all functions):**
- `items`: Raw input items to compare (e.g., sensor_outputs shape (m, n)).
  Required if `distance_matrix` is None. Each column (axis=1) or row (axis=0)
  represents an item to compare.
- `distance_matrix`: Pre-computed pairwise distance matrix, shape (n, n).
  If provided, used directly (efficient for multiple scores). If None, computed from items.
- `distance_func`: Distance function f(item1, item2) -> float.
  Required if `distance_matrix` is None. Default: spectral_angle_mapper
- `axis`: For 2D array input, axis representing items (0=rows, 1=columns, default: 1).
  Ignored if `distance_matrix` is provided.
- `**kwargs`: Additional arguments passed to compute_distance_matrix if items provided.

**Usage Patterns:**
>>> # Convenient: pass raw inputs
>>> score = min_based_dissimilarity_score(sensor_outputs, distance_func=spectral_angle_mapper)
>>>
>>> # Efficient: compute distance matrix once, reuse for multiple scores
>>> distance_matrix = compute_distance_matrix(sensor_outputs, spectral_angle_mapper)
>>> score1 = min_based_dissimilarity_score(distance_matrix=distance_matrix)
>>> score2 = group_based_dissimilarity_score(groups=groups, distance_matrix=distance_matrix)
"""

from collections.abc import Callable

import numpy as np

from lwi_microbolometer_design.analysis.distance_matrix import compute_distance_matrix
from lwi_microbolometer_design.analysis.distance_metrics import spectral_angle_mapper


def _ensure_distance_matrix(
    items: np.ndarray | list | None,
    distance_matrix: np.ndarray | None,
    distance_func: Callable | None,
    axis: int = 1,
    **kwargs,
) -> np.ndarray:
    """Ensure distance matrix is available, computing it if necessary.

    Helper function to reduce duplication across scoring functions.
    If distance_matrix is provided, returns it directly. Otherwise,
    computes it from items using the provided distance function.

    Parameters
    ----------
    items : np.ndarray | list, optional
        Raw input items to compare. Required if distance_matrix is None.
    distance_matrix : np.ndarray, optional
        Pre-computed pairwise distance matrix. If provided, returned directly.
    distance_func : Callable, optional
        Distance function. Required if distance_matrix is None.
    axis : int, optional
        For 2D array input, axis representing items (default: 1).
    **kwargs
        Additional arguments passed to compute_distance_matrix.

    Returns
    -------
    np.ndarray
        Distance matrix, shape (n, n)

    Raises
    ------
    ValueError
        If neither items nor distance_matrix is provided, or if items provided
        without distance_func.
    """
    if distance_matrix is not None:
        return distance_matrix

    if items is None:
        raise ValueError('Must provide either items or distance_matrix')

    if distance_func is None:
        distance_func = spectral_angle_mapper

    return compute_distance_matrix(items, distance_func=distance_func, axis=axis, **kwargs)


def min_based_dissimilarity_score(
    items: np.ndarray | list | None = None,
    distance_matrix: np.ndarray | None = None,
    distance_func: Callable | None = None,
    axis: int = 1,
    **kwargs,
) -> float:
    """Compute dissimilarity score based on minimum off-diagonal distance.

    Returns the minimum separation between any pair of substances. A conservative
    metric that ensures all substances can be distinguished with at least this distance.

    Parameters
    ----------
    items, distance_matrix, distance_func, axis, **kwargs
        See module docstring for common parameters.

    Returns
    -------
    float
        Minimum off-diagonal distance (worst-case separation).

    Notes
    -----
    Useful when misclassification of any substance pair is equally problematic.
    """
    distance_matrix = _ensure_distance_matrix(items, distance_matrix, distance_func, axis, **kwargs)

    n = distance_matrix.shape[0]
    # Extract off-diagonal values (exclude diagonal elements)
    # Use np.eye without dtype parameter, then convert to bool for indexing
    eye_mask = np.eye(n).astype(bool)
    off_diag_values = distance_matrix[~eye_mask]

    # Find the minimum distance among all pairs
    min_distance = float(np.min(off_diag_values))

    return min_distance


def mean_min_based_dissimilarity_score(
    items: np.ndarray | list | None = None,
    distance_matrix: np.ndarray | None = None,
    distance_func: Callable | None = None,
    axis: int = 1,
    alpha: float = 1.0,
    **kwargs,
) -> float:
    """Compute dissimilarity score combining mean and minimum with adjustable penalty.

    Combines average separation (mean) with worst-case separation (minimum).
    Formula: mean_distance * (min_distance ** alpha)

    Parameters
    ----------
    alpha : float, optional
        Exponent for penalizing minimum distance (default: 1.0).
        - alpha = 0: Only mean distance
        - alpha = 1: Equal weighting
        - alpha > 1: Heavy penalty on poor minimum
    items, distance_matrix, distance_func, axis, **kwargs
        See module docstring for common parameters.

    Returns
    -------
    float
        Mean-min based dissimilarity score.
    """
    distance_matrix = _ensure_distance_matrix(items, distance_matrix, distance_func, axis, **kwargs)

    n = distance_matrix.shape[0]
    # Extract off-diagonal values
    # Use np.eye without dtype parameter, then convert to bool for indexing
    eye_mask = np.eye(n).astype(bool)
    off_diag_values = distance_matrix[~eye_mask]

    # Calculate mean of off-diagonal values
    mean_distance = np.mean(off_diag_values)

    # Calculate minimum off-diagonal value
    min_distance = np.min(off_diag_values)

    # Weighted combined metric: mean * (min ^ alpha)
    score = float(mean_distance * (min_distance**alpha))
    return score


def group_based_dissimilarity_score(
    groups: list[list[int]],
    items: np.ndarray | list | None = None,
    distance_matrix: np.ndarray | None = None,
    distance_func: Callable | None = None,
    axis: int = 1,
    min_groups: int = 2,
    **kwargs,
) -> float:
    """Compute dissimilarity score based on mean inter-group distance.

    Evaluates separation between groups rather than individual substances.
    Useful when substances can be categorized into meaningful groups
    (e.g., hazardous vs. non-hazardous materials).

    Parameters
    ----------
    groups : List[List[int]]
        List of groups, where each group is a list of substance indices.
        Must have at least `min_groups` groups.
    items, distance_matrix, distance_func, axis, **kwargs
        See module docstring for common parameters.
    min_groups : int, optional
        Minimum number of groups required to compute dissimilarity (default: 2).

    Returns
    -------
    float
        Mean inter-group distance.

    Raises
    ------
    ValueError
        If fewer than `min_groups` groups are provided.
    """
    distance_matrix = _ensure_distance_matrix(items, distance_matrix, distance_func, axis, **kwargs)

    num_groups = len(groups)
    if num_groups < min_groups:
        raise ValueError(f'At least {min_groups} groups are required to compute dissimilarity.')

    # Calculate inter-group distances
    inter_group_distances = []
    for g1 in range(num_groups):
        for g2 in range(g1 + 1, num_groups):  # Only consider distinct group pairs
            for i in groups[g1]:
                for j in groups[g2]:
                    inter_group_distances.append(distance_matrix[i, j])

    # Calculate mean inter-group distance
    # Convert list to array for np.mean
    distances_array = np.array(inter_group_distances, dtype=np.float64)
    mean_inter_group_distance = float(np.mean(distances_array))

    return mean_inter_group_distance


def weighted_mean_min_dissimilarity_score(
    items: np.ndarray | list | None = None,
    distance_matrix: np.ndarray | None = None,
    distance_func: Callable | None = None,
    axis: int = 1,
    beta: float = 0.5,
    **kwargs,
) -> float:
    """Combine mean and minimum distances with a linear weighting factor.

    Formula: beta * mean_distance + (1 - beta) * min_distance

    Parameters
    ----------
    beta : float, optional
        Weight for mean distance (0 <= beta <= 1, default: 0.5).
        - beta = 0: Only minimum (conservative)
        - beta = 0.5: Equal weighting
        - beta = 1: Only mean (optimistic)
    items, distance_matrix, distance_func, axis, **kwargs
        See module docstring for common parameters.

    Returns
    -------
    float
        Weighted dissimilarity score.
    """
    distance_matrix = _ensure_distance_matrix(items, distance_matrix, distance_func, axis, **kwargs)

    n = distance_matrix.shape[0]
    # Use np.eye without dtype parameter, then convert to bool for indexing
    eye_mask = np.eye(n).astype(bool)
    off_diag_values = distance_matrix[~eye_mask]
    mean_distance = float(np.mean(off_diag_values))
    min_distance = float(np.min(off_diag_values))
    score = float(beta * mean_distance + (1 - beta) * min_distance)
    return score
