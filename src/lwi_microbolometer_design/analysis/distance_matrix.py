"""Distance matrix computation utilities for post-hoc analysis.

Compute pairwise distance matrices between collections of objects or arrays
using arbitrary distance functions. Used for diversity analysis, clustering,
and visualization.

Supports optimal pairing mode for grouped parameters (e.g., [(mu1, sigma1), ...])
where group ordering shouldn't affect distance calculations.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

# Internal import for optimal pairing (hidden from users)
from lwi_microbolometer_design.analysis.optimal_pairing_distance import (
    calculate_optimal_pairing_distance,
)


def _extract_data_from_items(
    items: list[Any] | list[np.ndarray] | np.ndarray,
    attribute: str | None = None,
    axis: int | None = None,
) -> list[Any]:
    """Extract data list from items (handles numpy arrays and lists with attributes)."""
    # Handle numpy array input (any dimensionality)
    if isinstance(items, np.ndarray):
        if axis is None:
            axis = 0  # Default: compare along first dimension

        # Validate axis is within array dimensions
        if axis >= items.ndim:
            raise ValueError(f"axis={axis} is out of bounds for array with {items.ndim} dimensions")

        # Extract items along the specified axis
        # Use advanced indexing instead of np.take to avoid mypy stub issues
        num_items = items.shape[axis]
        if axis == 0:
            data = [items[idx] for idx in range(num_items)]
        elif axis == 1:
            data = [items[:, idx] for idx in range(num_items)]
        else:
            # For higher dimensions, use np.moveaxis to bring target axis to front, then index
            # This avoids mypy issues with tuple indexing containing slices
            items_moved = np.moveaxis(items, axis, 0)
            data = [items_moved[idx] for idx in range(num_items)]

        # Attribute doesn't make sense for arrays
        if attribute is not None:
            raise ValueError("attribute parameter cannot be used with numpy array input")
        return data

    # Handle list input
    if axis is not None:
        raise ValueError("axis parameter can only be used with numpy array input")

    # Extract data from objects if needed, otherwise use items directly
    if attribute is not None:
        return [getattr(item, attribute) for item in items]

    return items


def _validate_optimal_pairing_params(
    use_optimal_pairing: bool,
    params_per_group: int | None,
    metric: str | None,
    distance_func: Callable | None,
) -> None:
    """Validate parameters for optimal pairing mode."""
    if not use_optimal_pairing:
        return

    if params_per_group is None:
        raise ValueError("params_per_group must be provided when use_optimal_pairing=True")
    if metric is None and distance_func is None:
        raise ValueError(
            "Either metric or distance_func must be provided when use_optimal_pairing=True"
        )


def _compute_optimal_pairing_distances(
    data: list[Any],
    params_per_group: int,
    metric: str | None,
    metric_params: dict | None,
    distance_func: Callable | None,
) -> np.ndarray:
    """Compute distance matrix using optimal pairing mode."""
    n = len(data)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Reshape into groups (ensure they're arrays)
            item_i = data[i]
            item_j = data[j]
            # Convert to arrays with explicit type checking
            array_i = item_i if isinstance(item_i, np.ndarray) else np.array(item_i)
            array_j = item_j if isinstance(item_j, np.ndarray) else np.array(item_j)

            # Validate array length is divisible by params_per_group
            if len(array_i) % params_per_group != 0 or len(array_j) % params_per_group != 0:
                raise ValueError(
                    f"Array lengths ({len(array_i)}, {len(array_j)}) must be "
                    f"divisible by params_per_group ({params_per_group})"
                )

            groups_i = array_i.reshape(-1, params_per_group)
            groups_j = array_j.reshape(-1, params_per_group)

            # Convert to list of tuples for optimal pairing
            items_i = [tuple(group) for group in groups_i]
            items_j = [tuple(group) for group in groups_j]

            # Compute optimal pairing distance
            if metric is not None:
                dist = calculate_optimal_pairing_distance(
                    items_i, items_j, metric=metric, metric_params=metric_params
                )
            else:
                # Use custom distance function
                if distance_func is None:
                    raise ValueError("distance_func must be provided when metric is None")

                def pairwise_distance_func(item_a: Any, item_b: Any) -> float:
                    # Convert to arrays with explicit type checking
                    arr_a = item_a if isinstance(item_a, np.ndarray) else np.array(item_a)
                    arr_b = item_b if isinstance(item_b, np.ndarray) else np.array(item_b)
                    return float(distance_func(arr_a, arr_b))

                dist = calculate_optimal_pairing_distance(
                    items_i, items_j, distance_func=pairwise_distance_func
                )

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def _compute_standard_distances(data: list[Any], distance_func: Callable) -> np.ndarray:
    """Compute distance matrix using standard distance function."""
    n = len(data)
    dist_matrix = np.zeros((n, n))

    # Compute upper triangle only (diagonal stays 0)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_func(data[i], data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def compute_distance_matrix(
    items: list[Any] | list[np.ndarray] | np.ndarray,
    distance_func: Callable | None = None,
    attribute: str | None = None,
    axis: int | None = None,
    use_optimal_pairing: bool = False,
    params_per_group: int | None = None,
    metric: str | None = None,
    metric_params: dict | None = None,
) -> np.ndarray:
    """Compute symmetric pairwise distance matrix for items.

    Unified function that handles:
    - Lists of objects or arrays
    - Numpy arrays of any dimensionality (with axis parameter to specify
      which dimension represents items)

    Supports optimal pairing mode for grouped parameters where group ordering
    shouldn't affect distance calculations.

    Parameters
    ----------
    items : list or np.ndarray
        Items to compute distances between. Can be:
        - List of objects/arrays: Each element is an item to compare
        - Numpy array: Use `axis` parameter to specify which dimension represents items
    distance_func : callable, optional
        Distance function f(item1, item2) -> float.
        Required if use_optimal_pairing=False and metric=None.
        Ignored if use_optimal_pairing=True or metric is provided.
    attribute : str, optional
        If provided, extracts this attribute from objects before distance computation.
        If None, treats items as arrays directly (default for arrays).
        Common values: "genes" (for GA solutions), "parameters", etc.
        Ignored if items is a numpy array.
    axis : int, optional
        For numpy array input, specifies which axis represents items to compare:
        - axis=0 (default): Compare along first dimension (e.g., rows for 2D)
        - axis=1: Compare along second dimension (e.g., columns for 2D)
        - axis=N: Compare along Nth dimension (requires array.ndim > N)
        Ignored if items is a list.
    use_optimal_pairing : bool, default=False
        If True, uses optimal pairing (Hungarian algorithm) to match groups
        within each array before computing distances. Requires params_per_group.
        Each pair comparison gets its own optimal matching (pairwise computation).
    params_per_group : int, optional
        Number of parameters per group (e.g., 2 for (mu, sigma)).
        Required if use_optimal_pairing=True.
    metric : str, optional
        Scipy distance metric name (e.g., 'euclidean', 'manhattan').
        Alternative to distance_func. Used with optimal pairing when provided.
    metric_params : dict, optional
        Parameters for metric (e.g., {"p": 3} for Minkowski).

    Returns
    -------
    np.ndarray
        Symmetric distance matrix (n x n), diagonal is zero

    Examples
    --------
    >>> # Standard distance matrix with custom function (list input)
    >>> parameter_sets = [params1, params2, params3]
    >>> dist_matrix = compute_distance_matrix(parameter_sets, euclidean_dist)

    >>> # Array input - compare along first dimension (rows for 2D)
    >>> sensor_params = np.array([[1, 2], [3, 4], [5, 6]])  # 3 sensors, 2 params each
    >>> dist_matrix = compute_distance_matrix(sensor_params, euclidean_dist, axis=0)

    >>> # Array input - compare along second dimension (columns for 2D)
    >>> sensor_outputs = np.array([[1, 2, 3], [4, 5, 6]])  # 2 basis funcs, 3 substances
    >>> dist_matrix = compute_distance_matrix(
    ...     sensor_outputs,
    ...     spectral_angle_mapper,
    ...     axis=1,  # Compare along second dimension (substances)
    ... )

    >>> # With optimal pairing (grouped parameters)
    >>> dist_matrix = compute_distance_matrix(
    ...     parameter_sets, metric='euclidean', use_optimal_pairing=True, params_per_group=2
    ... )

    >>> # With objects (e.g., GA solutions with .genes attribute)
    >>> solutions = [sol1, sol2, sol3]
    >>> dist_matrix = compute_distance_matrix(solutions, euclidean_dist, attribute='genes')

    Notes
    -----
    When use_optimal_pairing=True:
    - Arrays are reshaped into groups based on params_per_group
    - Each pair comparison uses optimal pairing (Hungarian algorithm)
    - This is slower than standard vectorized computation but necessary
      because optimal matching is pair-specific and cannot be vectorized
    """
    # Extract data from items (handles numpy arrays, lists, and object attributes)
    data = _extract_data_from_items(items, attribute=attribute, axis=axis)

    # Validate optimal pairing parameters
    _validate_optimal_pairing_params(use_optimal_pairing, params_per_group, metric, distance_func)

    # Dispatch to appropriate computation mode
    if use_optimal_pairing:
        # After validation, params_per_group is guaranteed to be int
        if params_per_group is None:
            raise ValueError("params_per_group must be provided when use_optimal_pairing=True")
        return _compute_optimal_pairing_distances(
            data, params_per_group, metric, metric_params, distance_func
        )

    # Standard mode: regular distance computation
    if distance_func is None:
        raise ValueError("distance_func must be provided when use_optimal_pairing=False")

    return _compute_standard_distances(data, distance_func)
