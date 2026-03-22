"""
Unified diversity and distance calculation utilities for GA populations.

This module provides a single source of truth for computing population diversity
and distance matrices, supporting both standard metrics (compatible with regular GA)
and optimal pairing (for grouped parameters).

**Architecture Note:**
- This module uses `analysis.distance_matrix` with `use_optimal_pairing=True` option
- Optimal pairing is hidden as an implementation detail - users don't need to know about it
- This module provides GA-specific integration that maintains consistency with GA niching behavior

All diversity calculations in the GA package should use functions from this module
to ensure consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

from lwi_microbolometer_design.analysis import compute_distance_matrix

if TYPE_CHECKING:
    from lwi_microbolometer_design.ga import NichingConfig

# Minimum population size for diversity calculations
MIN_POPULATION_FOR_DIVERSITY = 2


def compute_population_distance_matrix(
    population: np.ndarray,
    niching_config: NichingConfig | None = None,
) -> np.ndarray:
    """
    Compute pairwise distance matrix for a population.

    This is the unified function for computing distance matrices. It respects
    the GA's niching configuration to ensure consistency across all diversity
    calculations.

    **Routing Logic:**
    - If optimal pairing is enabled: delegates to `_compute_optimal_pairing_distance_matrix()`
    - Otherwise: uses standard scipy distance metrics (euclidean, manhattan, etc.)

    Parameters
    ----------
    population : np.ndarray
        Population array of shape (n_individuals, n_genes)
    niching_config : NichingConfig | None, optional
        Niching configuration from AdvancedGA. If provided and optimal pairing
        is enabled, uses optimal pairing distance. Otherwise uses standard metric.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape (n_individuals, n_individuals)

    Notes
    -----
    - If `niching_config.use_optimal_pairing=True`, uses optimal pairing mode
      (hidden implementation detail via distance_matrix.py)
    - Otherwise uses standard scipy distance metric from `niching_config.distance_metric`
    - Defaults to Euclidean if no config provided
    - This matches the logic used by AdvancedGA for niching calculations
    """
    if len(population) < MIN_POPULATION_FOR_DIVERSITY:
        return np.zeros((len(population), len(population)))

    # Use optimal pairing if configured (matches GA behavior)
    if niching_config and niching_config.enabled and niching_config.use_optimal_pairing:
        # Use distance_matrix.py with optimal pairing option (hides implementation)

        # Convert population to list of arrays
        items = [population[i] for i in range(len(population))]

        # Use distance_matrix with optimal pairing option
        return compute_distance_matrix(
            items,
            metric=niching_config.optimal_pairing_metric or "euclidean",
            use_optimal_pairing=True,
            params_per_group=niching_config.params_per_group,
        )

    # Use standard distance metric
    metric = (
        niching_config.distance_metric
        if niching_config and niching_config.distance_metric
        else "euclidean"
    )
    result = cdist(population, population, metric=metric)
    return np.asarray(result, dtype=np.float64)


def calculate_population_diversity(
    population: np.ndarray,
    niching_config: NichingConfig | None = None,
) -> float:
    """
    Calculate mean pairwise diversity for a population.

    This is the unified function for computing population diversity as a scalar.
    Uses the same distance calculation as `compute_population_distance_matrix()`.

    Parameters
    ----------
    population : np.ndarray
        Population array of shape (n_individuals, n_genes)
    niching_config : NichingConfig | None, optional
        Niching configuration from AdvancedGA. Determines distance calculation method.

    Returns
    -------
    float
        Mean pairwise distance between all individuals

    Examples
    --------
    Basic usage (no config - uses Euclidean):
        >>> diversity = calculate_population_diversity(population)

    With GA's niching config (respects optimal pairing settings):
        >>> diversity = calculate_population_diversity(population, ga.niching_config)
    """
    if len(population) < MIN_POPULATION_FOR_DIVERSITY:
        return 0.0

    # Compute distance matrix using unified function
    distance_matrix = compute_population_distance_matrix(population, niching_config)

    # Extract upper triangle (excluding diagonal) and compute mean
    n = len(population)
    upper_triangle_indices = np.triu_indices(n, k=1)
    distances = distance_matrix[upper_triangle_indices]

    return float(np.mean(distances)) if len(distances) > 0 else 0.0


# NOTE: _compute_optimal_pairing_distance_matrix() removed
# Now uses distance_matrix.py with use_optimal_pairing=True option
# This hides optimal pairing as an implementation detail
