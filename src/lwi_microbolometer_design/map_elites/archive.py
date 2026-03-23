"""MAP-Elites archive: feature extraction, binning, and initialisation."""

from typing import Any

import numpy as np


def extract_features(chromosome: np.ndarray) -> tuple[float, float]:
    """Extract feature descriptors: (smallest_mu, second_smallest_mu).

    Parameters
    ----------
    chromosome : np.ndarray
        Chromosome where even-indexed genes are mu (wavelength) parameters.

    Returns
    -------
    tuple[float, float]
        (smallest_mu, second_smallest_mu) sorted ascending.
    """
    mus = [float(chromosome[i]) for i in range(0, len(chromosome), 2)]
    sorted_mus = sorted(mus)
    return sorted_mus[0], sorted_mus[1]


def bin_coordinates(
    mu_1: float,
    mu_2: float,
    grid_resolution: int,
    mu_range: tuple[float, float],
) -> tuple[int, int]:
    """Convert feature coordinates to discrete bin indices.

    Parameters
    ----------
    mu_1, mu_2 : float
        Feature values (wavelength positions).
    grid_resolution : int
        Number of bins per dimension.
    mu_range : tuple[float, float]
        (min, max) range for mu values.

    Returns
    -------
    tuple[int, int]
        (x_bin, y_bin) indices clamped to [0, grid_resolution - 1].
    """
    mu_min, mu_max = mu_range

    mu_1_clamped = max(mu_min, min(mu_max, mu_1))
    mu_2_clamped = max(mu_min, min(mu_max, mu_2))

    x_bin = int((mu_1_clamped - mu_min) / (mu_max - mu_min) * grid_resolution)
    y_bin = int((mu_2_clamped - mu_min) / (mu_max - mu_min) * grid_resolution)

    x_bin = max(0, min(grid_resolution - 1, x_bin))
    y_bin = max(0, min(grid_resolution - 1, y_bin))

    return x_bin, y_bin


def reachable_cell_count(grid_resolution: int) -> int:
    """Return the number of reachable archive cells for the current descriptor.

    The archive descriptor is ``(smallest_mu, second_smallest_mu)``, so the
    corresponding bin coordinates always satisfy ``x_bin <= y_bin``.  Only the
    upper-triangular part of the nominal square grid can therefore be filled.
    """
    if grid_resolution <= 0:
        raise ValueError("grid_resolution must be positive.")
    return grid_resolution * (grid_resolution + 1) // 2


def archive_coverage_pct(archive_size: int, grid_resolution: int) -> float:
    """Compute archive coverage relative to reachable cells."""
    return archive_size / reachable_cell_count(grid_resolution) * 100.0


def initialize_archive(
    num_initial: int,
    gene_space: list[dict[str, float]],
    fitness_func: Any,
    grid_resolution: int,
    mu_range: tuple[float, float],
    random_seed: int = 42,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Seed the archive with uniformly random solutions.

    Parameters
    ----------
    num_initial : int
        Number of random solutions to generate.
    gene_space : list
        Gene space bounds, each element ``{"low": float, "high": float}``.
    fitness_func : callable
        ``fitness_func(ga_instance, chromosome, solution_idx) -> float``.
    grid_resolution : int
        Bins per dimension.
    mu_range : tuple[float, float]
        Range for mu values.
    random_seed : int
        Random seed.

    Returns
    -------
    dict
        Archive mapping ``(x_bin, y_bin) -> {"chromosome", "fitness", "mu_1", "mu_2"}``.
    """
    np.random.seed(random_seed)
    archive: dict[tuple[int, int], dict[str, Any]] = {}

    print(f"Initializing archive with {num_initial} random solutions...")

    for i in range(num_initial):
        chromosome = np.array([np.random.uniform(g["low"], g["high"]) for g in gene_space])

        fitness = fitness_func(None, chromosome, 0)
        mu_1, mu_2 = extract_features(chromosome)
        x_bin, y_bin = bin_coordinates(mu_1, mu_2, grid_resolution, mu_range)

        key = (x_bin, y_bin)
        if key not in archive or fitness > archive[key]["fitness"]:
            archive[key] = {
                "chromosome": chromosome.copy(),
                "fitness": float(fitness),
                "mu_1": mu_1,
                "mu_2": mu_2,
            }

        if (i + 1) % 100 == 0:
            print(f"  Initialized {i + 1}/{num_initial} solutions, archive size: {len(archive)}")

    total_reachable = reachable_cell_count(grid_resolution)
    print(f"Archive initialized: {len(archive)}/{total_reachable} reachable cells filled")
    return archive
