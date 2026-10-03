"""MAP-Elites archive: feature extraction, binning, and initialisation."""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm


def _validate_archive_inputs(
    num_initial: int,
    gene_space: list[dict[str, float]],
    grid_resolution: int,
    mu_range: tuple[float, float],
) -> None:
    """Validate the public archive geometry, seed count and Gaussian gene bounds."""
    if not isinstance(num_initial, (int, np.integer)) or num_initial < 0:
        raise ValueError("num_initial must be a nonnegative integer.")
    if not isinstance(grid_resolution, (int, np.integer)) or grid_resolution < 1:
        raise ValueError("grid_resolution must be a positive integer.")
    if len(mu_range) != 2 or not np.all(np.isfinite(mu_range)) or mu_range[0] >= mu_range[1]:
        raise ValueError("mu_range must contain finite increasing bounds.")
    if len(gene_space) < 4 or len(gene_space) % 2:
        raise ValueError("gene_space must contain at least two complete (mu, sigma) pairs.")
    for bounds in gene_space:
        low, high = bounds["low"], bounds["high"]
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError("Each gene must have finite bounds with low <= high.")


def _evaluate_fitness(fitness_func: Any, chromosome: np.ndarray) -> float:
    """Evaluate a candidate without admitting nonfinite scores to an archive."""
    fitness = float(fitness_func(None, chromosome, 0))
    if not np.isfinite(fitness):
        raise ValueError("fitness_func must return a finite scalar.")
    return fitness


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
    *,
    show_progress: bool = False,
    progress_desc: str = "MAP-Elites archive init",
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
    show_progress : bool, optional
        If True, show a tqdm bar (ETA, rate) while seeding.
    progress_desc : str, optional
        Bar description when *show_progress* is True.

    Returns
    -------
    dict
        Archive mapping ``(x_bin, y_bin) -> {"chromosome", "fitness", "mu_1", "mu_2"}``.
    """
    _validate_archive_inputs(num_initial, gene_space, grid_resolution, mu_range)
    np.random.seed(random_seed)
    archive: dict[tuple[int, int], dict[str, Any]] = {}

    pbar: tqdm | None = None
    if show_progress:
        pbar = tqdm(
            total=num_initial,
            desc=progress_desc,
            unit="seed",
            dynamic_ncols=True,
            mininterval=0.2,
        )
    else:
        print(f"Initializing archive with {num_initial} random solutions...")

    for i in range(num_initial):
        chromosome = np.array([np.random.uniform(g["low"], g["high"]) for g in gene_space])

        fitness = _evaluate_fitness(fitness_func, chromosome)
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

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(archive=len(archive), refresh=False)
        elif (i + 1) % 100 == 0:
            print(f"  Initialized {i + 1}/{num_initial} solutions, archive size: {len(archive)}")

    if pbar is not None:
        pbar.close()

    total_reachable = reachable_cell_count(grid_resolution)
    print(f"Archive initialized: {len(archive)}/{total_reachable} reachable cells filled")
    return archive
