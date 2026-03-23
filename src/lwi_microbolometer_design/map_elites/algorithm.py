"""MAP-Elites core algorithm: mutation and main loop."""

from typing import Any

import numpy as np

from .archive import (
    archive_coverage_pct,
    bin_coordinates,
    extract_features,
    initialize_archive,
    reachable_cell_count,
)


def mutate_chromosome(
    chromosome: np.ndarray,
    gene_space: list[dict[str, float]],
    mutation_probability: float = 0.1,
) -> np.ndarray:
    """Mutate a chromosome using Gaussian perturbation.

    Wavelength parameters (mu, even indices) use ``sigma=0.75`` to allow
    movement across archive bins.  Width parameters (sigma, odd indices)
    use ``sigma=0.2`` for smaller perturbation.

    Parameters
    ----------
    chromosome : np.ndarray
        Parent chromosome.
    gene_space : list
        Gene space bounds.
    mutation_probability : float
        Per-gene probability of mutation.

    Returns
    -------
    np.ndarray
        Mutated child chromosome.
    """
    child = chromosome.copy()

    for i, gene_bounds in enumerate(gene_space):
        if np.random.random() < mutation_probability:
            low = gene_bounds["low"]
            high = gene_bounds["high"]
            range_size = high - low

            mutation_sigma = 0.75 if i % 2 == 0 else 0.2
            mutation = np.random.normal(0, mutation_sigma * range_size)
            child[i] = max(low, min(high, child[i] + mutation))

    return child


def run_map_elites(
    fitness_func: Any,
    gene_space: list[dict[str, float]],
    grid_resolution: int = 20,
    mu_range: tuple[float, float] = (4.0, 20.0),
    num_initial: int = 1000,
    num_iterations: int = 200000,
    mutation_probability: float = 0.1,
    random_seed: int = 42,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Run the MAP-Elites quality-diversity algorithm.

    Parameters
    ----------
    fitness_func : callable
        ``fitness_func(ga_instance, chromosome, solution_idx) -> float``.
    gene_space : list
        Gene space bounds.
    grid_resolution : int
        Bins per dimension.
    mu_range : tuple[float, float]
        Range for mu values.
    num_initial : int
        Number of random solutions to seed the archive.
    num_iterations : int
        Main-loop iterations.
    mutation_probability : float
        Per-gene mutation probability.
    random_seed : int
        Random seed.

    Returns
    -------
    dict
        Final archive mapping ``(x_bin, y_bin) -> best individual dict``.
    """
    np.random.seed(random_seed)

    archive = initialize_archive(
        num_initial=num_initial,
        gene_space=gene_space,
        fitness_func=fitness_func,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        random_seed=random_seed,
    )

    archive_list = list(archive.values())

    print(f"\nRunning MAP-Elites for {num_iterations} iterations...")

    for iteration in range(num_iterations):
        if len(archive_list) == 0:
            parent_chromosome = np.array(
                [np.random.uniform(g["low"], g["high"]) for g in gene_space]
            )
        else:
            parent = archive_list[np.random.randint(len(archive_list))]
            parent_chromosome = parent["chromosome"]

        child_chromosome = mutate_chromosome(
            parent_chromosome,
            gene_space,
            mutation_probability,
        )

        child_fitness = fitness_func(None, child_chromosome, 0)
        mu_1, mu_2 = extract_features(child_chromosome)
        x_bin, y_bin = bin_coordinates(mu_1, mu_2, grid_resolution, mu_range)

        key = (x_bin, y_bin)
        if key not in archive or child_fitness > archive[key]["fitness"]:
            archive[key] = {
                "chromosome": child_chromosome.copy(),
                "fitness": float(child_fitness),
                "mu_1": mu_1,
                "mu_2": mu_2,
            }
            archive_list = list(archive.values())

        if (iteration + 1) % 5000 == 0:
            filled_cells = len(archive)
            total_cells = reachable_cell_count(grid_resolution)
            coverage = archive_coverage_pct(filled_cells, grid_resolution)
            best_fitness = max(ind["fitness"] for ind in archive.values())
            print(
                f"  Iter: {iteration + 1} | Archive: {filled_cells}/{total_cells} "
                f"({coverage:.1f}%) | Best Fitness: {best_fitness:.2f}"
            )

    total_cells = reachable_cell_count(grid_resolution)
    print(f"\nMAP-Elites complete: {len(archive)}/{total_cells} reachable cells filled")
    return archive
