"""Post-hoc polishing strategies for MAP-Elites elites.

Two strategies are provided:

* **Hill Climbing** — greedy local search with optional adaptive iteration budgets.
* **CMA-ES** — learns the local fitness landscape shape and searches along
  the most promising directions.
"""

from typing import Any

import numpy as np

from .normalization import UnitCubeScaler


# ---------------------------------------------------------------------------
# Hill-climbing polish
# ---------------------------------------------------------------------------


def polish_single_elite_hc(
    elite_id: int,
    chromosome: np.ndarray,
    initial_fitness: float,
    fitness_func: Any,
    gene_space: list[dict[str, float]],
    num_iterations: int = 4000,
    mutation_sigma: float = 0.05,
    mutation_probability: float = 0.15,
    adaptive_iterations: bool = True,
) -> dict[str, Any]:
    """Polish a single elite using greedy hill climbing.

    Parameters
    ----------
    elite_id : int
        Identifier for this elite.
    chromosome : np.ndarray
        Starting chromosome.
    initial_fitness : float
        Starting fitness.
    fitness_func : callable
        ``fitness_func(ga_instance, chromosome, solution_idx) -> float``.
    gene_space : list
        Gene space bounds.
    num_iterations : int
        Base iteration budget.
    mutation_sigma : float
        Gaussian sigma as a fraction of the gene range.
    mutation_probability : float
        Per-gene mutation probability.
    adaptive_iterations : bool
        Scale budget based on proximity to a target fitness.

    Returns
    -------
    dict
        ``{"elite_id", "initial_chromosome", "initial_fitness",
        "polished_chromosome", "polished_fitness", "fitness_gain"}``.
    """
    if adaptive_iterations:
        target_fitness = 59.0
        fitness_gap = target_fitness - initial_fitness
        if fitness_gap < 3.0:
            actual_iterations = int(num_iterations * 4.0)
        elif fitness_gap < 5.0:
            actual_iterations = int(num_iterations * 3.0)
        elif fitness_gap < 8.0:
            actual_iterations = int(num_iterations * 2.5)
        else:
            actual_iterations = int(num_iterations * 2.0)
    else:
        actual_iterations = num_iterations

    current_chromosome = chromosome.copy()
    current_fitness = initial_fitness
    best_chromosome = current_chromosome.copy()
    best_fitness = current_fitness

    for _iteration in range(actual_iterations):
        candidate = current_chromosome.copy()

        for i, gene_bounds in enumerate(gene_space):
            if np.random.random() < mutation_probability:
                low = gene_bounds["low"]
                high = gene_bounds["high"]
                range_size = high - low

                mutation = np.random.normal(0, mutation_sigma * range_size)
                candidate[i] = max(low, min(high, candidate[i] + mutation))

        candidate_fitness = fitness_func(None, candidate, 0)

        if candidate_fitness > current_fitness:
            current_chromosome = candidate.copy()
            current_fitness = candidate_fitness
            if candidate_fitness > best_fitness:
                best_chromosome = candidate.copy()
                best_fitness = candidate_fitness

    return {
        "elite_id": elite_id,
        "initial_chromosome": chromosome.copy(),
        "initial_fitness": initial_fitness,
        "polished_chromosome": best_chromosome,
        "polished_fitness": best_fitness,
        "fitness_gain": best_fitness - initial_fitness,
    }


# ---------------------------------------------------------------------------
# CMA-ES polish
# ---------------------------------------------------------------------------


def polish_single_elite_cma(
    elite_id: int,
    chromosome: np.ndarray,
    initial_fitness: float,
    fitness_func: Any,
    gene_space: list[dict[str, float]],
    max_fevals: int = 3000,
    initial_sigma: float = 0.15,
    population_size: int | None = None,
) -> dict[str, Any]:
    """Polish a single elite using CMA-ES.

    Parameters
    ----------
    elite_id : int
        Identifier for this elite.
    chromosome : np.ndarray
        Starting chromosome.
    initial_fitness : float
        Starting fitness.
    fitness_func : callable
        ``fitness_func(ga_instance, chromosome, solution_idx) -> float``.
        CMA-ES minimises, so the function value is negated internally.
    gene_space : list
        Gene space bounds.
    max_fevals : int
        Maximum fitness evaluations.
    initial_sigma : float
        Initial step size in normalized unit-cube coordinates.
    population_size : int | None
        CMA-ES population size.  ``None`` uses the library default.

    Returns
    -------
    dict
        ``{"elite_id", "initial_chromosome", "initial_fitness",
        "polished_chromosome", "polished_fitness", "fitness_gain",
        "fevals_used"}``.
    """
    import cma

    bounds_low = [g["low"] for g in gene_space]
    bounds_high = [g["high"] for g in gene_space]
    scaler = UnitCubeScaler.from_bounds(bounds_low, bounds_high)
    sigma0 = float(initial_sigma)
    if not 0.0 < sigma0 <= 1.0:
        raise ValueError("initial_sigma must lie in (0, 1] when CMA uses normalized genes.")

    opts: dict[str, Any] = {
        "bounds": [
            np.zeros(scaler.dimension, dtype=float).tolist(),
            np.ones(scaler.dimension, dtype=float).tolist(),
        ],
        "maxfevals": max_fevals,
        "verbose": -9,
        "tolfun": 1e-9,
        "tolx": 1e-9,
    }
    if population_size is not None:
        opts["popsize"] = population_size

    normalized_x0 = scaler.normalize(chromosome)
    es = cma.CMAEvolutionStrategy(normalized_x0.tolist(), sigma0, opts)

    best_chromosome = chromosome.copy()
    best_fitness = initial_fitness

    while not es.stop():
        normalized_candidates = es.ask()
        fitnesses = []
        for x in normalized_candidates:
            candidate = scaler.denormalize(np.asarray(x))
            f = fitness_func(None, candidate, 0)
            fitnesses.append(-f)
            if f > best_fitness:
                best_fitness = f
                best_chromosome = candidate.copy()
        es.tell(normalized_candidates, fitnesses)

    cma_result = es.result
    final_chromosome = scaler.denormalize(np.asarray(cma_result.xbest))
    final_fitness = fitness_func(None, final_chromosome, 0)
    if final_fitness > best_fitness:
        best_fitness = final_fitness
        best_chromosome = final_chromosome.copy()

    return {
        "elite_id": elite_id,
        "initial_chromosome": chromosome.copy(),
        "initial_fitness": initial_fitness,
        "polished_chromosome": best_chromosome,
        "polished_fitness": best_fitness,
        "fitness_gain": best_fitness - initial_fitness,
        "fevals_used": cma_result.evaluations,
    }
