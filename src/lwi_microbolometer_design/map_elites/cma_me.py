"""CMA-ME (Covariance Matrix Adaptation MAP-Elites) main loop.

Orchestrates multiple :class:`OptimizingEmitter` instances around a shared
MAP-Elites archive.  Each emitter generates candidate batches, evaluates
them via the provided fitness function, computes archive improvement, and
updates its CMA-ES distribution accordingly.  When an emitter converges it
is restarted from a randomly selected archive elite, maintaining perpetual
exploration pressure.

References
----------
Fontaine, M. C., et al. (2020). "Covariance Matrix Adaptation for the
Rapid Illumination of Behavior Space." *GECCO 2020*.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tqdm import trange

from .archive import (
    archive_coverage_pct,
    bin_coordinates,
    extract_features,
    initialize_archive,
    reachable_cell_count,
)
from .emitters import OptimizingEmitter

logger = logging.getLogger(__name__)


def run_cma_me(
    fitness_func: Any,
    gene_space: list[dict[str, float]],
    grid_resolution: int = 20,
    mu_range: tuple[float, float] = (4.0, 20.0),
    num_initial: int = 1000,
    total_evals: int = 500_000,
    num_emitters: int = 5,
    batch_size: int | None = None,
    initial_sigma: float = 0.2,
    restart_patience: int = 30,
    log_interval: int = 5000,
    random_seed: int = 42,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    """Run CMA-ME quality-diversity optimisation.

    Parameters
    ----------
    fitness_func : callable
        ``fitness_func(ga_instance, chromosome, solution_idx) -> float``.
    gene_space : list[dict[str, float]]
        Per-gene bounds ``{"low": float, "high": float}``.
    grid_resolution : int
        Archive bins per feature dimension.
    mu_range : tuple[float, float]
        Feature value range for both axes.
    num_initial : int
        Random solutions used to seed the archive.
    total_evals : int
        Total fitness evaluation budget (including initialisation).
    num_emitters : int
        Number of concurrent :class:`OptimizingEmitter` instances.
    batch_size : int | None
        CMA-ES population size per emitter.  ``None`` uses the library
        default :math:`4 + \\lfloor 3 \\ln n \\rfloor`.
    initial_sigma : float
        Initial step-size in normalized unit-cube coordinates.
    restart_patience : int
        Restart an emitter after this many consecutive non-improving
        generations.
    log_interval : int
        Print progress every *log_interval* fitness evaluations.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[dict, dict]
        ``(archive, metadata)`` where *archive* maps
        ``(x_bin, y_bin) -> {"chromosome", "fitness", "mu_1", "mu_2"}``
        and *metadata* contains run statistics and convergence history.
    """
    rng = np.random.default_rng(random_seed)
    np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # 1. Seed the archive with random solutions
    # ------------------------------------------------------------------
    archive = initialize_archive(
        num_initial=num_initial,
        gene_space=gene_space,
        fitness_func=fitness_func,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        random_seed=random_seed,
    )

    # ------------------------------------------------------------------
    # 2. Prepare bounds for normalized CMA-ES search
    # ------------------------------------------------------------------
    bounds_low = [g["low"] for g in gene_space]
    bounds_high = [g["high"] for g in gene_space]
    sigma0 = float(initial_sigma)
    if not 0.0 < sigma0 <= 1.0:
        raise ValueError("initial_sigma must lie in (0, 1] when CMA uses normalized genes.")

    # ------------------------------------------------------------------
    # 3. Create emitters, each seeded from a random archive elite
    # ------------------------------------------------------------------
    archive_list = list(archive.values())
    emitters: list[OptimizingEmitter] = []
    for i in range(num_emitters):
        elite = archive_list[rng.integers(len(archive_list))]
        emitters.append(
            OptimizingEmitter(
                x0=elite["chromosome"].copy(),
                sigma0=sigma0,
                bounds_low=bounds_low,
                bounds_high=bounds_high,
                batch_size=batch_size,
                restart_patience=restart_patience,
                seed=random_seed + i + 1,
            )
        )

    evals_per_batch = emitters[0].batch_size
    evals_per_round = evals_per_batch * num_emitters
    remaining_evals = total_evals - num_initial
    num_rounds = max(1, remaining_evals // evals_per_round)

    # ------------------------------------------------------------------
    # 4. Tracking
    # ------------------------------------------------------------------
    evals_used = num_initial
    total_improvements = 0
    total_new_cells = 0
    last_log_evals = evals_used

    history: dict[str, list[float]] = {
        "evals": [],
        "archive_size": [],
        "best_fitness": [],
        "coverage_pct": [],
    }
    total_cells = reachable_cell_count(grid_resolution)

    def _snapshot() -> None:
        best = max(ind["fitness"] for ind in archive.values())
        history["evals"].append(float(evals_used))
        history["archive_size"].append(float(len(archive)))
        history["best_fitness"].append(float(best))
        history["coverage_pct"].append(archive_coverage_pct(len(archive), grid_resolution))

    _snapshot()

    # ------------------------------------------------------------------
    # 5. Main CMA-ME loop
    # ------------------------------------------------------------------
    print(
        f"\nRunning CMA-ME with {num_emitters} emitters "
        f"(batch_size={evals_per_batch}) for {total_evals:,} total evals..."
    )

    for _round in trange(num_rounds, desc="CMA-ME", unit="round"):
        if evals_used >= total_evals:
            break

        for emitter in emitters:
            if evals_used >= total_evals:
                break

            solutions = emitter.ask()
            fitnesses: list[float] = []
            improvements: list[float] = []

            for solution in solutions:
                fitness = float(fitness_func(None, solution, 0))
                fitnesses.append(fitness)

                mu_1, mu_2 = extract_features(solution)
                key = bin_coordinates(mu_1, mu_2, grid_resolution, mu_range)

                if key not in archive:
                    imp = fitness
                    total_new_cells += 1
                elif fitness > archive[key]["fitness"]:
                    imp = fitness - archive[key]["fitness"]
                else:
                    imp = 0.0

                if imp > 0:
                    archive[key] = {
                        "chromosome": solution.copy(),
                        "fitness": fitness,
                        "mu_1": mu_1,
                        "mu_2": mu_2,
                    }
                    total_improvements += 1

                improvements.append(imp)

            emitter.tell(solutions, improvements, fitnesses)
            evals_used += len(solutions)

            if emitter.converged:
                archive_list = list(archive.values())
                elite = archive_list[rng.integers(len(archive_list))]
                emitter.restart(elite["chromosome"].copy())

        if evals_used - last_log_evals >= log_interval:
            _snapshot()
            last_log_evals = evals_used

    _snapshot()

    # ------------------------------------------------------------------
    # 6. Collect metadata
    # ------------------------------------------------------------------
    total_restarts = sum(e.total_restarts for e in emitters)
    coverage = len(archive) / total_cells * 100
    best_fitness = max(ind["fitness"] for ind in archive.values())

    metadata: dict[str, Any] = {
        "total_evals": evals_used,
        "total_improvements": total_improvements,
        "total_new_cells": total_new_cells,
        "total_restarts": total_restarts,
        "archive_size": len(archive),
        "reachable_cells": total_cells,
        "grid_resolution": grid_resolution,
        "coverage_pct": coverage,
        "best_fitness": best_fitness,
        "num_emitters": num_emitters,
        "batch_size": evals_per_batch,
        "history": history,
    }

    print(
        f"\nCMA-ME complete: {len(archive)}/{total_cells} reachable cells filled "
        f"({coverage:.1f}% coverage) | Best: {best_fitness:.2f} | "
        f"Restarts: {total_restarts}"
    )

    return archive, metadata
