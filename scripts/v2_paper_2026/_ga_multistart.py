"""Pickle-friendly single GA run for multi-start (ProcessPoolExecutor workers)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data import SceneConfig
import _budgets

from lwi_microbolometer_design.ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    create_ga_config,
)


def run_single_ga_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one independent GA; accepts a plain dict for robust pickling across spawn."""
    # Spawned workers inherit a fresh interpreter; keep logs quiet so the parent tqdm stays readable.
    logging.getLogger("lwi_microbolometer_design").setLevel(logging.WARNING)

    run_id = int(payload["run_id"])
    scene: SceneConfig = payload["scene"]
    gene_space: list[dict[str, float]] = payload["gene_space"]
    params_per_basis_function = int(payload["params_per_basis_function"])
    num_generations = int(payload["num_generations"])
    sol_per_pop = int(payload["sol_per_pop"])
    random_seed = int(payload["random_seed"])

    np.random.seed(random_seed)

    fitness_func = MinDissimilarityFitnessEvaluator(
        scene=scene,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    parents_frac = float(
        payload.get("parents_mating_fraction", _budgets.GA_PARENTS_MATING_FRACTION_STANDARD)
    )
    ga_config = create_ga_config(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=max(2, int(sol_per_pop * parents_frac)),
        mutation_type=str(payload.get("mutation_type", "random")),
        mutation_probability=float(payload.get("mutation_probability", 0.15)),
        niching_enabled=False,
        random_seed=random_seed,
        stop_criteria=str(payload.get("stop_criteria", "saturate_10000")),
        # Avoid PyGAD's per-generation best-solution list (memory + warning) — we only need the final best.
        save_best_solutions=False,
    )
    ga_config["num_genes"] = len(gene_space)
    ga_config["gene_space"] = gene_space
    ga_config["fitness_func"] = fitness_func

    ga = AdvancedGA(**ga_config)
    ga.run()
    best_chromosome, best_fitness, _best_idx = ga.best_solution()

    return {
        "run_id": run_id,
        "best_chromosome": np.asarray(best_chromosome, dtype=float),
        "best_fitness": float(best_fitness),
        "random_seed": random_seed,
    }
