#!/usr/bin/env python3
"""v2_paper_2026 step 02: Advanced GA with niching + diversity-preserving mutation.

Per-run budget and mating/mutation rates match step 01 (standard GA); only niching and
mutation operator differ.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

_SUITE_DIR = Path(__file__).resolve().parent
if str(_SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUITE_DIR))

import numpy as np
from tqdm import tqdm

import _artifacts
import _budgets
import _experiment_body
import _paths
import _plotting
import _progress
from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    calculate_population_diversity,
    create_ga_config,
    diversity_preserving_mutation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("v2_paper.run_02")

mp.set_start_method("spawn", force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2_paper_2026 step 02: advanced GA (niching)")
    _budgets.add_quickrun_arg(parser)
    args = parser.parse_args()

    if args.quick:
        num_generations, sol_per_pop = _budgets.quickrun_ga_baseline()
    else:
        num_generations = _budgets.GA_NUM_GENERATIONS_STANDARD
        sol_per_pop = _budgets.GA_SOL_PER_POP_STANDARD

    out_dir = _paths.step_output_dir(_paths.STEP_ADVANCED_GA)
    tqdm.write(
        f"\n{'=' * 72}\n"
        f"  STEP 02 | Advanced GA (niching) | generations={num_generations} | pop={sol_per_pop}\n"
        f"{'=' * 72}\n"
    )

    scene = _experiment_body.load_nominal_scene()
    gene_space = _experiment_body.default_gene_space()
    grid_resolution, mu_range = _experiment_body.default_map_elites_geometry()
    wavelengths_array = _experiment_body.wavelengths_array(scene)
    npp = _experiment_body.num_params_per_basis_function()

    fitness_func = MinDissimilarityFitnessEvaluator(
        scene=scene,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=npp,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    ga_config = create_ga_config(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=max(2, int(sol_per_pop * _budgets.GA_PARENTS_MATING_FRACTION_STANDARD)),
        mutation_type=diversity_preserving_mutation,
        mutation_probability=0.15,
        niching_enabled=True,
        stop_criteria=_budgets.GA_STOP_CRITERIA,
        random_seed=42,
    )
    ga_config["num_genes"] = len(gene_space)
    ga_config["gene_space"] = gene_space
    ga_config["fitness_func"] = fitness_func

    ga_progress = _progress.PresentationGAProgressBar(
        num_generations, "[02] Advanced GA (niching) | generation"
    )
    ga_config["on_generation"] = ga_progress.on_generation

    t0 = time.perf_counter()
    try:
        ga = AdvancedGA(**ga_config)
        ga.run()
    finally:
        ga_progress.close()
    wall_s = time.perf_counter() - t0

    population = np.array(ga.population, dtype=float)
    fitness = np.array(ga.last_generation_fitness, dtype=float)
    best_chrom, best_fit, _ = ga.best_solution()

    with (out_dir / "final_population.pkl").open("wb") as f:
        pickle.dump({"population": population, "fitness": fitness}, f)

    with (out_dir / "best_chromosome.pkl").open("wb") as f:
        pickle.dump(
            {"chromosome": np.asarray(best_chrom, dtype=float), "fitness": float(best_fit)},
            f,
        )

    diversity = calculate_population_diversity(population, ga.niching_config)
    unique_bins = _plotting.count_unique_descriptor_bins(population, mu_range, grid_resolution)

    _plotting.apply_presentation_style()
    _plotting.plot_descriptor_scatter(
        population,
        fitness,
        mu_range,
        grid_resolution,
        out_dir / "descriptor_space_final.png",
        title="Advanced GA (niching) — final population (descriptor space)",
    )
    individuals = [
        {"chromosome": np.asarray(population[i], dtype=float), "fitness": float(fitness[i])}
        for i in range(len(population))
    ]
    _plotting.export_presentation_top_elite_spectra_sweep(
        individuals,
        wavelengths_array,
        gaussian_parameters_to_unit_amplitude_curves,
        out_dir,
        "advanced_ga_niching",
        "Advanced GA / niching",
    )

    est_evals = int(num_generations * sol_per_pop)

    _artifacts.save_run_summary(
        out_dir / "run_summary.json",
        {
            "step": "02_advanced_ga_niching",
            "method": "advanced_GA_niching",
            "budgets": {
                "num_generations": num_generations,
                "sol_per_pop": sol_per_pop,
                "stop_criteria": _budgets.GA_STOP_CRITERIA,
                "parents_mating_fraction": _budgets.GA_PARENTS_MATING_FRACTION_STANDARD,
                "mutation_probability": 0.15,
                "mutation": "diversity_preserving_mutation",
                "niching_enabled": True,
            },
            "wall_time_seconds": wall_s,
            "best_fitness": float(best_fit),
            "mean_final_fitness": float(np.mean(fitness)),
            "final_population_diversity": diversity,
            "unique_descriptor_bins": unique_bins,
            "total_fitness_calls_estimate": est_evals,
            "output_files": [
                "final_population.pkl",
                "best_chromosome.pkl",
                "descriptor_space_final.png",
                *_plotting.elite_spectrum_png_filenames("advanced_ga_niching"),
            ],
        },
    )
    logger.info("Saved outputs under %s", out_dir)


if __name__ == "__main__":
    main()
