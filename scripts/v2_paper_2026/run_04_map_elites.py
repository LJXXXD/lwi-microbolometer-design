#!/usr/bin/env python3
"""v2_paper_2026 step 04: MAP-Elites (budget matched to step 01 standard GA)."""

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
from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.ga import (
    MinDissimilarityFitnessEvaluator,
    calculate_population_diversity,
)
from lwi_microbolometer_design.map_elites import (
    archive_coverage_pct,
    plot_map_elites_heatmap,
    reachable_cell_count,
    run_map_elites,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("v2_paper.run_04")

mp.set_start_method("spawn", force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2_paper_2026 step 04: MAP-Elites")
    _budgets.add_quickrun_arg(parser)
    args = parser.parse_args()

    if args.quick:
        num_iterations, num_initial = _budgets.quickrun_map_elites()
    else:
        num_iterations = _budgets.MAP_ELITES_ITERATIONS
        num_initial = _budgets.MAP_ELITES_NUM_INITIAL

    total_evals = num_initial + num_iterations
    out_dir = _paths.step_output_dir(_paths.STEP_MAP_ELITES)
    tqdm.write(
        f"\n{'=' * 72}\n"
        f"  STEP 04 | MAP-Elites | seed={num_initial:,} | mutations={num_iterations:,} "
        f"| total_evals≈{total_evals:,}\n"
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

    reachable_cells = reachable_cell_count(grid_resolution)
    t0 = time.perf_counter()
    archive = run_map_elites(
        fitness_func=fitness_func,
        gene_space=gene_space,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        num_initial=num_initial,
        num_iterations=num_iterations,
        mutation_probability=0.15,
        random_seed=42,
        show_progress=True,
        progress_desc_init="[04] MAP-Elites | seed archive",
        progress_desc_main="[04] MAP-Elites | mutations",
    )
    wall_s = time.perf_counter() - t0

    archive_path = out_dir / "map_elites_archive.pkl"
    with archive_path.open("wb") as f:
        pickle.dump(archive, f)

    archive_individuals = sorted(
        archive.values(),
        key=lambda ind: float(ind["fitness"]),
        reverse=True,
    )
    population = np.array(
        [np.asarray(ind["chromosome"], dtype=float) for ind in archive_individuals],
        dtype=float,
    )
    fitness = np.array([float(ind["fitness"]) for ind in archive_individuals], dtype=float)
    best_idx = int(np.argmax(fitness)) if len(fitness) else 0
    best_chrom = population[best_idx] if len(population) else np.array([], dtype=float)
    best_fit = float(fitness[best_idx]) if len(fitness) else 0.0

    with (out_dir / "final_population.pkl").open("wb") as f:
        pickle.dump({"population": population, "fitness": fitness}, f)

    with (out_dir / "best_chromosome.pkl").open("wb") as f:
        pickle.dump(
            {"chromosome": np.asarray(best_chrom, dtype=float), "fitness": float(best_fit)},
            f,
        )

    diversity = calculate_population_diversity(population, None) if len(population) else 0.0
    unique_bins = (
        _plotting.count_unique_descriptor_bins(population, mu_range, grid_resolution)
        if len(population)
        else 0
    )

    _plotting.apply_presentation_style()
    _plotting.plot_descriptor_scatter(
        population,
        fitness,
        mu_range,
        grid_resolution,
        out_dir / "descriptor_space_final.png",
        title="MAP-Elites — archive (descriptor space)",
    )
    plot_map_elites_heatmap(
        archive=archive,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        output_path=out_dir / "map_elites_heatmap.png",
    )
    _plotting.export_presentation_top_elite_spectra_sweep(
        archive_individuals,
        wavelengths_array,
        gaussian_parameters_to_unit_amplitude_curves,
        out_dir,
        "map_elites",
        "MAP-Elites",
    )

    _artifacts.save_run_summary(
        out_dir / "run_summary.json",
        {
            "step": "04_map_elites",
            "method": "MAP-Elites_vanilla",
            "budgets": {
                "num_iterations": num_iterations,
                "num_initial": num_initial,
                "total_fitness_evals_target": total_evals,
                "grid_resolution": grid_resolution,
                "mutation_probability": 0.15,
            },
            "wall_time_seconds": wall_s,
            "best_fitness": best_fit,
            "mean_final_fitness": float(np.mean(fitness)) if len(fitness) else None,
            "final_population_diversity": diversity,
            "unique_descriptor_bins": unique_bins,
            "archive_cells_filled": len(archive),
            "reachable_cells": reachable_cells,
            "coverage_pct": archive_coverage_pct(len(archive), grid_resolution),
            "total_fitness_calls_estimate": total_evals,
            "output_files": [
                "map_elites_archive.pkl",
                "final_population.pkl",
                "best_chromosome.pkl",
                "descriptor_space_final.png",
                "map_elites_heatmap.png",
                *_plotting.elite_spectrum_png_filenames("map_elites"),
            ],
        },
    )
    logger.info("Saved outputs under %s", out_dir)


if __name__ == "__main__":
    main()
