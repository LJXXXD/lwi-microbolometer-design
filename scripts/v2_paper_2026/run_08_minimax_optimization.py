#!/usr/bin/env python3
"""v2_paper_2026 step 08: vanilla MAP-Elites under minimax (worst-case) fitness."""

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
from lwi_microbolometer_design import gaussian_parameters_to_unit_amplitude_curves
from lwi_microbolometer_design.data import load_substance_atmosphere_data
from lwi_microbolometer_design.ga import calculate_population_diversity
from lwi_microbolometer_design.ga.experiment import (
    create_fitness_evaluator_from_experiment,
    create_gene_space_from_experiment,
    load_experiment_config,
)
from lwi_microbolometer_design.map_elites import (
    archive_coverage_pct,
    plot_map_elites_heatmap,
    reachable_cell_count,
    run_map_elites,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("v2_paper.run_08")

mp.set_start_method("spawn", force=True)

DEFAULT_YAML = Path("experiments/v2_paper_2026_minimax.yaml")


def _count_conditions(exp) -> int:
    loaded = load_substance_atmosphere_data(
        spectral_data_file=Path(exp.data["spectral_data_file"]),
        air_transmittance_file=Path(exp.data["air_transmittance_file"]),
        atmospheric_distance_ratio=exp.data.get("atmospheric_distance_ratio", 0.11),
        temperature_kelvin=exp.data.get("temperature_kelvin", 293.15),
        air_refractive_index=exp.data.get("air_refractive_index", 1.0),
    )
    return len(loaded) if isinstance(loaded, list) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="v2_paper_2026 step 08: minimax MAP-Elites")
    parser.add_argument(
        "--experiment-yaml",
        type=Path,
        default=None,
        help="Experiment YAML (default: experiments/v2_paper_2026_minimax.yaml).",
    )
    parser.add_argument(
        "--budget-multiplier",
        type=float,
        default=None,
        help="Scale MAP-Elites fitness-eval budget by this factor (default: PRES_MINIMAX_MULT or 1).",
    )
    _budgets.add_quickrun_arg(parser)
    args = parser.parse_args()

    yaml_path = args.experiment_yaml or (_paths.project_root() / DEFAULT_YAML)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Experiment YAML not found: {yaml_path}")

    experiment = load_experiment_config(yaml_path)
    fitness_func = create_fitness_evaluator_from_experiment(experiment)
    gene_space = create_gene_space_from_experiment(experiment)
    num_conditions = _count_conditions(experiment)

    mult = (
        args.budget_multiplier
        if args.budget_multiplier is not None
        else _budgets.MINIMAX_BUDGET_MULTIPLIER
    )
    if args.quick:
        total_budget = int(_budgets.quickrun_minimax() * mult)
        num_initial = _budgets.quickrun_cma_me()[1]
    else:
        total_budget = int(_budgets.MINIMAX_CMA_ME_TOTAL_EVALS * mult)
        num_initial = _budgets.MAP_ELITES_NUM_INITIAL

    num_initial = min(int(num_initial), int(total_budget))
    num_iterations = max(0, int(total_budget) - num_initial)

    out_dir = _paths.step_output_dir(_paths.STEP_MINIMAX)
    tqdm.write(
        f"\n{'=' * 72}\n"
        f"  STEP 08 | Minimax MAP-Elites | seed={num_initial:,} | mutations={num_iterations:,} "
        f"| total_evals≈{total_budget:,} | conditions={num_conditions} (min aggregation)\n"
        f"{'=' * 72}\n"
    )

    grid_resolution, mu_range = _experiment_body.default_map_elites_geometry()
    scene0 = _experiment_body.load_nominal_scene()
    wavelengths_array = _experiment_body.wavelengths_array(scene0)

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
        progress_desc_init="[08] Minimax MAP-Elites | seed archive",
        progress_desc_main="[08] Minimax MAP-Elites | mutations",
    )
    wall_s = time.perf_counter() - t0

    total_calls = num_initial + num_iterations
    physics_estimate = int(total_calls * num_conditions)

    archive_path = out_dir / "map_elites_archive.pkl"
    with archive_path.open("wb") as f:
        pickle.dump(archive, f)

    archive_individuals_sorted = sorted(
        archive.values(),
        key=lambda ind: float(ind["fitness"]),
        reverse=True,
    )
    population = np.array(
        [np.asarray(ind["chromosome"], dtype=float) for ind in archive_individuals_sorted],
        dtype=float,
    )
    fitness = np.array(
        [float(ind["fitness"]) for ind in archive_individuals_sorted],
        dtype=float,
    )
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
        title="Minimax MAP-Elites — archive (descriptor space)",
    )
    plot_map_elites_heatmap(
        archive=archive,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        output_path=out_dir / "map_elites_heatmap.png",
    )
    _plotting.export_presentation_top_elite_spectra_sweep(
        archive_individuals_sorted,
        wavelengths_array,
        gaussian_parameters_to_unit_amplitude_curves,
        out_dir,
        "minimax_map_elites",
        "Minimax MAP-Elites",
    )

    _artifacts.save_run_summary(
        out_dir / "run_summary.json",
        {
            "step": "08_minimax_optimization",
            "method": "MAP-Elites_minimax_fitness",
            "experiment_yaml": str(yaml_path.resolve()),
            "budgets": {
                "num_iterations": num_iterations,
                "num_initial": num_initial,
                "total_fitness_evals_target": total_budget,
                "grid_resolution": grid_resolution,
                "mutation_probability": 0.15,
                "num_conditions": num_conditions,
                "total_physics_forward_passes_estimate": physics_estimate,
                "budget_multiplier": mult,
            },
            "wall_time_seconds": wall_s,
            "best_fitness": best_fit,
            "mean_final_fitness": float(np.mean(fitness)) if len(fitness) else None,
            "final_population_diversity": diversity,
            "unique_descriptor_bins": unique_bins,
            "archive_cells_filled": len(archive),
            "reachable_cells": reachable_cells,
            "coverage_pct": archive_coverage_pct(len(archive), grid_resolution),
            "robustness_aggregation": experiment.data.get("robustness_aggregation"),
            "total_fitness_calls_estimate": total_calls,
            "output_files": [
                "map_elites_archive.pkl",
                "final_population.pkl",
                "best_chromosome.pkl",
                "descriptor_space_final.png",
                "map_elites_heatmap.png",
                *_plotting.elite_spectrum_png_filenames("minimax_map_elites"),
            ],
        },
    )
    logger.info(
        "Minimax MAP-Elites done; ~%s physics passes; outputs in %s",
        f"{physics_estimate:,}",
        out_dir,
    )


if __name__ == "__main__":
    main()
