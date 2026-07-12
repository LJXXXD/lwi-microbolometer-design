#!/usr/bin/env python3
"""v2_paper_2026 step 05: MAP-Elites + hill-climbing (QD budget matched to step 01)."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
    compute_population_distance_matrix,
)
from lwi_microbolometer_design.map_elites import (
    archive_coverage_pct,
    plot_map_elites_heatmap,
    plot_polished_elites,
    polish_single_elite_hc,
    reachable_cell_count,
    run_map_elites,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("v2_paper.run_05")

mp.set_start_method("spawn", force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2_paper_2026 step 05: MAP-Elites + HC")
    parser.add_argument(
        "--hc-workers",
        type=int,
        default=None,
        help=(
            "Max parallel HC workers (default: min(CPU, HC_MAX_WORKERS_CAP); "
            "see _budgets.HC_MAX_WORKERS_CAP)."
        ),
    )
    _budgets.add_quickrun_arg(parser)
    args = parser.parse_args()

    if args.quick:
        num_iterations, num_initial = _budgets.quickrun_map_elites()
        hc_workers = min(2, mp.cpu_count())
        max_polish = 8
        fitness_threshold = 0.0
    else:
        num_iterations = _budgets.MAP_ELITES_ITERATIONS
        num_initial = _budgets.MAP_ELITES_NUM_INITIAL
        hc_workers = args.hc_workers or _budgets.HC_MAX_WORKERS
        max_polish = _budgets.HC_MAX_ELITES
        fitness_threshold = _budgets.HC_FITNESS_THRESHOLD

    total_map_evals = num_initial + num_iterations
    out_dir = _paths.step_output_dir(_paths.STEP_MAP_ELITES_HC)
    tqdm.write(
        f"\n{'=' * 72}\n"
        f"  STEP 05 | MAP-Elites + HC | seed={num_initial:,} | mutations={num_iterations:,} "
        f"| total_evals≈{total_map_evals:,} | HC_workers={hc_workers} | polish_cap={max_polish}\n"
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

    t_map = time.perf_counter()
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
        progress_desc_init="[05] MAP-Elites+HC | seed archive",
        progress_desc_main="[05] MAP-Elites+HC | mutations",
    )
    map_wall_s = time.perf_counter() - t_map

    archive_individuals_sorted = sorted(
        archive.values(),
        key=lambda ind: float(ind["fitness"]),
        reverse=True,
    )
    population = np.array(
        [np.asarray(ind["chromosome"], dtype=float) for ind in archive_individuals_sorted],
        dtype=float,
    )
    fitness_arch = np.array(
        [float(ind["fitness"]) for ind in archive_individuals_sorted],
        dtype=float,
    )
    best_arch_idx = int(np.argmax(fitness_arch)) if len(fitness_arch) else 0
    best_arch_chrom = population[best_arch_idx] if len(population) else np.array([], dtype=float)
    best_arch_fit = float(fitness_arch[best_arch_idx]) if len(fitness_arch) else 0.0

    with (out_dir / "final_population.pkl").open("wb") as f:
        pickle.dump({"population": population, "fitness": fitness_arch}, f)

    with (out_dir / "best_chromosome.pkl").open("wb") as f:
        pickle.dump(
            {
                "chromosome": np.asarray(best_arch_chrom, dtype=float),
                "fitness": float(best_arch_fit),
            },
            f,
        )

    diversity_arch = calculate_population_diversity(population, None) if len(population) else 0.0
    unique_bins_arch = (
        _plotting.count_unique_descriptor_bins(population, mu_range, grid_resolution)
        if len(population)
        else 0
    )

    _plotting.apply_presentation_style()
    _plotting.plot_descriptor_scatter(
        population,
        fitness_arch,
        mu_range,
        grid_resolution,
        out_dir / "descriptor_space_final.png",
        title="MAP-Elites + HC — archive after QD (descriptor space)",
    )
    _plotting.export_presentation_top_elite_spectra_sweep(
        archive_individuals_sorted,
        wavelengths_array,
        gaussian_parameters_to_unit_amplitude_curves,
        out_dir,
        "map_elites_hc",
        "MAP-Elites + HC",
    )

    all_individuals = archive_individuals_sorted
    promising = [ind for ind in all_individuals if ind["fitness"] > fitness_threshold]
    num_elites_to_polish = min(max_polish, len(promising))
    top_elites_initial = promising[:num_elites_to_polish]

    polished_results: list[dict[str, Any]] = []
    t_hc = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(num_elites_to_polish, hc_workers)) as executor:
        future_to_elite = {
            executor.submit(
                polish_single_elite_hc,
                elite_id=i,
                chromosome=elite["chromosome"],
                initial_fitness=elite["fitness"],
                fitness_func=fitness_func,
                gene_space=gene_space,
                num_iterations=5000,
                mutation_sigma=0.05,
                mutation_probability=0.2,
                adaptive_iterations=True,
            ): (i, elite)
            for i, elite in enumerate(top_elites_initial)
        }
        for future in tqdm(
            as_completed(future_to_elite),
            total=num_elites_to_polish,
            desc="[05] MAP-Elites+HC | hill climb",
            unit="elite",
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.05,
        ):
            polished_results.append(future.result())
    hc_wall_s = time.perf_counter() - t_hc

    polished_results.sort(key=lambda x: x["polished_fitness"], reverse=True)

    archive_path = out_dir / "map_elites_archive.pkl"
    with archive_path.open("wb") as f:
        pickle.dump(archive, f)

    results_path = out_dir / "polished_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump(
            {
                "polished_results": polished_results,
                "top_elites_initial": top_elites_initial,
                "wavelengths": wavelengths_array,
            },
            f,
        )

    _plotting.apply_presentation_style()
    plot_map_elites_heatmap(
        archive=archive,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        output_path=out_dir / "map_elites_hc_heatmap.png",
    )
    for top_n in _plotting.SUITE_TOP_ELITE_COUNTS:
        plot_polished_elites(
            initial_elites=top_elites_initial,
            polished_elites=polished_results,
            wavelengths=wavelengths_array,
            parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
            output_path=out_dir / f"map_elites_hc_top{top_n}_elites.png",
            top_n=top_n,
            title_prefix="MAP-Elites + HC",
        )

    best_arch = best_arch_fit
    best_polished = (
        float(np.max([r["polished_fitness"] for r in polished_results]))
        if polished_results
        else 0.0
    )

    diversity_top: float | None = None
    top_performers = [r for r in polished_results if r["polished_fitness"] >= 58.0]
    if len(top_performers) > 1:
        chroms = np.array([r["polished_chromosome"] for r in top_performers])
        dm = compute_population_distance_matrix(chroms, None)
        n = len(top_performers)
        diversity_top = float(np.mean(dm[np.triu_indices(n, k=1)]))

    _artifacts.save_run_summary(
        out_dir / "run_summary.json",
        {
            "step": "05_map_elites_hc",
            "method": "MAP-Elites_plus_hill_climbing",
            "budgets": {
                "num_iterations": num_iterations,
                "num_initial": num_initial,
                "total_fitness_evals_target": total_map_evals,
                "grid_resolution": grid_resolution,
                "mutation_probability": 0.15,
                "elites_polished": num_elites_to_polish,
                "hc_workers": hc_workers,
            },
            "wall_time_seconds_map_elites": map_wall_s,
            "wall_time_seconds_hc": hc_wall_s,
            "best_fitness_archive": best_arch,
            "mean_final_fitness": float(np.mean(fitness_arch)) if len(fitness_arch) else None,
            "final_population_diversity": diversity_arch,
            "unique_descriptor_bins": unique_bins_arch,
            "total_fitness_calls_estimate": total_map_evals,
            "best_fitness_polished": best_polished,
            "mean_polished_fitness": float(
                np.mean([r["polished_fitness"] for r in polished_results])
            )
            if polished_results
            else None,
            "diversity_among_polished_fitness_ge_58": diversity_top,
            "archive_cells_filled": len(archive),
            "reachable_cells": reachable_cell_count(grid_resolution),
            "coverage_pct": archive_coverage_pct(len(archive), grid_resolution),
            "output_files": [
                "map_elites_archive.pkl",
                "final_population.pkl",
                "best_chromosome.pkl",
                "descriptor_space_final.png",
                "polished_results.pkl",
                "map_elites_hc_heatmap.png",
                *_plotting.elite_spectrum_png_filenames("map_elites_hc"),
                *[f"map_elites_hc_top{n}_elites.png" for n in _plotting.SUITE_TOP_ELITE_COUNTS],
            ],
        },
    )
    logger.info("Saved outputs under %s", out_dir)


if __name__ == "__main__":
    main()
