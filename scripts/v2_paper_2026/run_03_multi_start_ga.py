#!/usr/bin/env python3
"""v2_paper_2026 step 03: multi-start GA (independent runs, champion pool).

Each worker runs the same GA configuration as step 01 (standard GA); seeds differ per run.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_SUITE_DIR = Path(__file__).resolve().parent
if str(_SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUITE_DIR))

import numpy as np
from tqdm import tqdm

import _artifacts
import _budgets
import _experiment_body
import _ga_multistart
import _paths
import _plotting
import _plotting_multi
from lwi_microbolometer_design import gaussian_parameters_to_unit_amplitude_curves
from lwi_microbolometer_design.ga import compute_population_distance_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("v2_paper.run_03")

mp.set_start_method("spawn", force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2_paper_2026 step 03: multi-start GA")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Process pool size (default: min(runs, min(CPU, cap)); "
            "see _budgets.MULTI_START_MAX_WORKERS_CAP)."
        ),
    )
    _budgets.add_quickrun_arg(parser)
    args = parser.parse_args()

    if args.quick:
        num_runs, num_generations, sol_per_pop = _budgets.quickrun_multi_start()
        max_workers = min(2, num_runs)
    else:
        num_runs = _budgets.MULTI_START_NUM_RUNS
        num_generations = _budgets.MULTI_START_GENERATIONS
        sol_per_pop = _budgets.MULTI_START_POP
        max_workers = min(num_runs, _budgets.MULTI_START_MAX_WORKERS)

    if args.max_workers is not None:
        max_workers = args.max_workers
    if min(num_runs, num_generations, sol_per_pop, max_workers) < 1:
        parser.error("Runs, generations, population size and workers must be positive.")
    max_workers = min(num_runs, max_workers)

    out_dir = _paths.step_output_dir(_paths.STEP_MULTI_START_GA)
    tqdm.write(
        f"\n{'=' * 72}\n"
        f"  STEP 03 | Multi-start GA | runs={num_runs} | gen/run={num_generations} | "
        f"workers={max_workers}\n"
        f"{'=' * 72}\n"
        "  Note: the bar below advances once per *completed* run (not per generation); "
        "the first tick can take as long as one full GA.\n"
    )

    scene = _experiment_body.load_nominal_scene()
    gene_space = _experiment_body.default_gene_space()
    npp = _experiment_body.num_params_per_basis_function()
    wavelengths_array = _experiment_body.wavelengths_array(scene)
    base_seed = 42

    payloads = [
        {
            "run_id": run_id,
            "scene": scene,
            "gene_space": gene_space,
            "params_per_basis_function": npp,
            "num_generations": num_generations,
            "sol_per_pop": sol_per_pop,
            "random_seed": base_seed + run_id,
            "stop_criteria": _budgets.GA_STOP_CRITERIA,
            "parents_mating_fraction": _budgets.GA_PARENTS_MATING_FRACTION_STANDARD,
            "mutation_type": "random",
            "mutation_probability": 0.15,
        }
        for run_id in range(num_runs)
    ]

    champions: list[np.ndarray] = []
    champion_fitness: list[float] = []
    run_records: list[dict] = []

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {
            executor.submit(_ga_multistart.run_single_ga_worker, p): p["run_id"] for p in payloads
        }
        for future in tqdm(
            as_completed(future_to_run),
            total=num_runs,
            desc="[03] Multi-start GA | completed runs",
            unit="run",
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.05,
        ):
            result = future.result()
            champions.append(result["best_chromosome"])
            champion_fitness.append(result["best_fitness"])
            run_records.append(
                {
                    "run_id": int(result["run_id"]),
                    "best_fitness": result["best_fitness"],
                    "random_seed": result["random_seed"],
                }
            )
    wall_s = time.perf_counter() - t0

    run_records.sort(key=lambda r: r["run_id"])
    champions_array = np.stack(champions, axis=0)
    champion_fitness_array = np.array(champion_fitness, dtype=float)

    if len(champions_array) >= 2:
        dm = compute_population_distance_matrix(champions_array, None)
        n = len(champions_array)
        mean_dist = float(np.mean(dm[np.triu_indices(n, k=1)]))
    else:
        mean_dist = 0.0

    with (out_dir / "champions.pkl").open("wb") as f:
        pickle.dump(
            {
                "champions": champions_array,
                "champion_fitness": champion_fitness_array,
                "per_run": run_records,
            },
            f,
        )

    individuals = [
        {
            "chromosome": np.asarray(champions_array[i], dtype=float),
            "fitness": float(champion_fitness_array[i]),
        }
        for i in range(len(champions_array))
    ]
    _plotting.export_presentation_top_elite_spectra_sweep(
        individuals,
        wavelengths_array,
        gaussian_parameters_to_unit_amplitude_curves,
        out_dir,
        "multi_start_ga",
        "Multi-start GA",
    )
    _plotting_multi.plot_multi_start_champions(
        champions_array,
        champion_fitness_array,
        wavelengths_array,
        mean_dist,
        out_dir / "multi_start_champions_overlay.png",
        num_runs=num_runs,
    )

    _artifacts.save_json(out_dir / "per_run_fitness.json", {"runs": run_records})

    _artifacts.save_run_summary(
        out_dir / "run_summary.json",
        {
            "step": "03_multi_start_ga",
            "method": "multi_start_GA",
            "budgets": {
                "num_independent_runs": num_runs,
                "num_generations_per_run": num_generations,
                "sol_per_pop": sol_per_pop,
                "max_workers": max_workers,
                "stop_criteria": _budgets.GA_STOP_CRITERIA,
                "parents_mating_fraction": _budgets.GA_PARENTS_MATING_FRACTION_STANDARD,
                "mutation_probability": 0.15,
                "mutation": "random",
                "niching_enabled": False,
            },
            "wall_time_seconds": wall_s,
            "best_fitness": float(np.max(champion_fitness_array)),
            "mean_champion_fitness": float(np.mean(champion_fitness_array)),
            "champion_pool_mean_pairwise_distance": mean_dist,
            "total_fitness_calls_estimate": int(num_runs * num_generations * sol_per_pop),
            "output_files": [
                "champions.pkl",
                *_plotting.elite_spectrum_png_filenames("multi_start_ga"),
                "multi_start_champions_overlay.png",
                "per_run_fitness.json",
            ],
        },
    )
    logger.info("Saved outputs under %s", out_dir)


if __name__ == "__main__":
    main()
