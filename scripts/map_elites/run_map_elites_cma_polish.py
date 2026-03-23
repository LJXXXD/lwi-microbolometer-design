#!/usr/bin/env python3
"""MAP-Elites + CMA-ES Polish.

Takes the top elites from a MAP-Elites archive and refines them using CMA-ES
(Covariance Matrix Adaptation Evolution Strategy), which learns the local
fitness landscape shape to optimise far more efficiently than random hill
climbing.
"""

import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data import SceneConfig, load_substance_atmosphere_data
from lwi_microbolometer_design.ga import (
    MinDissimilarityFitnessEvaluator,
    compute_population_distance_matrix,
)
from lwi_microbolometer_design.map_elites import (
    extract_features,
    plot_map_elites_heatmap,
    plot_polished_elites,
    polish_single_elite_cma,
    run_map_elites,
)

mp.set_start_method("spawn", force=True)


def main() -> None:
    """Run CMA-ES polish on MAP-Elites elites."""
    print("=" * 80)
    print("MAP-ELITES + CMA-ES POLISH: Refining Winners with Learned Search")
    print("=" * 80)
    print("Strategy: Take top MAP-Elites elites and polish with CMA-ES")
    print("          CMA-ES learns the local fitness landscape shape (covariance")
    print("          matrix) and searches along the most promising directions.")

    # ------------------------------------------------------------------
    # [1/4] Load data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading data...")
    spectral_data_file = Path("data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")
    air_transmittance_file = Path("data/Test 3 - 4 White Powers/Air transmittance.xlsx")

    loaded_data = load_substance_atmosphere_data(
        spectral_data_file=spectral_data_file,
        air_transmittance_file=air_transmittance_file,
        atmospheric_distance_ratio=0.11,
        temperature_kelvin=293.15,
        air_refractive_index=1.0,
    )

    if isinstance(loaded_data, list):
        scene: SceneConfig = loaded_data[0]
    else:
        scene = loaded_data

    num_basis_functions = 4
    num_params_per_basis_function = 2
    param_bounds = [
        {"low": 4.0, "high": 20.0},
        {"low": 0.1, "high": 4.0},
    ]
    gene_space = param_bounds * num_basis_functions
    wavelengths_array = np.asarray(scene.wavelengths)

    fitness_func = MinDissimilarityFitnessEvaluator(
        scene=scene,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=num_params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # ------------------------------------------------------------------
    # [2/4] Populate MAP-Elites archive
    # ------------------------------------------------------------------
    print("\n[2/4] Running MAP-Elites to populate archive...")
    print("      (This may take a few minutes...)")

    archive = run_map_elites(
        fitness_func=fitness_func,
        gene_space=gene_space,
        grid_resolution=20,
        mu_range=(4.0, 20.0),
        num_initial=1000,
        num_iterations=200000,
        random_seed=42,
    )

    all_individuals = list(archive.values())
    all_individuals.sort(key=lambda x: x["fitness"], reverse=True)

    promising_elites = [ind for ind in all_individuals if ind["fitness"] > 45.0]
    num_elites_to_polish = min(120, len(promising_elites))
    top_elites_initial = promising_elites[:num_elites_to_polish]

    print("\nArchive Statistics:")
    print(f"  Total archive size: {len(all_individuals)} cells")
    print(f"  Best fitness:       {max(ind['fitness'] for ind in all_individuals):.2f}")
    print(f"  Mean fitness:       {np.mean([ind['fitness'] for ind in all_individuals]):.2f}")
    print(
        f"  Filtered to {len(top_elites_initial)} promising elites (fitness > 45) "
        f"from {len(all_individuals)} total"
    )

    print(f"\nTop {min(10, num_elites_to_polish)} Initial Elites (Before Polish):")
    for i, ind in enumerate(top_elites_initial[:10], 1):
        print(
            f"  {i}. Fitness={ind['fitness']:.2f}, "
            f"\u03bc\u2081={ind['mu_1']:.2f}, \u03bc\u2082={ind['mu_2']:.2f}"
        )
    if len(top_elites_initial) > 10:
        print(f"  ... and {len(top_elites_initial) - 10} more")

    # ------------------------------------------------------------------
    # [3/4] Polish with CMA-ES
    # ------------------------------------------------------------------
    max_fevals_per_elite = 3000
    initial_sigma = 0.15
    print(f"\n[3/4] Polishing top {num_elites_to_polish} elites with CMA-ES...")
    print(f"      Max {max_fevals_per_elite} fitness evaluations per elite")
    print(f"      Initial sigma {initial_sigma} in normalized unit-cube coordinates")
    print(f"      Using all {mp.cpu_count()} CPU cores\n")

    polished_results: list[dict[str, Any]] = []
    max_workers = min(num_elites_to_polish, mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_elite = {
            executor.submit(
                polish_single_elite_cma,
                elite_id=i,
                chromosome=elite["chromosome"],
                initial_fitness=elite["fitness"],
                fitness_func=fitness_func,
                gene_space=gene_space,
                max_fevals=max_fevals_per_elite,
                initial_sigma=initial_sigma,
            ): (i, elite)
            for i, elite in enumerate(top_elites_initial)
        }

        completed = 0
        for future in as_completed(future_to_elite):
            elite_id, elite = future_to_elite[future]
            try:
                result = future.result()
                polished_results.append(result)
                completed += 1
                if completed <= 10 or completed % 10 == 0:
                    print(
                        f"  [{completed}/{num_elites_to_polish}] Elite {elite_id + 1}: "
                        f"{result['initial_fitness']:.2f} -> {result['polished_fitness']:.2f} "
                        f"(+{result['fitness_gain']:.2f}, {result['fevals_used']} evals)"
                    )
            except Exception as exc:
                print(f"  Failed Elite {elite_id + 1}: {exc}")
                raise

    polished_results.sort(key=lambda x: x["polished_fitness"], reverse=True)

    # ------------------------------------------------------------------
    # [4/4] Save & Visualise
    # ------------------------------------------------------------------
    output_dir = Path("outputs/map_elites/cma")
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / "map_elites_cma_archive.pkl"
    with archive_path.open("wb") as f:
        pickle.dump(archive, f)
    print(f"\nSaved archive to: {archive_path}")

    results_path = output_dir / "map_elites_cma_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump(
            {
                "polished_results": polished_results,
                "top_elites_initial": top_elites_initial,
                "wavelengths": wavelengths_array,
            },
            f,
        )
    print(f"Saved results to: {results_path}")

    print("\n[4/4] Creating visualizations...")

    plot_map_elites_heatmap(
        archive=archive,
        grid_resolution=20,
        mu_range=(4.0, 20.0),
        output_path=output_dir / "map_elites_cma_heatmap.png",
    )

    for top_n in (10, 20, 50):
        plot_polished_elites(
            initial_elites=top_elites_initial,
            polished_elites=polished_results,
            wavelengths=wavelengths_array,
            parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
            output_path=output_dir / f"map_elites_cma_top{top_n}_elites.png",
            top_n=top_n,
            title_prefix="MAP-Elites + CMA-ES Polish",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    initial_fitnesses = [r["initial_fitness"] for r in polished_results]
    polished_fitnesses = [r["polished_fitness"] for r in polished_results]
    fitness_gains = [r["fitness_gain"] for r in polished_results]
    fevals = [r["fevals_used"] for r in polished_results]

    print("\n" + "=" * 80)
    print("CMA-ES POLISH RESULTS: Before vs After (Top 50)")
    print("=" * 80)
    print(
        f"\n{'Rank':<6} | {'Before':>10} | {'After':>10} | {'Gain':>8} | {'Evals':>6} | "
        f"{'\u03bc\u2081':>6} | {'\u03bc\u2082':>6}"
    )
    print("-" * 72)

    for i, result in enumerate(polished_results[:50], 1):
        mu_1, mu_2 = extract_features(result["polished_chromosome"])
        marker = " *" if result["polished_fitness"] >= 58.0 else ""
        print(
            f"{i:<6} | "
            f"{result['initial_fitness']:>10.4f} | "
            f"{result['polished_fitness']:>10.4f} | "
            f"{result['fitness_gain']:>8.4f} | "
            f"{result['fevals_used']:>6} | "
            f"{mu_1:>6.2f} | "
            f"{mu_2:>6.2f}{marker}"
        )

    print("\n" + "-" * 72)
    print("SUMMARY:")
    print("-" * 72)
    print(f"  Mean Initial Fitness:  {np.mean(initial_fitnesses):.4f}")
    print(f"  Mean Polished Fitness: {np.mean(polished_fitnesses):.4f}")
    print(f"  Mean Fitness Gain:     {np.mean(fitness_gains):.4f}")
    print(f"  Best Polished Fitness: {np.max(polished_fitnesses):.4f}")
    print(f"  Mean Evals Used:       {np.mean(fevals):.0f}")
    print(f"\n  Fitness > 50: {np.sum(np.array(polished_fitnesses) > 50)}/{len(polished_results)}")
    print(f"  Fitness > 55: {np.sum(np.array(polished_fitnesses) > 55)}/{len(polished_results)}")
    print(f"  Fitness > 58: {np.sum(np.array(polished_fitnesses) > 58)}/{len(polished_results)}")
    print(f"  Fitness > 59: {np.sum(np.array(polished_fitnesses) > 59)}/{len(polished_results)}")

    top_performers_threshold = 58.0
    top_performers = [
        r for r in polished_results if r["polished_fitness"] >= top_performers_threshold
    ]
    if len(top_performers) > 1:
        top_chromosomes = np.array([r["polished_chromosome"] for r in top_performers])
        dist_matrix = compute_population_distance_matrix(top_chromosomes, None)
        n = len(top_performers)
        distances = dist_matrix[np.triu_indices(n, k=1)]
        diversity = float(np.mean(distances))
        print(f"\n  Top Performers (fitness >= {top_performers_threshold}): {len(top_performers)}")
        print(f"  Diversity Among Top Performers: {diversity:.4f}")

    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
