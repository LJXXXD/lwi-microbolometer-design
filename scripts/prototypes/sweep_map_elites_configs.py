#!/usr/bin/env python3
"""Parallel sweep of MAP-Elites configurations.

Tests different combinations of grid resolution, iterations, and initial population
to find optimal settings.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data import SceneConfig, load_substance_atmosphere_data
from lwi_microbolometer_design.ga import MinDissimilarityFitnessEvaluator

# Required for multiprocessing
mp.set_start_method("spawn", force=True)


def run_single_map_elites_config(
    config: dict[str, Any],
    fitness_func: Any,
    gene_space: list[dict[str, float]],
) -> dict[str, Any]:
    """Run MAP-Elites with a specific configuration.

    Parameters
    ----------
    config : dict
        Configuration with keys: grid_resolution, num_initial, num_iterations, config_name
    fitness_func : callable
        Fitness function
    gene_space : list
        Gene space bounds

    Returns
    -------
    dict
        Results including best fitness, archive size, coverage, etc.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from polish_map_elites import run_map_elites_quick

    grid_resolution = config["grid_resolution"]
    num_initial = config["num_initial"]
    num_iterations = config["num_iterations"]
    config_name = config["config_name"]

    print(
        f"  [{config_name}] Starting: grid={grid_resolution}x{grid_resolution}, "
        f"init={num_initial}, iter={num_iterations}"
    )

    archive = run_map_elites_quick(
        fitness_func=fitness_func,
        gene_space=gene_space,
        grid_resolution=grid_resolution,
        mu_range=(4.0, 20.0),
        num_initial=num_initial,
        num_iterations=num_iterations,
        random_seed=42,
    )

    # Analyze archive
    all_individuals = list(archive.values())
    fitnesses = [ind["fitness"] for ind in all_individuals]

    total_cells = grid_resolution * grid_resolution
    coverage = len(archive) / total_cells * 100

    # Count high performers
    high_performers_58 = sum(1 for f in fitnesses if f >= 58.0)
    high_performers_59 = sum(1 for f in fitnesses if f >= 59.0)

    result = {
        "config_name": config_name,
        "grid_resolution": grid_resolution,
        "num_initial": num_initial,
        "num_iterations": num_iterations,
        "archive_size": len(archive),
        "total_cells": total_cells,
        "coverage": coverage,
        "best_fitness": max(fitnesses) if fitnesses else 0.0,
        "mean_fitness": np.mean(fitnesses) if fitnesses else 0.0,
        "high_performers_58": high_performers_58,
        "high_performers_59": high_performers_59,
    }

    print(
        f"  [{config_name}] Complete: best={result['best_fitness']:.2f}, "
        f"coverage={coverage:.1f}%, ≥58={high_performers_58}, ≥59={high_performers_59}"
    )

    return result


def main() -> None:
    """Run parallel sweep of MAP-Elites configurations."""
    print("=" * 80)
    print("MAP-ELITES CONFIGURATION SWEEP")
    print("=" * 80)
    print("Testing different grid resolutions, iterations, and initial populations")

    # Load data
    print("\n[1/3] Loading data...")
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

    # Sensor configuration
    num_basis_functions = 4
    num_params_per_basis_function = 2
    param_bounds = [
        {"low": 4.0, "high": 20.0},  # mu (wavelength center)
        {"low": 0.1, "high": 4.0},  # sigma (width)
    ]
    gene_space = param_bounds * num_basis_functions

    fitness_func = MinDissimilarityFitnessEvaluator(
        wavelengths=scene.wavelengths,
        emissivity_curves=scene.emissivity_curves,
        temperature_k=scene.temperature_k,
        atmospheric_distance_ratio=scene.atmospheric_distance_ratio,
        air_refractive_index=scene.air_refractive_index,
        air_transmittance=scene.air_transmittance,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=num_params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # Define configurations to test - only the best configuration
    configs = []

    # Best configuration from sweep: 30x30 grid, 2500 initial, 1M iterations
    configs.append(
        {
            "config_name": "G30_I2.5k_Iter1000k_BEST",
            "grid_resolution": 30,
            "num_initial": 2500,
            "num_iterations": 1000000,
        }
    )

    print(f"\n[2/3] Testing {len(configs)} configurations in parallel...")
    print(f"      Using {mp.cpu_count()} CPU cores\n")

    # Run configurations in parallel
    results = []
    max_workers = min(len(configs), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(
                run_single_map_elites_config,
                config=config,
                fitness_func=fitness_func,
                gene_space=gene_space,
            ): config
            for config in configs
        }

        # Collect results
        completed = 0
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                print(f"  [{completed}/{len(configs)}] ✓ {config['config_name']}")
            except Exception as exc:
                print(f"  ✗ Failed {config['config_name']}: {exc}")
                raise

    # Sort results by best fitness
    results.sort(key=lambda x: x["best_fitness"], reverse=True)

    # Print summary table
    print("\n" + "=" * 80)
    print("CONFIGURATION SWEEP RESULTS")
    print("=" * 80)
    print(
        f"\n{'Rank':<6} | {'Config':<20} | {'Grid':<6} | {'Init':<6} | {'Iter':<8} | "
        f"{'Best Fit':>9} | {'Mean Fit':>9} | {'Coverage':>9} | {'≥58':<5} | {'≥59':<5}"
    )
    print("-" * 80)

    for i, result in enumerate(results, 1):
        print(
            f"{i:<6} | "
            f"{result['config_name']:<20} | "
            f"{result['grid_resolution']:<6} | "
            f"{result['num_initial']:<6} | "
            f"{result['num_iterations'] // 1000}k{'':<4} | "
            f"{result['best_fitness']:>9.2f} | "
            f"{result['mean_fitness']:>9.2f} | "
            f"{result['coverage']:>8.1f}% | "
            f"{result['high_performers_58']:<5} | "
            f"{result['high_performers_59']:<5}"
        )

    # Find best configuration
    best_config = results[0]
    print("\n" + "-" * 80)
    print("BEST CONFIGURATION:")
    print("-" * 80)
    print(f"  Config Name: {best_config['config_name']}")
    print(f"  Grid Resolution: {best_config['grid_resolution']}x{best_config['grid_resolution']}")
    print(f"  Initial Population: {best_config['num_initial']}")
    print(f"  Iterations: {best_config['num_iterations']}")
    print(f"  Best Fitness: {best_config['best_fitness']:.2f}")
    print(f"  Mean Fitness: {best_config['mean_fitness']:.2f}")
    print(f"  Archive Coverage: {best_config['coverage']:.1f}%")
    print(f"  High Performers (≥58): {best_config['high_performers_58']}")
    print(f"  High Performers (≥59): {best_config['high_performers_59']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
