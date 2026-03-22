#!/usr/bin/env python3
"""Diversity Stress Test: Parallel parameter search for distinct solution families.

This script runs multiple GA configurations in parallel to find settings that
produce diverse, distinct solution families rather than converging to a single optimum.
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
from lwi_microbolometer_design.ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    calculate_population_diversity,
    create_ga_config,
    diversity_preserving_mutation,
)

# Required for multiprocessing on Windows/macOS
mp.set_start_method("spawn", force=True)


def run_single_config(
    config_name: str,
    scene: SceneConfig,
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    mutation_probability: float,
    niching_sigma_share: float,
    keep_elitism: int,
    num_generations: int = 100,
    sol_per_pop: int = 100,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run a single GA configuration and return results.

    Parameters
    ----------
    config_name : str
        Name identifier for this configuration
    scene : SceneConfig
        Loaded physical scene for fitness evaluation
    gene_space : list
        Gene space bounds
    params_per_basis_function : int
        Parameters per basis function
    mutation_probability : float
        Mutation probability (0.0 to 1.0)
    niching_sigma_share : float
        Niching radius (larger = more diverse niches)
    keep_elitism : int
        Number of elite solutions to preserve
    num_generations : int
        Number of generations to run
    sol_per_pop : int
        Population size
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Results dictionary with config name, fitness, and diversity metrics
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    fitness_func = MinDissimilarityFitnessEvaluator(
        wavelengths=scene.wavelengths,
        emissivity_curves=scene.emissivity_curves,
        temperature_k=scene.temperature_k,
        atmospheric_distance_ratio=scene.atmospheric_distance_ratio,
        air_refractive_index=scene.air_refractive_index,
        air_transmittance=scene.air_transmittance,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # Track diversity history
    diversity_history: list[float] = []

    def on_generation(ga_instance) -> None:
        """Track diversity per generation."""
        diversity = calculate_population_diversity(ga_instance.population)
        diversity_history.append(diversity)

    # Create GA configuration
    ga_config = create_ga_config(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=int(sol_per_pop * 0.5),  # 50% of population as parents
        keep_elitism=keep_elitism,
        mutation_type=diversity_preserving_mutation,
        mutation_probability=mutation_probability,
        niching_enabled=True,
        niching_sigma_share=niching_sigma_share,
        niching_alpha=1.0,
        random_seed=random_seed,
    )

    # Add required parameters
    ga_config["num_genes"] = len(gene_space)
    ga_config["gene_space"] = gene_space
    ga_config["fitness_func"] = fitness_func
    ga_config["on_generation"] = on_generation

    # Run GA
    ga = AdvancedGA(**ga_config)
    ga.run()

    # Extract results
    best_chromosome, best_fitness, _best_idx = ga.best_solution()
    final_fitness_scores = ga.last_generation_fitness
    final_diversity = diversity_history[-1] if diversity_history else 0.0

    # Calculate number of distinct high-quality solutions (fitness > 45)
    high_quality_threshold = 45.0
    high_quality_count = int(np.sum(final_fitness_scores >= high_quality_threshold))

    # Calculate fitness spread (std)
    fitness_std = float(np.std(final_fitness_scores))

    return {
        "config_name": config_name,
        "best_fitness": float(best_fitness),
        "final_diversity": float(final_diversity),
        "high_quality_count": high_quality_count,
        "fitness_std": fitness_std,
        "mean_fitness": float(np.mean(final_fitness_scores)),
        "final_population": ga.population,
        "final_fitness_scores": final_fitness_scores,
    }


def main() -> None:
    """Run diversity stress test with multiple configurations."""
    print("=" * 80)
    print("DIVERSITY STRESS TEST: Finding Configurations for Distinct Solution Families")
    print("=" * 80)

    # Load data
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

    # Sensor configuration
    num_basis_functions = 4
    num_params_per_basis_function = 2
    param_bounds = [
        {"low": 4.0, "high": 20.0},  # mu (wavelength center)
        {"low": 0.1, "high": 4.0},  # sigma (width)
    ]
    gene_space = param_bounds * num_basis_functions

    # Define test configurations spanning conservative to chaotic
    # Strategy: Vary mutation_probability, niching_sigma_share, and keep_elitism
    configurations = [
        {
            "name": "Conservative",
            "mutation_probability": 0.05,
            "niching_sigma_share": 0.3,
            "keep_elitism": 10,
            "description": "Low mutation, tight niching, high elitism",
        },
        {
            "name": "Baseline",
            "mutation_probability": 0.1,
            "niching_sigma_share": 0.5,
            "keep_elitism": 5,
            "description": "Balanced default settings",
        },
        {
            "name": "High Mutation",
            "mutation_probability": 0.3,
            "niching_sigma_share": 0.5,
            "keep_elitism": 2,
            "description": "High exploration, low elitism",
        },
        {
            "name": "Strong Niching",
            "mutation_probability": 0.1,
            "niching_sigma_share": 2.0,
            "keep_elitism": 5,
            "description": "Large niche radius for diverse solutions",
        },
        {
            "name": "Balanced Aggressive",
            "mutation_probability": 0.2,
            "niching_sigma_share": 1.5,
            "keep_elitism": 1,
            "description": "Moderate mutation + large niches + low elitism",
        },
        {
            "name": "Chaotic",
            "mutation_probability": 0.5,
            "niching_sigma_share": 3.0,
            "keep_elitism": 0,
            "description": "Maximum exploration, no elitism",
        },
    ]

    print(f"\n[2/4] Running {len(configurations)} configurations in parallel...")
    print(f"      Each config: {100} generations, population size 100\n")

    # Run configurations in parallel
    results: list[dict[str, Any]] = []
    max_workers = min(len(configurations), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(
                run_single_config,
                config_name=config["name"],
                scene=scene,
                gene_space=gene_space,
                params_per_basis_function=num_params_per_basis_function,
                mutation_probability=config["mutation_probability"],
                niching_sigma_share=config["niching_sigma_share"],
                keep_elitism=config["keep_elitism"],
                num_generations=100,  # Quick but shows trends
                sol_per_pop=100,
                random_seed=42,  # Same seed for fair comparison
            ): config
            for config in configurations
        }

        # Collect results as they complete
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                print(f"  ✓ Completed: {config['name']}")
            except Exception as exc:
                print(f"  ✗ Failed: {config['name']} - {exc}")
                raise

    # Sort by diversity (descending) to see which produces most diverse solutions
    results.sort(key=lambda x: x["final_diversity"], reverse=True)

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS: Comparison Table")
    print("=" * 80)
    print(
        f"\n{'Config Name':<20} | {'Best Fitness':>12} | {'Diversity':>10} | {'High-Quality':>12} | {'Fitness Std':>12}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['config_name']:<20} | "
            f"{result['best_fitness']:>12.4f} | "
            f"{result['final_diversity']:>10.4f} | "
            f"{result['high_quality_count']:>12} | "
            f"{result['fitness_std']:>12.4f}"
        )

    # Print detailed configuration settings
    print("\n" + "=" * 80)
    print("CONFIGURATION DETAILS")
    print("=" * 80)
    for config in configurations:
        print(f"\n{config['name']}:")
        print(f"  Description: {config['description']}")
        print(f"  Mutation Probability: {config['mutation_probability']}")
        print(f"  Niching Sigma Share: {config['niching_sigma_share']}")
        print(f"  Keep Elitism: {config['keep_elitism']}")

    # Find best configuration for diversity
    best_diversity_config = results[0]
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nBest configuration for diversity: {best_diversity_config['config_name']}")
    print(f"  Final Diversity Score: {best_diversity_config['final_diversity']:.4f}")
    print(f"  Best Fitness: {best_diversity_config['best_fitness']:.4f}")
    print(f"  High-Quality Solutions: {best_diversity_config['high_quality_count']}")
    print(f"  Fitness Standard Deviation: {best_diversity_config['fitness_std']:.4f}")

    # Find configuration with best balance
    # Score = diversity * (high_quality_count / 10) * (fitness_std / 5)
    for result in results:
        balance_score = (
            result["final_diversity"]
            * (result["high_quality_count"] / 10.0)
            * (result["fitness_std"] / 5.0)
        )
        result["balance_score"] = balance_score

    results.sort(key=lambda x: x["balance_score"], reverse=True)
    best_balance_config = results[0]

    print(f"\nBest balanced configuration: {best_balance_config['config_name']}")
    print(f"  Balance Score: {best_balance_config['balance_score']:.4f}")
    print(f"  Final Diversity: {best_balance_config['final_diversity']:.4f}")
    print(f"  Best Fitness: {best_balance_config['best_fitness']:.4f}")
    print(f"  High-Quality Solutions: {best_balance_config['high_quality_count']}")

    print("\n" + "=" * 80)
    print("STRESS TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
