#!/usr/bin/env python3
"""Module 2 Comparative Experiments Script.

Runs Basic GA and Enhanced (Niching) GA 10 times each, saving final populations.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from lwi_microbolometer_design.analysis.distance_metrics import spectral_angle_mapper
from lwi_microbolometer_design.data import load_substance_atmosphere_data
from lwi_microbolometer_design.ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    NichingConfig,
    create_ga_config,
    diversity_preserving_mutation,
)
from lwi_microbolometer_design.simulation.gaussian_parameter_to_curves import (
    gaussian_parameters_to_unit_amplitude_curves,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def save_population(population: np.ndarray, filepath: Path) -> None:
    """Save population to pickle file.

    Parameters
    ----------
    population : np.ndarray
        Population array of shape (n_individuals, n_genes)
    filepath : Path
        Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(population, f)
    logger.info(f"Saved population ({len(population)} individuals) to {filepath}")


def run_basic_ga(
    fitness_func,
    gene_space: list[dict[str, float]],
    random_seed: int,
) -> np.ndarray:
    """Run Basic GA (no niching) and return final population.

    Parameters
    ----------
    fitness_func : Callable
        Fitness function for GA
    gene_space : list[dict[str, float]]
        Gene space configuration
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Final population array
    """
    np.random.seed(random_seed)

    # Basic GA configuration (niching disabled)
    ga_config = create_ga_config(
        num_generations=2000,
        num_parents_mating=50,
        sol_per_pop=200,
        parent_selection_type="tournament",
        K_tournament=3,
        keep_elitism=5,
        crossover_type="uniform",
        crossover_probability=0.8,
        mutation_type=diversity_preserving_mutation,
        mutation_probability=0.1,
        save_best_solutions=True,
        stop_criteria="saturate_200",
        niching_enabled=False,  # Explicitly disable niching for Basic GA
        random_seed=random_seed,
    )

    ga_params = ga_config.copy()
    ga_params["num_genes"] = len(gene_space)
    ga_params["gene_space"] = gene_space
    ga_params["fitness_func"] = fitness_func

    ga_instance = AdvancedGA(**ga_params)
    ga_instance.run()

    # Return final population
    return ga_instance.population.copy()


def run_niching_ga(
    fitness_func,
    gene_space: list[dict[str, float]],
    random_seed: int,
    params_per_group: int = 2,
) -> np.ndarray:
    """Run Enhanced (Niching) GA and return final population.

    Parameters
    ----------
    fitness_func : Callable
        Fitness function for GA
    gene_space : list[dict[str, float]]
        Gene space configuration
    random_seed : int
        Random seed for reproducibility
    params_per_group : int
        Number of parameters per group for optimal pairing (default: 2 for mu, sigma)

    Returns
    -------
    np.ndarray
        Final population array
    """
    np.random.seed(random_seed)

    # Configure niching with optimal pairing for grouped parameters
    niching_config = NichingConfig(
        enabled=True,
        use_optimal_pairing=True,  # Use optimal pairing for [mu, sigma] groups
        sigma_share=1.0,  # Niche radius
        alpha=1.0,  # Sharing power parameter
        params_per_group=params_per_group,
        optimal_pairing_metric="euclidean",
    )

    # Enhanced GA configuration (niching enabled)
    ga_config = create_ga_config(
        num_generations=2000,
        num_parents_mating=50,
        sol_per_pop=200,
        parent_selection_type="tournament",
        K_tournament=3,
        keep_elitism=5,
        crossover_type="uniform",
        crossover_probability=0.8,
        mutation_type=diversity_preserving_mutation,
        mutation_probability=0.1,
        save_best_solutions=True,
        stop_criteria="saturate_200",
        niching_enabled=True,
        niching_use_optimal_pairing=True,
        niching_params_per_group=params_per_group,
        niching_sigma_share=1.0,
        niching_alpha=1.0,
        niching_optimal_pairing_metric="euclidean",
        random_seed=random_seed,
    )

    ga_params = ga_config.copy()
    ga_params["num_genes"] = len(gene_space)
    ga_params["gene_space"] = gene_space
    ga_params["fitness_func"] = fitness_func

    ga_instance = AdvancedGA(**ga_params)
    ga_instance.run()

    # Return final population
    return ga_instance.population.copy()


def main() -> None:
    """Run comparative experiments for Module 2."""
    parser = argparse.ArgumentParser(
        description="Run Basic GA and Enhanced (Niching) GA comparison experiments for Module 2."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of times to run each GA variant (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/module2/populations"),
        help="Output directory for population pickle files (default: outputs/module2/populations)",
    )
    parser.add_argument(
        "--random_seed_base",
        type=int,
        default=100,
        help="Base random seed for reproducibility (default: 100)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("      Module 2: Comparative Experiments (Basic vs. Niching GA)")
    logger.info("=" * 60)
    logger.info(f"Number of runs per variant: {args.num_runs}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed base: {args.random_seed_base}")

    # --- Data Configuration (same as Module 1) ---
    spectral_data_file = Path("data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")
    air_transmittance_file = Path("data/Test 3 - 4 White Powers/Air transmittance.xlsx")
    atmospheric_distance_ratio = 0.11
    temperature_kelvin = 293.15
    air_refractive_index = 1.0

    # Sensor configuration
    num_basis_functions = 4
    num_params_per_basis_function = 2  # mu, sigma
    param_bounds = [
        {"low": 4.0, "high": 20.0},  # mu (wavelength center)
        {"low": 0.1, "high": 4.0},  # sigma (width)
    ]
    gene_space = param_bounds * num_basis_functions

    try:
        # Load data
        logger.info("\n=== Loading Data ===")
        data = load_substance_atmosphere_data(
            spectral_data_file=spectral_data_file,
            air_transmittance_file=air_transmittance_file,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            temperature_kelvin=temperature_kelvin,
            air_refractive_index=air_refractive_index,
        )
        logger.info("✓ Data loaded successfully.")

        # Create fitness function
        logger.info("\n=== Creating Fitness Evaluator ===")
        fitness_evaluator = MinDissimilarityFitnessEvaluator(
            wavelengths=data["wavelengths"],
            emissivity_curves=data["emissivity_curves"],
            temperature_K=data["temperature_K"],
            atmospheric_distance_ratio=data["atmospheric_distance_ratio"],
            air_refractive_index=data["air_refractive_index"],
            air_transmittance=data["air_transmittance"],
            parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
            params_per_basis_function=num_params_per_basis_function,
            distance_metric=spectral_angle_mapper,
        )
        fitness_func = fitness_evaluator.fitness_func
        logger.info("✓ Fitness evaluator created successfully.")

    except Exception as e:
        logger.error(f"Failed to initialize data or fitness function: {e}")
        return

    # --- Run Basic GA experiments ---
    logger.info("\n" + "=" * 60)
    logger.info("Running Basic GA experiments...")
    logger.info("=" * 60)

    for i in range(args.num_runs):
        run_seed = args.random_seed_base + i
        logger.info(f"\n--- Basic GA Run {i + 1}/{args.num_runs} (seed: {run_seed}) ---")

        try:
            population = run_basic_ga(fitness_func, gene_space, run_seed)
            output_file = args.output_dir / f"basic_pop_run_{i + 1}.pkl"
            save_population(population, output_file)
        except Exception as e:
            logger.error(f"Error in Basic GA run {i + 1}: {e}")
            continue

    # --- Run Niching GA experiments ---
    logger.info("\n" + "=" * 60)
    logger.info("Running Enhanced (Niching) GA experiments...")
    logger.info("=" * 60)

    for i in range(args.num_runs):
        run_seed = args.random_seed_base + 1000 + i  # Use different seed range
        logger.info(f"\n--- Niching GA Run {i + 1}/{args.num_runs} (seed: {run_seed}) ---")

        try:
            population = run_niching_ga(
                fitness_func, gene_space, run_seed, params_per_group=num_params_per_basis_function
            )
            output_file = args.output_dir / f"niching_pop_run_{i + 1}.pkl"
            save_population(population, output_file)
        except Exception as e:
            logger.error(f"Error in Niching GA run {i + 1}: {e}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info("All experiments completed!")
    logger.info(f"Populations saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
