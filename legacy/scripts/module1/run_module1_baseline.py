#!/usr/bin/env python3
"""Module 1 Baseline Experiment Script.

Runs Basic GA (no niching) 20 times and logs best_fitness_score and
total_runtime_seconds to module1_baseline_results.csv.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lwi_microbolometer_design.analysis.distance_metrics import spectral_angle_mapper
from lwi_microbolometer_design.data import load_substance_atmosphere_data
from lwi_microbolometer_design.ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    create_ga_config,
    diversity_preserving_mutation,
)
from lwi_microbolometer_design.simulation.gaussian_parameter_to_curves import (
    gaussian_parameters_to_unit_amplitude_curves,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_NUM_RUNS = 20
DEFAULT_RANDOM_SEED_BASE = 42

# Data configuration
SPECTRAL_DATA_FILE = Path("data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")
AIR_TRANSMITTANCE_FILE = Path("data/Test 3 - 4 White Powers/Air transmittance.xlsx")
ATMOSPHERIC_DISTANCE_RATIO = 0.11
TEMPERATURE_KELVIN = 293.15
AIR_REFRACTIVE_INDEX = 1.0

# Sensor configuration
NUM_BASIS_FUNCTIONS = 4
PARAMS_PER_BASIS_FUNCTION = 2  # mu and sigma for each Gaussian
PARAM_BOUNDS = [
    {"low": 4.0, "high": 20.0},  # mu (wavelength center)
    {"low": 0.1, "high": 4.0},  # sigma (width)
]

# GA configuration (Basic GA - no niching)
# Using optimized hyperparameters from tuning results that achieve high fitness
# Key differences from basic defaults:
# - num_parents_mating: 60 (vs 50) - more parents = better exploration
# - mutation_probability: 0.05 (vs 0.1) - lower mutation preserves good solutions
# - crossover_probability: 0.9 (vs 0.8) - higher crossover = more exploration
# - K_tournament: 5 (vs 3) - larger tournament = stronger selection pressure
GA_CONFIG = {
    "num_generations": 2000,
    "num_parents_mating": 60,  # Optimized: was 50
    "sol_per_pop": 200,
    "parent_selection_type": "tournament",
    "K_tournament": 5,  # Optimized: was 3
    "keep_elitism": 5,
    "crossover_type": "uniform",
    "crossover_probability": 0.9,  # Optimized: was 0.8
    "mutation_type": diversity_preserving_mutation,
    "mutation_probability": 0.05,  # Optimized: was 0.1
    "save_best_solutions": True,
    "stop_criteria": "saturate_200",
    "niching_enabled": False,  # Basic GA - no niching (for Module 1 comparison)
}


def run_single_baseline_run(
    data: dict,
    gene_space: list[dict[str, float]],
    run_idx: int,
    random_seed: int,
) -> dict[str, float]:
    """Run a single Basic GA run and return results.

    Parameters
    ----------
    data : dict
        Loaded spectral data dictionary
    gene_space : list[dict[str, float]]
        Gene space bounds
    run_idx : int
        Run index (for logging)
    random_seed : int
        Random seed for this run

    Returns
    -------
    dict[str, float]
        Dictionary with 'best_fitness_score' and 'total_runtime_seconds'
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Create fitness function
    fitness_func = MinDissimilarityFitnessEvaluator(
        wavelengths=data["wavelengths"],
        emissivity_curves=data["emissivity_curves"],
        temperature_K=data["temperature_K"],
        atmospheric_distance_ratio=data["atmospheric_distance_ratio"],
        air_refractive_index=data["air_refractive_index"],
        air_transmittance=data["air_transmittance"],
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=PARAMS_PER_BASIS_FUNCTION,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # Build GA config
    # CRITICAL: Match demo_ga.py behavior exactly
    # demo_ga.py calls create_ga_config() which defaults to niching_enabled=True
    # For Module 1, we want Basic GA (no niching), but we need to match all other defaults
    ga_config_dict = create_ga_config(
        num_generations=GA_CONFIG["num_generations"],
        num_parents_mating=GA_CONFIG["num_parents_mating"],
        sol_per_pop=GA_CONFIG["sol_per_pop"],
        parent_selection_type=GA_CONFIG["parent_selection_type"],
        K_tournament=GA_CONFIG["K_tournament"],
        keep_elitism=GA_CONFIG["keep_elitism"],
        crossover_type=GA_CONFIG["crossover_type"],
        crossover_probability=GA_CONFIG["crossover_probability"],
        mutation_type=GA_CONFIG["mutation_type"],
        mutation_probability=GA_CONFIG["mutation_probability"],
        save_best_solutions=GA_CONFIG["save_best_solutions"],
        stop_criteria=GA_CONFIG["stop_criteria"],
        niching_enabled=GA_CONFIG["niching_enabled"],  # False for Basic GA
        # Match demo_ga.py defaults for niching params (used when enabled)
        niching_use_optimal_pairing=True,
        niching_params_per_group=PARAMS_PER_BASIS_FUNCTION,
        niching_sigma_share=0.5,  # demo_ga.py default
        niching_alpha=0.5,  # demo_ga.py default
        niching_optimal_pairing_metric="euclidean",
        random_seed=random_seed,
    )

    # Add runtime-specific parameters
    ga_params = ga_config_dict.copy()
    ga_params["num_genes"] = len(gene_space)
    ga_params["gene_space"] = gene_space
    ga_params["fitness_func"] = fitness_func

    # Run GA with timing
    start_time = time.time()
    ga = AdvancedGA(**ga_params)
    ga.run()
    end_time = time.time()
    total_runtime_seconds = end_time - start_time

    # Extract best fitness
    best_chromosome, best_fitness, _best_idx = ga.best_solution()

    logger.info(
        f"Run {run_idx + 1}: best_fitness={best_fitness:.4f}, runtime={total_runtime_seconds:.2f}s"
    )

    return {
        "run_idx": run_idx + 1,
        "best_fitness_score": float(best_fitness),
        "total_runtime_seconds": total_runtime_seconds,
    }


def main() -> None:
    """Run Module 1 baseline experiments."""
    parser = argparse.ArgumentParser(description="Run Basic GA baseline experiments for Module 1.")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Number of runs (default: {DEFAULT_NUM_RUNS})",
    )
    parser.add_argument(
        "--random-seed-base",
        type=int,
        default=DEFAULT_RANDOM_SEED_BASE,
        help=f"Base random seed (default: {DEFAULT_RANDOM_SEED_BASE})",
    )
    parser.add_argument(
        "--enable-niching",
        action="store_true",
        help="Enable niching (like demo_ga.py default) to test if that explains 58-59 scores",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("module1_baseline_results.csv"),
        help="Output CSV file (default: module1_baseline_results.csv)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Module 1: Baseline GA Experiments")
    logger.info("=" * 60)

    if args.enable_niching:
        logger.warning("=" * 60)
        logger.warning("WARNING: NICHING ENABLED - This is for testing only!")
        logger.warning("demo_ga.py gets 58-59 because it uses niching by default.")
        logger.warning("This test will show if niching is what makes the difference.")
        logger.warning("=" * 60)
        GA_CONFIG["niching_enabled"] = True
    else:
        logger.info(
            "NOTE: Using Basic GA (no niching) to compare against v1 grid search.\n"
            "DEMO_GA.PY GETS 58-59 BECAUSE IT USES NICHING (enabled by default).\n"
            "To test if niching is the difference, run with --enable-niching flag."
        )

    # Load data
    logger.info("\nLoading spectral data...")
    data = load_substance_atmosphere_data(
        spectral_data_file=SPECTRAL_DATA_FILE,
        air_transmittance_file=AIR_TRANSMITTANCE_FILE,
        atmospheric_distance_ratio=ATMOSPHERIC_DISTANCE_RATIO,
        temperature_kelvin=TEMPERATURE_KELVIN,
        air_refractive_index=AIR_REFRACTIVE_INDEX,
    )

    # Handle multi-condition data (should be single condition)
    if isinstance(data, list):
        if len(data) > 1:
            logger.warning("Multi-condition data detected, using first condition only")
        data = data[0]

    # Create gene space
    gene_space = PARAM_BOUNDS * NUM_BASIS_FUNCTIONS
    logger.info(f"Gene space: {len(gene_space)} genes ({NUM_BASIS_FUNCTIONS} basis functions)")

    # Run experiments
    logger.info(f"\nRunning {args.num_runs} baseline GA runs...")
    results = []

    for run_idx in range(args.num_runs):
        random_seed = args.random_seed_base + run_idx
        result = run_single_baseline_run(data, gene_space, run_idx, random_seed)
        results.append(result)

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    df_results.to_csv(args.output_csv, index=False)
    logger.info(f"\nResults saved to: {args.output_csv}")

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics")
    logger.info("=" * 60)
    logger.info(f"Mean best_fitness_score: {df_results['best_fitness_score'].mean():.4f}")
    logger.info(f"Std best_fitness_score: {df_results['best_fitness_score'].std():.4f}")
    logger.info(f"Min best_fitness_score: {df_results['best_fitness_score'].min():.4f}")
    logger.info(f"Max best_fitness_score: {df_results['best_fitness_score'].max():.4f}")
    logger.info(f"\nMean total_runtime_seconds: {df_results['total_runtime_seconds'].mean():.2f}")
    logger.info(f"Std total_runtime_seconds: {df_results['total_runtime_seconds'].std():.2f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
