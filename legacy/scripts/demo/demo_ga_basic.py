#!/usr/bin/env python3
"""Basic GA Demonstration Script - Standard PyGAD Only.

This script demonstrates a simple GA run using ONLY standard PyGAD features:
- No AdvancedGA (uses pygad.GA directly)
- No custom mutations (uses PyGAD built-in mutation operators)
- No niching/fitness sharing
- No optimal pairing distance
- Just plain vanilla GA optimization

This is useful for:
1. Baseline comparisons
2. Debugging
3. Understanding pure GA performance
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pygad

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data import load_substance_atmosphere_data
from lwi_microbolometer_design.ga import MinDissimilarityFitnessEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run basic GA demonstration."""
    parser = argparse.ArgumentParser(
        description="Basic GA demonstration using only standard PyGAD (no enhancements)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2000,
        help="Number of generations to run (default: 2000)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=200,
        help="Population size (default: 200)",
    )
    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.1,
        help="Mutation probability (default: 0.1)",
    )
    parser.add_argument(
        "--mutation-type",
        type=str,
        default="random",
        choices=["random", "adaptive", "swap", "inversion", "scramble"],
        help="Mutation type (default: random)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("      Basic GA Demonstration (Standard PyGAD Only)")
    logger.info("=" * 60)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Population size: {args.population_size}")
    logger.info(f"Mutation type: {args.mutation_type}")
    logger.info(f"Mutation probability: {args.mutation_probability}")
    logger.info("=" * 60)

    # Set random seed
    np.random.seed(args.seed)

    # Data configuration
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
        loaded = load_substance_atmosphere_data(
            spectral_data_file=spectral_data_file,
            air_transmittance_file=air_transmittance_file,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            temperature_kelvin=temperature_kelvin,
            air_refractive_index=air_refractive_index,
        )
        # Handle multi-condition data
        if isinstance(loaded, list):
            if len(loaded) > 1:
                logger.warning("Multi-condition data detected, using first condition only")
            scene = loaded[0]
        else:
            scene = loaded
        logger.info("✓ Data loaded successfully.")

        # Create fitness function
        logger.info("\n=== Creating Fitness Evaluator ===")
        fitness_evaluator = MinDissimilarityFitnessEvaluator(
            wavelengths=scene.wavelengths,
            emissivity_curves=scene.emissivity_curves,
            temperature_k=scene.temperature_k,
            atmospheric_distance_ratio=scene.atmospheric_distance_ratio,
            air_refractive_index=scene.air_refractive_index,
            air_transmittance=scene.air_transmittance,
            parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
            params_per_basis_function=num_params_per_basis_function,
            distance_metric=spectral_angle_mapper,
        )
        fitness_func = fitness_evaluator.fitness_func
        logger.info("✓ Fitness evaluator created successfully.")

        # Track fitness over generations
        fitness_history: list[float] = []

        def on_generation(ga_instance: pygad.GA) -> None:
            """Track fitness per generation."""
            best_fitness = float(np.max(ga_instance.last_generation_fitness))
            mean_fitness = float(np.mean(ga_instance.last_generation_fitness))
            fitness_history.append(best_fitness)

            # Log every 100 generations
            if ga_instance.generations_completed % 100 == 0:
                logger.info(
                    f"Gen {ga_instance.generations_completed}: "
                    f"Best={best_fitness:.2f}, "
                    f"Mean={mean_fitness:.2f}"
                )

        # Configure basic PyGAD (no AdvancedGA features)
        logger.info("\n=== Running Basic GA ===")
        ga_instance = pygad.GA(
            num_generations=args.generations,
            num_parents_mating=int(args.population_size * 0.3),  # 30% of population
            fitness_func=fitness_func,
            sol_per_pop=args.population_size,
            num_genes=len(gene_space),
            gene_space=gene_space,
            parent_selection_type="tournament",
            K_tournament=3,
            keep_parents=1,  # Keep best parent
            keep_elitism=5,  # Keep top 5 elite solutions
            crossover_type="uniform",
            crossover_probability=0.8,
            mutation_type=args.mutation_type,  # Use standard PyGAD mutation
            mutation_probability=args.mutation_probability,
            mutation_by_replacement=False,
            save_best_solutions=True,
            on_generation=on_generation,
            random_seed=args.seed,
        )

        # Run optimization
        ga_instance.run()

        # Get best solution
        best_chromosome, best_fitness, best_idx = ga_instance.best_solution()

        logger.info("\n" + "=" * 60)
        logger.info("     BASIC GA COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nBest fitness achieved: {best_fitness:.4f}")
        logger.info(f"Best chromosome: {best_chromosome}")
        logger.info("\nFitness history:")
        logger.info(f"  Initial best: {fitness_history[0]:.2f}")
        logger.info(f"  Final best: {fitness_history[-1]:.2f}")
        logger.info(f"  Improvement: {fitness_history[-1] - fitness_history[0]:.2f}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"GA demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
