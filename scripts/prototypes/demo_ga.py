#!/usr/bin/env python3
"""Comprehensive GA Demonstration Script.

This script demonstrates the complete GA workflow:
1. Loads optimal hyperparameters from tuning results
2. Runs GA with optimized configuration
3. Visualizes multiple distinct optimal solutions
4. Creates publication-ready visualizations

This is the MVP deliverable for demonstrating GA optimization of microbolometer sensors.
"""

import argparse
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
    load_ga_configuration_from_csv,
    visualize_ga_results,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Plotting style
plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


# Data loading function moved to lwi_microbolometer_design.data.substance_atmosphere_data
# Imported above: load_substance_atmosphere_data


# Configuration functions moved to lwi_microbolometer_design.ga.ga_configuration
# Imported above: load_ga_configuration_from_csv, create_ga_config


def run_optimized_ga(
    scene: SceneConfig,
    gene_space: list[dict[str, float]],
    ga_config: dict,
    params_per_basis_function: int,
    high_fitness_threshold: float = 50.0,
    good_fitness_threshold: float = 45.0,
) -> dict:
    """Run GA with optimized configuration.

    Parameters
    ----------
    high_fitness_threshold : float, optional
        Threshold for tracking high-fitness solutions during evolution (default: 50.0)
    good_fitness_threshold : float, optional
        Threshold for reporting good-quality solutions at completion (default: 45.0)
    """
    logger.info("\n=== Running Optimized GA ===")

    # Create fitness function using class-based evaluator (pickleable for multiprocessing)
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

    # Track detailed metrics per generation for multimodal analysis
    mean_fitness_history: list[float] = []
    diversity_history: list[float] = []
    high_fitness_count_history: list[int] = []
    fitness_std_history: list[float] = []

    def on_generation(ga_instance) -> None:
        """Track detailed metrics per generation."""
        # Mean fitness
        mean_fitness = float(np.mean(ga_instance.last_generation_fitness))
        mean_fitness_history.append(mean_fitness)

        # Population diversity
        diversity = calculate_population_diversity(ga_instance.population)
        diversity_history.append(diversity)

        # Count of high-fitness solutions
        high_fitness_count = int(
            np.sum(ga_instance.last_generation_fitness >= high_fitness_threshold)
        )
        high_fitness_count_history.append(high_fitness_count)

        # Fitness standard deviation (spread)
        fitness_std = np.std(ga_instance.last_generation_fitness)
        fitness_std_history.append(fitness_std)

        # Log every 100 generations
        if ga_instance.generations_completed % 100 == 0:
            logger.info(
                f"Gen {ga_instance.generations_completed}: "
                f"Best={ga_instance.best_solutions_fitness[-1]:.2f}, "
                f"Mean={mean_fitness:.2f}, "
                f"Diversity={diversity:.4f}, "
                f"High-fitness count={high_fitness_count}, "
                f"Std={fitness_std:.2f}"
            )

    # Start with ga_config
    ga_params = ga_config.copy()

    # Add runtime-specific parameters
    ga_params["num_genes"] = len(gene_space)
    ga_params["gene_space"] = gene_space
    ga_params["fitness_func"] = fitness_func
    ga_params["on_generation"] = on_generation

    # Ensure we use the custom mutation operator regardless of CSV defaults
    ga_params["mutation_type"] = diversity_preserving_mutation
    # PyGAD expects a scalar mutation_probability; coerce lists/tuples to a scalar
    mutation_probability_value = ga_params.get("mutation_probability", 0.1)
    if isinstance(mutation_probability_value, (list, tuple, np.ndarray)):
        try:
            ga_params["mutation_probability"] = float(mutation_probability_value[-1])
        except Exception:
            float_list = [float(x) for x in mutation_probability_value]
            ga_params["mutation_probability"] = float(np.mean(np.array(float_list)))
    else:
        ga_params["mutation_probability"] = float(mutation_probability_value)

    ga = AdvancedGA(**ga_params)

    # Run optimization
    ga.run()

    # Get best solution
    best_chromosome, best_fitness, _best_idx = ga.best_solution()

    # Get fitness scores
    final_fitness_scores = ga.last_generation_fitness

    # Build result dictionary with enhanced metrics
    result = {
        "best_fitness": best_fitness,
        "best_chromosome": best_chromosome,
        "final_population": ga.population,
        "final_fitness_scores": final_fitness_scores,
        "diversity_history": diversity_history
        if diversity_history
        else [diversity_history[-1] if diversity_history else 0.0],
        "best_fitness_history": ga.best_solutions_fitness
        if hasattr(ga, "best_solutions_fitness")
        else [best_fitness],
        "mean_fitness_history": mean_fitness_history
        if mean_fitness_history
        else [float(np.mean(np.array(final_fitness_scores)))],
        "high_fitness_count_history": high_fitness_count_history,
        "fitness_std_history": fitness_std_history,
    }

    logger.info("GA completed successfully!")
    logger.info(f"  Best fitness: {result['best_fitness']:.4f}")
    logger.info(f"  Final diversity: {result['diversity_history'][-1]:.4f}")
    good_solutions = np.sum(result["final_fitness_scores"] >= good_fitness_threshold)
    logger.info(f"  High-quality solutions: {good_solutions}")

    return result


# Visualization functions moved to lwi_microbolometer_design.ga.visualization
# Imported above: visualize_ga_results


def run_single_experiment(
    config_info: dict[str, Any],
    scene: SceneConfig,
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    high_fitness_threshold: float = 50.0,
    good_fitness_threshold: float = 45.0,
) -> dict[str, Any]:
    """Run a single GA configuration in a separate process."""
    try:
        result = run_optimized_ga(
            scene,
            gene_space,
            config_info["config"],
            params_per_basis_function,
            high_fitness_threshold,
            good_fitness_threshold,
        )
        return {"name": config_info["name"], "config": config_info["config"], "result": result}
    except Exception as e:
        logger.error(f"Error in {config_info['name']}: {e}")
        raise


def run_multiple_experiments(
    scene: SceneConfig,
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    random_seed: int | None = None,
    high_fitness_threshold: float = 50.0,
    good_fitness_threshold: float = 45.0,
) -> dict[str, dict[str, Any]]:
    """
    Run multiple GA configurations for comparison.

    Executes multiple GA experiments in parallel and returns results
    for performance comparison.

    Parameters
    ----------
    scene : SceneConfig
        Loaded physical scene for fitness evaluation
    gene_space : list
        Gene space bounds for GA
    params_per_basis_function : int
        Number of parameters per basis function
    random_seed : int | None, optional
        Random seed for reproducibility (default: None)
    high_fitness_threshold : float, optional
        Threshold for high-fitness solutions (default: 50.0)
    good_fitness_threshold : float, optional
        Threshold for good-quality solutions (default: 45.0)

    Returns
    -------
    dict
        Dictionary mapping experiment names to their results
    """
    # Base configuration - most experiments share these values
    # Use explicit parameters instead of dict unpacking to satisfy type checker
    base_num_generations = 2000
    base_mutation_type: str | None = "adaptive"  # Use PyGAD built-in for comparison

    # Define experiment configurations using create_ga_config()
    configurations: list[dict[str, Any]] = [
        {
            "name": "Baseline GA (No Niching)",
            "config": create_ga_config(
                num_generations=base_num_generations,
                mutation_type=base_mutation_type,
                mutation_probability=0.1,
                niching_enabled=False,
            ),
        },
        {
            "name": "Advanced GA (With Niching)",
            "config": create_ga_config(
                num_generations=base_num_generations,
                mutation_type=base_mutation_type,
                mutation_probability=0.1,
                niching_enabled=True,
                niching_sigma_share=0.5,
                niching_alpha=0.5,
            ),
        },
        {
            "name": "Advanced GA (Moderate Niching)",
            "config": create_ga_config(
                num_generations=base_num_generations,
                mutation_type=base_mutation_type,
                mutation_probability=0.1,
                niching_enabled=True,
                niching_sigma_share=1.0,
                niching_alpha=0.5,
            ),
        },
        {
            "name": "High Mutation Rate",
            "config": create_ga_config(
                num_generations=base_num_generations,
                mutation_type=base_mutation_type,
                mutation_probability=0.2,
                niching_enabled=False,
            ),
        },
        {
            "name": "High Diversity Niching",
            "config": create_ga_config(
                num_generations=base_num_generations,
                mutation_type=base_mutation_type,
                mutation_probability=0.15,
                niching_enabled=True,
                niching_sigma_share=2.0,
                niching_alpha=0.5,
            ),
        },
        {
            "name": "Ultra-Diverse Niching (Multi-Pattern)",
            "config": create_ga_config(
                num_generations=base_num_generations,
                mutation_type=base_mutation_type,
                mutation_probability=0.15,
                niching_enabled=True,
                niching_sigma_share=3.0,
                niching_alpha=1.0,
            ),
        },
    ]

    # Add random_seed to all configs if provided
    if random_seed is not None:
        for config_dict in configurations:
            config_dict["config"]["random_seed"] = random_seed

    logger.info(f"\n=== Running {len(configurations)} Experimental Configurations ===")
    max_workers = min(len(configurations), mp.cpu_count())
    logger.info(f"Using {max_workers} parallel workers")

    results: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(
                run_single_experiment,
                config_info,
                scene,
                gene_space,
                params_per_basis_function,
                high_fitness_threshold,
                good_fitness_threshold,
            ): config_info
            for config_info in configurations
        }

        # Collect results as they complete
        for future in as_completed(future_to_config):
            config_info = future_to_config[future]
            try:
                result_data = future.result()
                results[result_data["name"]] = result_data["result"]

                # Log results
                logger.info(f"\n✓ Completed {result_data['name']}")
                logger.info(f"  Best fitness: {result_data['result']['best_fitness']:.4f}")
                logger.info(
                    f"  Final diversity: {result_data['result']['diversity_history'][-1]:.4f}"
                )
                good_count = np.sum(
                    result_data["result"]["final_fitness_scores"] >= good_fitness_threshold
                )
                logger.info(f"  High-quality solutions: {good_count}")

            except Exception as exc:
                logger.error(f"Configuration {config_info['name']} generated an exception: {exc}")
                raise

    # Final comparison summary
    logger.info("\n=== Experiment Comparison Summary ===")
    for name, result in results.items():
        logger.info(
            f"{name}: "
            f"fitness={result['best_fitness']:.4f}, "
            f"diversity={result['diversity_history'][-1]:.4f}, "
            f"high-quality={np.sum(result['final_fitness_scores'] >= good_fitness_threshold)}"
        )

    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(
        description="Comprehensive GA Demonstration for Microbolometer Sensor Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python demo_ga_comprehensive.py

  # Run with different random seed
  python demo_ga_comprehensive.py --seed 123

  # Run with custom fitness thresholds
  python demo_ga_comprehensive.py --high-fitness-threshold 55.0 --good-fitness-threshold 48.0

  # Run multiple experiments for comparison
  python demo_ga_comprehensive.py --multiple-experiments

  # Suppress warnings (if needed)
  python -W ignore::FutureWarning demo_ga_comprehensive.py
  python -W ignore demo_ga_comprehensive.py  # Suppress all warnings
        """,
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--no-seed", action="store_true", help="Disable random seed (non-reproducible runs)"
    )

    parser.add_argument(
        "--multiple-experiments",
        action="store_true",
        default=False,
        help="Run multiple experimental configurations for comparison (default: False)",
    )

    parser.add_argument(
        "--generations", type=int, default=2000, help="Number of generations to run (default: 2000)"
    )

    parser.add_argument(
        "--high-fitness-threshold",
        type=float,
        default=50.0,
        help="Threshold for high-fitness solutions in fitness tracking (default: 50.0)",
    )

    parser.add_argument(
        "--good-fitness-threshold",
        type=float,
        default=45.0,
        help="Threshold for good-quality solutions in reporting (default: 45.0)",
    )

    return parser.parse_args()


def main() -> None:
    """Run main demonstration."""
    # Parse command-line arguments
    args = parse_arguments()

    # Determine configuration from arguments
    use_random_seed = not args.no_seed
    random_seed = args.seed if use_random_seed else None
    run_multi_flag = args.multiple_experiments
    high_fitness_threshold = args.high_fitness_threshold
    good_fitness_threshold = args.good_fitness_threshold

    logger.info("=" * 60)
    logger.info("      Comprehensive GA Demonstration")
    logger.info("=" * 60)
    logger.info(f"High fitness threshold: {high_fitness_threshold}")
    logger.info(f"Good fitness threshold: {good_fitness_threshold}")

    # Set random seed for reproducibility
    if use_random_seed and random_seed is not None:
        np.random.seed(random_seed)
        logger.info(f"Random seed: {random_seed} (reproducible)")
    else:
        logger.info("Random seed: None (non-reproducible)")

    # Data configuration
    spectral_data_file = Path("data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")
    air_transmittance_file = Path("data/Test 3 - 4 White Powers/Air transmittance.xlsx")
    atmospheric_distance_ratio = 0.11
    temperature_kelvin = 293.15
    air_refractive_index = 1.0

    # GA configuration file (optional)
    ga_config_file = Path("outputs/tuning/tuning_results_20251020_175031.csv")

    # Sensor configuration
    num_basis_functions = 4
    num_params_per_basis_function = (
        2  # Each Gaussian basis function has 2 params: mu (mean) and sigma (std)
    )
    # Parameter bounds for Gaussian basis functions (mu, sigma)
    param_bounds = [
        {"low": 4.0, "high": 20.0},  # mu (wavelength center)
        {"low": 0.1, "high": 4.0},  # sigma (width)
    ]

    # Create output directory
    output_dir = Path("outputs/ga/comprehensive_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        logger.info("\n=== Step 1: Loading Data ===")
        loaded_data = load_substance_atmosphere_data(
            spectral_data_file=spectral_data_file,
            air_transmittance_file=air_transmittance_file,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            temperature_kelvin=temperature_kelvin,
            air_refractive_index=air_refractive_index,
        )

        # Handle case where data might be a list (multi-condition)
        if isinstance(loaded_data, list):
            if len(loaded_data) > 1:
                logger.warning("Multi-condition data detected, using first condition only")
            scene: SceneConfig = loaded_data[0]
        else:
            scene = loaded_data

        gene_space = param_bounds * num_basis_functions

        if run_multi_flag:
            # Run multiple experiments for comparison
            logger.info("\n=== Running Multiple Experiments ===")
            all_results = run_multiple_experiments(
                scene,
                gene_space,
                num_params_per_basis_function,
                random_seed,
                high_fitness_threshold,
                good_fitness_threshold,
            )

            # Use the best result for visualization
            best_exp_name = max(all_results.keys(), key=lambda k: all_results[k]["best_fitness"])
            logger.info(f"\nBest experiment: {best_exp_name}")
            ga_result = all_results[best_exp_name]

            # Save comparison summary
            comparison_summary = {
                "timestamp": datetime.now().isoformat(),
                "experiments": {
                    name: {
                        "best_fitness": float(result["best_fitness"]),
                        "final_diversity": float(result["diversity_history"][-1]),
                        "high_quality_solutions": int(
                            np.sum(result["final_fitness_scores"] >= good_fitness_threshold)
                        ),
                    }
                    for name, result in all_results.items()
                },
                "best_experiment": best_exp_name,
            }

            with open(output_dir / "00_experiment_comparison.json", "w") as f:
                json.dump(comparison_summary, f, indent=2)
        else:
            # Run single optimized GA
            logger.info("\n=== Step 2: Loading Optimal Configuration ===")
            ga_config = load_ga_configuration_from_csv(ga_config_file)

            # If no config file found, use defaults
            if ga_config is None:
                logger.info("Using default GA configuration")
                if random_seed is not None:
                    np.random.seed(random_seed)
                    logger.info(f"Random seed set to: {random_seed} (reproducible)")
                ga_config = create_ga_config(num_generations=args.generations)
            else:
                # Override generations from command line if provided
                ga_config["num_generations"] = args.generations

            logger.info("\n=== Step 3: Running Optimized GA ===")
            ga_result = run_optimized_ga(
                scene,
                gene_space,
                ga_config,
                num_params_per_basis_function,
                high_fitness_threshold,
                good_fitness_threshold,
            )

        # Visualize results
        logger.info("\n=== Step 4: Visualizing Results ===")
        visualize_ga_results(ga_result, scene, output_dir, high_fitness_threshold)

        # Save summary
        logger.info("\n=== Step 5: Saving Summary ===")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "best_fitness": float(ga_result["best_fitness"]),
            "final_diversity": float(ga_result["diversity_history"][-1]),
            "high_quality_solutions": int(
                np.sum(ga_result["final_fitness_scores"] >= good_fitness_threshold)
            ),
            "random_seed": random_seed,
            "reproducible": use_random_seed,
            "multiple_experiments": run_multi_flag,
            "high_fitness_threshold": high_fitness_threshold,
            "good_fitness_threshold": good_fitness_threshold,
        }

        if not run_multi_flag and ga_config is not None:
            # Add config details for single experiment by copying ga_config
            # and flattening niching_config into niching_* keys for JSON friendliness
            # Filter out non-serializable values (functions, objects)
            flat_config: dict[str, str | int | float | bool | None] = {}
            for k, v in ga_config.items():
                if k == "niching_config":
                    continue
                elif callable(v):
                    flat_config[k] = (
                        str(v.__name__) if hasattr(v, "__name__") else str(type(v).__name__)
                    )
                elif isinstance(v, (int, float, str, bool, type(None))):
                    flat_config[k] = v
                else:
                    flat_config[k] = str(v)

            niching = ga_config["niching_config"]
            flat_config["niching_enabled"] = niching.enabled
            flat_config["niching_sigma_share"] = niching.sigma_share
            flat_config["niching_alpha"] = niching.alpha
            flat_config["niching_distance_metric"] = niching.distance_metric
            summary["config"] = flat_config

        with open(output_dir / "00_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("     DEMONSTRATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"Best fitness achieved: {ga_result['best_fitness']:.4f}")
        good_found = np.sum(ga_result["final_fitness_scores"] >= good_fitness_threshold)
        logger.info(f"High-quality solutions found: {good_found}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    mp.set_start_method("spawn", force=True)
    main()
