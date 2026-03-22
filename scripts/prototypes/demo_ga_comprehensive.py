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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lwi_microbolometer_design import (
    compute_distance_matrix,
    gaussian_parameters_to_unit_amplitude_curves,
    ivat_transform,
    spectral_angle_mapper,
    vat_reorder,
)
from lwi_microbolometer_design.data import SceneConfig, load_substance_atmosphere_data
from lwi_microbolometer_design.ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    NichingConfig,
    diversity_preserving_mutation,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants for algorithm behavior (not fitness-related)
MIN_POPULATION_FOR_DIVERSITY = 2  # Minimum population size to calculate diversity
MIN_SOLUTIONS_FOR_IVAT = 10  # Minimum solutions needed for IVAT visualization

# Plotting style
plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def load_data(
    spectral_data_file: Path,
    air_transmittance_file: Path,
    atmospheric_distance_ratio: float = 0.11,
    temperature_kelvin: float = 293.15,
    air_refractive_index: float = 1.0,
) -> SceneConfig:
    """
    Load spectral data and configuration parameters via the package loader.

    Parameters
    ----------
    spectral_data_file : Path
        Path to the Excel file containing spectral data.
    air_transmittance_file : Path
        Path to the Excel file containing air transmittance data.
    atmospheric_distance_ratio : float, optional
        Atmospheric distance ratio. Default: 0.11
    temperature_kelvin : float, optional
        Scene temperature in Kelvin. Default: 293.15
    air_refractive_index : float, optional
        Air refractive index. Default: 1.0

    Returns
    -------
    SceneConfig
        Canonical scene for the given scalar parameters (first condition if expanded).
    """
    loaded = load_substance_atmosphere_data(
        spectral_data_file=spectral_data_file,
        air_transmittance_file=air_transmittance_file,
        atmospheric_distance_ratio=atmospheric_distance_ratio,
        temperature_kelvin=temperature_kelvin,
        air_refractive_index=air_refractive_index,
    )
    if isinstance(loaded, list):
        if len(loaded) > 1:
            logger.warning("Multi-condition data detected, using first condition only")
        return loaded[0]
    return loaded


def default_ga_configuration(num_generations: int = 2000) -> dict:
    """
    Return default GA configuration when no CSV file is available.

    Parameters
    ----------
    num_generations : int, optional
        Number of generations to run (default: 2000)

    Returns
    -------
    dict
        Default GA configuration dictionary ready for AdvancedGA.
    """
    return {
        "num_generations": num_generations,
        "num_parents_mating": 50,
        "sol_per_pop": 200,
        "parent_selection_type": "tournament",
        "K_tournament": 3,
        "keep_elitism": 5,
        "crossover_type": "uniform",
        "crossover_probability": 0.8,
        # Use custom diversity-preserving mutation
        "mutation_type": diversity_preserving_mutation,
        "mutation_probability": 0.1,
        "save_best_solutions": True,
        "stop_criteria": "saturate_200",
        "niching_config": NichingConfig(
            enabled=True,
            use_optimal_pairing=False,
            sigma_share=0.5,
            alpha=0.5,
            params_per_group=2,
            optimal_pairing_metric="euclidean",
        ),
    }


def load_ga_configuration_from_csv(
    file_path: Path, sort_by: str = "best_fitness", ascending: bool = False, row_index: int = 0
) -> dict | None:
    """
    Load GA configuration from CSV and return complete configuration dictionary.

    All CSV columns are optional. Missing parameters use defaults with warnings.

    Supported columns: num_generations, num_parents_mating, sol_per_pop,
    parent_selection_type, K_tournament, keep_elitism,
    crossover_type, crossover_probability,
    mutation_type, mutation_probability,
    save_best_solutions, stop_criteria,
    niching_enabled, niching_sigma_share, niching_alpha, niching_distance_metric.

    Note: 'best_fitness' is informational only.

    Parameters
    ----------
    file_path : Path
        Path to CSV file with GA configuration parameters
    sort_by : str, optional
        Column to sort by (default: 'best_fitness')
    ascending : bool, optional
        Sort order (default: False)
    row_index : int, optional
        Row index after sorting (default: 0)

    Returns
    -------
    dict | None
        Complete GA configuration ready for AdvancedGA, or None if file not found
    """
    if not file_path.exists():
        logger.warning(f"Configuration file not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # Sort if sort_by column exists
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Get the specified row
    if row_index >= len(df):
        logger.warning(f"Row index {row_index} out of range. Using first row.")
        row_index = 0

    raw_config = df.iloc[row_index]

    # Create default configuration once
    default_ga_config = default_ga_configuration()

    # Start with empty configuration
    ga_config = {}

    # Track missing parameters
    missing_params = []

    # Helper function to extract parameter from CSV or use default
    def get_param(param_name: str, param_type: type, default_value: object) -> object:
        """Extract parameter from CSV or return default."""
        if param_name in raw_config.index:
            value = raw_config[param_name]
            # Handle boolean strings from CSV (pd.read_csv doesn't auto-convert them)
            # Without this, CSV "False" → Python True (wrong!) because bool("False") = True
            if param_type is bool and isinstance(value, str):
                lower_val = value.lower()
                if lower_val in ["true", "1", "yes"]:
                    return True
                elif lower_val in ["false", "0", "no"]:
                    return False
                else:
                    # Unknown string, use default type conversion
                    return param_type(value)
            return param_type(value)
        else:
            missing_params.append(param_name)
            return default_value

    # Fill ga_config with CSV values or defaults
    ga_config["num_generations"] = get_param(
        "num_generations", int, default_ga_config["num_generations"]
    )
    ga_config["num_parents_mating"] = get_param(
        "num_parents_mating", int, default_ga_config["num_parents_mating"]
    )
    ga_config["sol_per_pop"] = get_param("sol_per_pop", int, default_ga_config["sol_per_pop"])
    ga_config["parent_selection_type"] = get_param(
        "parent_selection_type", str, default_ga_config["parent_selection_type"]
    )
    ga_config["K_tournament"] = get_param("K_tournament", int, default_ga_config["K_tournament"])
    ga_config["keep_elitism"] = get_param("keep_elitism", int, default_ga_config["keep_elitism"])
    ga_config["crossover_type"] = get_param(
        "crossover_type", str, default_ga_config["crossover_type"]
    )
    ga_config["crossover_probability"] = get_param(
        "crossover_probability", float, default_ga_config["crossover_probability"]
    )
    ga_config["mutation_type"] = get_param("mutation_type", str, default_ga_config["mutation_type"])
    ga_config["mutation_probability"] = get_param(
        "mutation_probability", float, default_ga_config["mutation_probability"]
    )
    ga_config["save_best_solutions"] = get_param(
        "save_best_solutions", bool, default_ga_config["save_best_solutions"]
    )
    ga_config["stop_criteria"] = get_param("stop_criteria", str, default_ga_config["stop_criteria"])

    # Extract niching parameters and rebuild NichingConfig
    niching_enabled = get_param(
        "niching_enabled", bool, default_ga_config["niching_config"].enabled
    )
    niching_sigma_share = get_param(
        "niching_sigma_share", float, default_ga_config["niching_config"].sigma_share
    )
    niching_alpha = get_param("niching_alpha", float, default_ga_config["niching_config"].alpha)
    niching_distance_metric = get_param(
        "niching_distance_metric", str, default_ga_config["niching_config"].distance_metric
    )
    # Determine optimal pairing setting from CSV or use default
    niching_use_optimal_pairing = get_param(
        "niching_use_optimal_pairing",
        bool,
        default_ga_config["niching_config"].use_optimal_pairing,
    )
    niching_params_per_group = get_param(
        "niching_params_per_group",
        int,
        default_ga_config["niching_config"].params_per_group,
    )
    niching_optimal_pairing_metric = get_param(
        "niching_optimal_pairing_metric",
        str,
        default_ga_config["niching_config"].optimal_pairing_metric,
    )

    ga_config["niching_config"] = NichingConfig(
        enabled=niching_enabled,
        use_optimal_pairing=niching_use_optimal_pairing,
        sigma_share=niching_sigma_share,
        alpha=niching_alpha,
        distance_metric=niching_distance_metric,
        params_per_group=niching_params_per_group,
        optimal_pairing_metric=niching_optimal_pairing_metric,
    )

    # Warn about missing parameters
    if missing_params:
        logger.warning("=" * 60)
        logger.warning("MISSING CONFIGURATION PARAMETERS - USING DEFAULTS")
        logger.warning("=" * 60)
        for param in missing_params:
            # Get default value (niching params are nested)
            if param.startswith("niching_"):
                attr_name = param.replace("niching_", "")  # Remove prefix
                default_value = getattr(default_ga_config["niching_config"], attr_name)
            else:
                default_value = default_ga_config[param]
            logger.warning(f"  ⚠️  '{param}' → default: {default_value}")
        logger.warning("=" * 60)

    logger.info(f"Loaded GA configuration from: {file_path}")
    logger.info(f"  Row index: {row_index}")
    if "best_fitness" in raw_config:
        logger.info(f"  Best fitness: {raw_config['best_fitness']:.4f}")

    logger.info("\n  Loaded parameters:")
    logger.info(f"    num_generations: {ga_config['num_generations']}")
    logger.info(f"    num_parents_mating: {ga_config['num_parents_mating']}")
    logger.info(f"    sol_per_pop: {ga_config['sol_per_pop']}")
    logger.info(f"    parent_selection_type: {ga_config['parent_selection_type']}")
    logger.info(f"    K_tournament: {ga_config['K_tournament']}")
    logger.info(f"    keep_elitism: {ga_config['keep_elitism']}")
    logger.info(f"    crossover_type: {ga_config['crossover_type']}")
    logger.info(f"    crossover_probability: {ga_config['crossover_probability']}")
    logger.info(f"    mutation_type: {ga_config['mutation_type']}")
    logger.info(f"    mutation_probability: {ga_config['mutation_probability']}")
    logger.info(f"    save_best_solutions: {ga_config['save_best_solutions']}")
    logger.info(f"    stop_criteria: {ga_config['stop_criteria']}")
    niching_config = ga_config["niching_config"]
    logger.info(f"    niching_enabled: {niching_config.enabled}")
    logger.info(f"    niching_use_optimal_pairing: {niching_config.use_optimal_pairing}")
    logger.info(f"    niching_sigma_share: {niching_config.sigma_share}")
    logger.info(f"    niching_alpha: {niching_config.alpha}")
    if niching_config.use_optimal_pairing:
        logger.info(f"    niching_params_per_group: {niching_config.params_per_group}")
        logger.info(f"    niching_optimal_pairing_metric: {niching_config.optimal_pairing_metric}")
    else:
        logger.info(f"    niching_distance_metric: {niching_config.distance_metric}")

    return ga_config


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
        scene=scene,
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
        "diversity_history": diversity_history if diversity_history else [0.0],
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


def calculate_population_diversity(population: np.ndarray) -> float:
    """Calculate population diversity using pairwise distances."""
    if len(population) < MIN_POPULATION_FOR_DIVERSITY:
        return 0.0

    total_distance = 0.0
    comparisons = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = float(np.linalg.norm(population[i] - population[j]))
            total_distance += distance
            comparisons += 1

    return total_distance / comparisons if comparisons > 0 else 0.0


def visualize_ga_results(
    result: dict,
    scene: SceneConfig,
    output_dir: Path,
    high_fitness_threshold: float = 50.0,
) -> None:
    """Create comprehensive visualizations of GA results.

    Parameters
    ----------
    high_fitness_threshold : float, optional
        Threshold for determining which solutions to visualize (default: 50.0)
    """
    logger.info("\n=== Creating Visualizations ===")

    wavelengths: np.ndarray = np.asarray(scene.wavelengths)
    fitness_threshold = high_fitness_threshold

    # Get high-quality solutions
    high_quality_mask = result["final_fitness_scores"] >= fitness_threshold
    high_quality_population = result["final_population"][high_quality_mask]
    high_quality_fitness = result["final_fitness_scores"][high_quality_mask]

    # 1. Fitness Evolution
    plt.figure(figsize=(12, 6))
    plt.plot(result["best_fitness_history"], label="Best Fitness", linewidth=2, color="#1f77b4")
    plt.plot(
        result["mean_fitness_history"],
        label="Mean Fitness",
        linewidth=1.5,
        color="#ff7f0e",
        linestyle="--",
    )
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness Score", fontsize=14)
    plt.title("GA Fitness Evolution", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "01_fitness_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Diversity Evolution
    plt.figure(figsize=(12, 6))
    plt.plot(
        result["diversity_history"], label="Population Diversity", linewidth=2, color="#2ca02c"
    )
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Diversity Score", fontsize=14)
    plt.title("GA Population Diversity Evolution", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "02_diversity_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Final Fitness Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(
        result["final_fitness_scores"],
        bins=30,
        alpha=0.7,
        color="#9467bd",
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Fitness Score", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Final Population Fitness Distribution", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "03_fitness_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Top Sensor Designs
    if len(high_quality_population) > 0:
        plot_top_sensor_designs(
            high_quality_population,
            high_quality_fitness,
            wavelengths,
            fitness_threshold,
            output_dir,
        )

    # 5. IVAT Diversity Analysis
    if len(high_quality_population) >= MIN_SOLUTIONS_FOR_IVAT:
        plot_ivat_analysis(high_quality_population, high_quality_fitness, output_dir)

    # 6. High-Fitness Count Evolution (Multimodal Analysis)
    if "high_fitness_count_history" in result and len(result["high_fitness_count_history"]) > 0:
        plot_high_fitness_evolution(result, output_dir)

    # 7. Fitness Distribution Evolution (Heatmap)
    if "fitness_std_history" in result and len(result["fitness_std_history"]) > 0:
        plot_fitness_spread_evolution(result, output_dir)

    logger.info(f"Visualizations saved to {output_dir}")


def plot_top_sensor_designs(
    high_quality_population: np.ndarray,
    high_quality_fitness: np.ndarray,
    wavelengths: np.ndarray,
    fitness_threshold: float,
    output_dir: Path,
) -> None:
    """Plot top sensor designs with improved formatting."""
    sorted_indices = np.argsort(high_quality_fitness)[::-1]
    top_10 = high_quality_population[sorted_indices[:10]]
    top_10_fitness = high_quality_fitness[sorted_indices[:10]]

    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_10)))

    for i, (chromosome, _fitness) in enumerate(zip(top_10, top_10_fitness, strict=False)):
        # Convert chromosome to list of tuples for gaussian curves
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)
        vertical_offset = i * 0.1

        # Plot each basis function
        for _j, basis_func in enumerate(basis_functions.T):
            scaled_basis = basis_func * 0.3
            plt.plot(
                wavelengths,
                scaled_basis + vertical_offset,
                color=colors[i],
                alpha=0.8,
                linewidth=1.5,
            )

    plt.xlabel("Wavelength (µm)", fontsize=14)
    plt.ylabel("Absorptivity (Offset Applied)", fontsize=14)
    plt.title(
        f"Top 10 Sensor Designs (Fitness ≥ {fitness_threshold})", fontsize=16, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)

    # Add fitness annotations
    for i, fitness in enumerate(top_10_fitness):
        plt.text(
            0.02,
            0.98 - i * 0.09,
            f"Rank {i + 1}: {fitness:.2f}",
            transform=plt.gca().transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "04_top_sensor_designs.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Also plot the best design
    plot_best_design(top_10[0], top_10_fitness[0], wavelengths, output_dir)


def plot_best_design(
    best_chromosome: np.ndarray, best_fitness: float, wavelengths: np.ndarray, output_dir: Path
) -> None:
    """Plot the single best sensor design."""
    plt.figure(figsize=(12, 6))

    # Convert chromosome array to list of tuples
    gaussian_params = [
        (best_chromosome[j], best_chromosome[j + 1]) for j in range(0, len(best_chromosome), 2)
    ]
    basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)

    # Plot each basis function
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for j, basis_func in enumerate(basis_functions.T):
        plt.plot(
            wavelengths,
            basis_func,
            color=colors[j % len(colors)],
            alpha=0.8,
            linewidth=2,
            label=f"Basis Function {j + 1}",
        )

    plt.xlabel("Wavelength (µm)", fontsize=14)
    plt.ylabel("Absorptivity", fontsize=14)
    plt.title(f"Best Sensor Design (Fitness: {best_fitness:.2f})", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "05_best_design.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ivat_analysis(
    high_quality_population: np.ndarray, _high_quality_fitness: np.ndarray, output_dir: Path
) -> None:
    """Create IVAT visualizations with fixed color range."""
    # Use arrays directly (no need for Chromosome class)
    parameter_sets = [np.array(genes) for genes in high_quality_population]
    # Use optimal pairing mode for grouped parameters (mu, sigma pairs)
    distance_matrix = compute_distance_matrix(
        parameter_sets,
        metric="euclidean",
        use_optimal_pairing=True,
        params_per_group=2,  # Each basis function has 2 params: (mu, sigma)
    )

    # Determine color range (exclude diagonal zeros)
    all_distances = distance_matrix.flatten()
    all_distances = all_distances[all_distances > 0]

    if len(all_distances) == 0:
        logger.warning("No valid distances found for IVAT analysis")
        return

    global_vmin = np.percentile(all_distances, 5)
    global_vmax = np.percentile(all_distances, 95)

    # Compute VAT and IVAT
    vat_matrix, _reorder = vat_reorder(distance_matrix)
    ivat_matrix = ivat_transform(vat_matrix)

    # Create subplot with 3 panels
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original distance matrix
    im1 = axes[0].imshow(distance_matrix, cmap="viridis", vmin=global_vmin, vmax=global_vmax)
    axes[0].set_title("Original Distance Matrix", fontsize=12)
    axes[0].set_xlabel("Solution Index")
    axes[0].set_ylabel("Solution Index")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # VAT matrix
    im2 = axes[1].imshow(vat_matrix, cmap="viridis", vmin=global_vmin, vmax=global_vmax)
    axes[1].set_title("VAT Matrix", fontsize=12)
    axes[1].set_xlabel("Solution Index")
    axes[1].set_ylabel("Solution Index")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # IVAT matrix
    im3 = axes[2].imshow(ivat_matrix, cmap="viridis", vmin=global_vmin, vmax=global_vmax)
    axes[2].set_title("IVAT Matrix (Clustering)", fontsize=12)
    axes[2].set_xlabel("Solution Index")
    axes[2].set_ylabel("Solution Index")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"IVAT Diversity Analysis (Top {len(high_quality_population)} Solutions)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "06_ivat_diversity.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_high_fitness_evolution(result: dict, output_dir: Path) -> None:
    """Plot evolution of high-fitness solution count over generations."""
    plt.figure(figsize=(12, 6))

    plt.plot(result["high_fitness_count_history"], linewidth=2, color="#e377c2")
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Count of Solutions with Fitness ≥ 50", fontsize=14)
    plt.title(
        "Evolution of High-Fitness Solutions (Multimodal Discovery)", fontsize=16, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "07_high_fitness_count.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_fitness_spread_evolution(result: dict, output_dir: Path) -> None:
    """Plot fitness standard deviation over generations."""
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Diversity and fitness std
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(
        result["diversity_history"], linewidth=2, color="#2ca02c", label="Population Diversity"
    )
    line2 = ax1_twin.plot(
        result["fitness_std_history"], linewidth=2, color="#d62728", label="Fitness Std Dev"
    )

    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Population Diversity", fontsize=12, color="#2ca02c")
    ax1_twin.set_ylabel("Fitness Std Dev", fontsize=12, color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax1_twin.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Population Diversity vs Fitness Spread", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=10)

    # Bottom panel: Best, mean, and std bands
    # Ensure all arrays have the same length (use shortest)
    min_len = min(
        len(result["best_fitness_history"]),
        len(result["mean_fitness_history"]),
        len(result["fitness_std_history"]),
    )
    generations = np.arange(min_len)
    best = np.array(result["best_fitness_history"][:min_len])
    mean = np.array(result["mean_fitness_history"][:min_len])
    std = np.array(result["fitness_std_history"][:min_len])

    ax2.plot(generations, best, linewidth=2, color="#1f77b4", label="Best Fitness")
    ax2.plot(generations, mean, linewidth=2, color="#ff7f0e", label="Mean Fitness")
    ax2.fill_between(
        generations, mean - std, mean + std, alpha=0.3, color="#ff7f0e", label="±1 Std Dev"
    )

    ax2.set_xlabel("Generation", fontsize=12)
    ax2.set_ylabel("Fitness Score", fontsize=12)
    ax2.set_title("Fitness Evolution with Spread", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "08_fitness_spread.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_single_experiment(
    config_info: dict,
    scene: SceneConfig,
    gene_space: list,
    params_per_basis_function: int,
    high_fitness_threshold: float = 50.0,
    good_fitness_threshold: float = 45.0,
) -> dict:
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
    gene_space: list,
    params_per_basis_function: int,
    random_seed: int | None = None,
    high_fitness_threshold: float = 50.0,
    good_fitness_threshold: float = 45.0,
) -> dict:
    """Run multiple GA configurations for comparison."""
    configurations = [
        {
            "name": "Baseline GA (No Niching)",
            "config": {
                "num_generations": 2000,
                "sol_per_pop": 200,
                "num_parents_mating": 50,
                "mutation_probability": 0.1,
                "crossover_probability": 0.8,
                "keep_elitism": 5,
                "parent_selection_type": "tournament",
                "K_tournament": 3,
                "crossover_type": "uniform",
                "mutation_type": "adaptive",
                "save_best_solutions": True,
                "stop_criteria": "saturate_200",
                "niching_config": NichingConfig(enabled=False, use_optimal_pairing=False),
                "random_seed": random_seed,
            },
        },
        {
            "name": "Advanced GA (With Niching)",
            "config": {
                "num_generations": 2000,
                "sol_per_pop": 200,
                "num_parents_mating": 50,
                "mutation_probability": 0.1,
                "crossover_probability": 0.8,
                "keep_elitism": 5,
                "parent_selection_type": "tournament",
                "K_tournament": 3,
                "crossover_type": "uniform",
                "mutation_type": "adaptive",
                "save_best_solutions": True,
                "stop_criteria": "saturate_200",
                "niching_config": NichingConfig(
                    enabled=True,
                    use_optimal_pairing=False,
                    sigma_share=0.5,
                    alpha=0.5,
                    distance_metric="euclidean",
                ),
                "random_seed": random_seed,
            },
        },
        {
            "name": "Advanced GA (Moderate Niching)",
            "config": {
                "num_generations": 2000,
                "sol_per_pop": 200,
                "num_parents_mating": 50,
                "mutation_probability": 0.1,
                "crossover_probability": 0.8,
                "keep_elitism": 5,
                "parent_selection_type": "tournament",
                "K_tournament": 3,
                "crossover_type": "uniform",
                "mutation_type": "adaptive",
                "save_best_solutions": True,
                "stop_criteria": "saturate_200",
                "niching_config": NichingConfig(
                    enabled=True,
                    use_optimal_pairing=False,
                    sigma_share=1.0,
                    alpha=0.5,
                    distance_metric="euclidean",
                ),
                "random_seed": random_seed,
            },
        },
        {
            "name": "High Mutation Rate",
            "config": {
                "num_generations": 2000,
                "sol_per_pop": 200,
                "num_parents_mating": 50,
                "mutation_probability": 0.2,
                "crossover_probability": 0.8,
                "keep_elitism": 5,
                "parent_selection_type": "tournament",
                "K_tournament": 3,
                "crossover_type": "uniform",
                "mutation_type": "adaptive",
                "save_best_solutions": True,
                "stop_criteria": "saturate_200",
                "niching_config": NichingConfig(enabled=False, use_optimal_pairing=False),
                "random_seed": random_seed,
            },
        },
        {
            "name": "High Diversity Niching",
            "config": {
                "num_generations": 2000,
                "sol_per_pop": 200,
                "num_parents_mating": 50,
                "mutation_probability": 0.15,
                "crossover_probability": 0.8,
                "keep_elitism": 5,
                "parent_selection_type": "tournament",
                "K_tournament": 3,
                "crossover_type": "uniform",
                "mutation_type": "adaptive",
                "save_best_solutions": True,
                "stop_criteria": "saturate_200",
                "niching_config": NichingConfig(
                    enabled=True,
                    use_optimal_pairing=False,
                    sigma_share=2.0,
                    alpha=0.5,
                    distance_metric="euclidean",
                ),
                "random_seed": random_seed,
            },
        },
        {
            "name": "Ultra-Diverse Niching (Multi-Pattern)",
            "config": {
                "num_generations": 2000,
                "sol_per_pop": 200,
                "num_parents_mating": 50,
                "mutation_probability": 0.15,
                "crossover_probability": 0.8,
                "keep_elitism": 5,
                "parent_selection_type": "tournament",
                "K_tournament": 3,
                "crossover_type": "uniform",
                "mutation_type": "adaptive",
                "save_best_solutions": True,
                "stop_criteria": "saturate_200",
                "niching_config": NichingConfig(
                    enabled=True,
                    use_optimal_pairing=False,
                    sigma_share=3.0,
                    alpha=1.0,
                    distance_metric="euclidean",
                ),
                "random_seed": random_seed,
            },
        },
    ]

    logger.info(f"\n=== Running {len(configurations)} Experimental Configurations ===")
    results = {}

    # Use ProcessPoolExecutor for parallel execution
    max_workers = min(len(configurations), mp.cpu_count())
    logger.info(f"Using {max_workers} parallel workers")

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
                logger.info(f"Best fitness: {result_data['result']['best_fitness']:.4f}")
                final_div = result_data["result"]["diversity_history"][-1]
                logger.info(f"Final diversity: {final_div:.4f}")
                fitness_scores = result_data["result"]["final_fitness_scores"]
                good_count = np.sum(fitness_scores >= good_fitness_threshold)
                logger.info(f"High-quality solutions: {good_count}")

            except Exception as exc:
                logger.error(f"Configuration {config_info['name']} generated an exception: {exc}")
                raise

    # Compare results
    logger.info("\n=== Experiment Comparison ===")
    for name, result in results.items():
        logger.info(f"{name}:")
        logger.info(f"  Best fitness: {result['best_fitness']:.4f}")
        logger.info(f"  Final diversity: {result['diversity_history'][-1]:.4f}")
        good_sols = np.sum(result["final_fitness_scores"] >= good_fitness_threshold)
        logger.info(f"  High-quality solutions: {good_sols}")

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
    logger.info("      MVP Deliverable")
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
        scene = load_data(
            spectral_data_file=spectral_data_file,
            air_transmittance_file=air_transmittance_file,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            temperature_kelvin=temperature_kelvin,
            air_refractive_index=air_refractive_index,
        )
        gene_space = param_bounds * num_basis_functions

        # Initialize ga_config for later use
        ga_config: dict | None = None

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
                ga_config = default_ga_configuration(num_generations=args.generations)
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
