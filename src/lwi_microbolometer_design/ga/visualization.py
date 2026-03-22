"""
GA visualization utilities for AdvancedGA results.

This module provides functions to create publication-ready visualizations
of GA optimization results, including fitness evolution, diversity analysis,
and solution comparisons.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np

from lwi_microbolometer_design.analysis import (
    compute_distance_matrix,
    ivat_transform,
    vat_reorder,
)
from lwi_microbolometer_design.data.scene_config import SceneConfig
from lwi_microbolometer_design.simulation import gaussian_parameters_to_unit_amplitude_curves

logger = logging.getLogger(__name__)

# Maps decoded per-basis parameters (e.g. list of (mu, sigma)) to a (wavelengths, n_basis) curve matrix.
ParametersToCurves: TypeAlias = Callable[[list[tuple[float, float]], np.ndarray], np.ndarray]

# Constants for visualization behavior
MIN_SOLUTIONS_FOR_IVAT = 10  # Minimum solutions needed for IVAT visualization


def visualize_ga_results(
    result: dict[str, Any],
    scene: SceneConfig,
    output_dir: Path,
    high_fitness_threshold: float = 50.0,
    *,
    parameters_to_curves: ParametersToCurves = gaussian_parameters_to_unit_amplitude_curves,
) -> None:
    """
    Create comprehensive visualizations of GA results.

    Generates multiple plots including fitness evolution, diversity metrics,
    top solutions, and IVAT analysis.

    Parameters
    ----------
    result : dict[str, Any]
        GA result dictionary containing:
        - best_fitness_history: list of best fitness per generation
        - mean_fitness_history: list of mean fitness per generation (optional)
        - diversity_history: list of diversity scores per generation (optional)
        - final_fitness_scores: array of final generation fitness scores
        - final_population: array of final generation chromosomes
        - high_fitness_count_history: (optional) list of high-fitness counts
        - fitness_std_history: (optional) list of fitness std dev per generation
    scene : SceneConfig
        Loaded scene configuration (wavelength grid and substance spectra).
    output_dir : Path
        Directory to save visualization plots
    high_fitness_threshold : float, optional
        Threshold for determining which solutions to visualize (default: 50.0)
    parameters_to_curves : callable, optional
        Maps ``(parameter_tuples, wavelengths)`` to an array of shape
        ``(n_wavelengths, n_basis)``. Default: Gaussian unit-amplitude curves.

    Notes
    -----
    - Optional fields (high_fitness_count_history, fitness_std_history) will
      be skipped if not available
    - To extract results from an AdvancedGA instance, use extract_basic_results()
      before calling this function
    """
    logger.info("\n=== Creating Visualizations ===")

    wavelengths: np.ndarray = np.asarray(scene.wavelengths)
    fitness_threshold = high_fitness_threshold

    # Validate required fields
    required_fields = ["final_fitness_scores", "final_population"]
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        msg = f"Missing required fields in result: {missing_fields}"
        raise ValueError(msg)

    # Get high-quality solutions (convert to arrays with explicit type checking)
    fitness_data = result["final_fitness_scores"]
    population_data = result["final_population"]
    final_fitness = (
        np.array(fitness_data) if not isinstance(fitness_data, np.ndarray) else fitness_data
    )
    final_population = (
        np.array(population_data)
        if not isinstance(population_data, np.ndarray)
        else population_data
    )

    if final_fitness.size == 0 or final_population.size == 0:
        logger.warning("Empty population or fitness scores - skipping visualizations")
        return

    high_quality_mask = final_fitness >= fitness_threshold
    high_quality_population = final_population[high_quality_mask]
    high_quality_fitness = final_fitness[high_quality_mask]

    # 1. Fitness Evolution
    if "best_fitness_history" in result and len(result["best_fitness_history"]) > 0:
        _plot_fitness_evolution(result, output_dir)
    else:
        logger.warning("best_fitness_history not available - skipping fitness evolution plot")

    # 2. Diversity Evolution
    if "diversity_history" in result and len(result["diversity_history"]) > 0:
        _plot_diversity_evolution(result, output_dir)
    else:
        logger.warning("diversity_history not available - skipping diversity evolution plot")

    # 3. Final Fitness Distribution
    _plot_fitness_distribution(result, output_dir)

    # 4. Top Sensor Designs
    if len(high_quality_population) > 0:
        plot_top_sensor_designs(
            high_quality_population,
            high_quality_fitness,
            wavelengths,
            fitness_threshold,
            output_dir,
            parameters_to_curves=parameters_to_curves,
        )

    # 5. IVAT Diversity Analysis
    if len(high_quality_population) >= MIN_SOLUTIONS_FOR_IVAT:
        plot_ivat_analysis(high_quality_population, high_quality_fitness, output_dir)

    # 6. High-Fitness Count Evolution (Multimodal Analysis)
    if (
        "high_fitness_count_history" in result
        and result["high_fitness_count_history"] is not None
        and len(result["high_fitness_count_history"]) > 0
    ):
        plot_high_fitness_evolution(result, output_dir)
    else:
        logger.debug("high_fitness_count_history not available - skipping high-fitness count plot")

    # 7. Fitness Distribution Evolution (Heatmap)
    if (
        "fitness_std_history" in result
        and result["fitness_std_history"] is not None
        and len(result["fitness_std_history"]) > 0
    ):
        plot_fitness_spread_evolution(result, output_dir)
    else:
        logger.debug("fitness_std_history not available - skipping fitness spread plot")

    logger.info(f"Visualizations saved to {output_dir}")


def _plot_fitness_evolution(result: dict[str, np.ndarray | list[float]], output_dir: Path) -> None:
    """Plot fitness evolution over generations."""
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


def _plot_diversity_evolution(result: dict[str, list[float]], output_dir: Path) -> None:
    """Plot population diversity evolution over generations."""
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


def _plot_fitness_distribution(result: dict[str, np.ndarray], output_dir: Path) -> None:
    """Plot final population fitness distribution."""
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


def plot_top_sensor_designs(
    high_quality_population: np.ndarray,
    high_quality_fitness: np.ndarray,
    wavelengths: np.ndarray,
    fitness_threshold: float,
    output_dir: Path,
    *,
    parameters_to_curves: ParametersToCurves = gaussian_parameters_to_unit_amplitude_curves,
) -> None:
    """
    Plot top sensor designs with improved formatting.

    Parameters
    ----------
    high_quality_population : np.ndarray
        Array of high-quality chromosomes (shape: [n_solutions, n_genes])
    high_quality_fitness : np.ndarray
        Array of fitness scores for high-quality solutions
    wavelengths : np.ndarray
        Wavelength array for plotting basis functions
    fitness_threshold : float
        Fitness threshold used to filter solutions
    output_dir : Path
        Directory to save the plot
    parameters_to_curves : callable, optional
        Same contract as :func:`visualize_ga_results` (default: Gaussian curves).
    """
    sorted_indices = np.argsort(high_quality_fitness)[::-1]
    top_10 = high_quality_population[sorted_indices[:10]]
    top_10_fitness = high_quality_fitness[sorted_indices[:10]]

    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_10)))

    for i, (chromosome, _fitness) in enumerate(zip(top_10, top_10_fitness, strict=False)):
        # Convert chromosome to list of tuples for gaussian curves
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = parameters_to_curves(gaussian_params, wavelengths)
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
    plot_best_design(
        top_10[0],
        top_10_fitness[0],
        wavelengths,
        output_dir,
        parameters_to_curves=parameters_to_curves,
    )


def plot_best_design(
    best_chromosome: np.ndarray,
    best_fitness: float,
    wavelengths: np.ndarray,
    output_dir: Path,
    *,
    parameters_to_curves: ParametersToCurves = gaussian_parameters_to_unit_amplitude_curves,
) -> None:
    """
    Plot the single best sensor design.

    Parameters
    ----------
    best_chromosome : np.ndarray
        Best chromosome found by GA
    best_fitness : float
        Fitness score of the best chromosome
    wavelengths : np.ndarray
        Wavelength array for plotting basis functions
    output_dir : Path
        Directory to save the plot
    parameters_to_curves : callable, optional
        Same contract as :func:`plot_top_sensor_designs` (default: Gaussian curves).
    """
    plt.figure(figsize=(12, 6))

    # Convert chromosome array to list of tuples
    gaussian_params = [
        (best_chromosome[j], best_chromosome[j + 1]) for j in range(0, len(best_chromosome), 2)
    ]
    basis_functions = parameters_to_curves(gaussian_params, wavelengths)

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
    """
    Create IVAT visualizations with fixed color range.

    IVAT (Improved Visual Assessment of Cluster Tendency) helps visualize
    clustering structure in high-quality solutions.

    Parameters
    ----------
    high_quality_population : np.ndarray
        Array of high-quality chromosomes
    _high_quality_fitness : np.ndarray
        Array of fitness scores (unused, kept for API consistency)
    output_dir : Path
        Directory to save the plot
    """
    # Use arrays directly (no need for Chromosome class)
    parameter_sets = [np.array(genes) for genes in high_quality_population]
    # Use optimal pairing mode (hidden implementation detail)
    distance_matrix = compute_distance_matrix(
        parameter_sets,
        metric="euclidean",
        use_optimal_pairing=True,
        params_per_group=2,  # Assume (mu, sigma) pairs - could be configurable
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


def plot_high_fitness_evolution(result: dict[str, list[int]], output_dir: Path) -> None:
    """
    Plot evolution of high-fitness solution count over generations.

    Parameters
    ----------
    result : dict[str, list[int]]
        Result dictionary containing 'high_fitness_count_history'
    output_dir : Path
        Directory to save the plot
    """
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


def plot_fitness_spread_evolution(result: dict[str, list[float]], output_dir: Path) -> None:
    """
    Plot fitness standard deviation over generations.

    Shows how fitness spread evolves, indicating convergence or diversity.

    Parameters
    ----------
    result : dict[str, list[float]]
        Result dictionary containing diversity_history, fitness_std_history,
        best_fitness_history, and mean_fitness_history
    output_dir : Path
        Directory to save the plot
    """
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
