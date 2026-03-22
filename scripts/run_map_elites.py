#!/usr/bin/env python3
"""MAP-Elites: Multi-dimensional Archive of Phenotypic Elites.

This script implements a Quality-Diversity algorithm that maintains an archive
of elite solutions organized by feature descriptors (mu_1, mu_2).
Instead of converging to a single peak, it illuminates the entire design space.
"""

import multiprocessing as mp
import pickle
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
    MinDissimilarityFitnessEvaluator,
)

# Required for multiprocessing
mp.set_start_method("spawn", force=True)


def extract_features(chromosome: np.ndarray) -> tuple[float, float]:
    """Extract feature descriptors from chromosome.

    Features: smallest and second smallest means (mu values)
    This ensures consistent ordering regardless of which basis function has which mean.

        Parameters
        ----------
        chromosome : np.ndarray
        Chromosome with format [mu1, sigma1, mu2, sigma2, mu3, sigma3, mu4, sigma4]

        Returns
        -------
        tuple[float, float]
        (smallest_mu, second_smallest_mu) feature descriptors
    """
    mu_1 = float(chromosome[0])  # First Gaussian center
    mu_2 = float(chromosome[2])  # Second Gaussian center
    mu_3 = float(chromosome[4])  # Third Gaussian center
    mu_4 = float(chromosome[6])  # Fourth Gaussian center

    # Sort to get smallest and second smallest
    sorted_mus = sorted([mu_1, mu_2, mu_3, mu_4])
    return sorted_mus[0], sorted_mus[1]  # Smallest and second smallest


def bin_coordinates(
    mu_1: float, mu_2: float, grid_resolution: int, mu_range: tuple[float, float]
) -> tuple[int, int]:
    """Convert feature coordinates to bin indices.

        Parameters
        ----------
    mu_1, mu_2 : float
        Feature values
    grid_resolution : int
        Number of bins per dimension
    mu_range : tuple[float, float]
        (min, max) range for mu values

        Returns
        -------
        tuple[int, int]
        (x_bin, y_bin) indices
    """
    mu_min, mu_max = mu_range

    # Clamp values to range
    mu_1_clamped = max(mu_min, min(mu_max, mu_1))
    mu_2_clamped = max(mu_min, min(mu_max, mu_2))

    # Convert to bin indices
    x_bin = int((mu_1_clamped - mu_min) / (mu_max - mu_min) * grid_resolution)
    y_bin = int((mu_2_clamped - mu_min) / (mu_max - mu_min) * grid_resolution)

    # Ensure indices are in valid range
    x_bin = max(0, min(grid_resolution - 1, x_bin))
    y_bin = max(0, min(grid_resolution - 1, y_bin))

    return x_bin, y_bin


def initialize_archive(
    num_initial: int,
    gene_space: list[dict[str, float]],
    fitness_func: Any,
    grid_resolution: int,
    mu_range: tuple[float, float],
    random_seed: int = 42,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Initialize archive with random solutions.

    Parameters
    ----------
    num_initial : int
        Number of random solutions to generate
    gene_space : list
        Gene space bounds
    fitness_func : callable
        Fitness function
    grid_resolution : int
        Grid resolution (bins per dimension)
    mu_range : tuple[float, float]
        Range for mu values
    random_seed : int
        Random seed

        Returns
        -------
    dict
        Archive mapping (x_bin, y_bin) -> best individual dict
    """
    np.random.seed(random_seed)
    archive: dict[tuple[int, int], dict[str, Any]] = {}

    print(f"Initializing archive with {num_initial} random solutions...")

    for i in range(num_initial):
        # Generate random chromosome
        chromosome = []
        for gene_bounds in gene_space:
            low = gene_bounds["low"]
            high = gene_bounds["high"]
            chromosome.append(np.random.uniform(low, high))
        chromosome = np.array(chromosome)

        # Evaluate fitness
        fitness = fitness_func(None, chromosome, 0)

        # Extract features
        mu_1, mu_2 = extract_features(chromosome)
        x_bin, y_bin = bin_coordinates(mu_1, mu_2, grid_resolution, mu_range)

        # Place in archive if better than current occupant (or empty)
        key = (x_bin, y_bin)
        if key not in archive or fitness > archive[key]["fitness"]:
            archive[key] = {
                "chromosome": chromosome.copy(),
                "fitness": float(fitness),
                "mu_1": mu_1,
                "mu_2": mu_2,
            }

        if (i + 1) % 100 == 0:
            print(f"  Initialized {i + 1}/{num_initial} solutions, archive size: {len(archive)}")

    print(f"Archive initialized: {len(archive)}/{grid_resolution * grid_resolution} cells filled")
    return archive


def mutate_chromosome(
    chromosome: np.ndarray,
    gene_space: list[dict[str, float]],
    mutation_probability: float = 0.1,
) -> np.ndarray:
    """Mutate a chromosome using Gaussian mutation with proper scaling.

    For wavelength parameters (mu): Uses sigma ~0.5-1.0 to allow movement
    between bins while preserving fitness.
    For width parameters (sigma): Uses smaller perturbation.

        Parameters
        ----------
    chromosome : np.ndarray
        Parent chromosome
    gene_space : list
        Gene space bounds
    mutation_probability : float
        Probability of mutating each gene

        Returns
        -------
        np.ndarray
            Mutated chromosome
    """
    child = chromosome.copy()

    for i, gene_bounds in enumerate(gene_space):
        if np.random.random() < mutation_probability:
            low = gene_bounds["low"]
            high = gene_bounds["high"]
            range_size = high - low

            # Determine mutation strength based on parameter type
            # Wavelength parameters (mu): indices 0, 2, 4, 6 - need stronger mutation
            # Width parameters (sigma): indices 1, 3, 5, 7 - need weaker mutation
            if i % 2 == 0:  # Wavelength parameter (mu)
                # Use sigma ~0.75 for wavelength parameters to allow bin movement
                mutation_sigma = 0.75
            else:  # Width parameter (sigma)
                # Use sigma ~0.2 for width parameters to preserve fitness
                mutation_sigma = 0.2

            # Gaussian mutation scaled by range
            mutation = np.random.normal(0, mutation_sigma * range_size)
            child[i] = child[i] + mutation

            # Clip to bounds
            child[i] = max(low, min(high, child[i]))

    return child


def run_map_elites(
    fitness_func: Any,
    gene_space: list[dict[str, float]],
    grid_resolution: int = 20,
    mu_range: tuple[float, float] = (4.0, 20.0),
    num_initial: int = 1000,
    num_iterations: int = 200000,  # Scaled up to match GA compute budget
    mutation_probability: float = 0.1,
    random_seed: int = 42,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Run MAP-Elites algorithm.

        Parameters
        ----------
    fitness_func : callable
        Fitness function
    gene_space : list
        Gene space bounds
    grid_resolution : int
        Grid resolution (bins per dimension)
    mu_range : tuple[float, float]
        Range for mu values
        num_initial : int
        Number of initial random solutions
    num_iterations : int
        Number of MAP-Elites iterations
    mutation_probability : float
        Probability of mutating each gene
    mutation_strength : float
        Standard deviation of Gaussian mutation
    random_seed : int
        Random seed

        Returns
        -------
    dict
        Final archive mapping (x_bin, y_bin) -> best individual dict
    """
    np.random.seed(random_seed)

    # Initialize archive
    archive = initialize_archive(
        num_initial=num_initial,
        gene_space=gene_space,
        fitness_func=fitness_func,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        random_seed=random_seed,
    )

    # Convert archive to list for random selection
    archive_list = list(archive.values())

    print(f"\nRunning MAP-Elites for {num_iterations} iterations...")

    # MAP-Elites main loop
    for iteration in range(num_iterations):
        # Selection: Randomly select a parent from archive
        if len(archive_list) == 0:
            # If archive is empty, generate random solution
            parent_chromosome = np.array(
                [
                    np.random.uniform(gene_bounds["low"], gene_bounds["high"])
                    for gene_bounds in gene_space
                ]
            )
        else:
            parent = archive_list[np.random.randint(len(archive_list))]
            parent_chromosome = parent["chromosome"]

            # Mutation: Create child
        child_chromosome = mutate_chromosome(
            parent_chromosome,
            gene_space,
            mutation_probability,
        )

        # Evaluation: Calculate fitness and features
        child_fitness = fitness_func(None, child_chromosome, 0)
        mu_1, mu_2 = extract_features(child_chromosome)
        x_bin, y_bin = bin_coordinates(mu_1, mu_2, grid_resolution, mu_range)

        # Placement: Place in archive if better than current occupant (or empty)
        key = (x_bin, y_bin)
        if key not in archive or child_fitness > archive[key]["fitness"]:
            archive[key] = {
                "chromosome": child_chromosome.copy(),
                "fitness": float(child_fitness),
                "mu_1": mu_1,
                "mu_2": mu_2,
            }
            # Update archive list for selection
            archive_list = list(archive.values())

        # Progress reporting (every 5,000 iterations)
        if (iteration + 1) % 5000 == 0:
            filled_cells = len(archive)
            total_cells = grid_resolution * grid_resolution
            coverage = (filled_cells / total_cells) * 100
            best_fitness = max(ind["fitness"] for ind in archive.values())
            print(
                f"  Iter: {iteration + 1} | Archive Size: {filled_cells}/{total_cells} ({coverage:.1f}%) | Global Best Fitness: {best_fitness:.2f}"
            )

    print(f"\nMAP-Elites complete: {len(archive)}/{grid_resolution * grid_resolution} cells filled")
    archive_path = Path("results/map_elites_archive.pkl")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as f:
        pickle.dump(archive, f)
    return archive


def plot_map_elites_results(
    archive: dict[tuple[int, int], dict[str, Any]],
    grid_resolution: int,
    mu_range: tuple[float, float],
    wavelengths: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate MAP-Elites visualizations.

    Creates:
    1. Heatmap showing fitness landscape
    2. Top 10 elite solutions plot
    """
    mu_min, mu_max = mu_range

    # Create fitness heatmap
    fitness_grid = np.full((grid_resolution, grid_resolution), np.nan)
    for (x_bin, y_bin), individual in archive.items():
        fitness_grid[x_bin, y_bin] = individual["fitness"]

    # Plot 1: Fitness Heatmap
    fig1, ax1 = plt.subplots(figsize=(12, 10))

    # Create meshgrid for plotting
    mu_1_bins = np.linspace(mu_min, mu_max, grid_resolution)
    mu_2_bins = np.linspace(mu_min, mu_max, grid_resolution)
    X, Y = np.meshgrid(mu_1_bins, mu_2_bins)

    # Plot heatmap
    im = ax1.imshow(
        fitness_grid.T,
        origin="lower",
        extent=[mu_min, mu_max, mu_min, mu_max],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    ax1.set_xlabel("Smallest μ (µm)", fontsize=14)
    ax1.set_ylabel("Second Smallest μ (µm)", fontsize=14)
    ax1.set_title(
        f"MAP-Elites Fitness Landscape\n"
        f"Archive Coverage: {len(archive)}/{grid_resolution * grid_resolution} cells",
        fontsize=16,
        fontweight="bold",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Fitness", fontsize=12)

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "map_elites_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to: {output_dir / 'map_elites_heatmap.png'}")
    plt.close()

    # Plot 2: Top 10 Elite Solutions
    all_individuals = list(archive.values())
    all_individuals.sort(key=lambda x: x["fitness"], reverse=True)
    top_10 = all_individuals[:10]

    fig2, ax2 = plt.subplots(figsize=(14, 8))

    num_individuals = len(top_10)

    for i, individual in enumerate(top_10):
        chromosome = individual["chromosome"]
        fitness = individual["fitness"]

        # Convert chromosome to Gaussian parameters
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)

        # Reverse offset: rank 1 gets highest offset (at top)
        vertical_offset = (num_individuals - 1 - i) * 0.1

        # Plot each basis function separately (matching tune_ga.py style)
        for _j, basis_func in enumerate(basis_functions.T):
            scaled_basis = basis_func * 0.3
            ax2.plot(
                wavelengths,
                scaled_basis + vertical_offset,
                color="red",
                alpha=0.8,
                linewidth=1.5,
            )

    ax2.set_xlabel("Wavelength (µm)", fontsize=14)
    ax2.set_ylabel("Absorptivity (Offset Applied)", fontsize=14)
    ax2.set_title(
        "MAP-Elites: Top 10 Elite Solutions",
        fontsize=16,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    # Add fitness annotations
    for i, individual in enumerate(top_10):
        fitness = individual["fitness"]
        mu_1 = individual["mu_1"]
        mu_2 = individual["mu_2"]
        ax2.text(
            0.02,
            0.98 - i * 0.09,
            f"Rank {i + 1}: F={fitness:.2f} (μ₁={mu_1:.1f}, μ₂={mu_2:.1f})",
            transform=ax2.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "map_elites_top10_elites.png", dpi=300, bbox_inches="tight")
    print(f"Saved top 10 elites plot to: {output_dir / 'map_elites_top10_elites.png'}")
    plt.close()


def main() -> None:
    """Run MAP-Elites algorithm."""
    print("=" * 80)
    print("MAP-ELITES: Multi-dimensional Archive of Phenotypic Elites")
    print("=" * 80)
    print("Strategy: Illuminate design space by maintaining elites for each")
    print("         combination of feature descriptors (mu_1, mu_2)")

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

    wavelengths_array = np.asarray(scene.wavelengths)

    fitness_func = MinDissimilarityFitnessEvaluator(
        scene=scene,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=num_params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # MAP-Elites parameters
    grid_resolution = 20  # 20x20 = 400 cells
    mu_range = (4.0, 20.0)
    num_initial = 1000
    num_iterations = 200000  # Scaled up to match GA compute budget (200,000 evaluations)

    print("\n[2/4] MAP-Elites Configuration:")
    print(
        f"  Grid Resolution: {grid_resolution}x{grid_resolution} = {grid_resolution * grid_resolution} cells"
    )
    print(f"  Feature 1 (mu_1): Range [{mu_range[0]}, {mu_range[1]}] µm")
    print(f"  Feature 2 (mu_2): Range [{mu_range[0]}, {mu_range[1]}] µm")
    print(f"  Initial Solutions: {num_initial}")
    print(f"  Iterations: {num_iterations} (matches GA compute budget: 200,000 evaluations)")
    print("  Mutation: Gaussian with sigma=0.75 for wavelengths, sigma=0.2 for widths")

    # Run MAP-Elites
    print("\n[3/4] Running MAP-Elites algorithm...")
    archive = run_map_elites(
        fitness_func=fitness_func,
        gene_space=gene_space,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        num_initial=num_initial,
        num_iterations=num_iterations,
        mutation_probability=0.1,
        random_seed=42,
    )

    # Create output directory
    output_dir = Path("outputs/map_elites/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\n[4/4] Creating visualizations...")
    plot_map_elites_results(
        archive=archive,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        wavelengths=wavelengths_array,
        output_dir=output_dir,
    )

    # Analysis
    all_individuals = list(archive.values())
    all_fitness = [ind["fitness"] for ind in all_individuals]

    print("\n" + "=" * 80)
    print("MAP-ELITES ANALYSIS")
    print("=" * 80)
    print("\nArchive Statistics:")
    print(f"  Filled Cells: {len(archive)}/{grid_resolution * grid_resolution}")
    print(f"  Coverage: {(len(archive) / (grid_resolution * grid_resolution)) * 100:.1f}%")
    print(f"  Best Fitness: {np.max(all_fitness):.4f}")
    print(f"  Mean Fitness: {np.mean(all_fitness):.4f}")
    print(f"  Std Fitness: {np.std(all_fitness):.4f}")
    print(f"  High Performers (Fitness > 50): {np.sum(np.array(all_fitness) > 50)}/{len(archive)}")

    # Top 10 summary
    all_individuals.sort(key=lambda x: x["fitness"], reverse=True)
    top_10 = all_individuals[:10]
    print("\nTop 10 Elite Solutions:")
    for i, ind in enumerate(top_10, 1):
        print(f"  {i}. Fitness={ind['fitness']:.2f}, mu₁={ind['mu_1']:.2f}, mu₂={ind['mu_2']:.2f}")

    print(f"\nPlots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
