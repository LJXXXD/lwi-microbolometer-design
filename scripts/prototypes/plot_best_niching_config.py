#!/usr/bin/env python3
"""Plot top performers for the best niching configuration from parameter sweep.

This script runs the best configuration (σ=10.0_mut=0.1_sus) and generates
the visualization showing top 20 performers.
"""

import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data import load_substance_atmosphere_data
from lwi_microbolometer_design.ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    compute_population_distance_matrix,
    create_ga_config,
    diversity_preserving_mutation,
)

# Required for multiprocessing
mp.set_start_method("spawn", force=True)


def calculate_elite_diversity(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    top_n: int = 20,
    niching_config=None,
) -> float:
    """Calculate diversity score for only the top N elite individuals."""
    if len(population) < 2:
        return 0.0

    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite_indices = sorted_indices[: min(top_n, len(population))]
    elite_population = population[elite_indices]

    if len(elite_population) < 2:
        return 0.0

    distance_matrix = compute_population_distance_matrix(elite_population, niching_config)
    n = len(elite_population)
    upper_triangle_indices = np.triu_indices(n, k=1)
    distances = distance_matrix[upper_triangle_indices]

    return float(np.mean(distances)) if len(distances) > 0 else 0.0


def plot_population_curves(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    wavelengths: np.ndarray,
    output_path: Path,
) -> None:
    """Plot Top 20 individuals matching tune_ga.py style."""
    # Always select Top 20 individuals
    sorted_indices = np.argsort(fitness_scores)[::-1]
    top_20_indices = sorted_indices[: min(20, len(population))]
    selected_pop = population[top_20_indices]
    selected_fitness = fitness_scores[top_20_indices]

    print("\nPlotting Top 20 individuals")
    print(f"  High performers (Fitness > 50): {np.sum(selected_fitness > 50)}")
    print(f"  Lower performers (Fitness ≤ 50): {np.sum(selected_fitness <= 50)}")

    # Create figure matching tune_ga.py style
    plt.figure(figsize=(14, 8))

    num_individuals = len(selected_pop)

    # Use color coding: Red for high performers, Blue for lower performers
    for i, (chromosome, fitness) in enumerate(zip(selected_pop, selected_fitness, strict=False)):
        # Convert chromosome to list of tuples for gaussian curves
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)
        # Reverse offset: rank 1 gets highest offset (at top), rank 20 gets lowest (at bottom)
        vertical_offset = (num_individuals - 1 - i) * 0.1

        # Color logic: Red if Fitness > 50, Blue if ≤ 50
        if fitness > 50:
            color = "red"
        else:
            color = "steelblue"

        # Plot each basis function separately (matching tune_ga.py style)
        for _j, basis_func in enumerate(basis_functions.T):
            scaled_basis = basis_func * 0.3
            plt.plot(
                wavelengths,
                scaled_basis + vertical_offset,
                color=color,
                alpha=0.8,
                linewidth=1.5,
            )

    plt.xlabel("Wavelength (µm)", fontsize=14)
    plt.ylabel("Absorptivity (Offset Applied)", fontsize=14)
    plt.title(
        "Top 20 Sensor Designs - Best Niching Configuration\n"
        "Red = High Performers (Fitness > 50), Blue = Lower Performers (Fitness ≤ 50)",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Add fitness annotations (matching visual order: rank 1 at top)
    for i, fitness in enumerate(selected_fitness):
        plt.text(
            0.02,
            0.98 - i * 0.045,  # Adjusted spacing for 20 individuals
            f"Rank {i + 1}: {fitness:.2f}",
            transform=plt.gca().transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close()


def main() -> None:
    """Run best configuration and generate plot."""
    print("=" * 80)
    print("PLOTTING BEST NICHING CONFIGURATION")
    print("=" * 80)
    print("Configuration: σ=10.0, mutation=0.1, parent_selection=sus")
    print("(This was the best config with 4 high performers)")

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
        data = loaded_data[0]
    else:
        data = loaded_data

    # Sensor configuration
    num_basis_functions = 4
    num_params_per_basis_function = 2
    param_bounds = [
        {"low": 4.0, "high": 20.0},  # mu (wavelength center)
        {"low": 0.1, "high": 4.0},  # sigma (width)
    ]
    gene_space = param_bounds * num_basis_functions

    # Prepare fitness function
    wavelengths_val = data["wavelengths"]
    emissivity_val = data["emissivity_curves"]
    temp_k_val = data["temperature_K"]
    atm_dist_val = data["atmospheric_distance_ratio"]
    air_ref_idx_val = data["air_refractive_index"]
    air_trans_val = data["air_transmittance"]

    wavelengths_array = (
        wavelengths_val
        if isinstance(wavelengths_val, np.ndarray)
        else np.array([float(wavelengths_val)])
    )
    emissivity_array = (
        emissivity_val
        if isinstance(emissivity_val, np.ndarray)
        else np.array([float(emissivity_val)])
    )
    temperature_float = float(temp_k_val) if not isinstance(temp_k_val, float) else temp_k_val
    atm_dist_float = float(atm_dist_val) if not isinstance(atm_dist_val, float) else atm_dist_val
    air_ref_idx_float = (
        float(air_ref_idx_val) if not isinstance(air_ref_idx_val, float) else air_ref_idx_val
    )
    air_trans_array = (
        air_trans_val if isinstance(air_trans_val, np.ndarray) else np.array([float(air_trans_val)])
    )

    fitness_func = MinDissimilarityFitnessEvaluator(
        wavelengths=wavelengths_array,
        emissivity_curves=emissivity_array,
        temperature_k=temperature_float,
        atmospheric_distance_ratio=atm_dist_float,
        air_refractive_index=air_ref_idx_float,
        air_transmittance=air_trans_array,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=num_params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # Create best configuration
    print("\n[2/3] Running GA with best configuration (500 generations)...")
    ga_config = create_ga_config(
        num_generations=500,
        sol_per_pop=100,
        num_parents_mating=50,
        keep_elitism=1,
        parent_selection_type="sus",
        mutation_type=diversity_preserving_mutation,
        mutation_probability=0.1,
        niching_enabled=True,
        niching_sigma_share=10.0,  # Best from sweep
        niching_alpha=1.0,
        random_seed=42,
    )

    ga_config["num_genes"] = len(gene_space)
    ga_config["gene_space"] = gene_space
    ga_config["fitness_func"] = fitness_func

    # Run GA
    ga = AdvancedGA(**ga_config)
    ga.run()

    # Extract results
    best_chromosome, best_fitness, _best_idx = ga.best_solution()
    final_fitness_scores = ga.last_generation_fitness

    # Calculate Elite Diversity
    elite_diversity = calculate_elite_diversity(
        population=ga.population,
        fitness_scores=final_fitness_scores,
        top_n=20,
        niching_config=ga.niching_config,
    )

    print("\nResults:")
    print(f"  Best Fitness: {best_fitness:.4f}")
    print(f"  Elite Diversity (Top 20): {elite_diversity:.4f}")
    print(f"  High performers (Fitness > 50): {np.sum(final_fitness_scores >= 50)}")

    # Create output directory
    output_dir = Path("outputs/ga/best_niching_config")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    print("\n[3/3] Creating visualization...")
    plot_path = output_dir / "04_top_sensor_designs.png"
    plot_population_curves(
        population=ga.population,
        fitness_scores=final_fitness_scores,
        wavelengths=wavelengths_array,
        output_path=plot_path,
    )

    print("\n" + "=" * 80)
    print("PLOT GENERATED")
    print("=" * 80)
    print(f"\nPlot saved to: {plot_path}")
    print("\nCheck the plot to see if multiple distinct red groups (high performers) exist!")
    print("=" * 80)


if __name__ == "__main__":
    main()
