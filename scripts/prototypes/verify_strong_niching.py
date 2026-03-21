#!/usr/bin/env python3
"""Verify Strong Niching configuration with visualization.

This script runs the "Strong Niching" configuration with intermediate generations
and creates visualizations to verify that diversity scores match visual appearance.
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
    calculate_population_diversity,
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
    """Calculate diversity score for only the top N elite individuals.

    This metric focuses on whether the high-performing solutions are diverse,
    rather than including low-fitness individuals in the calculation.

    Parameters
    ----------
    population : np.ndarray
        Full population array
    fitness_scores : np.ndarray
        Fitness scores for each individual
    top_n : int
        Number of top individuals to consider (default: 20)
    niching_config : optional
        Niching configuration for distance calculation

    Returns
    -------
    float
        Mean pairwise distance within the elite group
    """
    if len(population) < 2:
        return 0.0

    # Sort by fitness descending
    sorted_indices = np.argsort(fitness_scores)[::-1]

    # Select top N elites
    elite_indices = sorted_indices[: min(top_n, len(population))]
    elite_population = population[elite_indices]

    if len(elite_population) < 2:
        return 0.0

    # Calculate distance matrix for elites only
    distance_matrix = compute_population_distance_matrix(elite_population, niching_config)

    # Extract upper triangle (excluding diagonal) and compute mean
    n = len(elite_population)
    upper_triangle_indices = np.triu_indices(n, k=1)
    distances = distance_matrix[upper_triangle_indices]

    return float(np.mean(distances)) if len(distances) > 0 else 0.0


def run_strong_niching_ga(
    data: dict[str, np.ndarray | float],
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    num_generations: int = 500,
    sol_per_pop: int = 100,
    random_seed: int = 42,
) -> dict:
    """Run Strong Niching GA configuration."""
    # Set random seed
    np.random.seed(random_seed)

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
        params_per_basis_function=params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # Track diversity history
    diversity_history: list[float] = []

    def on_generation(ga_instance) -> None:
        """Track diversity per generation."""
        diversity = calculate_population_diversity(
            ga_instance.population, ga_instance.niching_config
        )
        diversity_history.append(diversity)

    # Create Multimodal configuration (break elitism cloning)
    # Updated: keep_elitism=1 (minimal elitism) + more generations + SUS selection
    ga_config = create_ga_config(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=int(sol_per_pop * 0.5),
        keep_elitism=1,  # Minimal elitism (prevents total loss of best solution)
        parent_selection_type="sus",  # Stochastic universal selection (better balance)
        mutation_type=diversity_preserving_mutation,
        mutation_probability=0.1,  # Stabilize search
        niching_enabled=True,
        niching_sigma_share=2.0,  # Large niche radius for distinct peaks
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

    # Calculate Elite Diversity (Top 20 only)
    elite_diversity = calculate_elite_diversity(
        population=ga.population,
        fitness_scores=final_fitness_scores,
        top_n=20,
        niching_config=ga.niching_config,
    )

    return {
        "ga": ga,
        "best_fitness": float(best_fitness),
        "best_chromosome": best_chromosome,
        "final_population": ga.population,
        "final_fitness_scores": final_fitness_scores,
        "final_diversity": float(final_diversity),
        "elite_diversity": elite_diversity,
        "diversity_history": diversity_history,
        "wavelengths": wavelengths_array,
    }


def plot_population_curves(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    wavelengths: np.ndarray,
    best_fitness: float,
    output_path: Path,
) -> None:
    """Plot Top 20 individuals matching tune_ga.py style (04_top_sensor_designs.png).

    Shows individual basis functions with vertical offsets, color-coded by fitness.
    """
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

    # Reverse order: rank 1 at top (highest offset), rank 20 at bottom (lowest offset)
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
        "Top 20 Sensor Designs\n"
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
    """Run verification and visualization."""
    print("=" * 80)
    print("STRONG NICHING VERIFICATION: Diversity Metric Analysis & Visualization")
    print("=" * 80)

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

    # Explain diversity metric
    print("\n" + "=" * 80)
    print("DIVERSITY METRIC EXPLANATION")
    print("=" * 80)
    print("""
The diversity score is calculated as:
  1. Compute pairwise Euclidean distances between all individuals in parameter space
  2. Take the mean of all pairwise distances (excluding self-distances)

Parameter Space:
  - Each individual has 8 parameters: [μ₁, σ₁, μ₂, σ₂, μ₃, σ₃, μ₄, σ₄]
  - μ (mu) ranges: [4.0, 20.0] → range = 16.0 μm
  - σ (sigma) ranges: [0.1, 4.0] → range = 3.9 μm
  - Total parameter space: 8-dimensional

Interpretation:
  - Baseline (~2.0): Mean pairwise distance of 2.0 units in 8D space
    → Solutions are relatively clustered
  - Strong Niching (~5.8): Mean pairwise distance of 5.8 units
    → Solutions are ~2.9x more spread out than Baseline
    → This suggests distinct solution families
  - Chaotic (~11.4): Mean pairwise distance of 11.4 units
    → Solutions are very spread out, possibly too diverse

Physical Meaning:
  A diversity of 5.8 means on average, two random individuals differ by:
  - Roughly 5.8 units in the combined 8D parameter space
  - This could mean: different peak positions (μ), different widths (σ), or both
  - Visually: Sensor curves should show distinct shapes/positions

Does it align with visual distinctness?
  YES - Higher diversity should correspond to visually distinct sensor curves.
  However, we need to verify this matches what we see in the plots.
    """)

    # Run GA with improved multimodal configuration
    num_generations = 2000  # Increased to 2000 for better convergence
    print("\n[2/3] Running Improved Multimodal GA Configuration...")
    print("Configuration: keep_elitism=1, parent_selection=sus, niching_sigma_share=2.0")
    print(f"Generations: {num_generations} (increased for better convergence)")
    print("Goal: Find multiple distinct high-performing peaks (not clones)")
    print(f"Running GA: {num_generations} generations, population 100...")
    print("This may take a few minutes...")

    # For now, run single instance (can be parallelized if multiple configs needed)
    results = run_strong_niching_ga(
        data=data,
        gene_space=gene_space,
        params_per_basis_function=num_params_per_basis_function,
        num_generations=num_generations,
        sol_per_pop=100,
        random_seed=42,
    )

    print("\nResults:")
    print(f"  Best Fitness: {results['best_fitness']:.4f}")
    print(f"  Final Diversity (Full Population): {results['final_diversity']:.4f}")
    print(f"  Elite Diversity (Top 20 Only): {results['elite_diversity']:.4f}")

    # Create output directory
    output_dir = Path("outputs/ga/strong_niching_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    print("\n[3/3] Creating visualization...")
    plot_path = output_dir / "04_top_sensor_designs.png"
    plot_population_curves(
        population=results["final_population"],
        fitness_scores=results["final_fitness_scores"],
        wavelengths=results["wavelengths"],
        best_fitness=results["best_fitness"],
        output_path=plot_path,
    )

    # Analysis
    print("\n" + "=" * 80)
    print("VISUAL VERIFICATION ANALYSIS")
    print("=" * 80)
    print("\nDiversity Metrics:")
    print(f"  Full Population Diversity: {results['final_diversity']:.4f}")
    print(f"  Elite Diversity (Top 20): {results['elite_diversity']:.4f}")
    print(f"\nBest Fitness: {results['best_fitness']:.4f}")

    high_quality_count = np.sum(results["final_fitness_scores"] >= 50)
    print(f"High-quality solutions (fitness ≥ 50): {high_quality_count}")

    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    print("The Elite Diversity metric focuses on whether the WINNERS are diverse.")
    print("This is more meaningful than full population diversity.")
    print(f"\nElite Diversity Score: {results['elite_diversity']:.4f}")
    if results["elite_diversity"] < 2.0:
        print("  → Low diversity: Elite solutions are clustered (converged to similar designs)")
    elif results["elite_diversity"] < 5.0:
        print("  → Moderate diversity: Some variation among elite solutions")
    else:
        print("  → High diversity: Elite solutions are spread out (multiple distinct families)")

    # Check for multiple high-performing groups
    high_performers = results["final_fitness_scores"][results["final_fitness_scores"] > 50]
    print(f"\nHigh Performers (Fitness > 50): {len(high_performers)}")

    if len(high_performers) >= 2:
        print("  ✓ SUCCESS: Multiple high-performing solutions found!")
        print("  Check plot: Do red curves form distinct groups?")
    else:
        print("  ⚠ WARNING: Only one or no high-performing solution found")
        print("  May need more generations or different niching parameters")

    print("\nVisual Check: Look at the Smart Plot")
    print("  - Red curves (High performers): Do they form DISTINCT GROUPS?")
    print("    * SUCCESS: Multiple red groups at different wavelengths → Multimodal!")
    print("    * FAILURE: All red curves overlap → Still converging to one peak")
    print("  - Blue curves (Lower performers): Are they exploring different regions?")
    print("  - If red curves overlap heavily → Low elite diversity (converged)")
    print("  - If red curves are spread out → High elite diversity (multiple solutions)")

    print(f"\nPlot saved to: {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
