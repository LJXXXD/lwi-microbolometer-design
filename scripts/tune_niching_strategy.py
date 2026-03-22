#!/usr/bin/env python3
"""Parallel Parameter Sweep for Niching Strategy Optimization.

This script explores the niching parameter space in parallel to find configurations
that prevent convergence to a single peak and promote multiple distinct high-performing solutions.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for multiprocessing
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


def check_top5_cloning(
    population: np.ndarray, fitness_scores: np.ndarray, threshold: float = 1e-6
) -> bool:
    """Check if top 5 solutions are identical clones.

    Returns True if top 5 are identical (within threshold), False otherwise.
    """
    if len(population) < 5:
        return False

    sorted_indices = np.argsort(fitness_scores)[::-1]
    top5 = population[sorted_indices[:5]]

    # Check if all pairwise distances are below threshold
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            distance = np.linalg.norm(top5[i] - top5[j])
            if distance > threshold:
                return False  # Found a difference

    return True  # All are identical


def plot_config_results(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    wavelengths: np.ndarray,
    config_name: str,
    output_path: Path,
) -> None:
    """Plot Top 20 individuals for a configuration."""
    # Always select Top 20 individuals
    sorted_indices = np.argsort(fitness_scores)[::-1]
    top_20_indices = sorted_indices[: min(20, len(population))]
    selected_pop = population[top_20_indices]
    selected_fitness = fitness_scores[top_20_indices]

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
        f"Top 20 Sensor Designs - {config_name}\n"
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
    plt.close()


def run_single_config(
    config_name: str,
    scene: SceneConfig,
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    niching_sigma_share: float,
    mutation_probability: float,
    parent_selection_type: str,
    output_dir: Path,
    num_generations: int = 500,
    sol_per_pop: int = 100,
    keep_elitism: int = 1,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run a single GA configuration and return results."""
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    fitness_func = MinDissimilarityFitnessEvaluator(
        scene=scene,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # Create GA configuration
    ga_config = create_ga_config(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=int(sol_per_pop * 0.5),
        keep_elitism=keep_elitism,
        parent_selection_type=parent_selection_type,
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

    # Run GA
    ga = AdvancedGA(**ga_config)
    ga.run()

    # Extract results
    best_chromosome, best_fitness, _best_idx = ga.best_solution()
    final_fitness_scores = ga.last_generation_fitness

    # Calculate Elite Diversity (Top 20)
    elite_diversity = calculate_elite_diversity(
        population=ga.population,
        fitness_scores=final_fitness_scores,
        top_n=20,
        niching_config=ga.niching_config,
    )

    # Check for top 5 cloning
    top5_cloned = check_top5_cloning(ga.population, final_fitness_scores)

    # Count high performers
    high_performers_count = int(np.sum(final_fitness_scores >= 50.0))

    # Get top 5 fitness values for analysis
    sorted_indices = np.argsort(final_fitness_scores)[::-1]
    top5_fitness = final_fitness_scores[sorted_indices[:5]]

    wavelengths_array = np.asarray(scene.wavelengths)
    plot_filename = config_name.replace("=", "_").replace(".", "p") + "_top20.png"
    plot_path = output_dir / plot_filename

    plot_config_results(
        population=ga.population,
        fitness_scores=final_fitness_scores,
        wavelengths=wavelengths_array,
        config_name=config_name,
        output_path=plot_path,
    )

    return {
        "config_name": config_name,
        "best_fitness": float(best_fitness),
        "elite_diversity": elite_diversity,
        "high_performers_count": high_performers_count,
        "top5_cloned": top5_cloned,
        "top5_fitness": top5_fitness.tolist(),
        "top5_fitness_std": float(np.std(top5_fitness)),
        "plot_path": str(plot_path),
    }


def main() -> None:
    """Run parallel parameter sweep."""
    print("=" * 80)
    print("PARALLEL NICHING STRATEGY PARAMETER SWEEP")
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

    # Define parameter grid
    niching_sigma_share_values = [2.0, 5.0, 10.0]
    mutation_probability_values = [0.1, 0.2]
    parent_selection_types = ["sus", "rws"]

    # Generate all configurations
    configurations = []
    config_idx = 0
    for sigma_share in niching_sigma_share_values:
        for mut_prob in mutation_probability_values:
            for parent_sel in parent_selection_types:
                config_idx += 1
                config_name = f"σ={sigma_share:.1f}_mut={mut_prob:.1f}_{parent_sel}"
                configurations.append(
                    {
                        "name": config_name,
                        "niching_sigma_share": sigma_share,
                        "mutation_probability": mut_prob,
                        "parent_selection_type": parent_sel,
                    }
                )

    # Create output directory for plots
    output_dir = Path("outputs/ga/niching_strategy_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_generations = 5000  # Increased to 5000 for better convergence
    print(f"\n[2/3] Running {len(configurations)} configurations in parallel...")
    print(f"      Using all {mp.cpu_count()} CPU cores")
    print(f"      Each config: {num_generations} generations, population 100")
    print(f"      Plots will be saved to: {output_dir}")
    print("      This will take longer but should show better convergence...\n")

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
                niching_sigma_share=config["niching_sigma_share"],
                mutation_probability=config["mutation_probability"],
                parent_selection_type=config["parent_selection_type"],
                output_dir=output_dir,
                num_generations=5000,  # Increased to 5000 generations
                sol_per_pop=100,
                keep_elitism=1,
                random_seed=42,
            ): config
            for config in configurations
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                print(f"  [{completed}/{len(configurations)}] ✓ {config['name']}")
            except Exception as exc:
                print(f"  ✗ Failed: {config['name']} - {exc}")
                raise

    # Sort by Elite Diversity (descending)
    results.sort(key=lambda x: x["elite_diversity"], reverse=True)

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS: Summary Table (Sorted by Elite Diversity)")
    print("=" * 80)
    print(
        f"\n{'Config':<25} | {'Best Fitness':>12} | {'Elite Div':>10} | {'High Perf':>10} | {'Top5 Cloned':>12} | {'Top5 Std':>10}"
    )
    print("-" * 80)

    for result in results:
        cloned_str = "YES ⚠️" if result["top5_cloned"] else "NO ✓"
        print(
            f"{result['config_name']:<25} | "
            f"{result['best_fitness']:>12.4f} | "
            f"{result['elite_diversity']:>10.4f} | "
            f"{result['high_performers_count']:>10} | "
            f"{cloned_str:>12} | "
            f"{result['top5_fitness_std']:>10.4f}"
        )

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find best configuration (highest elite diversity, no cloning)
    best_configs = [r for r in results if not r["top5_cloned"]]
    if best_configs:
        best = best_configs[0]
        print(f"\nBest Configuration (No Cloning): {best['config_name']}")
        print(f"  Elite Diversity: {best['elite_diversity']:.4f}")
        print(f"  Best Fitness: {best['best_fitness']:.4f}")
        print(f"  High Performers: {best['high_performers_count']}")
        print(f"  Top 5 Fitness Values: {[f'{f:.2f}' for f in best['top5_fitness'][:5]]}")
    else:
        print("\n⚠️  WARNING: All configurations show top 5 cloning!")
        print("   Consider increasing niching_sigma_share or adjusting other parameters.")

    # Count configurations with multiple high performers
    multi_high_perf = [r for r in results if r["high_performers_count"] >= 2]
    print(f"\nConfigurations with ≥2 high performers: {len(multi_high_perf)}/{len(results)}")

    if multi_high_perf:
        print("  Top configurations:")
        for r in multi_high_perf[:3]:
            print(f"    - {r['config_name']}: {r['high_performers_count']} high performers")

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Total plots generated: {len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
