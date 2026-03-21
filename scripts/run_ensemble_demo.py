#!/usr/bin/env python3
"""Ensemble Strategy: Run 20 Independent GAs to Find Multiple Distinct Peaks.

This script runs multiple independent GA instances in parallel, each converging
to its own peak. This guarantees finding distinct high-performing solution families
without relying on niching within a single population.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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


def run_single_ga(
    run_id: int,
    data: dict[str, np.ndarray | float],
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    num_generations: int = 1000,
    sol_per_pop: int = 100,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run a single independent GA instance.

    Parameters
    ----------
    run_id : int
        Unique identifier for this run
    data : dict
        Simulation data dictionary
    gene_space : list
        Gene space bounds
    params_per_basis_function : int
        Parameters per basis function
    num_generations : int
        Number of generations
    sol_per_pop : int
        Population size
    random_seed : int
        Base random seed (will be modified by run_id)

    Returns
    -------
    dict
        Results with best solution and fitness
    """
    # Use different random seed for each run
    np.random.seed(random_seed + run_id)

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

    # Create GA configuration (NO NICHING - each run converges greedily)
    ga_config = create_ga_config(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=int(sol_per_pop * 0.5),
        keep_elitism=1,  # Anchor the best result
        mutation_type=diversity_preserving_mutation,
        mutation_probability=0.1,  # Standard
        niching_enabled=False,  # CRITICAL: Turn OFF niching - each run converges to its own peak
        random_seed=random_seed + run_id,
    )

    # Add required parameters
    ga_config["num_genes"] = len(gene_space)
    ga_config["gene_space"] = gene_space
    ga_config["fitness_func"] = fitness_func

    # Run GA
    ga = AdvancedGA(**ga_config)
    ga.run()

    # Extract BEST solution (Rank 1 Champion) from this run
    best_chromosome, best_fitness, _best_idx = ga.best_solution()

    return {
        "run_id": run_id,
        "best_chromosome": best_chromosome,
        "best_fitness": float(best_fitness),
        "random_seed": random_seed + run_id,
    }


def calculate_ensemble_diversity(champions: np.ndarray) -> float:
    """Calculate diversity between ensemble champions.

    Parameters
    ----------
    champions : np.ndarray
        Array of champion chromosomes (shape: [n_champions, n_genes])

    Returns
    -------
    float
        Mean pairwise distance between champions
    """
    if len(champions) < 2:
        return 0.0

    # Use standard Euclidean distance (no niching config needed)
    distance_matrix = compute_population_distance_matrix(champions, niching_config=None)

    # Extract upper triangle (excluding diagonal) and compute mean
    n = len(champions)
    upper_triangle_indices = np.triu_indices(n, k=1)
    distances = distance_matrix[upper_triangle_indices]

    return float(np.mean(distances)) if len(distances) > 0 else 0.0


def plot_ensemble_champions(
    champions: np.ndarray,
    champion_fitness: np.ndarray,
    wavelengths: np.ndarray,
    ensemble_diversity: float,
    output_path: Path,
) -> None:
    """Plot all 20 ensemble champions matching tune_ga.py style.

    Shows individual basis functions with vertical offsets, all in RED.

    Parameters
    ----------
    champions : np.ndarray
        Array of champion chromosomes (shape: [20, n_genes])
    champion_fitness : np.ndarray
        Fitness scores for each champion
    wavelengths : np.ndarray
        Wavelength array for plotting
    ensemble_diversity : float
        Diversity score between champions
    output_path : Path
        Output file path for plot
    """
    print(f"\nPlotting {len(champions)} ensemble champions...")
    print(f"  Ensemble Diversity: {ensemble_diversity:.4f}")
    print(f"  Fitness range: {np.min(champion_fitness):.2f} - {np.max(champion_fitness):.2f}")
    print(f"  High performers (Fitness > 50): {np.sum(champion_fitness > 50)}/{len(champions)}")

    # Sort by fitness descending for consistent ordering
    sorted_indices = np.argsort(champion_fitness)[::-1]
    sorted_champions = champions[sorted_indices]
    sorted_fitness = champion_fitness[sorted_indices]

    # Create figure matching tune_ga.py style
    plt.figure(figsize=(14, 8))

    num_individuals = len(sorted_champions)

    # Plot each champion's individual basis functions with vertical offsets
    # All champions in RED (since they're all high performers)
    for i, (chromosome, fitness) in enumerate(zip(sorted_champions, sorted_fitness, strict=False)):
        # Convert chromosome to list of tuples for gaussian curves
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)
        # Reverse offset: rank 1 gets highest offset (at top), rank 20 gets lowest (at bottom)
        vertical_offset = (num_individuals - 1 - i) * 0.1

        # Plot each basis function separately (matching tune_ga.py style)
        for _j, basis_func in enumerate(basis_functions.T):
            scaled_basis = basis_func * 0.3
            plt.plot(
                wavelengths,
                scaled_basis + vertical_offset,
                color="red",
                alpha=0.3,  # High transparency to see overlaps
                linewidth=1.5,
            )

    plt.xlabel("Wavelength (µm)", fontsize=14)
    plt.ylabel("Absorptivity (Offset Applied)", fontsize=14)
    plt.title(
        "Ensemble Strategy: 50 Independent GA Champions\n"
        f"Ensemble Diversity: {ensemble_diversity:.4f} | "
        f"High Performers: {np.sum(champion_fitness > 50)}/{len(champions)}",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Add fitness annotations (matching visual order: rank 1 at top)
    # Only annotate top 20 to avoid clutter
    num_annotations = min(20, len(sorted_fitness))
    for i in range(num_annotations):
        fitness = sorted_fitness[i]
        plt.text(
            0.02,
            0.98 - i * (0.98 / num_annotations),  # Adjusted spacing
            f"Rank {i + 1}: {fitness:.2f}",
            transform=plt.gca().transAxes,
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            verticalalignment="top",
        )

    # Add summary statistics
    stats_text = (
        f"Total Champions: {len(champions)}\n"
        f"High Performers (F>50): {np.sum(champion_fitness > 50)}\n"
        f"Best Fitness: {np.max(champion_fitness):.2f}\n"
        f"Mean Fitness: {np.mean(champion_fitness):.2f}\n"
        f"Ensemble Diversity: {ensemble_diversity:.4f}"
    )
    plt.text(
        0.98,
        0.02,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9},
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close()


def main() -> None:
    """Run ensemble strategy with 20 independent GAs."""
    print("=" * 80)
    print("ENSEMBLE STRATEGY: 50 Independent GA Runs")
    print("=" * 80)
    print("Strategy: Run 50 separate GAs, each converging to its own peak")
    print("         Extract best solution from each run (50 Champions)")
    print("         Visualize all champions to find distinct solution families")

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

    # Configuration parameters
    num_runs = 50  # Increased to 50 ensembles
    num_generations = 1000
    sol_per_pop = 100
    base_random_seed = 42

    print("\n[2/4] Configuration:")
    print(f"  Number of independent runs: {num_runs}")
    print(f"  Generations per run: {num_generations}")
    print(f"  Population size: {sol_per_pop}")
    print("  Niching: DISABLED (each run converges to its own peak)")
    print("  Keep elitism: 1 (anchor best result)")

    # Run 50 independent GAs in parallel
    print(f"\n[3/4] Running {num_runs} independent GAs in parallel...")
    print(f"      Using all {mp.cpu_count()} CPU cores")
    print("      This may take a few minutes...\n")

    champions: list[np.ndarray] = []
    champion_fitness: list[float] = []
    max_workers = min(num_runs, mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_run = {
            executor.submit(
                run_single_ga,
                run_id=run_id,
                data=data,
                gene_space=gene_space,
                params_per_basis_function=num_params_per_basis_function,
                num_generations=num_generations,
                sol_per_pop=sol_per_pop,
                random_seed=base_random_seed,
            ): run_id
            for run_id in range(num_runs)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_run):
            run_id = future_to_run[future]
            try:
                result = future.result()
                champions.append(result["best_chromosome"])
                champion_fitness.append(result["best_fitness"])
                completed += 1
                print(
                    f"  [{completed}/{num_runs}] ✓ Run {run_id + 1}: Fitness={result['best_fitness']:.2f}"
                )
            except Exception as exc:
                print(f"  ✗ Failed Run {run_id + 1}: {exc}")
                raise

    # Convert to numpy arrays
    champions_array = np.array(champions)
    champion_fitness_array = np.array(champion_fitness)

    # Calculate ensemble diversity
    ensemble_diversity = calculate_ensemble_diversity(champions_array)

    # Create output directory
    output_dir = Path("outputs/ga/ensemble_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare wavelengths for plotting
    wavelengths_array = (
        data["wavelengths"]
        if isinstance(data["wavelengths"], np.ndarray)
        else np.array([float(data["wavelengths"])])
    )

    # Generate visualization
    print("\n[4/4] Creating ensemble visualization...")
    plot_path = output_dir / "ensemble_champions_overlay.png"
    plot_ensemble_champions(
        champions=champions_array,
        champion_fitness=champion_fitness_array,
        wavelengths=wavelengths_array,
        ensemble_diversity=ensemble_diversity,
        output_path=plot_path,
    )

    # Analysis
    print("\n" + "=" * 80)
    print("ENSEMBLE ANALYSIS")
    print("=" * 80)
    print(f"\nEnsemble Diversity: {ensemble_diversity:.4f}")
    print(f"Best Fitness: {np.max(champion_fitness_array):.4f}")
    print(f"Mean Fitness: {np.mean(champion_fitness_array):.4f}")
    print(f"Std Fitness: {np.std(champion_fitness_array):.4f}")
    print(f"High Performers (Fitness > 50): {np.sum(champion_fitness_array > 50)}/{num_runs}")

    # Visual analysis guidance
    print("\n" + "-" * 80)
    print("VISUAL ANALYSIS:")
    print("-" * 80)
    print("Look at the plot to identify visually distinct families:")
    print("  - Do red curves cluster into groups?")
    print("  - How many distinct peaks/families can you see?")
    print("  - Are champions spread across different wavelength regions?")
    print(f"\nPlot saved to: {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
