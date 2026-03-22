#!/usr/bin/env python3
"""Local Search Polish for MAP-Elites Winners.

This script takes the top 10 elites from MAP-Elites and refines them using
greedy hill climbing to converge to local optima.
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
from lwi_microbolometer_design.data import SceneConfig, load_substance_atmosphere_data
from lwi_microbolometer_design.ga import MinDissimilarityFitnessEvaluator

# Required for multiprocessing
mp.set_start_method("spawn", force=True)


def polish_single_elite(
    elite_id: int,
    chromosome: np.ndarray,
    initial_fitness: float,
    fitness_func: Any,
    gene_space: list[dict[str, float]],
    num_iterations: int = 4000,
    mutation_sigma: float = 0.05,
    mutation_probability: float = 0.15,
    adaptive_iterations: bool = True,
) -> dict[str, Any]:
    """Polish a single elite using greedy hill climbing.

    Parameters
    ----------
    elite_id : int
        Identifier for this elite
    chromosome : np.ndarray
        Initial chromosome to polish
    initial_fitness : float
        Initial fitness value
    fitness_func : callable
        Fitness function
    gene_space : list
        Gene space bounds
    num_iterations : int
        Number of hill climbing iterations
    mutation_sigma : float
        Standard deviation for Gaussian mutation (small for polishing)
    mutation_probability : float
        Probability of mutating each gene

    Returns
    -------
    dict
        Results with polished chromosome and fitness
    """
    # Adaptive iterations: use more iterations for promising elites
    if adaptive_iterations:
        target_fitness = 59.0
        fitness_gap = target_fitness - initial_fitness
        if fitness_gap < 3.0:  # Very close to target (fitness > 56) - maximum polish
            actual_iterations = int(num_iterations * 4.0)  # 20000 iterations for elite ones
        elif fitness_gap < 5.0:  # Close to target (fitness > 54)
            actual_iterations = int(num_iterations * 3.0)  # 15000 iterations for promising ones
        elif fitness_gap < 8.0:  # Moderate gap (fitness > 51)
            actual_iterations = int(num_iterations * 2.5)  # 12500 iterations
        else:
            actual_iterations = int(num_iterations * 2.0)  # 10000 iterations
    else:
        actual_iterations = num_iterations

    current_chromosome = chromosome.copy()
    current_fitness = initial_fitness

    best_chromosome = current_chromosome.copy()
    best_fitness = current_fitness

    for iteration in range(actual_iterations):
        # Create candidate by mutating current solution
        candidate = current_chromosome.copy()

        for i, gene_bounds in enumerate(gene_space):
            if np.random.random() < mutation_probability:
                low = gene_bounds["low"]
                high = gene_bounds["high"]
                range_size = high - low

                # Small Gaussian mutation for polishing
                mutation = np.random.normal(0, mutation_sigma * range_size)
                candidate[i] = candidate[i] + mutation

                # Clip to bounds
                candidate[i] = max(low, min(high, candidate[i]))

        # Evaluate candidate
        candidate_fitness = fitness_func(None, candidate, 0)

        # Greedy acceptance: keep if better
        if candidate_fitness > current_fitness:
            current_chromosome = candidate.copy()
            current_fitness = candidate_fitness

            # Track best ever found
            if candidate_fitness > best_fitness:
                best_chromosome = candidate.copy()
                best_fitness = candidate_fitness

    return {
        "elite_id": elite_id,
        "initial_chromosome": chromosome.copy(),
        "initial_fitness": initial_fitness,
        "polished_chromosome": best_chromosome,
        "polished_fitness": best_fitness,
        "fitness_gain": best_fitness - initial_fitness,
    }


def extract_features(chromosome: np.ndarray) -> tuple[float, float]:
    """Extract feature descriptors from chromosome.

    Features: smallest and second smallest means (mu values)
    This ensures consistent ordering regardless of which basis function has which mean.
    """
    mu_1 = float(chromosome[0])  # First Gaussian center
    mu_2 = float(chromosome[2])  # Second Gaussian center
    mu_3 = float(chromosome[4])  # Third Gaussian center
    mu_4 = float(chromosome[6])  # Fourth Gaussian center

    # Sort to get smallest and second smallest
    sorted_mus = sorted([mu_1, mu_2, mu_3, mu_4])
    return sorted_mus[0], sorted_mus[1]  # Smallest and second smallest


def run_map_elites_quick(
    fitness_func: Any,
    gene_space: list[dict[str, float]],
    grid_resolution: int = 20,
    mu_range: tuple[float, float] = (4.0, 20.0),
    num_initial: int = 1000,
    num_iterations: int = 2000,
    random_seed: int = 42,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Run MAP-Elites to get top elites (reuse logic from run_map_elites.py)."""
    import sys

    _scripts_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_scripts_root))
    from run_map_elites import (
        bin_coordinates,
        initialize_archive,
        mutate_chromosome,
    )

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

    archive_list = list(archive.values())

    # Run MAP-Elites main loop (abbreviated for speed - we just need top 10)
    for iteration in range(num_iterations):
        if len(archive_list) == 0:
            parent_chromosome = np.array(
                [
                    np.random.uniform(gene_bounds["low"], gene_bounds["high"])
                    for gene_bounds in gene_space
                ]
            )
        else:
            parent = archive_list[np.random.randint(len(archive_list))]
            parent_chromosome = parent["chromosome"]

        child_chromosome = mutate_chromosome(parent_chromosome, gene_space, 0.1)
        child_fitness = fitness_func(None, child_chromosome, 0)
        mu_1, mu_2 = extract_features(child_chromosome)
        x_bin, y_bin = bin_coordinates(mu_1, mu_2, grid_resolution, mu_range)

        key = (x_bin, y_bin)
        if key not in archive or child_fitness > archive[key]["fitness"]:
            archive[key] = {
                "chromosome": child_chromosome.copy(),
                "fitness": float(child_fitness),
                "mu_1": mu_1,
                "mu_2": mu_2,
            }
            archive_list = list(archive.values())

    return archive


def plot_map_elites_heatmap(
    archive: dict[tuple[int, int], dict[str, Any]],
    grid_resolution: int,
    mu_range: tuple[float, float],
    output_path: Path,
) -> None:
    """Generate MAP-Elites fitness heatmap visualization."""
    mu_min, mu_max = mu_range

    # Create fitness heatmap
    fitness_grid = np.full((grid_resolution, grid_resolution), np.nan)
    for (x_bin, y_bin), individual in archive.items():
        fitness_grid[x_bin, y_bin] = individual["fitness"]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(
        fitness_grid.T,
        origin="lower",
        extent=[mu_min, mu_max, mu_min, mu_max],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    ax.set_xlabel("Smallest μ (µm)", fontsize=14)
    ax.set_ylabel("Second Smallest μ (µm)", fontsize=14)
    ax.set_title(
        f"MAP-Elites Fitness Landscape\n"
        f"Archive Coverage: {len(archive)}/{grid_resolution * grid_resolution} cells",
        fontsize=16,
        fontweight="bold",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fitness", fontsize=12)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved heatmap to: {output_path}")
    plt.close()


def plot_polished_elites(
    initial_elites: list[dict[str, Any]],
    polished_elites: list[dict[str, Any]],
    wavelengths: np.ndarray,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot polished top N elites matching tune_ga.py style."""
    # Sort by polished fitness
    sorted_indices = np.argsort([e["polished_fitness"] for e in polished_elites])[::-1]
    sorted_polished = [polished_elites[i] for i in sorted_indices]
    sorted_initial = [initial_elites[i] for i in sorted_indices]

    # Take top N for plotting
    top_n_actual = min(top_n, len(sorted_polished))
    sorted_polished = sorted_polished[:top_n_actual]
    sorted_initial = sorted_initial[:top_n_actual]

    plt.figure(figsize=(14, 10))  # Increased height for better spacing

    num_individuals = len(sorted_polished)

    # Fixed vertical offset spacing (increased for better separation)
    offset_step = 0.2
    # Calculate all offsets and find max for Y-axis limits
    all_offsets = []
    all_curve_maxes = []

    for i, (polished, initial) in enumerate(zip(sorted_polished, sorted_initial, strict=False)):
        chromosome = polished["polished_chromosome"]

        # Convert chromosome to Gaussian parameters
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)

        # Reverse offset: rank 1 gets highest offset (at top)
        vertical_offset = (num_individuals - 1 - i) * offset_step
        all_offsets.append(vertical_offset)

        # Track max curve height for this individual
        for basis_func in basis_functions.T:
            scaled_basis = basis_func * 0.3
            all_curve_maxes.append(float(np.max(scaled_basis + vertical_offset)))

    # Plot all curves
    for i, (polished, initial) in enumerate(zip(sorted_polished, sorted_initial, strict=False)):
        chromosome = polished["polished_chromosome"]

        # Convert chromosome to Gaussian parameters
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)

        # Reverse offset: rank 1 gets highest offset (at top)
        vertical_offset = (num_individuals - 1 - i) * offset_step

        # Plot each basis function separately (matching tune_ga.py style)
        for _j, basis_func in enumerate(basis_functions.T):
            scaled_basis = basis_func * 0.3
            plt.plot(
                wavelengths,
                scaled_basis + vertical_offset,
                color="red",
                alpha=0.7,  # Reduced from 0.8
                linewidth=1.5,
            )

    # Set Y-axis limits with proper padding
    y_min = -0.1
    y_max = max(all_curve_maxes) + 0.5  # Add padding so top curve isn't cut off
    plt.ylim(y_min, y_max)

    plt.xlabel("Wavelength (µm)", fontsize=14)
    plt.ylabel("Absorptivity (Offset Applied)", fontsize=14)
    plt.title(
        f"MAP-Elites + Local Search Polish: Top {top_n_actual} Refined Solutions",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Add fitness annotations showing improvement (smaller font, better spacing)
    for i, (polished, initial) in enumerate(zip(sorted_polished, sorted_initial, strict=False)):
        polished_fitness = polished["polished_fitness"]
        initial_fitness = initial["fitness"]
        fitness_gain = polished["fitness_gain"]

        # Calculate text position based on number of individuals
        text_y_pos = 0.98 - (i * (0.95 / num_individuals))  # Distribute evenly

        plt.text(
            0.02,
            text_y_pos,
            f"Rank {i + 1}: {initial_fitness:.2f} → {polished_fitness:.2f} (+{fitness_gain:.2f})",
            transform=plt.gca().transAxes,
            fontsize=8,  # Reduced from 9
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},  # Reduced alpha
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved polished elites plot to: {output_path}")
    plt.close()


def main() -> None:
    """Run local search polish on MAP-Elites top 10 elites."""
    print("=" * 80)
    print("LOCAL SEARCH POLISH: Refining MAP-Elites Winners")
    print("=" * 80)
    print("Strategy: Take top 100 MAP-Elites elites and polish with adaptive greedy hill climbing")
    print("Goal: Achieve 50+ solutions close to fitness ~59 with high diversity")
    print("      Compare to GA ensemble (50 champions)")

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
        wavelengths=scene.wavelengths,
        emissivity_curves=scene.emissivity_curves,
        temperature_k=scene.temperature_k,
        atmospheric_distance_ratio=scene.atmospheric_distance_ratio,
        air_refractive_index=scene.air_refractive_index,
        air_transmittance=scene.air_transmittance,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=num_params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # Get top 10 elites from MAP-Elites
    print("\n[2/4] Running MAP-Elites to get top 10 elites...")
    print("      (This may take a few minutes to reach convergence...)")

    archive = run_map_elites_quick(
        fitness_func=fitness_func,
        gene_space=gene_space,
        grid_resolution=30,  # 30x30 grid for finer-grained exploration (900 cells)
        mu_range=(4.0, 20.0),
        num_initial=2500,  # Best from sweep: G30_I2.5k_Iter1000k
        num_iterations=1000000,  # Best from sweep: 1M iterations (improved from 600k)
        random_seed=42,
    )

    # Extract top elites - use lower threshold to get more candidates
    all_individuals = list(archive.values())
    all_individuals.sort(key=lambda x: x["fitness"], reverse=True)

    # Use lower threshold (45 instead of 50) to get more promising elites
    # The polishing will improve them significantly
    promising_elites = [ind for ind in all_individuals if ind["fitness"] > 45.0]

    # Take top 120 from promising elites (increased to get maximum diverse high performers)
    num_elites_to_polish = min(120, len(promising_elites))
    top_elites_initial = promising_elites[:num_elites_to_polish]

    print("\nArchive Statistics:")
    print(f"  Total archive size: {len(all_individuals)} cells")
    print(f"  Best fitness in archive: {max(ind['fitness'] for ind in all_individuals):.2f}")
    print(f"  Mean fitness in archive: {np.mean([ind['fitness'] for ind in all_individuals]):.2f}")
    print(
        f"  Filtered to {len(top_elites_initial)} promising elites (fitness > 45) from {len(all_individuals)} total"
    )

    print(f"\nTop {num_elites_to_polish} Initial Elites (Before Polish):")
    for i, ind in enumerate(top_elites_initial[:10], 1):  # Show first 10
        print(f"  {i}. Fitness={ind['fitness']:.2f}, μ₁={ind['mu_1']:.2f}, μ₂={ind['mu_2']:.2f}")
    if len(top_elites_initial) > 10:
        print(f"  ... and {len(top_elites_initial) - 10} more")

    # Polish each elite in parallel
    print(f"\n[3/4] Polishing top {num_elites_to_polish} elites with adaptive local search...")
    print("      Base: 5000 iterations, adaptive up to 20000 for promising elites")
    print(f"      Using all {mp.cpu_count()} CPU cores\n")

    polished_results: list[dict[str, Any]] = []
    max_workers = min(num_elites_to_polish, mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all polishing jobs
        future_to_elite = {
            executor.submit(
                polish_single_elite,
                elite_id=i,
                chromosome=elite["chromosome"],
                initial_fitness=elite["fitness"],
                fitness_func=fitness_func,
                gene_space=gene_space,
                num_iterations=5000,  # Increased base iterations (adaptive will increase for promising elites)
                mutation_sigma=0.05,  # Small mutation for polishing
                mutation_probability=0.2,  # Increased mutation prob for better exploration
                adaptive_iterations=True,  # Use more iterations for elites closer to 59
            ): (i, elite)
            for i, elite in enumerate(top_elites_initial)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_elite):
            elite_id, elite = future_to_elite[future]
            try:
                result = future.result()
                polished_results.append(result)
                completed += 1
                if completed <= 10 or completed % 5 == 0:  # Show first 10, then every 5th
                    print(
                        f"  [{completed}/{num_elites_to_polish}] ✓ Elite {elite_id + 1}: "
                        f"{result['initial_fitness']:.2f} → {result['polished_fitness']:.2f} "
                        f"(+{result['fitness_gain']:.2f})"
                    )
            except Exception as exc:
                print(f"  ✗ Failed Elite {elite_id + 1}: {exc}")
                raise

    # Sort by polished fitness
    polished_results.sort(key=lambda x: x["polished_fitness"], reverse=True)

    # Create output directory
    output_dir = Path("outputs/map_elites/polished")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results for later plot regeneration
    import pickle

    results_file = output_dir / "polished_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(
            {
                "polished_results": polished_results,
                "top_elites_initial": top_elites_initial,
                "wavelengths": wavelengths_array,
            },
            f,
        )
    print(f"\nSaved results to: {results_file} (for plot regeneration)")

    # Generate visualization
    print("\n[4/4] Creating polished elites visualization...")

    # Generate heatmap from MAP-Elites archive
    print("      Generating MAP-Elites fitness heatmap...")
    plot_map_elites_heatmap(
        archive=archive,
        grid_resolution=30,  # Match the grid resolution used
        mu_range=(4.0, 20.0),
        output_path=output_dir / "map_elites_heatmap.png",
    )

    # Generate top 50 plot for comparison with GA ensemble
    plot_path_top50 = output_dir / "polished_top50_elites.png"
    print(f"      Generating top 50 plot: {plot_path_top50}")
    plot_polished_elites(
        initial_elites=top_elites_initial,
        polished_elites=polished_results,
        wavelengths=wavelengths_array,
        output_path=plot_path_top50,
        top_n=50,  # Plot top 50 for comparison with GA ensemble
    )

    # Also generate top 20 version
    plot_path_top20 = output_dir / "polished_top20_elites.png"
    print(f"      Generating top 20 plot: {plot_path_top20}")
    plot_polished_elites(
        initial_elites=top_elites_initial,
        polished_elites=polished_results,
        wavelengths=wavelengths_array,
        output_path=plot_path_top20,
        top_n=20,
    )

    # Calculate diversity among top performers
    top_performers_threshold = 58.0  # Consider solutions with fitness >= 58 as "top performers"
    top_performers = [
        r for r in polished_results if r["polished_fitness"] >= top_performers_threshold
    ]

    if len(top_performers) > 1:
        from lwi_microbolometer_design.ga import compute_population_distance_matrix

        top_performer_chromosomes = np.array([r["polished_chromosome"] for r in top_performers])
        top_performer_distances = compute_population_distance_matrix(
            top_performer_chromosomes, None
        )
        n = len(top_performers)
        upper_triangle_indices = np.triu_indices(n, k=1)
        top_performer_diversity = float(np.mean(top_performer_distances[upper_triangle_indices]))
    else:
        top_performer_diversity = 0.0

    # Print summary table (show top 50, highlight those >= 58)
    print("\n" + "=" * 80)
    print("POLISHING RESULTS: Before vs After Comparison (Top 50)")
    print("=" * 80)
    print(
        f"\n{'Rank':<6} | {'Before Polish':>12} | {'After Polish':>12} | {'Gain':>10} | {'μ₁':>8} | {'μ₂':>8}"
    )
    print("-" * 80)

    for i, result in enumerate(polished_results[:50], 1):  # Show top 50
        mu_1, mu_2 = extract_features(result["polished_chromosome"])
        marker = "★" if result["polished_fitness"] >= top_performers_threshold else " "
        print(
            f"{i:<6} | "
            f"{result['initial_fitness']:>12.4f} | "
            f"{result['polished_fitness']:>12.4f} | "
            f"{result['fitness_gain']:>10.4f} | "
            f"{mu_1:>8.2f} | "
            f"{mu_2:>8.2f} {marker}"
        )

    # Summary statistics
    initial_fitnesses = [r["initial_fitness"] for r in polished_results]
    polished_fitnesses = [r["polished_fitness"] for r in polished_results]
    fitness_gains = [r["fitness_gain"] for r in polished_results]

    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"  Mean Initial Fitness: {np.mean(initial_fitnesses):.4f}")
    print(f"  Mean Polished Fitness: {np.mean(polished_fitnesses):.4f}")
    print(f"  Mean Fitness Gain: {np.mean(fitness_gains):.4f}")
    print(f"  Best Polished Fitness: {np.max(polished_fitnesses):.4f}")
    print("\n  High Performers After Polish:")
    print(f"    Fitness > 50: {np.sum(np.array(polished_fitnesses) > 50)}/{len(polished_results)}")
    print(f"    Fitness > 55: {np.sum(np.array(polished_fitnesses) > 55)}/{len(polished_results)}")
    print(f"    Fitness > 58: {np.sum(np.array(polished_fitnesses) > 58)}/{len(polished_results)}")
    print(f"    Fitness > 59: {np.sum(np.array(polished_fitnesses) > 59)}/{len(polished_results)}")

    # Comparison with GA ensemble (50 champions)
    print("\n  Comparison with GA Ensemble (50 Champions):")
    top_50_polished = polished_results[:50]
    top_50_fitnesses = [r["polished_fitness"] for r in top_50_polished]
    print(f"    Top 50 Mean Fitness: {np.mean(top_50_fitnesses):.4f}")
    print(f"    Top 50 Best Fitness: {np.max(top_50_fitnesses):.4f}")
    print(f"    Top 50 with Fitness >= 58: {np.sum(np.array(top_50_fitnesses) >= 58)}/50")
    print(f"    Top 50 with Fitness >= 59: {np.sum(np.array(top_50_fitnesses) >= 59)}/50")

    # Calculate diversity of top 50
    if len(top_50_polished) > 1:
        from lwi_microbolometer_design.ga import compute_population_distance_matrix

        top_50_chromosomes = np.array([r["polished_chromosome"] for r in top_50_polished])
        top_50_distances = compute_population_distance_matrix(top_50_chromosomes, None)
        n = len(top_50_polished)
        upper_triangle_indices = np.triu_indices(n, k=1)
        top_50_diversity = float(np.mean(top_50_distances[upper_triangle_indices]))
        print(f"    Top 50 Diversity Score: {top_50_diversity:.4f}")
        print("      (Compare to GA ensemble diversity - higher = more diverse families)")

    # Diversity analysis
    print(f"\n  Diversity Analysis (Top Performers with Fitness >= {top_performers_threshold}):")
    print(f"    Number of Top Performers: {len(top_performers)}")
    if len(top_performers) > 1:
        print(f"    Diversity Among Top Performers: {top_performer_diversity:.4f}")
        print("    (Higher = more diverse, Lower = more similar)")

        # Analyze distinct families
        if top_performer_diversity > 5.0:
            print("    ✓ GOOD: Top performers are diverse (likely multiple distinct families)")
        elif top_performer_diversity > 2.0:
            print("    ⚠ MODERATE: Top performers show some diversity")
        else:
            print("    ✗ LOW: Top performers are similar (converging to one family)")

        # Show feature space distribution
        top_performer_mu1 = [extract_features(r["polished_chromosome"])[0] for r in top_performers]
        top_performer_mu2 = [extract_features(r["polished_chromosome"])[1] for r in top_performers]
        print("\n    Feature Space Distribution:")
        print(f"      μ₁ range: {np.min(top_performer_mu1):.2f} - {np.max(top_performer_mu1):.2f}")
        print(f"      μ₂ range: {np.min(top_performer_mu2):.2f} - {np.max(top_performer_mu2):.2f}")
    else:
        print(f"    ⚠ Only {len(top_performers)} top performer found - need more exploration")

    print("\nPlots saved to:")
    print(f"  - Top 50: {plot_path_top50}")
    print(f"  - Top 20: {plot_path_top20}")
    print("=" * 80)


if __name__ == "__main__":
    main()
