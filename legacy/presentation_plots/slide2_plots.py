#!/usr/bin/env python3
"""Slide 2: Limitations of Single-Population Optimization.

This script generates plots for Slide 2:

1. Left: Table summarizing fitness + diversity metrics for several GA
   configurations using niching and custom mutation (slide2-1_table.png).
2. Right: Population curves for one "Strong Niching + Custom Mutation"
   run, showing that only the top 1 converges while the rest are diverse
   but low fitness (slide2-2_strong_niching.png).

The goal is to visually support the narrative:
- Hypothesis: Can we force diversity using Niching and Custom Mutation?
- Observation: Strong Niching penalized fitness too heavily; Weak Niching
  failed to prevent convergence.
- Insight: Standard GA is fundamentally designed for convergence, making
  it inefficient for maintaining multiple simultaneous peaks.
"""

from __future__ import annotations

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
    calculate_population_diversity,
    compute_population_distance_matrix,
    create_ga_config,
    diversity_preserving_mutation,
)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def calculate_elite_diversity(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    top_n: int = 20,
    niching_config: Any | None = None,
) -> float:
    """Calculate diversity score for only the top N elite individuals.

    This focuses on whether *high-performing* solutions are diverse, rather
    than including low-fitness individuals.
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


# ---------------------------------------------------------------------------
# GA runner for a single configuration
# ---------------------------------------------------------------------------


def run_ga_with_config(
    config_name: str,
    data: dict[str, np.ndarray | float],
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    *,
    niching_enabled: bool,
    niching_sigma_share: float | None,
    mutation_probability: float,
    use_custom_mutation: bool,
    parent_selection_type: str,
    keep_elitism: int,
    num_generations: int = 1000,
    sol_per_pop: int = 100,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run GA with a specific niching/mutation configuration.

    Returns metrics used for the left-hand summary table and the right-hand
    population plot (for selected configurations).
    """
    np.random.seed(random_seed)

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

    # Choose mutation operator
    if use_custom_mutation:
        mutation_type: Any = diversity_preserving_mutation
    else:
        mutation_type = "random"

    # Build GA configuration using AdvancedGA for both niching and non-niching
    # When niching is disabled, we still need to provide a sigma_share value (it won't be used)
    ga_config = create_ga_config(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=int(sol_per_pop * 0.5),
        keep_elitism=keep_elitism,
        parent_selection_type=parent_selection_type,
        mutation_type=mutation_type,
        mutation_probability=mutation_probability,
        niching_enabled=niching_enabled,
        niching_sigma_share=niching_sigma_share
        if niching_enabled
        else 2.0,  # Default value when disabled
        niching_alpha=1.0,
        random_seed=random_seed,
    )

    ga_config["num_genes"] = len(gene_space)
    ga_config["gene_space"] = gene_space
    ga_config["fitness_func"] = fitness_func

    ga = AdvancedGA(**ga_config)
    ga.run()

    # Metrics
    best_chromosome, best_fitness, _ = ga.best_solution()
    final_population = ga.population
    final_fitness_scores = ga.last_generation_fitness

    # Diversity metrics
    full_diversity = calculate_population_diversity(final_population, ga.niching_config)
    elite_diversity = calculate_elite_diversity(
        population=final_population,
        fitness_scores=final_fitness_scores,
        top_n=20,
        niching_config=ga.niching_config,
    )

    # High performers
    high_perf_threshold = 50.0
    high_perf_count = int(np.sum(final_fitness_scores >= high_perf_threshold))

    # Top-5 stats
    sorted_indices = np.argsort(final_fitness_scores)[::-1]
    top5 = final_fitness_scores[sorted_indices[:5]]
    top5_std = float(np.std(top5)) if len(top5) > 0 else 0.0

    # Simple clone check: are top 5 chromosomes nearly identical?
    top5_pop = (
        final_population[sorted_indices[:5]] if len(sorted_indices) >= 5 else final_population
    )
    cloned = True
    if len(top5_pop) >= 2:
        for i in range(len(top5_pop)):
            for j in range(i + 1, len(top5_pop)):
                if np.linalg.norm(top5_pop[i] - top5_pop[j]) > 1e-6:
                    cloned = False
                    break
            if not cloned:
                break

    return {
        "config_name": config_name,
        "ga": ga,
        "wavelengths": wavelengths_array,
        "best_chromosome": best_chromosome,
        "best_fitness": float(best_fitness),
        "population": final_population,
        "fitness_scores": final_fitness_scores,
        "full_diversity": full_diversity,
        "elite_diversity": elite_diversity,
        "high_perf_count": high_perf_count,
        "top5_std": top5_std,
        "top5_cloned": cloned,
        "use_custom_mutation": use_custom_mutation,
        "niching_enabled": niching_enabled,
        "niching_sigma_share": niching_sigma_share if niching_enabled else None,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_results_table(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Create a table summarizing GA configurations and metrics.

    Columns:
    - Config
    - Best Fitness
    - Full Diversity
    - Elite Diversity (Top 20)
    - # High Performers (≥ 50)
    - Top-5 Std
    - Top-5 Cloned?
    """
    # Sort by config name for stable ordering (Baseline, Weak, Strong)
    rows = sorted(results, key=lambda r: r["config_name"])

    headers = [
        "Config",
        "Niching",
        "Mutation",
        "Best F",
        "Full Div",
        "Elite Div",
        "High Perf (≥50)",
        "Top5 Std",
        "Top5 Cloned",
    ]

    cell_text: list[list[str]] = []
    for r in rows:
        # Format niching info
        if r["niching_enabled"]:
            niching_str = f"σ={r['niching_sigma_share']:.1f}" if r["niching_sigma_share"] else "ON"
        else:
            niching_str = "OFF"

        # Format mutation info
        mutation_str = "Custom" if r["use_custom_mutation"] else "Standard"

        cell_text.append(
            [
                r["config_name"],
                niching_str,
                mutation_str,
                f"{r['best_fitness']:.2f}",
                f"{r['full_diversity']:.2f}",
                f"{r['elite_diversity']:.2f}",
                f"{r['high_perf_count']}",
                f"{r['top5_std']:.2f}",
                "YES" if r["top5_cloned"] else "NO",
            ]
        )

    # Table should be wider to accommodate more columns
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)

    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#E6E6FA")
        table[(0, i)].set_text_props(weight="bold")

    ax.set_title(
        "Niching + Custom Mutation Configurations\nFitness vs Diversity Trade-offs",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_population_curves(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    wavelengths: np.ndarray,
    output_path: Path,
    *,
    top_n: int = 20,
    score_min: float = 50.0,
    score_max: float = 60.0,
) -> None:
    """Plot Top N individuals with colorbar based on fitness (same as slide1-2).

    - Rank 1 at the top (highest offset)
    - Colors based on fitness score using viridis colormap
    - Shows colorbar indicating fitness range
    """
    # Ensure wavelengths is 1D
    if wavelengths.ndim > 1:
        wavelengths = wavelengths.flatten()

    sorted_indices = np.argsort(fitness_scores)[::-1]
    top_indices = sorted_indices[: min(top_n, len(population))]
    selected_pop = population[top_indices]
    selected_fitness = fitness_scores[top_indices]

    fig, ax = plt.subplots(figsize=(5, 4))

    # Create colormap using default matplotlib colormap (viridis, same as slide1-2)
    cmap = plt.colormaps["viridis"]

    num_individuals = len(selected_pop)
    offset_step = 0.1

    # Plot each individual's basis functions (colored by fitness score)
    for i, (chromosome, fitness) in enumerate(zip(selected_pop, selected_fitness, strict=False)):
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)

        vertical_offset = (num_individuals - 1 - i) * offset_step

        # Get color based on fitness score using viridis colormap
        color_norm = (fitness - score_min) / (score_max - score_min)
        color_norm = np.clip(color_norm, 0, 1)
        color = cmap(color_norm)

        # Plot all basis functions with the same color for this individual
        for j in range(basis_functions.shape[1]):
            scaled_basis = basis_functions[:, j] * 0.3
            ax.plot(
                wavelengths,
                scaled_basis + vertical_offset,
                color=color,
                alpha=0.8,
                linewidth=1.5,
            )

        # Add inline score text at the right side
        x_pos = wavelengths.max() + (wavelengths.max() - wavelengths.min()) * 0.05
        y_pos = vertical_offset

        ax.text(
            x_pos,
            y_pos,
            f"{fitness:.2f}",
            color=color,
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="left",
        )

    # Add colorbar (same as slide1-2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=score_min, vmax=score_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Fitness Score", pad=0.02)
    cbar.set_label("Fitness Score", fontsize=12, fontweight="bold")

    ax.set_xlabel("Wavelength (µm)", fontsize=12)
    ax.set_ylabel("Absorptivity (Offset Applied)", fontsize=12)
    ax.set_title(
        "Strong Niching + Custom Mutation\nTop 20 Sensor Designs",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Extend x-axis slightly to accommodate inline scores
    xlim_current = ax.get_xlim()
    ax.set_xlim(xlim_current[0], xlim_current[1] * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a small niching study and generate Slide 2 plots.

    This intentionally uses only a few configurations (Baseline, Weak Niching,
    Strong Niching) so it runs in a reasonable time while still showing the
    qualitative behavior we care about.
    """
    # Repo root (this file lives under legacy/presentation_plots/)
    project_root = Path(__file__).resolve().parent.parent.parent

    # Data paths
    spectral_data_file = (
        project_root / "data" / "Test 3 - 4 White Powers" / "white_powders_with_labels.xlsx"
    )
    air_transmittance_file = (
        project_root / "data" / "Test 3 - 4 White Powers" / "Air transmittance.xlsx"
    )

    # Load data
    print("=== Loading Data ===")
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

    # Sensor configuration (same as Slide 1)
    num_basis_functions = 4
    num_params_per_basis_function = 2
    param_bounds = [
        {"low": 4.0, "high": 20.0},  # mu (wavelength center)
        {"low": 0.1, "high": 4.0},  # sigma (width)
    ]
    gene_space = param_bounds * num_basis_functions

    # Define a small set of configurations for the table.
    # These are chosen to illustrate the narrative qualitatively.
    configs: list[dict[str, Any]] = [
        {
            "name": "Baseline GA",
            "niching_enabled": False,
            # Sigma is not used when niching is disabled, but the GA config
            # still expects a value. We provide a reasonable placeholder.
            "niching_sigma_share": 2.0,
            "mutation_probability": 0.1,
            "use_custom_mutation": False,
            "parent_selection_type": "rws",
            "keep_elitism": 5,
        },
        {
            "name": "Weak Niching",
            "niching_enabled": True,
            "niching_sigma_share": 5.0,  # Large radius, weak penalty
            "mutation_probability": 0.1,
            "use_custom_mutation": True,
            "parent_selection_type": "sus",
            "keep_elitism": 5,
        },
        {
            "name": "Strong Niching",
            "niching_enabled": True,
            "niching_sigma_share": 2.0,  # Stronger penalty in crowded peaks
            "mutation_probability": 0.1,
            "use_custom_mutation": True,
            "parent_selection_type": "sus",
            "keep_elitism": 1,
        },
    ]

    print("\n=== Running GA Configurations for Slide 2 ===")
    results: list[dict[str, Any]] = []

    for cfg in configs:
        print(f"\n--- Running config: {cfg['name']} ---")
        res = run_ga_with_config(
            config_name=cfg["name"],
            data=data,
            gene_space=gene_space,
            params_per_basis_function=num_params_per_basis_function,
            niching_enabled=cfg["niching_enabled"],
            niching_sigma_share=cfg["niching_sigma_share"],
            mutation_probability=cfg["mutation_probability"],
            use_custom_mutation=cfg["use_custom_mutation"],
            parent_selection_type=cfg["parent_selection_type"],
            keep_elitism=cfg["keep_elitism"],
            num_generations=500,
            sol_per_pop=100,
            random_seed=42,
        )
        results.append(res)

    # Output directory (unified under outputs/)
    output_dir = project_root / "outputs" / "results_archive" / "presentation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Left figure: table summarizing configs
    table_path = output_dir / "slide2-1_table.png"
    plot_results_table(results, output_path=table_path)

    # Right figure: strong niching population plot
    strong_result = next(r for r in results if r["config_name"] == "Strong Niching")
    pop_plot_path = output_dir / "slide2-2_strong_niching.png"
    plot_population_curves(
        population=strong_result["population"],
        fitness_scores=strong_result["fitness_scores"],
        wavelengths=strong_result["wavelengths"],
        output_path=pop_plot_path,
        top_n=20,
        score_min=50.0,
        score_max=60.0,
    )

    print("\n=== Slide 2 plots generated successfully! ===")
    print(f"  Table:   {table_path}")
    print(f"  Niching: {pop_plot_path}")


if __name__ == "__main__":
    main()
