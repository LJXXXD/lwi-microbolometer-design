#!/usr/bin/env python3
"""MAP-Elites Raw: Multi-dimensional Archive of Phenotypic Elites.

Thin script that wires up the data loading, fitness evaluator, and
MAP-Elites algorithm from the ``lwi_microbolometer_design`` package.
"""

import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data import SceneConfig, load_substance_atmosphere_data
from lwi_microbolometer_design.ga import MinDissimilarityFitnessEvaluator
from lwi_microbolometer_design.map_elites import (
    archive_coverage_pct,
    plot_map_elites_heatmap,
    plot_top_elites,
    reachable_cell_count,
    run_map_elites,
)

mp.set_start_method("spawn", force=True)


def main() -> None:
    """Run MAP-Elites algorithm."""
    print("=" * 80)
    print("MAP-ELITES RAW: Multi-dimensional Archive of Phenotypic Elites")
    print("=" * 80)
    print("Strategy: Illuminate design space by maintaining elites for each")
    print("          combination of feature descriptors (mu_1, mu_2)")

    # ------------------------------------------------------------------
    # [1/4] Load data
    # ------------------------------------------------------------------
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

    num_basis_functions = 4
    num_params_per_basis_function = 2
    param_bounds = [
        {"low": 4.0, "high": 20.0},
        {"low": 0.1, "high": 4.0},
    ]
    gene_space = param_bounds * num_basis_functions
    wavelengths_array = np.asarray(scene.wavelengths)

    fitness_func = MinDissimilarityFitnessEvaluator(
        scene=scene,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=num_params_per_basis_function,
        distance_metric=spectral_angle_mapper,
    ).fitness_func

    # ------------------------------------------------------------------
    # [2/4] Configuration
    # ------------------------------------------------------------------
    grid_resolution = 20
    mu_range = (4.0, 20.0)
    num_initial = 1000
    num_iterations = 200000
    reachable_cells = reachable_cell_count(grid_resolution)

    print("\n[2/4] MAP-Elites Configuration:")
    print(
        f"  Grid: {grid_resolution}x{grid_resolution} "
        f"= {grid_resolution * grid_resolution} nominal bins"
    )
    print(
        f"  Reachable Cells: {reachable_cells} (ordered \u03bc\u2081 \u2264 \u03bc\u2082 descriptor)"
    )
    print(f"  Feature Range: [{mu_range[0]}, {mu_range[1]}] \u00b5m")
    print(f"  Initial Solutions: {num_initial}")
    print(f"  Iterations: {num_iterations}")
    print("  Mutation: Gaussian sigma=0.75 (wavelengths), sigma=0.2 (widths)")

    # ------------------------------------------------------------------
    # [3/4] Run MAP-Elites
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # [4/4] Save & Visualise
    # ------------------------------------------------------------------
    output_dir = Path("outputs/map_elites/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / "map_elites_raw_archive.pkl"
    with archive_path.open("wb") as f:
        pickle.dump(archive, f)
    print(f"\nSaved archive to: {archive_path}")

    print("\n[4/4] Creating visualizations...")
    plot_map_elites_heatmap(
        archive=archive,
        grid_resolution=grid_resolution,
        mu_range=mu_range,
        output_path=output_dir / "map_elites_raw_heatmap.png",
    )
    for top_n in (10, 20, 50):
        plot_top_elites(
            archive=archive,
            wavelengths=wavelengths_array,
            parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
            output_path=output_dir / f"map_elites_raw_top{top_n}_elites.png",
            top_n=top_n,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_individuals = list(archive.values())
    all_fitness = [ind["fitness"] for ind in all_individuals]

    print("\n" + "=" * 80)
    print("MAP-ELITES RAW RESULTS")
    print("=" * 80)
    print("\nArchive Statistics:")
    print(f"  Filled Cells: {len(archive)}/{reachable_cells}")
    print(f"  Reachable Coverage: {archive_coverage_pct(len(archive), grid_resolution):.1f}%")
    print(f"  Best Fitness: {np.max(all_fitness):.4f}")
    print(f"  Mean Fitness: {np.mean(all_fitness):.4f}")
    print(f"  Std Fitness:  {np.std(all_fitness):.4f}")
    print(f"\n  Fitness > 50: {np.sum(np.array(all_fitness) > 50)}/{len(archive)}")
    print(f"  Fitness > 55: {np.sum(np.array(all_fitness) > 55)}/{len(archive)}")
    print(f"  Fitness > 58: {np.sum(np.array(all_fitness) > 58)}/{len(archive)}")

    all_individuals.sort(key=lambda x: x["fitness"], reverse=True)
    print("\nTop 10 Elite Solutions:")
    for i, ind in enumerate(all_individuals[:10], 1):
        print(
            f"  {i}. Fitness={ind['fitness']:.2f}, "
            f"\u03bc\u2081={ind['mu_1']:.2f}, \u03bc\u2082={ind['mu_2']:.2f}"
        )

    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
