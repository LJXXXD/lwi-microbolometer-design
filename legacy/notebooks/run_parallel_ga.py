#!/usr/bin/env python3
"""
Parallel GA execution module for Jupyter notebooks.

This module handles the multiprocessing execution of GA configurations
to avoid serialization issues in Jupyter notebooks.
"""

import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any

# Import the Advanced GA components
from lwi_microbolometer_design import (
    make_min_dissimilarity_fitness,
    spectral_angle_mapper,
)

# Import Advanced GA components from optimization module
from lwi_microbolometer_design.optimization import (
    AdvancedGA,
    AdvancedGAConfig,
    NichingConfig,
)

logger = logging.getLogger(__name__)


def load_data():
    """Load spectral data and configuration parameters."""
    # File paths
    spectral_data_file = Path("../data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")
    air_transmittance_file = Path("../data/Test 3 - 4 White Powers/Air transmittance.xlsx")

    # Parameters
    atmospheric_distance_ratio = 0.11
    temperature_K = 293.15
    air_refractive_index = 1

    # Load spectral data
    substances_spectral_data = pd.read_excel(spectral_data_file)
    wavelengths = substances_spectral_data.iloc[:, :1].to_numpy()
    substance_names = substances_spectral_data.columns[1:].to_numpy()
    emissivity_curves = substances_spectral_data.iloc[:, 1:].to_numpy()

    # Load air transmittance
    air_transmittance = np.array(pd.read_excel(air_transmittance_file, header=None))
    air_transmittance = air_transmittance[:, 1:]

    logger.info(f"Loaded spectral data for {len(substance_names)} substances")
    logger.info(f"Wavelength range: {wavelengths[0][0]:.1f} - {wavelengths[-1][0]:.1f} µm")

    return {
        "wavelengths": wavelengths,
        "substance_names": substance_names,
        "emissivity_curves": emissivity_curves,
        "air_transmittance": air_transmittance,
        "atmospheric_distance_ratio": atmospheric_distance_ratio,
        "temperature_K": temperature_K,
        "air_refractive_index": air_refractive_index,
    }


def create_fitness_function(data):
    """Create fitness function for sensor optimization."""
    return make_min_dissimilarity_fitness(
        wavelengths=data["wavelengths"],
        emissivity_curves=data["emissivity_curves"],
        temperature_K=data["temperature_K"],
        atmospheric_distance_ratio=data["atmospheric_distance_ratio"],
        air_refractive_index=data["air_refractive_index"],
        air_transmittance=data["air_transmittance"],
        distance_metric=spectral_angle_mapper,
    )


def create_gene_space():
    """Create gene space bounds for GA parameters."""
    MU_MIN, MU_MAX = 4.0, 20.0
    SIGMA_MIN, SIGMA_MAX = 0.1, 4.0

    gene_space = []
    for i in range(4):  # 4 basis functions
        gene_space.append({"low": MU_MIN, "high": MU_MAX})  # µ
        gene_space.append({"low": SIGMA_MIN, "high": SIGMA_MAX})  # σ

    return gene_space


def run_single_ga_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single GA configuration in a separate process.

    This function is designed to be called by multiprocessing workers.
    It creates the fitness function and runs the GA, returning results.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing 'name', 'config_dict', and 'random_seed'

    Returns
    -------
    dict
        Dictionary containing name, config, result, and runtime
    """
    import logging

    # Set up logging for this process
    logger = logging.getLogger(__name__)

    try:
        # Reconstruct the config object from dictionary
        config_dict_data = config_dict["config_dict"]
        config = AdvancedGAConfig(
            population_size=config_dict_data["population_size"],
            num_generations=config_dict_data["num_generations"],
            num_parents_mating=config_dict_data["num_parents_mating"],
            mutation_rate_base=config_dict_data["mutation_rate_base"],
            crossover_rate=config_dict_data["crossover_rate"],
            elitism_size=config_dict_data["elitism_size"],
            elitism_niche_radius=config_dict_data.get("elitism_niche_radius", 1.0),
            niching=NichingConfig(
                enabled=config_dict_data["niching"]["enabled"],
                sigma_share=config_dict_data["niching"]["sigma_share"],
                alpha=config_dict_data["niching"]["alpha"],
                distance_metric=config_dict_data["niching"]["distance_metric"],
            ),
            random_seed=config_dict["random_seed"],
        )

        # Load data and create fitness function
        data = load_data()
        fitness_func = create_fitness_function(data)
        gene_space = create_gene_space()

        # Run GA with timing
        start_time = time.time()
        ga = AdvancedGA(config, fitness_func, gene_space)
        result = ga.run()
        end_time = time.time()
        runtime = end_time - start_time

        return {"name": config_dict["name"], "config": config, "result": result, "runtime": runtime}

    except Exception as e:
        logger.error(f"Error in {config_dict['name']}: {e}")
        raise


def run_parallel_ga_configurations() -> Dict[str, Any]:
    """
    Run all GA configurations in parallel and return results.

    Returns
    -------
    dict
        Dictionary mapping configuration names to their results
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Define configurations as dictionaries for serialization
    configurations = [
        {
            "name": "Baseline GA (No Niching)",
            "config_dict": {
                "population_size": 200,
                "num_generations": 1000,
                "num_parents_mating": 50,
                "mutation_rate_base": 0.1,
                "crossover_rate": 0.8,
                "elitism_size": 5,
                "elitism_niche_radius": 1.0,
                "niching": {
                    "enabled": False,
                    "sigma_share": 0.5,
                    "alpha": 0.5,
                    "distance_metric": "euclidean",
                },
            },
            "random_seed": 42,
        },
        {
            "name": "Advanced GA (With Niching)",
            "config_dict": {
                "population_size": 200,
                "num_generations": 1000,
                "num_parents_mating": 50,
                "mutation_rate_base": 0.1,
                "crossover_rate": 0.8,
                "elitism_size": 5,
                "elitism_niche_radius": 1.0,
                "niching": {
                    "enabled": True,
                    "sigma_share": 0.5,
                    "alpha": 0.5,
                    "distance_metric": "euclidean",
                },
            },
            "random_seed": 42,
        },
        {
            "name": "Advanced GA (Moderate Niching)",
            "config_dict": {
                "population_size": 200,
                "num_generations": 1000,
                "num_parents_mating": 50,
                "mutation_rate_base": 0.1,
                "crossover_rate": 0.8,
                "elitism_size": 5,
                "elitism_niche_radius": 1.0,
                "niching": {
                    "enabled": True,
                    "sigma_share": 1.0,
                    "alpha": 0.5,
                    "distance_metric": "euclidean",
                },
            },
            "random_seed": 42,
        },
        {
            "name": "Diversity-Preserving Elitism Only",
            "config_dict": {
                "population_size": 200,
                "num_generations": 1000,
                "num_parents_mating": 50,
                "mutation_rate_base": 0.1,
                "crossover_rate": 0.8,
                "elitism_size": 5,
                "elitism_niche_radius": 2.0,
                "niching": {
                    "enabled": False,
                    "sigma_share": 0.5,
                    "alpha": 0.5,
                    "distance_metric": "euclidean",
                },
            },
            "random_seed": 42,
        },
        {
            "name": "Diversity-Preserving Elitism + Niching",
            "config_dict": {
                "population_size": 200,
                "num_generations": 1000,
                "num_parents_mating": 50,
                "mutation_rate_base": 0.1,
                "crossover_rate": 0.8,
                "elitism_size": 5,
                "elitism_niche_radius": 2.0,
                "niching": {
                    "enabled": True,
                    "sigma_share": 0.5,
                    "alpha": 0.5,
                    "distance_metric": "euclidean",
                },
            },
            "random_seed": 42,
        },
    ]

    # Run configurations in parallel
    logger.info("=== Advanced GA Demonstration (Parallel) ===")
    logger.info(f"Running {len(configurations)} configurations in parallel...")

    results = {}

    # Use ProcessPoolExecutor for parallel execution
    max_workers = min(len(configurations), mp.cpu_count())
    logger.info(f"Using {max_workers} parallel workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_ga_config, config_info): config_info
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
                logger.info(f"Runtime: {result_data['runtime']:.2f} seconds")
                logger.info(f"Best fitness: {result_data['result']['best_fitness']:.4f}")
                logger.info(
                    f"Final diversity: {result_data['result']['diversity_history'][-1]:.4f}"
                )
                logger.info(
                    f"High-quality solutions: {np.sum(result_data['result']['final_fitness_scores'] >= 45.0)}"
                )

                # Log niching configuration details
                if result_data["config"].niching.enabled:
                    logger.info(
                        f"Niching config: sigma_share={result_data['config'].niching.sigma_share}, "
                        f"alpha={result_data['config'].niching.alpha}, "
                        f"distance_metric={result_data['config'].niching.distance_metric}"
                    )

                # Assert diversity thresholds for niching configurations
                final_diversity = result_data["result"]["diversity_history"][-1]
                if result_data["config"].niching.enabled:
                    assert final_diversity >= 1.0, (
                        f"Diversity too low for niching config: {final_diversity:.4f}"
                    )
                    logger.info(f"✓ Niching diversity check passed: {final_diversity:.4f} >= 1.0")
                else:
                    logger.info(f"Baseline diversity (no niching): {final_diversity:.4f}")

            except Exception as exc:
                logger.error(f"Configuration {config_info['name']} generated an exception: {exc}")
                raise

    logger.info("\nAll GA runs completed successfully!")
    return results


if __name__ == "__main__":
    # Run the parallel GA configurations
    results = run_parallel_ga_configurations()
    print(f"Completed {len(results)} GA configurations successfully!")
