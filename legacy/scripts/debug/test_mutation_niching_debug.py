#!/usr/bin/env python3
"""Minimal test script to verify mutation and niching are functioning."""

import numpy as np
from pathlib import Path

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data import SceneConfig, load_substance_atmosphere_data
from lwi_microbolometer_design.ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    create_ga_config,
    diversity_preserving_mutation,
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("DEBUG TEST: Verifying Mutation and Niching Logic")
print("=" * 60)

# Load minimal data
spectral_data_file = Path("data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")
air_transmittance_file = Path("data/Test 3 - 4 White Powers/Air transmittance.xlsx")

print("\nLoading data...")
loaded_data = load_substance_atmosphere_data(
    spectral_data_file=spectral_data_file,
    air_transmittance_file=air_transmittance_file,
    atmospheric_distance_ratio=0.11,
    temperature_kelvin=293.15,
    air_refractive_index=1.0,
)

# Handle case where data might be a list
if isinstance(loaded_data, list):
    scene: SceneConfig = loaded_data[0]
else:
    scene = loaded_data

# Minimal sensor configuration
num_basis_functions = 2  # Reduced for speed
num_params_per_basis_function = 2
param_bounds = [
    {"low": 4.0, "high": 20.0},  # mu
    {"low": 0.1, "high": 4.0},  # sigma
]
gene_space = param_bounds * num_basis_functions

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

# Create GA config with TINY parameters for fast test
print("\nCreating GA config with tiny parameters (num_generations=2, sol_per_pop=5)...")
ga_config = create_ga_config(
    num_generations=2,  # TINY
    sol_per_pop=5,  # TINY
    num_parents_mating=4,  # Most of population as parents to ensure offspring
    keep_elitism=0,  # No elites, force new offspring creation
    crossover_probability=1.0,  # Always crossover to generate offspring
    mutation_type=diversity_preserving_mutation,  # Ensure custom mutation is used
    mutation_probability=0.5,  # Higher prob to ensure mutations happen
    niching_enabled=True,  # Ensure niching is enabled
    niching_sigma_share=1.0,
    niching_alpha=1.0,
)

# Add required parameters
ga_config["num_genes"] = len(gene_space)
ga_config["gene_space"] = gene_space
ga_config["fitness_func"] = fitness_func

print("\nInitializing GA...")
print(f"  num_generations: {ga_config['num_generations']}")
print(f"  sol_per_pop: {ga_config['sol_per_pop']}")
print(f"  mutation_type: {ga_config['mutation_type']}")
print(f"  niching_enabled: {ga_config['niching_config'].enabled}")

print("\n" + "=" * 60)
print("RUNNING GA - Watch for DEBUG prints...")
print("=" * 60 + "\n")

# Run GA
ga = AdvancedGA(**ga_config)
ga.run()

print("\n" + "=" * 60)
print("GA COMPLETE")
print("=" * 60)
print("\nIf you saw DEBUG prints above, wiring is GOOD!")
print("If not, check mutation_type parameter passing.")
