#!/usr/bin/env python3
"""Minimal test script to verify mutation and niching are functioning."""

import numpy as np
from pathlib import Path

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data import load_substance_atmosphere_data
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
    data = loaded_data[0]
else:
    data = loaded_data

# Minimal sensor configuration
num_basis_functions = 2  # Reduced for speed
num_params_per_basis_function = 2
param_bounds = [
    {"low": 4.0, "high": 20.0},  # mu
    {"low": 0.1, "high": 4.0},  # sigma
]
gene_space = param_bounds * num_basis_functions

# Create fitness function
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
    emissivity_val if isinstance(emissivity_val, np.ndarray) else np.array([float(emissivity_val)])
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
