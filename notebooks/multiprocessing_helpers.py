import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from tools import simulate_sensor_output, compute_distance_matrix, mean_min_based_dissimilarity_score, \
    group_based_dissimilarity_score, weighted_mean_min_dissimilarity_score, fom_sensor_noise_covariance

def process_subset(subset, gaussian_basis_functions, wavelengths, emissivity_curves,
                   temperature_K, atmospheric_distance_ratio, air_refractive_index,
                   air_transmittance, spectral_angle_mapper, groups):
    """
    Processes a single subset of basis functions to calculate scores.

    Parameters:
    - subset (tuple): A tuple of indices representing the subset of basis functions.

    Returns:
    - dict: A dictionary containing the subset and its calculated scores.
    """
    try:
        # Select the current subset of basis functions
        selected_basis_funcs = gaussian_basis_functions[:, list(subset)]

        # Generate sensor outputs for the selected basis functions
        sensor_outputs = simulate_sensor_output(
            wavelengths=wavelengths,
            substances_emissivity=emissivity_curves,
            basis_functions=selected_basis_funcs,
            temperature_K=temperature_K,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            air_refractive_index=air_refractive_index,
            air_transmittance=air_transmittance
        )

        # Compute the FOM sensor noise covariance score
        score_fom = fom_sensor_noise_covariance(sensor_outputs)

        # Compute the distance matrix using the SAM metric
        distance_matrix = compute_distance_matrix(sensor_outputs, spectral_angle_mapper)

        # Compute scores for the current subset
        score_mean_min = mean_min_based_dissimilarity_score(distance_matrix, alpha=3)
        score_group_based = group_based_dissimilarity_score(distance_matrix, groups)
        score_weighted_mean_min = weighted_mean_min_dissimilarity_score(distance_matrix)

        # Return the subset and scores
        return {
            "subset": subset,
            "mean_min_score": score_mean_min,
            "group_based_score": score_group_based,
            "weighted_mean_min_score": score_weighted_mean_min,
            "fom_score": score_fom
        }
    except Exception as e:
        print(f"Error in subset {subset}: {e}")
        return None