import sys
from pathlib import Path

# Legacy ML pipeline (`tools` package) lives under `legacy/` at repo root.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "legacy"))
from tools import (  # noqa: E402
    compute_distance_matrix,
    group_based_dissimilarity_score,
    min_based_dissimilarity_score,
    simulate_sensor_output,
)


def process_subset(
    basis_function_indices,
    gaussian_basis_functions,
    wavelengths,
    emissivity_curves,
    temperature_K,
    atmospheric_distance_ratio,
    air_refractive_index,
    air_transmittance,
    spectral_angle_mapper,
    groups,
):
    """
    Processes a single subset of basis functions to calculate scores.

    Parameters
    ----------
    - basis_function_indices (tuple): A tuple of indices representing the selected basis functions.

    Returns
    -------
    - dict: A dictionary containing the basis function indices and their calculated scores.
    """
    try:
        # Select the current subset of basis functions
        selected_basis_funcs = gaussian_basis_functions[:, list(basis_function_indices)]

        # Generate sensor outputs for the selected basis functions
        sensor_outputs = simulate_sensor_output(
            wavelengths=wavelengths,
            substances_emissivity=emissivity_curves,
            basis_functions=selected_basis_funcs,
            temperature_K=temperature_K,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            air_refractive_index=air_refractive_index,
            air_transmittance=air_transmittance,
        )

        # Compute the FOM sensor noise covariance score
        # score_fom = fom_sensor_noise_covariance(sensor_outputs)

        # Compute the distance matrix using the SAM metric
        distance_matrix = compute_distance_matrix(sensor_outputs, spectral_angle_mapper)

        # Compute scores for the current subset
        score_min = min_based_dissimilarity_score(distance_matrix)
        # score_mean_min = mean_min_based_dissimilarity_score(distance_matrix, alpha=3)
        score_group_based = group_based_dissimilarity_score(distance_matrix, groups)
        # score_weighted_mean_min = weighted_mean_min_dissimilarity_score(distance_matrix)

        # Return the basis function indices and scores
        return {
            "basis_function_indices": basis_function_indices,
            "min_score": score_min,
            # "mean_min_score": score_mean_min,
            "group_based_score": score_group_based,
            # "weighted_mean_min_score": score_weighted_mean_min,
            # "fom_score": score_fom
        }
    except Exception as e:
        print(f"Error in basis_function_indices {basis_function_indices}: {e}")
        return None
