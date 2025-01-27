
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.simulations.blackbody_emit import blackbody_emit




# def simulator(sim_params, mat_em):
#     bb_emit = blackbody_emit(sim_params.spectra, sim_params.temp_K, sim_params.air_RI)
#     tau_air = sim_params.air_trans ** sim_params.atm_dist_ratio
#     abso_spec = tau_air * bb_emit * mat_em * sim_params.basis_funcs
#     out = np.trapz(abso_spec, sim_params.spectra, axis=0)

#     return out


def simulate_sensor_output(wavelengths, substances_emissivity, basis_functions, temperature_K, 
                           atmospheric_distance_ratio, air_refractive_index, air_transmittance):
    """
    Simulates sensor output for one or multiple substances using given basis functions.

    Parameters:
    - wavelengths (2D array): Array of wavelengths (shape = (d, 1)).
    - substances_emissivity (2D array): Emissivity spectra of one or multiple substances (shape = (d, n)).
    - basis_functions (2D array): Basis functions of the sensor (shape = (d, m)).
    - temperature_K (float): Temperature of the substances in Kelvin.
    - atmospheric_distance_ratio (float): Factor modeling the effect of atmospheric distance on measurements.
    - air_refractive_index (float): Refractive index of the surrounding air.
    - air_transmittance (2D array): Transmission coefficients of air (shape = (d, 1)).

    Returns:
    - sensor_outputs (2D array): Sensor output values (shape = (m, n)), where n = 1 for a single substance.
    """
    # Ensure all inputs are NumPy arrays
    if not isinstance(wavelengths, np.ndarray):
        wavelengths = np.array(wavelengths)
    if not isinstance(substances_emissivity, np.ndarray):
        substances_emissivity = np.array(substances_emissivity)
    if not isinstance(basis_functions, np.ndarray):
        basis_functions = np.array(basis_functions)
    if not isinstance(air_transmittance, np.ndarray):
        air_transmittance = np.array(air_transmittance)

    # Reshape inputs to enforce correct dimensions
    if wavelengths.ndim == 1:
        wavelengths = wavelengths.reshape(-1, 1)  # Ensure shape is (d, 1)
    if air_transmittance.ndim == 1:
        air_transmittance = air_transmittance.reshape(-1, 1)  # Ensure shape is (d, 1)

    # Calculate blackbody emission for the given temperature and air refractive index
    bb_emit = blackbody_emit(wavelengths, temperature_K, air_refractive_index)
    if bb_emit.ndim == 1:
        bb_emit = bb_emit.reshape(-1, 1)  # Ensure shape is (d, 1)

    # Calculate atmospheric transmission factor
    tau_air = air_transmittance ** atmospheric_distance_ratio

    # Check if substances_emissivity is for one or multiple substances
    if substances_emissivity.ndim == 1:
        substances_emissivity = substances_emissivity.reshape(-1, 1)  # Shape = (d, 1)

    # Number of substances and basis functions
    n = substances_emissivity.shape[1]  # Number of substances
    m = basis_functions.shape[1]  # Number of basis functions

    # Initialize the output matrix (m x n)
    sensor_outputs = np.zeros((m, n))

    # Compute sensor outputs for each substance
    for i in range(n):
        # Extract the emissivity spectrum for the current substance
        emissivity_curve = substances_emissivity[:, i:i+1]  # Shape = (d, 1)

        # Compute the absorption spectrum
        abso_spec = tau_air * bb_emit * emissivity_curve * basis_functions  # Shape = (d, m)

        # Integrate over the wavelengths to get sensor outputs
        sensor_outputs[:, i] = np.trapz(abso_spec, wavelengths.flatten(), axis=0)  # Shape = (m,)

    return sensor_outputs


# def simulate_single_substance(wavelengths, substances_emissivity, basis_functions, temperature_K, 
#                               atmospheric_distance_ratio, air_refractive_index, air_transmittance):
#     """
#     Simulates sensor output for a single substance and given basis functions.

#     Parameters:
#     - wavelengths (2D array): Array of wavelengths (shape = (d, 1)).
#     - substances_emissivity (2D array): Emissivity spectrum of the substance (shape = (d, 1)).
#     - basis_functions (2D array): Basis functions of the sensor (shape = (d, m)).
#     - temperature_K (float): Temperature of the substance in Kelvin.
#     - atmospheric_distance_ratio (float): Factor modeling the effect of atmospheric distance on measurements.
#     - air_refractive_index (float): Refractive index of the surrounding air.
#     - air_transmittance (2D array): Transmission coefficients of air (shape = (d, 1)).

#     Returns:
#     - out (1D array): Sensor output values (length = m), representing the response for the given basis functions.
#     """
#     # Ensure all inputs are NumPy arrays
#     if not isinstance(wavelengths, np.ndarray):
#         wavelengths = np.array(wavelengths)
#     if not isinstance(substances_emissivity, np.ndarray):
#         substances_emissivity = np.array(substances_emissivity)
#     if not isinstance(basis_functions, np.ndarray):
#         basis_functions = np.array(basis_functions)
#     if not isinstance(air_transmittance, np.ndarray):
#         air_transmittance = np.array(air_transmittance)

#     # Reshape inputs to enforce correct dimensions
#     if wavelengths.ndim == 1:
#         wavelengths = wavelengths.reshape(-1, 1)  # Ensure shape is (d, 1)
#     if substances_emissivity.ndim == 1:
#         substances_emissivity = substances_emissivity.reshape(-1, 1)  # Ensure shape is (d, 1)
#     if air_transmittance.ndim == 1:
#         air_transmittance = air_transmittance.reshape(-1, 1)  # Ensure shape is (d, 1)
#     if basis_functions.ndim != 2:
#         raise ValueError("basis_functions must be a 2D array with shape (d, m)")

#     # Calculate blackbody emission for the given temperature and air refractive index
#     bb_emit = blackbody_emit(wavelengths, temperature_K, air_refractive_index)
#     if not isinstance(bb_emit, np.ndarray):
#         bb_emit = np.array(bb_emit)  # Ensure blackbody emission is a NumPy array
#     if bb_emit.ndim == 1:
#         bb_emit = bb_emit.reshape(-1, 1)  # Ensure shape is (d, 1)

#     # Calculate atmospheric transmission factor
#     tau_air = air_transmittance ** atmospheric_distance_ratio

#     # Compute the absorption spectrum
#     abso_spec = tau_air * bb_emit * substances_emissivity * basis_functions  # Broadcast to (d, m)

#     # Integrate over the wavelengths to get sensor outputs
#     out = np.trapz(abso_spec, wavelengths.flatten(), axis=0)  # Shape (m,)
#     return out


# def simulate_multiple_substances(wavelengths, substances_emissivity, basis_functions, temperature_K,
#                                   atmospheric_distance_ratio, air_refractive_index, air_transmittance):
#     """
#     Generate the A matrix for multiple substances and basis functions.

#     Parameters:
#     - wavelengths (2D array): Array of wavelengths (shape = (d, 1)).
#     - substances_emissivity (2D array): Emissivity spectra of all substances (shape = (d, n)).
#     - basis_functions (2D array): Basis functions of the sensor (shape = (d, m)).
#     - temperature_K (float): Temperature of the substances in Kelvin.
#     - atmospheric_distance_ratio (float): Factor modeling the effect of atmospheric distance on measurements.
#     - air_refractive_index (float): Refractive index of the surrounding air.
#     - air_transmittance (2D array): Transmission coefficients of air (shape = (d, 1)).

#     Returns:
#     - A_matrix (2D array): Sensor output values (shape = (m, n)), where each column corresponds to a substance.
#     """
#     # Number of substances (n) and basis functions (m)
#     n = substances_emissivity.shape[1]  # Number of substances
#     m = basis_functions.shape[1]  # Number of basis functions

#     # Initialize the A matrix (m x n)
#     A_matrix = np.zeros((m, n))

#     # Compute sensor outputs for each substance
#     for i in range(n):
#         # Get the current substance's emissivity spectrum
#         substance_emissivity = substances_emissivity[:, i:i+1]  # Shape = (d, 1)

#         # Simulate the sensor output for this substance
#         A_matrix[:, i] = simulate_single_substance(
#             wavelengths=wavelengths,
#             substances_emissivity=substance_emissivity,
#             basis_functions=basis_functions,
#             temperature_K=temperature_K,
#             atmospheric_distance_ratio=atmospheric_distance_ratio,
#             air_refractive_index=air_refractive_index,
#             air_transmittance=air_transmittance
#         )

#     return A_matrix


def visualize_sensor_output(sensor_outputs, substances_names=None, basis_funcs_labels=None, fontsize=10):
    """
    Visualizes sensor outputs as curves for different substances.

    Parameters:
    - sensor_outputs (2D array): Sensor output values (shape = (m, n)).
    - substances_names (list or array-like of str, optional): Names of substances (columns of sensor_outputs). Defaults to None.
    - basis_funcs_labels (list or array-like of str, optional): Labels for basis functions (rows of sensor_outputs). Defaults to None.
    - fontsize (int, optional): Font size for text in the plot. Defaults to 10.

    Returns:
    - None
    """
    m, n = sensor_outputs.shape  # m = number of basis functions, n = number of substances

    # X-axis values for the basis functions
    x = np.arange(1, m + 1)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot each substance's sensor output as a curve
    for i in range(n):
        label = substances_names[i] if substances_names is not None else f"Substance {i+1}"
        plt.plot(x, sensor_outputs[:, i], marker='o', label=label)

    # Add labels, title, and legend
    plt.xlabel("Basis Function Index", fontsize=fontsize)
    plt.ylabel("Sensor Output Values (Volt)", fontsize=fontsize)
    plt.title("Comparison of Substances Using Selected Basis Functions (Microbolometers)", fontsize=fontsize + 2, fontweight="bold")
    plt.legend(loc="best", fontsize=fontsize)

    # Optionally label x-ticks with basis function labels
    if basis_funcs_labels is not None:
        plt.xticks(ticks=x, labels=basis_funcs_labels, fontsize=fontsize)
    else:
        plt.xticks(ticks=x, fontsize=fontsize)

    # Show grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.tight_layout()
    plt.show()