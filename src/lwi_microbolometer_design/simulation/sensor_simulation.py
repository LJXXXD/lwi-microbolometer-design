"""Sensor simulation functions for microbolometer design."""

import numpy as np

from .blackbody import blackbody_emit


def simulate_sensor_output(
    wavelengths: np.ndarray | list | tuple,
    substances_emissivity: np.ndarray | list | tuple,
    basis_functions: np.ndarray | list | tuple,
    temperature_k: float,
    atmospheric_distance_ratio: float,
    air_refractive_index: float,
    air_transmittance: np.ndarray | list | tuple,
) -> np.ndarray:
    """
    Simulate sensor output for one or multiple substances using given basis functions.

    This function implements the physics-based simulation of microbolometer sensor
    responses to infrared radiation from substances with known emissivity spectra.

    Parameters
    ----------
    wavelengths : np.ndarray
        Array of wavelength values in micrometers (µm), shape (d, 1)
    substances_emissivity : np.ndarray
        Emissivity spectra of substances, shape (d, n) where n is number of substances
    basis_functions : np.ndarray
        Basis functions of the sensor, shape (d, m) where m is number of basis functions
    temperature_k : float
        Temperature of the substances in Kelvin (K)
    atmospheric_distance_ratio : float
        Factor modeling the effect of atmospheric distance on measurements
    air_refractive_index : float
        Refractive index of the surrounding air. Affects blackbody emission (scales with n²).
    air_transmittance : np.ndarray
        Transmission coefficients of air, shape (d, 1)

    Returns
    -------
    np.ndarray
        Sensor output values, shape (m, n) where m is number of basis functions
        and n is number of substances

    Notes
    -----
    The simulation follows the physics of infrared radiation:
    1. Calculate blackbody emission at given temperature
    2. Apply atmospheric transmission effects
    3. Multiply by substance emissivity
    4. Convolve with sensor basis functions
    5. Integrate over wavelength to get sensor response
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
    bb_emit = blackbody_emit(wavelengths, temperature_k, air_refractive_index)
    if bb_emit.ndim == 1:
        bb_emit = bb_emit.reshape(-1, 1)  # Ensure shape is (d, 1)

    # Calculate atmospheric transmission factor
    tau_air = air_transmittance**atmospheric_distance_ratio

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
        emissivity_curve = substances_emissivity[:, i : i + 1]  # Shape = (d, 1)

        # Compute the absorption spectrum
        abso_spec = tau_air * bb_emit * emissivity_curve * basis_functions  # Shape = (d, m)

        # Integrate over the wavelengths to get sensor outputs
        sensor_outputs[:, i] = np.trapz(abso_spec, wavelengths.flatten(), axis=0)  # Shape = (m,)

    return sensor_outputs
