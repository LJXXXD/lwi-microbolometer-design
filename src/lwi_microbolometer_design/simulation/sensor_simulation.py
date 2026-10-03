"""Sensor simulation functions for microbolometer design."""

import numpy as np

from lwi_microbolometer_design.simulation.blackbody import blackbody_emit
from lwi_microbolometer_design.simulation._validation import (
    emissivity_matrix,
    positive_scalar,
    spectral_vector,
    validate_transmittance,
    validate_wavelengths,
)


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
        Array of wavelength values in micrometers (µm), shape (d,) or (d, 1)
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
        Transmission coefficients of air, shape (d,) or (d, 1)

    Returns
    -------
    np.ndarray
        Integrated model response in W/(m²·sr) for dimensionless basis functions,
        shape (m, n) where m is number of basis functions
        and n is number of substances

    Notes
    -----
    The simulation follows the physics of infrared radiation:
    1. Calculate blackbody emission at given temperature
    2. Apply atmospheric transmission effects
    3. Multiply by substance emissivity
    4. Weight by the sensor basis functions
    5. Integrate over vacuum wavelength in µm using the trapezoidal rule.

    Distance ratio must be finite and nonnegative. At ratio zero, attenuation
    is one at every wavelength (including zero-transmission samples). Finite
    signed measured emissivity is preserved without clipping; negative values
    do not establish physically valid passive radiance. Basis functions must
    be finite and aligned, but may be signed for custom linear response models.
    """
    wl = spectral_vector(wavelengths, "wavelengths")
    validate_wavelengths(wl)
    em = np.asarray(substances_emissivity, dtype=np.float64)
    if em.ndim == 1:
        em = em[:, None]
    em = emissivity_matrix(em, wl.size)
    basis = np.asarray(basis_functions, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != wl.size or basis.shape[1] == 0:
        raise ValueError("basis_functions must have shape (d, m) aligned with wavelengths.")
    if not np.all(np.isfinite(basis)):
        raise ValueError("basis_functions must be finite.")
    air = spectral_vector(air_transmittance, "air_transmittance")
    if air.size != wl.size:
        raise ValueError("air_transmittance length must match wavelengths.")
    validate_transmittance(air)
    ratio = positive_scalar(
        atmospheric_distance_ratio, "atmospheric_distance_ratio", allow_zero=True
    )
    radiance = blackbody_emit(wl, temperature_k, air_refractive_index)

    # Nonuniform trapezoid weights preserve wavelength-density integration.
    dx = np.diff(wl)
    weights = np.empty_like(wl)
    weights[0], weights[-1] = dx[0] / 2, dx[-1] / 2
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2
    outputs = basis.T @ ((weights * air**ratio * radiance)[:, None] * em)
    if not np.all(np.isfinite(outputs)):
        raise ValueError("Sensor integration produced nonfinite outputs.")
    return outputs
