"""Blackbody radiation calculations for microbolometer sensor simulation."""

from math import pi

import numpy as np


def blackbody_emit(
    spectra: np.ndarray, temperature_k: float, refractive_index: float = 1.0
) -> np.ndarray:
    """
    Calculate blackbody emission spectrum for given wavelengths and temperature.

    This function implements Planck's law for blackbody radiation emission,
    which is fundamental to microbolometer sensor response calculations.

    Parameters
    ----------
    spectra : np.ndarray
        Array of wavelength values in micrometers (µm)
    temperature_k : float
        Temperature in Kelvin (K)
    refractive_index : float, optional
        Refractive index of the medium (default: 1.0 for vacuum/air).
        For blackbody radiation in a medium, emission scales with n².

    Returns
    -------
    np.ndarray
        Blackbody emission spectrum in W/(m²·µm·sr)

    Notes
    -----
    The function uses Planck's law for blackbody radiation in a medium:
    B(λ,T) = n^2 * (2πhc^2/λ^5) / (exp(hc/(λkT)) - 1)

    Where:
    - h = Planck constant (6.62606957e-34 J·s)
    - c = Speed of light (299792458 m/s)
    - k = Boltzmann constant (1.3806488e-23 J/K)
    - n = refractive index of the medium
    - λ = wavelength (µm)
    - T = temperature (K)

    The n² factor accounts for the increased density of states in the medium.
    For air (n ≈ 1.0003), the effect is small (~0.06%) but physically correct.
    """
    # Physical constants
    h = 6.626_069_57e-34  # Planck constant (J·s)
    c = 299_792_458  # Speed of light (m/s)
    k = 1.380_648_8e-23  # Boltzmann constant (J/K)

    # Planck's law constants
    c1 = 2 * pi * h * c**2
    c2 = h * c / k

    # Convert wavelengths to meters and calculate blackbody emission
    # The factor 1e24 converts from m² to µm² and adjusts units
    # Multiply by n² for emission in a medium (n² accounts for density of states)
    n_squared = refractive_index**2
    return (
        n_squared
        * (c1 * 1e24 / (pi * spectra**5))
        * (1 / (np.exp(c2 * 1e6 / (temperature_k * spectra)) - 1))
    )
