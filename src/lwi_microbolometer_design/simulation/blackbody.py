"""Blackbody radiation calculations for microbolometer sensor simulation."""

import numpy as np

from lwi_microbolometer_design.simulation._validation import positive_scalar


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
        Finite positive vacuum wavelengths in micrometers (µm). Shape is preserved.
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
    B(λ,T) = n² * (2hc² * 1e24 / λ_um⁵) / expm1(hc * 1e6 / (λ_um kT))

    Where:
    - h = Planck constant (6.62606957e-34 J·s)
    - c = Speed of light (299792458 m/s)
    - k = Boltzmann constant (1.3806488e-23 J/K)
    - n = refractive index of the medium
    - λ = wavelength (µm)
    - T = temperature (K)

    The n² factor assumes a homogeneous, isotropic, non-attenuating medium
    with fixed index and a vacuum-wavelength spectral density; it does not model
    interfaces or dispersive atmospheric propagation. The factor 1e24 combines
    λ_m = 1e-6 λ_um with the per-meter to per-micrometer density factor 1e-6.
    Constants use CODATA 2010 values for compatibility with archived runs.

    References
    ----------
    NIST Technical Note 910-8, Eq. (12.3a):
    https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote910-8.pdf
    """
    wavelengths = np.asarray(spectra, dtype=np.float64)
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("spectra must contain finite positive wavelengths.")
    temperature = positive_scalar(temperature_k, "temperature_k")
    index = positive_scalar(refractive_index, "refractive_index")
    h = 6.626_069_57e-34
    c = 299_792_458
    k = 1.380_648_8e-23

    # Evaluate the radiance in log space so only the final result can underflow.
    log_wavelength = np.log(wavelengths)
    log_x = np.log(h * c / k * 1e6) - np.log(temperature) - log_wavelength
    with np.errstate(over="ignore", under="ignore"):
        x = np.exp(log_x)
    log_denominator = np.empty_like(x)
    small = x < 1e-5
    large = x > 50.0
    middle = ~(small | large)
    with np.errstate(under="ignore"):
        log_denominator[small] = log_x[small] + np.log1p(x[small] / 2 + x[small] ** 2 / 6)
    log_denominator[middle] = np.log(np.expm1(x[middle]))
    log_denominator[large] = x[large]  # exp(-x) is below float64 relative precision.
    log_radiance = (
        2 * np.log(index) + np.log(2 * h * c**2 * 1e24) - 5 * log_wavelength - log_denominator
    )
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return np.exp(log_radiance)
