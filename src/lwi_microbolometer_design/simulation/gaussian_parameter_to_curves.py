"""
Response curve generation for microbolometer sensor simulation.

This module contains functions for generating Gaussian response curves from
mathematical parameter sets, used in sensor design optimization.
"""

import numpy as np


def gaussian_parameters_to_unit_amplitude_curves(
    gaussian_parameters: list[tuple[float, float]],
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Convert Gaussian parameters to curves.

    Generates Gaussian curves from (mu, sigma) parameters using formula:
    exp(-(x - mu_aligned)^2 / (2*sigma^2))

    Key features:
    - Mean is aligned to nearest discrete wavelength for exact peak value
    - Vectorized for fast computation of multiple curves
    - Supports single or multiple curves
    - Peak value is 1.0 (unit-amplitude)

    Parameters
    ----------
    gaussian_parameters : List[Tuple[float, float]]
        List of (mu, sigma) parameters for each Gaussian curve.
        Single: [(mean, sigma)]
        Multiple: [(mu1, sigma1), (mu2, sigma2), ...]
    wavelengths : np.ndarray
        1D array of discrete wavelength values (in μm).

    Returns
    -------
    np.ndarray
        Array of shape (num_wavelengths, num_curves) where each column
        is a Gaussian curve.

    Notes
    -----
    Mean alignment: μ is adjusted to the nearest discrete wavelength to ensure
    the peak value is exactly 1.0. Without this, peaks might be < 1.0 on discrete grids.

    Examples
    --------
    >>> wavelengths = np.linspace(4, 20, 100)
    >>> curve = gaussian_parameters_to_unit_amplitude_curves([(10.0, 2.0)], wavelengths)
    >>> curves = gaussian_parameters_to_unit_amplitude_curves(
    ...     [(10.0, 2.0), (12.0, 1.5)], wavelengths
    ... )
    """
    wavelengths_1d = np.asarray(wavelengths, dtype=np.float64).reshape(-1)
    if wavelengths_1d.size == 0 or not np.all(np.isfinite(wavelengths_1d)):
        raise ValueError("wavelengths must be nonempty and finite.")
    parameters = np.asarray(gaussian_parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 2 or parameters.shape[0] == 0:
        raise ValueError("gaussian_parameters must contain one or more (mu, sigma) pairs.")
    if not np.all(np.isfinite(parameters)) or np.any(parameters[:, 1] <= 0):
        raise ValueError("Gaussian means must be finite and sigmas finite and positive.")
    means, sigmas = parameters.T
    aligned_indices = np.argmin(np.abs(wavelengths_1d[:, None] - means), axis=0)
    aligned_means = wavelengths_1d[aligned_indices]
    return np.exp(-0.5 * ((wavelengths_1d[:, None] - aligned_means) / sigmas) ** 2)
