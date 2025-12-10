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
    # Ensure wavelengths is 1D
    wavelengths_1d = wavelengths.flatten() if wavelengths.ndim > 1 else wavelengths
    num_wavelengths = len(wavelengths_1d)
    num_subpixels = len(gaussian_parameters)

    # Pre-allocate output array
    response_curves = np.zeros((num_wavelengths, num_subpixels))

    # Extract means and sigmas
    means = np.array([mean for mean, _ in gaussian_parameters])
    sigmas = np.array([sigma for _, sigma in gaussian_parameters])

    # Align each mean to the closest discrete wavelength
    # Use broadcasting to find closest wavelength for each mean
    wavelength_diff = np.abs(wavelengths_1d[:, np.newaxis] - means[np.newaxis, :])
    aligned_indices = np.argmin(wavelength_diff, axis=0)
    aligned_means = wavelengths_1d[aligned_indices]

    # Vectorized computation: compute all curves simultaneously
    # Shape: (num_wavelengths, num_subpixels)
    # For each wavelength (row) and each Gaussian (column):
    #   curve[w, s] = exp(-(wavelengths[w] - aligned_means[s])^2 / (2 * sigmas[s]^2))

    # Broadcasting: (num_wavelengths, num_subpixels) - (num_subpixels,)
    # -> (num_wavelengths, num_subpixels)
    wavelength_matrix = wavelengths_1d[:, np.newaxis]  # Shape: (num_wavelengths, 1)
    aligned_means_matrix = aligned_means[np.newaxis, :]  # Shape: (1, num_subpixels)
    sigmas_matrix = sigmas[np.newaxis, :]  # Shape: (1, num_subpixels)

    # Compute all curves at once using broadcasting
    exponents = -((wavelength_matrix - aligned_means_matrix) ** 2) / (2 * sigmas_matrix**2)
    response_curves = np.exp(exponents)

    return response_curves
