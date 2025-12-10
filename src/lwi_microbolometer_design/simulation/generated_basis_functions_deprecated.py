"""Basis function generation utilities.

**DEPRECATION NOTICE:**
This module contains code directly copied from the deprecated
tools/simulations/generate_basis_functions.py.
This is a temporary measure to ensure compatibility with existing code (e.g., SAM_demo notebook).

**Status:** Needs verification and potential refactoring
- [ ] Verify correctness of implementation
- [ ] Review if this function fits the simulation package's purpose
- [ ] Consider refactoring to align with package conventions
- [ ] Update or remove after verification

**Original Location:** tools/simulations/generate_basis_functions.py
**Copied Date:** 2024 (approximate)
**Action Required:** Review and either refactor or remove in future cleanup
"""

import numpy as np


def generate_structured_gaussian_basis_functions(
    wavelengths: np.ndarray | list,
    mean_range: tuple[float, float] = (4, 20),
    step: float = 1,
    widths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate structured Gaussian basis functions over a given spectrum.

    Generates Gaussian basis functions with means aligned to the nearest wavelength.
    Each mean is paired with each width to create a structured set of basis functions.

    Parameters
    ----------
    wavelengths : np.ndarray | list
        The wavelength range (e.g., from 4 to 20 in the IR band), shape (d, 1).
    mean_range : Tuple[float, float], optional
        Range for the means of the Gaussian curves (start, end), default: (4, 20)
    step : float, optional
        Step size for generating evenly spaced means within the range, default: 1
    widths : Tuple[float, ...], optional
        Set of standard deviations (widths) for the Gaussian curves, default: (0.5, 1.0, 2.0, 4.0)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - basis_functions: 2D array with shape (d, num_functions), where each column
          is a Gaussian basis function.
        - means: Array of means (centers) aligned to the nearest wavelength values.
        - sigmas: Array of standard deviations for the Gaussian curves.

    Notes
    -----
    This function is a direct copy from the deprecated `tools` module.
    See module docstring for deprecation notice and review status.
    """
    # Ensure wavelengths is a (d, 1) array
    wavelengths = np.array(wavelengths).reshape(-1, 1)

    # Generate evenly spaced means within the range using the step size
    evenly_spaced_means = np.arange(mean_range[0], mean_range[1] + step, step)

    # Find the closest actual wavelengths to the evenly spaced means
    aligned_means = [
        wavelengths[np.abs(wavelengths - mu).argmin()][0] for mu in evenly_spaced_means
    ]

    # Initialize storage for basis functions, means, and widths
    basis_functions_list = []
    all_means_list = []
    all_sigmas_list = []

    # Generate Gaussian curves for each mean and width
    for mu in aligned_means:
        for sigma in widths:
            gaussian = np.exp(-((wavelengths.flatten() - mu) ** 2) / (2 * sigma**2))
            basis_functions_list.append(gaussian)
            all_means_list.append(mu)
            all_sigmas_list.append(sigma)

    # Convert to NumPy arrays
    basis_functions = np.array(basis_functions_list).T  # Shape (d, num_functions)
    all_means = np.array(all_means_list)
    all_sigmas = np.array(all_sigmas_list)

    return basis_functions, all_means, all_sigmas
