
import numpy as np

def generate_gaussian_basis_functions(wavelengths, num_functions, mean_range=(6, 18), sigma_range=(0.5, 3.0), random_seed=None):
    """
    Generates Gaussian basis functions over a given spectrum.
    
    Parameters:
    - spectra (array-like): The wavelength range (e.g., from 4 to 20 in the IR band), shape (d, 1).
    - num_functions (int): Number of Gaussian basis functions to generate.
    - mean_range (tuple): Range for the means of the Gaussian curves.
    - sigma_range (tuple): Range for the standard deviations of the Gaussian curves.
    - random_seed (int, optional): Seed for reproducibility.
    
    Returns:
    - basis_functions (ndarray): 2D array with shape (d, num_functions), where each column is a Gaussian basis function.
    - means (ndarray): Array of means (centers) for the Gaussian curves.
    - sigmas (ndarray): Array of standard deviations for the Gaussian curves.
    """
    # Ensure spectra is a (d, 1) array
    wavelengths = np.array(wavelengths).reshape(-1, 1)

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random means and standard deviations
    means = np.random.uniform(*mean_range, size=num_functions)
    sigmas = np.random.uniform(*sigma_range, size=num_functions)

    # Generate Gaussian curves
    basis_functions = np.zeros((wavelengths.shape[0], num_functions))  # (d, num_functions)
    for i, (mu, sigma) in enumerate(zip(means, sigmas)):
        basis_functions[:, i] = np.exp(-((wavelengths.flatten() - mu)**2) / (2 * sigma**2))

    return basis_functions, means, sigmas




def generate_structured_gaussian_basis_functions(wavelengths, mean_range=(4, 20), step=1, widths=(0.5, 1.0, 2.0, 4.0)):
    """
    Generates structured Gaussian basis functions over a given spectrum with means aligned to the nearest wavelength.

    Parameters:
    - wavelengths (array-like): The wavelength range (e.g., from 4 to 20 in the IR band), shape (d, 1).
    - mean_range (tuple): Range for the means of the Gaussian curves (start, end).
    - step (float): Step size for generating evenly spaced means within the range.
    - widths (tuple): Set of standard deviations (widths) for the Gaussian curves.

    Returns:
    - basis_functions (ndarray): 2D array with shape (d, num_functions), where each column is a Gaussian basis function.
    - means (ndarray): Array of means (centers) aligned to the nearest wavelength values.
    - sigmas (ndarray): Array of standard deviations for the Gaussian curves.
    """
    # Ensure wavelengths is a (d, 1) array
    wavelengths = np.array(wavelengths).reshape(-1, 1)

    # Generate evenly spaced means within the range using the step size
    evenly_spaced_means = np.arange(mean_range[0], mean_range[1] + step, step)

    # Find the closest actual wavelengths to the evenly spaced means
    aligned_means = [wavelengths[np.abs(wavelengths - mu).argmin()][0] for mu in evenly_spaced_means]

    # Initialize storage for basis functions, means, and widths
    basis_functions = []
    all_means = []
    all_sigmas = []

    # Generate Gaussian curves for each mean and width
    for mu in aligned_means:
        for sigma in widths:
            gaussian = np.exp(-((wavelengths.flatten() - mu) ** 2) / (2 * sigma ** 2))
            basis_functions.append(gaussian)
            all_means.append(mu)
            all_sigmas.append(sigma)

    # Convert to NumPy arrays
    basis_functions = np.array(basis_functions).T  # Shape (d, num_functions)
    all_means = np.array(all_means)
    all_sigmas = np.array(all_sigmas)

    return basis_functions, all_means, all_sigmas