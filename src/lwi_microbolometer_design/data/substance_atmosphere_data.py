"""
Substance and atmosphere data loading utilities.

This module provides functions to load substance spectral emissivity data and
atmospheric transmittance data from Excel files, along with simulation parameters,
for use in sensor simulation and optimization.

The loaded data represents the fixed scene/environment conditions (substances
and atmospheric parameters) needed for sensor simulation. Sensor basis functions
are not included here as they are typically optimized or loaded separately.

Supports both single-condition and multi-condition simulation setups.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_substance_atmosphere_data(
    spectral_data_file: Path,
    air_transmittance_file: Path,
    atmospheric_distance_ratio: float | list[float] | np.ndarray = 0.11,
    temperature_kelvin: float | list[float] | np.ndarray = 293.15,
    air_refractive_index: float | list[float] | np.ndarray = 1.0,
) -> dict[str, np.ndarray | float] | list[dict[str, np.ndarray | float]]:
    """
    Load substance spectral data and atmospheric data from Excel files.

    Loads spectral emissivity curves for substances and atmospheric transmittance
    data, along with simulation parameters, into a structured dictionary (or list
    of dictionaries for multi-condition setups) for use in sensor simulation.

    This function loads the fixed scene/environment data needed for simulation.
    Sensor basis functions must be provided separately as they are typically
    optimized or loaded independently.

    Parameters
    ----------
    spectral_data_file : Path
        Path to the Excel file containing spectral data.
        Expected format: First column contains wavelengths, subsequent columns
        contain emissivity values for different substances.
    air_transmittance_file : Path
        Path to the Excel file containing air transmittance data.
        Expected format: Second column onwards contains transmittance values.
    atmospheric_distance_ratio : float | list[float] | np.ndarray, optional
        Atmospheric distance ratio(s) for simulation (default: 0.11)
        If multiple values provided, creates one dataset per value
    temperature_kelvin : float | list[float] | np.ndarray, optional
        Scene temperature(s) in Kelvin (default: 293.15)
        If multiple values provided, creates one dataset per value
    air_refractive_index : float | list[float] | np.ndarray, optional
        Air refractive index value(s) (default: 1.0)
        If multiple values provided, creates one dataset per value

    Returns
    -------
    dict[str, np.ndarray | float] | list[dict[str, np.ndarray | float]]
        If all parameters are scalars: single dictionary containing:
        - wavelengths: np.ndarray - Array of wavelengths (µm), shape (n_points, 1)
        - substance_names: np.ndarray - Array of substance names, shape (n_substances,)
        - emissivity_curves: np.ndarray - Matrix of emissivity values,
          shape (n_points, n_substances)
        - air_transmittance: np.ndarray - Array of air transmittance values
        - atmospheric_distance_ratio: float - Atmospheric distance ratio parameter
        - temperature_K: float - Scene temperature in Kelvin
        - air_refractive_index: float - Air refractive index parameter

        If any parameter is a list/array: list of dictionaries, one per condition.
        Conditions are generated from all combinations of provided parameter values.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Single condition
    >>> spectral_file = Path('data/spectra.xlsx')
    >>> transmittance_file = Path('data/air_transmittance.xlsx')
    >>> data = load_substance_atmosphere_data(spectral_file, transmittance_file)
    >>> print(f'Loaded {len(data["substance_names"])} substances')
    >>> # Multiple temperatures
    >>> data_list = load_substance_atmosphere_data(
    ...     spectral_file, transmittance_file, temperature_kelvin=[273.15, 293.15, 313.15]
    ... )
    >>> print(f'Created {len(data_list)} simulation conditions')
    """
    # Load spectral data (same for all conditions)
    substances_spectral_data = pd.read_excel(spectral_data_file)
    wavelengths = substances_spectral_data.iloc[:, :1].to_numpy()
    substance_names = substances_spectral_data.columns[1:].to_numpy()
    emissivity_curves = substances_spectral_data.iloc[:, 1:].to_numpy()

    # Load air transmittance (same for all conditions)
    air_transmittance_df = pd.read_excel(air_transmittance_file, header=None)
    air_transmittance: np.ndarray = air_transmittance_df.to_numpy()[:, 1:]

    logger.info(f'Loaded spectral data for {len(substance_names)} substances')
    logger.info(f'Wavelength range: {wavelengths[0][0]:.1f} - {wavelengths[-1][0]:.1f} µm')

    # Convert parameters to numpy arrays for easier handling
    def to_array(value: float | list[float] | np.ndarray) -> np.ndarray:
        """Convert scalar or list to numpy array."""
        if isinstance(value, (list, tuple)):
            return np.array(value)
        elif isinstance(value, np.ndarray):
            return value
        else:
            return np.array([value])

    atm_ratios = to_array(atmospheric_distance_ratio)
    temps = to_array(temperature_kelvin)
    ref_indices = to_array(air_refractive_index)

    # Check if we have multiple conditions
    is_multi_condition = len(atm_ratios) > 1 or len(temps) > 1 or len(ref_indices) > 1

    if not is_multi_condition:
        # Single condition - return single dict (backward compatible)
        return {
            'wavelengths': wavelengths,
            'substance_names': substance_names,
            'emissivity_curves': emissivity_curves,
            'air_transmittance': air_transmittance,
            'atmospheric_distance_ratio': float(atm_ratios[0]),
            'temperature_K': float(temps[0]),
            'air_refractive_index': float(ref_indices[0]),
        }

    # Multiple conditions - generate all combinations
    # Use meshgrid to create all combinations
    atm_mesh, temp_mesh, ref_mesh = np.meshgrid(atm_ratios, temps, ref_indices, indexing='ij')

    # Flatten to get all combinations
    conditions = list(
        zip(atm_mesh.flatten(), temp_mesh.flatten(), ref_mesh.flatten(), strict=False)
    )

    logger.info(f'Creating {len(conditions)} simulation conditions from parameter combinations')

    # Create list of data dicts, one per condition
    data_list: list[dict[str, np.ndarray | float]] = []
    for atm_ratio, temp, ref_idx in conditions:
        data_list.append(
            {
                'wavelengths': wavelengths,
                'substance_names': substance_names,
                'emissivity_curves': emissivity_curves,
                'air_transmittance': air_transmittance,
                'atmospheric_distance_ratio': float(atm_ratio),
                'temperature_K': float(temp),
                'air_refractive_index': float(ref_idx),
            }
        )

    return data_list
