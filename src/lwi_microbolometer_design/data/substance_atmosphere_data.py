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
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd

from .scene_config import SceneConfig

logger = logging.getLogger(__name__)


@overload
def load_substance_atmosphere_data(
    spectral_data_file: Path,
    air_transmittance_file: Path,
    atmospheric_distance_ratio: float = 0.11,
    temperature_kelvin: float = 293.15,
    air_refractive_index: float = 1.0,
) -> SceneConfig: ...


@overload
def load_substance_atmosphere_data(
    spectral_data_file: Path,
    air_transmittance_file: Path,
    atmospheric_distance_ratio: float | Sequence[float] | np.ndarray = 0.11,
    temperature_kelvin: float | Sequence[float] | np.ndarray = 293.15,
    air_refractive_index: float | Sequence[float] | np.ndarray = 1.0,
) -> list[SceneConfig]: ...


def load_substance_atmosphere_data(
    spectral_data_file: Path,
    air_transmittance_file: Path,
    atmospheric_distance_ratio: float | list[float] | np.ndarray = 0.11,
    temperature_kelvin: float | list[float] | np.ndarray = 293.15,
    air_refractive_index: float | list[float] | np.ndarray = 1.0,
) -> SceneConfig | list[SceneConfig]:
    """
    Load substance spectral data and atmospheric data from Excel files.

    Loads spectral emissivity curves for substances and atmospheric transmittance
    data, along with simulation parameters, into a :class:`SceneConfig` (or list
    thereof for multi-condition setups) for use in sensor simulation.

    This function loads the fixed scene/environment data needed for simulation.
    Sensor basis functions must be provided separately as they are typically
    optimized or loaded independently.

    Wavelengths and air transmittance are canonicalized to 1D ``(d,)`` inside
    :class:`SceneConfig`.

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
    SceneConfig | list[SceneConfig]
        If all parameters are scalars: a single :class:`SceneConfig`.

        If any parameter is a list/array: list of :class:`SceneConfig`, one per
        condition. Conditions are generated from all combinations of provided
        parameter values.

        Type checkers treat any ``Sequence[float]`` or ``ndarray`` argument as
        the multi-condition overload (returning ``list[SceneConfig]``). At
        runtime, length-1 sequences still yield a single :class:`SceneConfig`.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Single condition
    >>> spectral_file = Path('data/spectra.xlsx')
    >>> transmittance_file = Path('data/air_transmittance.xlsx')
    >>> scene = load_substance_atmosphere_data(spectral_file, transmittance_file)
    >>> print(f'Loaded {len(scene.substance_names)} substances')
    >>> # Multiple temperatures
    >>> scene_list = load_substance_atmosphere_data(
    ...     spectral_file, transmittance_file, temperature_kelvin=[273.15, 293.15, 313.15]
    ... )
    >>> print(f'Created {len(scene_list)} simulation conditions')
    """
    # Load spectral data (same for all conditions)
    substances_spectral_data = pd.read_excel(spectral_data_file)
    wavelengths = substances_spectral_data.iloc[:, :1].to_numpy()
    substance_names = substances_spectral_data.columns[1:].to_numpy()
    emissivity_curves = substances_spectral_data.iloc[:, 1:].to_numpy()

    # Load air transmittance (same for all conditions)
    air_transmittance_df = pd.read_excel(air_transmittance_file, header=None)
    air_transmittance: np.ndarray = air_transmittance_df.to_numpy()[:, 1:]

    logger.info(f"Loaded spectral data for {len(substance_names)} substances")
    wl_flat = np.squeeze(np.asarray(wavelengths, dtype=np.float64))
    if wl_flat.ndim == 1:
        logger.info(f"Wavelength range: {wl_flat[0]:.1f} - {wl_flat[-1]:.1f} µm")
    else:
        logger.info("Wavelength range: (non-vector layout; see SceneConfig)")

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

    def build_scene(
        atm_ratio: float,
        temp_k: float,
        ref_idx: float,
    ) -> SceneConfig:
        return SceneConfig(
            wavelengths=wavelengths,
            substance_names=substance_names,
            emissivity_curves=emissivity_curves,
            air_transmittance=air_transmittance,
            temperature_k=float(temp_k),
            atmospheric_distance_ratio=float(atm_ratio),
            air_refractive_index=float(ref_idx),
        )

    if not is_multi_condition:
        return build_scene(float(atm_ratios[0]), float(temps[0]), float(ref_indices[0]))

    atm_mesh, temp_mesh, ref_mesh = np.meshgrid(atm_ratios, temps, ref_indices, indexing="ij")

    conditions = list(
        zip(atm_mesh.flatten(), temp_mesh.flatten(), ref_mesh.flatten(), strict=False)
    )

    logger.info(f"Creating {len(conditions)} simulation conditions from parameter combinations")

    return [
        build_scene(float(atm_ratio), float(temp), float(ref_idx))
        for atm_ratio, temp, ref_idx in conditions
    ]
