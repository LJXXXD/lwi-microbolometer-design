"""Shared nominal scene, gene space, and MAP-Elites geometry (presentation suite)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _paths import project_root

from lwi_microbolometer_design.data import SceneConfig, load_substance_atmosphere_data


def spectral_data_path() -> Path:
    return project_root() / Path("data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")


def air_transmittance_path() -> Path:
    return project_root() / Path("data/Test 3 - 4 White Powers/Air transmittance.xlsx")


def load_nominal_scene() -> SceneConfig:
    """Single nominal training scene (matches legacy map_elites scripts)."""
    loaded = load_substance_atmosphere_data(
        spectral_data_file=spectral_data_path(),
        air_transmittance_file=air_transmittance_path(),
        atmospheric_distance_ratio=0.11,
        temperature_kelvin=293.15,
        air_refractive_index=1.0,
    )
    return loaded[0] if isinstance(loaded, list) else loaded


def default_gene_space() -> list[dict[str, float]]:
    num_basis_functions = 4
    param_bounds = [
        {"low": 4.0, "high": 20.0},
        {"low": 0.1, "high": 4.0},
    ]
    return param_bounds * num_basis_functions


def default_map_elites_geometry() -> tuple[int, tuple[float, float]]:
    """(grid_resolution, mu_range)."""
    return 20, (4.0, 20.0)


def num_params_per_basis_function() -> int:
    return 2


def wavelengths_array(scene: SceneConfig) -> np.ndarray:
    return np.asarray(scene.wavelengths)
