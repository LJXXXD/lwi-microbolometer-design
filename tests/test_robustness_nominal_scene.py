"""Tests for nominal-scene matching in environmental robustness grids."""

from __future__ import annotations

import numpy as np
import pytest

from lwi_microbolometer_design.analysis.robustness import find_nominal_scene_index
from lwi_microbolometer_design.data.scene_config import SceneConfig


def _minimal_scene(
    temperature_k: float,
    atmospheric_distance_ratio: float,
    air_refractive_index: float,
) -> SceneConfig:
    wl = np.array([1.0, 2.0], dtype=np.float64)
    em = np.zeros((2, 1), dtype=np.float64)
    at = np.ones(2, dtype=np.float64)
    return SceneConfig(
        wavelengths=wl,
        emissivity_curves=em,
        air_transmittance=at,
        temperature_k=temperature_k,
        atmospheric_distance_ratio=atmospheric_distance_ratio,
        air_refractive_index=air_refractive_index,
        substance_names=np.array(["s0"], dtype=object),
    )


def test_find_nominal_scene_index_matches_training_tuple() -> None:
    nominal = _minimal_scene(293.15, 0.11, 1.0)
    scenes = [
        _minimal_scene(273.15, 0.11, 1.0),
        nominal,
        _minimal_scene(293.15, 0.20, 1.0),
    ]
    assert find_nominal_scene_index(scenes, nominal) == 1


def test_find_nominal_scene_index_raises_when_missing() -> None:
    nominal = _minimal_scene(293.15, 0.11, 1.0)
    scenes = [_minimal_scene(300.0, 0.11, 1.0)]
    with pytest.raises(ValueError, match="No robustness grid scene"):
        find_nominal_scene_index(scenes, nominal)
