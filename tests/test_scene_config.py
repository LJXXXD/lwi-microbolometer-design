"""Tests for SceneConfig shape canonicalization and validation."""

import numpy as np
import pytest

from lwi_microbolometer_design.data.scene_config import SceneConfig


def _base_arrays(d: int = 5, n: int = 3):
    wl = np.linspace(4.0, 20.0, d).reshape(-1, 1)
    em = np.random.default_rng(0).random((d, n))
    air = np.ones((d, 1)) * 0.9
    names = np.array([f"s{i}" for i in range(n)], dtype=object)
    return wl, em, air, names


def test_scene_config_squeezes_wavelengths_and_transmittance():
    wl, em, air, names = _base_arrays()
    scene = SceneConfig(
        wavelengths=wl,
        emissivity_curves=em,
        air_transmittance=air,
        temperature_k=293.15,
        atmospheric_distance_ratio=0.11,
        air_refractive_index=1.0,
        substance_names=names,
    )
    assert scene.wavelengths.shape == (5,)
    assert scene.wavelengths.ndim == 1
    assert scene.air_transmittance.shape == (5,)
    assert scene.air_transmittance.ndim == 1


def test_scene_config_first_column_when_transmittance_multi_column():
    d, n = 4, 2
    wl = np.linspace(1.0, 4.0, d)
    em = np.ones((d, n))
    air = np.stack([np.linspace(0.5, 1.0, d), np.linspace(0.0, 0.1, d)], axis=1)
    names = np.array(["a", "b"], dtype=object)
    scene = SceneConfig(
        wavelengths=wl,
        emissivity_curves=em,
        air_transmittance=air,
        temperature_k=300.0,
        atmospheric_distance_ratio=0.1,
        air_refractive_index=1.0,
        substance_names=names,
    )
    np.testing.assert_allclose(scene.air_transmittance, air[:, 0])


def test_scene_config_rejects_mismatched_emissivity_rows():
    wl = np.array([1.0, 2.0])
    em = np.ones((3, 2))
    air = np.ones(3)
    names = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="first axis"):
        SceneConfig(
            wavelengths=wl,
            emissivity_curves=em,
            air_transmittance=air,
            temperature_k=300.0,
            atmospheric_distance_ratio=0.1,
            air_refractive_index=1.0,
            substance_names=names,
        )


def test_scene_config_rejects_mismatched_air_transmittance_length():
    wl = np.array([1.0, 2.0, 3.0])
    em = np.ones((3, 2))
    air = np.ones(2)
    names = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="air_transmittance length"):
        SceneConfig(
            wavelengths=wl,
            emissivity_curves=em,
            air_transmittance=air,
            temperature_k=300.0,
            atmospheric_distance_ratio=0.1,
            air_refractive_index=1.0,
            substance_names=names,
        )


def test_scene_config_rejects_substance_name_count_mismatch():
    wl = np.array([1.0, 2.0, 3.0])
    em = np.ones((3, 2))
    air = np.ones(3)
    names = np.array(["only_one"], dtype=object)
    with pytest.raises(ValueError, match="substance_names length"):
        SceneConfig(
            wavelengths=wl,
            emissivity_curves=em,
            air_transmittance=air,
            temperature_k=300.0,
            atmospheric_distance_ratio=0.1,
            air_refractive_index=1.0,
            substance_names=names,
        )
