"""Input and numerical contracts independent of optimizer trajectories."""

from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwi_microbolometer_design.data import SceneConfig, load_substance_atmosphere_data
from lwi_microbolometer_design.simulation import blackbody_emit, simulate_sensor_output
from lwi_microbolometer_design.simulation import sensor_simulation


def scene_inputs():
    return dict(
        wavelengths=np.array([4.0, 6.0, 9.0]),
        emissivity_curves=np.array([[0.5, 0.8], [-0.01, 0.6], [0.4, 0.7]]),
        air_transmittance=np.array([0.0, 0.5, 1.0]),
        temperature_k=300.0,
        atmospheric_distance_ratio=0.1,
        air_refractive_index=1.0,
        substance_names=np.array(["a", "b"]),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("wavelengths", [4.0, 4.0, 9.0]),
        ("wavelengths", [9.0, 6.0, 4.0]),
        ("wavelengths", [0.0, 6.0, 9.0]),
        ("wavelengths", [4.0, np.nan, 9.0]),
        ("temperature_k", 0.0),
        ("temperature_k", np.inf),
        ("atmospheric_distance_ratio", -0.1),
        ("atmospheric_distance_ratio", np.nan),
        ("air_refractive_index", 0.0),
        ("air_refractive_index", np.inf),
        ("air_transmittance", [0.0, -1.0, 1.0]),
        ("air_transmittance", [0.0, 1.01, 1.0]),
        ("air_transmittance", [0.0, np.nan, 1.0]),
        ("emissivity_curves", [[0.5, 0.8], [np.nan, 0.6], [0.4, 0.7]]),
    ],
)
def test_scene_rejects_invalid_domains(field, value):
    inputs = scene_inputs()
    inputs[field] = value
    with pytest.raises(ValueError):
        SceneConfig(**inputs)


def test_scene_preserves_signed_measurements_and_zero_path():
    inputs = scene_inputs()
    inputs["atmospheric_distance_ratio"] = 0.0
    scene = SceneConfig(**inputs)
    np.testing.assert_array_equal(scene.emissivity_curves, inputs["emissivity_curves"])
    assert scene.atmospheric_distance_ratio == 0.0


def test_scene_selects_only_consumed_transmission_column():
    inputs = scene_inputs()
    inputs["air_transmittance"] = np.column_stack([inputs["air_transmittance"], [np.nan] * 3])
    scene = SceneConfig(**inputs)
    np.testing.assert_array_equal(scene.air_transmittance, [0.0, 0.5, 1.0])


def workbooks(tmp_path, atmosphere_wavelengths=None):
    spectral, air = tmp_path / "spectral.xlsx", tmp_path / "air.xlsx"
    pd.DataFrame(
        {"wavelength": [4.0, 6.0, 9.0], "a": [0.5, -0.01, 0.4], "b": [0.8, 0.6, 0.7]}
    ).to_excel(spectral, index=False)
    wl = [4.0, 6.0, 9.0] if atmosphere_wavelengths is None else atmosphere_wavelengths
    pd.DataFrame({"wavelength": wl, "tau": [0.1] * len(wl)}).to_excel(
        air, index=False, header=False
    )
    return spectral, air


@pytest.mark.parametrize("grid", [[4.1, 6.1, 9.1], [4.0, 6.0]])
def test_loader_rejects_shifted_or_different_length_grid(tmp_path, grid):
    with pytest.raises(ValueError, match="exactly match"):
        load_substance_atmosphere_data(*workbooks(tmp_path, grid))


def test_loader_cartesian_order_and_singleton_compatibility(tmp_path):
    files = workbooks(tmp_path)
    singleton = load_substance_atmosphere_data(*files, temperature_kelvin=np.array(300.0))
    assert isinstance(singleton, SceneConfig)
    assert isinstance(
        load_substance_atmosphere_data(*files, temperature_kelvin=(300.0,)), SceneConfig
    )
    scenes = load_substance_atmosphere_data(
        *files,
        atmospheric_distance_ratio=(0.0, 0.2),
        temperature_kelvin=[280.0, 300.0],
        air_refractive_index=[1.0, 1.1],
    )
    assert [
        (s.atmospheric_distance_ratio, s.temperature_k, s.air_refractive_index) for s in scenes
    ] == [
        (0.0, 280.0, 1.0),
        (0.0, 280.0, 1.1),
        (0.0, 300.0, 1.0),
        (0.0, 300.0, 1.1),
        (0.2, 280.0, 1.0),
        (0.2, 280.0, 1.1),
        (0.2, 300.0, 1.0),
        (0.2, 300.0, 1.1),
    ]
    assert singleton.emissivity_curves[1, 0] == -0.01


@pytest.mark.parametrize("conditions", [[], [[300.0, 310.0]]])
def test_loader_rejects_empty_and_matrix_conditions(tmp_path, conditions):
    with pytest.raises(ValueError, match="nonempty scalar or 1D"):
        load_substance_atmosphere_data(*workbooks(tmp_path), temperature_kelvin=conditions)


@pytest.mark.parametrize("temperature", [200.0, 300.0, 800.0])
def test_planck_absolute_radiance_against_si(temperature):
    # SI meter-density reference, then per-meter -> per-micrometer conversion.
    meters = np.array([4.0, 10.0, 20.0]) * 1e-6
    expected = (
        2
        * 6.62607015e-34
        * 299792458**2
        / meters**5
        / np.expm1(6.62607015e-34 * 299792458 / (meters * 1.380649e-23 * temperature))
    ) * 1e-6
    # CODATA 2010 constants differ from exact SI by < 1.2 ppm on this grid.
    np.testing.assert_allclose(
        blackbody_emit(meters * 1e6, temperature), expected, rtol=2e-6, atol=0.0
    )


def test_planck_full_spectrum_integral_is_radiance_not_exitance():
    wl = np.geomspace(0.01, 1e5, 40000)
    actual = np.trapezoid(blackbody_emit(wl, 300.0), wl)
    expected = 5.670374419e-8 * 300.0**4 / np.pi
    # Includes the historical-constant difference, tail truncation and grid error.
    assert actual == pytest.approx(expected, rel=1e-6)


def test_planck_wien_tail_avoids_intermediate_exponential_overflow():
    with localcontext() as ctx:
        ctx.prec = 60
        wl = Decimal("1e-9")
        h, c, k = Decimal("6.62606957e-34"), Decimal(299792458), Decimal("1.3806488e-23")
        temperature = float(h * c / (wl * k * Decimal(720)))
        x = h * c / (wl * k * Decimal(str(temperature)))
        expected = float(2 * h * c * c / (wl**5 * (x.exp() - 1)) * Decimal("1e-6"))
    with np.errstate(all="raise"):
        actual = blackbody_emit(np.array([0.001]), temperature)[0]
    assert actual > 0.0
    assert actual == pytest.approx(expected, rel=2e-12, abs=0.0)
    assert blackbody_emit(np.array([0.001]), 1.0)[0] == 0.0


def test_planck_rayleigh_jeans_and_index_scaling():
    wl = np.array([1e9])
    expected = 2 * 299792458 * 1.3806488e-23 * 300 * 1e18 / wl**4
    np.testing.assert_allclose(blackbody_emit(wl, 300.0), expected, rtol=3e-8, atol=0.0)
    np.testing.assert_allclose(
        blackbody_emit([10.0], 300.0, 1.1), blackbody_emit([10.0], 300.0) * 1.1**2, rtol=1e-14
    )


@pytest.mark.parametrize(
    "wl,t,n",
    [([0.0], 300.0, 1.0), ([np.nan], 300.0, 1.0), ([10.0], 0.0, 1.0), ([10.0], 300.0, np.inf)],
)
def test_planck_rejects_invalid_inputs(wl, t, n):
    with pytest.raises(ValueError):
        blackbody_emit(wl, t, n)


def test_nonuniform_quadrature_with_signed_spectra_and_nonconstant_atmosphere(monkeypatch):
    monkeypatch.setattr(
        sensor_simulation, "blackbody_emit", lambda wl, t, n: np.full(wl.shape, 2.0)
    )
    actual = simulate_sensor_output(
        [4, 6, 9],
        [[0.5, 1, 0.1], [1, 0.5, -0.2], [0, 1, 0.4]],
        [[1, 2], [2, 1], [3, 0]],
        300.0,
        0.5,
        1.0,
        [1, 0.25, 0],
    )
    np.testing.assert_allclose(
        actual, [[6.0, 4.5, -0.8], [4.5, 5.25, -0.1]], rtol=1e-14, atol=1e-14
    )


def test_zero_distance_ratio_removes_attenuation_including_zero_transmission():
    inputs = scene_inputs()
    basis = np.ones((3, 2))
    zero_path = simulate_sensor_output(
        inputs["wavelengths"], inputs["emissivity_curves"], basis, 300.0, 0.0, 1.0, [0.0, 0.5, 1.0]
    )
    transparent = simulate_sensor_output(
        inputs["wavelengths"], inputs["emissivity_curves"], basis, 300.0, 1.0, 1.0, [1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(zero_path, transparent)


def test_real_nominal_workbook_is_preserved():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/Test 3 - 4 White Powers"
    scene = load_substance_atmosphere_data(
        data / "white_powders_with_labels.xlsx", data / "Air transmittance.xlsx"
    )
    assert scene.wavelengths.shape == (142,)
    assert scene.emissivity_curves.min() == pytest.approx(-0.013948)
