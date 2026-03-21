"""Tests for simulation functions."""

import numpy as np

from lwi_microbolometer_design.simulation import blackbody_emit, simulate_sensor_output


class TestBlackbodyEmit:
    """Test blackbody emission calculations."""

    def test_blackbody_emit_basic(self):
        """Test basic blackbody emission calculation."""
        wavelengths = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        temp_K = 300.0

        result = blackbody_emit(wavelengths, temp_K)

        assert isinstance(result, np.ndarray)
        assert result.shape == wavelengths.shape
        assert np.all(result > 0)  # All values should be positive

    def test_blackbody_emit_temperature_dependence(self):
        """Test that higher temperatures produce higher emissions."""
        wavelengths = np.array([5.0])
        temp_low = 300.0
        temp_high = 500.0

        emission_low = blackbody_emit(wavelengths, temp_low)
        emission_high = blackbody_emit(wavelengths, temp_high)

        assert emission_high > emission_low

    def test_blackbody_emit_wavelength_dependence(self):
        """Test that emission varies with wavelength."""
        wavelengths = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        temp_K = 300.0

        result = blackbody_emit(wavelengths, temp_K)

        # Should not be constant across wavelengths
        assert not np.allclose(result, result[0])


class TestSimulateSensorOutput:
    """Test sensor output simulation."""

    def test_simulate_sensor_output_basic(self):
        """Test basic sensor output simulation."""
        wavelengths = np.array([[4.0], [5.0], [6.0], [7.0], [8.0]])
        substances_emissivity = np.array(
            [[0.8, 0.9], [0.7, 0.8], [0.6, 0.7], [0.5, 0.6], [0.4, 0.5]]
        )
        basis_functions = np.array([[1.0, 0.0], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
        temperature_K = 300.0
        atmospheric_distance_ratio = 0.1
        air_refractive_index = 1.0
        air_transmittance = np.array([[0.9], [0.8], [0.7], [0.6], [0.5]])

        result = simulate_sensor_output(
            wavelengths=wavelengths,
            substances_emissivity=substances_emissivity,
            basis_functions=basis_functions,
            temperature_k=temperature_K,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            air_refractive_index=air_refractive_index,
            air_transmittance=air_transmittance,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)  # 2 basis functions, 2 substances
        assert np.all(np.isfinite(result))  # All values should be finite

    def test_simulate_sensor_output_single_substance(self):
        """Test sensor output simulation with single substance."""
        wavelengths = np.array([[4.0], [5.0], [6.0]])
        substances_emissivity = np.array([[0.8], [0.7], [0.6]])
        basis_functions = np.array([[1.0], [0.8], [0.6]])
        temperature_K = 300.0
        atmospheric_distance_ratio = 0.1
        air_refractive_index = 1.0
        air_transmittance = np.array([[0.9], [0.8], [0.7]])

        result = simulate_sensor_output(
            wavelengths=wavelengths,
            substances_emissivity=substances_emissivity,
            basis_functions=basis_functions,
            temperature_k=temperature_K,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            air_refractive_index=air_refractive_index,
            air_transmittance=air_transmittance,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)  # 1 basis function, 1 substance
        assert np.all(np.isfinite(result))
