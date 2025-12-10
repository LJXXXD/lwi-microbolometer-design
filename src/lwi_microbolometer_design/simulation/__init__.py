"""Sensor simulation module for microbolometer design."""

# Blackbody radiation
from .blackbody import blackbody_emit

# Gaussian basis functions
from .gaussian_parameter_to_curves import gaussian_parameters_to_unit_amplitude_curves

# Deprecated basis functions
from .generated_basis_functions_deprecated import generate_structured_gaussian_basis_functions

# Sensor simulation
from .sensor_simulation import simulate_sensor_output

__all__ = [
    # Blackbody radiation
    'blackbody_emit',
    # Gaussian basis functions
    'gaussian_parameters_to_unit_amplitude_curves',
    # Deprecated basis functions
    'generate_structured_gaussian_basis_functions',  # TODO: Review - temporary copy
    # Sensor simulation
    'simulate_sensor_output',
]
