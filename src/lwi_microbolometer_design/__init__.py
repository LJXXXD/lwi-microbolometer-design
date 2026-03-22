"""LWI Microbolometer Design and Optimization Package.

This package provides tools for designing and optimizing infrared microbolometer sensors
for substance detection and identification using spectral analysis and optimization algorithms.
"""

__version__ = "0.1.0"

# Analysis functions
from .analysis import (
    compute_distance_matrix,
    group_based_dissimilarity_score,
    ivat_transform,
    min_based_dissimilarity_score,
    spectral_angle_mapper,
    vat_reorder,
)

# GA functions
from .ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    NichingConfig,
    analyze_population_diversity,
)

# Simulation functions
from .simulation import (
    blackbody_emit,
    gaussian_parameters_to_unit_amplitude_curves,
    simulate_sensor_output,
)

# Visualization functions
from .visualization import (
    visualize_distance_matrix,
    visualize_distance_matrix_large,
    visualize_distance_matrix_simple,
    visualize_sensor_output,
)

__all__ = [
    # Package metadata
    "__version__",
    # Analysis functions
    "compute_distance_matrix",
    "group_based_dissimilarity_score",
    "ivat_transform",
    "min_based_dissimilarity_score",
    "spectral_angle_mapper",
    "vat_reorder",
    # GA functions and classes
    "AdvancedGA",
    "MinDissimilarityFitnessEvaluator",
    "NichingConfig",
    "analyze_population_diversity",
    # Simulation functions
    "blackbody_emit",
    "gaussian_parameters_to_unit_amplitude_curves",
    "simulate_sensor_output",
    # Visualization functions
    "visualize_distance_matrix",
    "visualize_distance_matrix_large",
    "visualize_distance_matrix_simple",
    "visualize_sensor_output",
]
