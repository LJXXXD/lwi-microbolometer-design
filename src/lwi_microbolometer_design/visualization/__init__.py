"""Visualization module for plotting sensor outputs and analysis results."""

# Distance matrix visualization
from .distance_matrix_visualization import (
    visualize_distance_matrix,
    visualize_distance_matrix_large,  # Backward compatibility alias
    visualize_distance_matrix_simple,
)

# Robustness visualization
from .robustness_visualization import (
    plot_condition_sensitivity,
    plot_fitness_degradation_heatmap,
    plot_fitness_distribution_by_condition,
    plot_retention_histogram,
    plot_worst_case_vs_nominal,
)

# Sensor output visualization
from .sensor_output_visualization import visualize_sensor_output

__all__ = [
    # Distance matrix visualization
    "visualize_distance_matrix",
    "visualize_distance_matrix_large",  # Backward compatibility
    "visualize_distance_matrix_simple",
    # Robustness visualization
    "plot_condition_sensitivity",
    "plot_fitness_degradation_heatmap",
    "plot_fitness_distribution_by_condition",
    "plot_retention_histogram",
    "plot_worst_case_vs_nominal",
    # Sensor output visualization
    "visualize_sensor_output",
]
