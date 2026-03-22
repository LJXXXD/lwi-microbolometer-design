"""Visualization module for plotting sensor outputs and analysis results."""

# Distance matrix visualization
from .distance_matrix_visualization import (
    visualize_distance_matrix,
    visualize_distance_matrix_large,  # Backward compatibility alias
    visualize_distance_matrix_simple,
)

# Sensor output visualization
from .sensor_output_visualization import visualize_sensor_output

__all__ = [
    # Distance matrix visualization
    "visualize_distance_matrix",
    "visualize_distance_matrix_large",  # Backward compatibility
    "visualize_distance_matrix_simple",
    # Sensor output visualization
    "visualize_sensor_output",
]
