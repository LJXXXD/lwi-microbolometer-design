"""LWI Microbolometer Design and Optimization Package.

This package provides tools for designing and optimizing infrared microbolometer sensors
for substance detection and identification using spectral analysis and optimization algorithms.
"""

__version__ = "0.1.0"

# Analysis functions
from .analysis import (
    ConditionLabel,
    RobustnessResult,
    compute_distance_matrix,
    evaluate_archive_robustness,
    evaluate_elite_robustness,
    evaluate_solutions_robustness,
    group_based_dissimilarity_score,
    ivat_transform,
    min_based_dissimilarity_score,
    spectral_angle_mapper,
    summarise_robustness,
    vat_reorder,
)

# GA functions
from .ga import (
    AdvancedGA,
    MinDissimilarityFitnessEvaluator,
    NichingConfig,
    analyze_population_diversity,
)

# MAP-Elites functions
from .map_elites import (
    EmitterBase,
    OptimizingEmitter,
    plot_cma_me_progress,
    plot_map_elites_heatmap,
    plot_polished_elites,
    plot_top_elites,
    polish_single_elite_cma,
    polish_single_elite_hc,
    run_cma_me,
    run_map_elites,
)

# Simulation functions
from .simulation import (
    blackbody_emit,
    gaussian_parameters_to_unit_amplitude_curves,
    simulate_sensor_output,
)

# Visualization functions
from .visualization import (
    plot_condition_sensitivity,
    plot_fitness_degradation_heatmap,
    plot_fitness_distribution_by_condition,
    plot_retention_histogram,
    plot_worst_case_vs_nominal,
    visualize_distance_matrix,
    visualize_distance_matrix_large,
    visualize_distance_matrix_simple,
    visualize_sensor_output,
)

__all__ = [
    # Package metadata
    "__version__",
    # Analysis functions
    "ConditionLabel",
    "RobustnessResult",
    "compute_distance_matrix",
    "evaluate_archive_robustness",
    "evaluate_elite_robustness",
    "evaluate_solutions_robustness",
    "group_based_dissimilarity_score",
    "ivat_transform",
    "min_based_dissimilarity_score",
    "spectral_angle_mapper",
    "summarise_robustness",
    "vat_reorder",
    # GA functions and classes
    "AdvancedGA",
    "MinDissimilarityFitnessEvaluator",
    "NichingConfig",
    "analyze_population_diversity",
    # MAP-Elites functions
    "EmitterBase",
    "OptimizingEmitter",
    "plot_cma_me_progress",
    "plot_map_elites_heatmap",
    "plot_polished_elites",
    "plot_top_elites",
    "polish_single_elite_cma",
    "polish_single_elite_hc",
    "run_cma_me",
    "run_map_elites",
    # Simulation functions
    "blackbody_emit",
    "gaussian_parameters_to_unit_amplitude_curves",
    "simulate_sensor_output",
    # Visualization functions
    "plot_condition_sensitivity",
    "plot_fitness_degradation_heatmap",
    "plot_fitness_distribution_by_condition",
    "plot_retention_histogram",
    "plot_worst_case_vs_nominal",
    "visualize_distance_matrix",
    "visualize_distance_matrix_large",
    "visualize_distance_matrix_simple",
    "visualize_sensor_output",
]
