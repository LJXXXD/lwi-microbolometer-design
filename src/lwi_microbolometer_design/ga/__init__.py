"""Genetic Algorithm package for sensor design optimization.

This package provides a complete GA system including:
- Core GA runtime with niching (AdvancedGA)
- Custom mutation operators (diversity_preserving_mutation)
- Fitness evaluators (MinDissimilarityFitnessEvaluator)
- Population analysis tools (analyze_population_diversity)
- Configuration utilities (create_ga_config, load_ga_configuration_from_csv)
- Visualization utilities (visualize_ga_results, plot_top_sensor_designs)
- Hyperparameter tuning (HyperparameterTuner)
"""

# Core GA classes
from .advanced_ga import (
    AdvancedGA,
    NichingConfig,
    compute_chromosome_distance,
    niche_sharing_coefficient,
)

# Analysis
from .analysis import AnalysisConfig, analyze_population_diversity

# Diversity
from .diversity import calculate_population_diversity, compute_population_distance_matrix

# Fitness
from .fitness import MinDissimilarityFitnessEvaluator

# Configuration
from .ga_configuration import (
    create_ga_config,
    load_ga_configuration_from_csv,
)

# Mutations
from .mutations import MutationConfig, diversity_preserving_mutation

# Results
from .result_extraction import extract_basic_results

# Tuning
from .tuning import (
    HyperparameterSearchSpace,
    HyperparameterTuner,
    create_default_search_space,
    create_focused_search_space,
)

# Visualization
from .visualization import (
    plot_best_design,
    plot_fitness_spread_evolution,
    plot_high_fitness_evolution,
    plot_ivat_analysis,
    plot_top_sensor_designs,
    visualize_ga_results,
)

__all__ = [
    # Core GA classes
    'AdvancedGA',
    'NichingConfig',
    'compute_chromosome_distance',
    'niche_sharing_coefficient',
    # Analysis
    'AnalysisConfig',
    'analyze_population_diversity',
    # Diversity
    'calculate_population_diversity',
    'compute_population_distance_matrix',
    # Fitness
    'MinDissimilarityFitnessEvaluator',
    # Configuration
    'create_ga_config',
    'load_ga_configuration_from_csv',
    # Mutations
    'MutationConfig',
    'diversity_preserving_mutation',
    # Results
    'extract_basic_results',
    # Tuning
    'HyperparameterSearchSpace',
    'HyperparameterTuner',
    'create_default_search_space',
    'create_focused_search_space',
    # Visualization
    'plot_best_design',
    'plot_fitness_spread_evolution',
    'plot_high_fitness_evolution',
    'plot_ivat_analysis',
    'plot_top_sensor_designs',
    'visualize_ga_results',
]
