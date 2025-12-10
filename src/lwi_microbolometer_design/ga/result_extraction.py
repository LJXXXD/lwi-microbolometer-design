"""
GA result extraction utilities.

Provides functions to extract basic results from AdvancedGA instances into
standardized dictionary formats for visualization, analysis, and reporting.

This module focuses on minimal, fast extraction of essential GA state.
For comprehensive population analysis, see the `analysis` module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lwi_microbolometer_design.ga.advanced_ga import AdvancedGA

from lwi_microbolometer_design.ga import calculate_population_diversity

logger = logging.getLogger(__name__)


def extract_basic_results(ga_instance: AdvancedGA) -> dict[str, Any]:
    """
    Extract basic result dictionary from AdvancedGA instance.

    Extracts the minimal set of fields needed for visualization and basic analysis.
    This function provides a baseline structure that works with standard visualization
    functions. For custom analysis needs, users can extend this or build their own
    extraction functions.

    Parameters
    ----------
    ga_instance : AdvancedGA
        GA instance that has been run

    Returns
    -------
    dict[str, Any]
        Result dictionary containing:
        - best_fitness: float - Best fitness value found
        - best_chromosome: np.ndarray - Best solution chromosome
        - final_population: np.ndarray - Final generation population
        - final_fitness_scores: np.ndarray - Final generation fitness scores
        - best_fitness_history: list[float] - Best fitness per generation (if tracked)
        - mean_fitness_history: list[float] - Mean fitness per generation
          (approximated if not tracked)
        - diversity_history: list[float] - Diversity scores per generation
          (approximated if not tracked)

    Notes
    -----
    - History fields (mean_fitness_history, diversity_history) are approximated
      from final generation data if not tracked during evolution
    - For accurate history data, implement on_generation callback to track these metrics
    - Users can extend this result dict with additional fields as needed

    Examples
    --------
    Basic usage:
        >>> ga = AdvancedGA(...)
        >>> ga.run()
        >>> results = extract_basic_results(ga)
        >>> visualize_ga_results(results, data, output_dir)

    Extended usage (adding custom fields):
        >>> results = extract_basic_results(ga)
        >>> results['custom_metric'] = compute_custom_metric(ga)
        >>> results['high_fitness_count_history'] = my_tracked_history
    """
    # Extract best fitness history (PyGAD tracks this automatically)
    best_fitness_history = (
        list(ga_instance.best_solutions_fitness)
        if hasattr(ga_instance, 'best_solutions_fitness')
        and ga_instance.best_solutions_fitness is not None
        else []
    )

    # Get final population and fitness
    final_population = (
        ga_instance.population if ga_instance.population is not None else np.array([])
    )
    final_fitness_scores = (
        ga_instance.last_generation_fitness
        if hasattr(ga_instance, 'last_generation_fitness')
        and ga_instance.last_generation_fitness is not None
        else np.array([])
    )

    # Build mean fitness history from best fitness history if not tracked
    # (simple approximation: use current mean for all generations)
    num_generations = len(best_fitness_history)
    if num_generations > 0 and final_fitness_scores.size > 0:
        current_mean = float(np.mean(final_fitness_scores))
        mean_fitness_history = [current_mean] * num_generations
    else:
        mean_fitness_history = []

    # Calculate final diversity if population exists
    # Use unified diversity calculation that respects GA's niching configuration
    if final_population.size > 0:
        final_diversity = calculate_population_diversity(
            final_population, ga_instance.niching_config if ga_instance.niching_config else None
        )
        diversity_history = (
            [final_diversity] * num_generations if num_generations > 0 else [final_diversity]
        )
    else:
        diversity_history = []

    # Get best solution
    if hasattr(ga_instance, 'best_solution'):
        try:
            best_chromosome, best_fitness, _best_idx = ga_instance.best_solution()
        except Exception:
            best_chromosome = np.array([])
            best_fitness = 0.0
    else:
        best_chromosome = np.array([])
        best_fitness = 0.0

    # Build result dict
    result_dict: dict[str, Any] = {
        'best_fitness': best_fitness,
        'best_chromosome': best_chromosome,
        'final_population': final_population,
        'final_fitness_scores': final_fitness_scores,
        'best_fitness_history': best_fitness_history,
        'mean_fitness_history': mean_fitness_history,
        'diversity_history': diversity_history,
    }

    return result_dict
