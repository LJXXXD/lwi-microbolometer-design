"""
GA configuration utilities for AdvancedGA.

This module provides functions to create, load, and validate GA configurations
from various sources (defaults, CSV files, programmatic specification).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .advanced_ga import NichingConfig
from .mutations import diversity_preserving_mutation

logger = logging.getLogger(__name__)


def create_ga_config(
    num_generations: int = 2000,
    num_parents_mating: int = 50,
    sol_per_pop: int = 200,
    parent_selection_type: str = 'tournament',
    k_tournament: int = 3,
    keep_elitism: int = 5,
    crossover_type: str = 'uniform',
    crossover_probability: float = 0.8,
    mutation_type: str
    | Callable
    | None = None,  # Defaults to diversity_preserving_mutation if None
    mutation_probability: float = 0.1,
    save_best_solutions: bool = True,
    stop_criteria: str = 'saturate_1000',
    niching_enabled: bool = True,
    niching_use_optimal_pairing: bool = True,
    niching_params_per_group: int = 2,
    niching_sigma_share: float = 0.5,
    niching_alpha: float = 0.5,
    niching_optimal_pairing_metric: str = 'euclidean',
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Create GA configuration dictionary programmatically.

    This function provides an easy way to specify GA parameters directly in code,
    with sensible defaults. All parameters are optional and can be overridden.

    Parameters
    ----------
    num_generations : int, optional
        Number of generations to run (default: 2000)
    num_parents_mating : int, optional
        Number of parents selected for mating (default: 50)
    sol_per_pop : int, optional
        Population size (default: 200)
    parent_selection_type : str, optional
        Parent selection method (default: 'tournament')
    k_tournament : int, optional
        Tournament size for tournament selection (default: 3)
    keep_elitism : int, optional
        Number of elite solutions to preserve (default: 5)
    crossover_type : str, optional
        Crossover operator type (default: 'uniform')
    crossover_probability : float, optional
        Probability of crossover (default: 0.8)
    mutation_type : str | Callable | None, optional
        Mutation operator. Can be:
        - A string like 'adaptive', 'random', etc. (PyGAD built-in operators)
        - A callable function like diversity_preserving_mutation
        - None (defaults to diversity_preserving_mutation)
    mutation_probability : float, optional
        Probability of mutation (default: 0.1)
    save_best_solutions : bool, optional
        Whether to track best solutions over generations (default: True)
    stop_criteria : str, optional
        Stopping criteria (default: 'saturate_200')
    niching_enabled : bool, optional
        Whether to enable fitness sharing/niching (default: True)
    niching_use_optimal_pairing : bool, optional
        Whether to use optimal pairing for grouped parameters (default: True)
    niching_params_per_group : int, optional
        Number of parameters per group for optimal pairing (default: 2)
    niching_sigma_share : float, optional
        Niche radius for fitness sharing (default: 0.5)
    niching_alpha : float, optional
        Sharing power parameter (default: 0.5)
    niching_optimal_pairing_metric : str, optional
        Distance metric for optimal pairing (default: 'euclidean')
    **kwargs : Any
        Additional parameters passed to AdvancedGA

    Returns
    -------
    dict[str, Any]
        Complete GA configuration dictionary ready for AdvancedGA

    Examples
    --------
    >>> # Minimal configuration (uses all defaults)
    >>> config = create_ga_config()
    >>> # Custom configuration
    >>> config = create_ga_config(
    ...     num_generations=3000,
    ...     sol_per_pop=300,
    ...     mutation_probability=0.15,
    ...     niching_sigma_share=1.0,
    ... )
    >>> # Use PyGAD built-in mutation
    >>> config = create_ga_config(mutation_type='adaptive')
    >>> # Use custom mutation function
    >>> from lwi_microbolometer_design.ga import diversity_preserving_mutation
    >>> config = create_ga_config(mutation_type=diversity_preserving_mutation)
    """
    # Use default mutation if not specified
    if mutation_type is None:
        mutation_type = diversity_preserving_mutation

    # Build niching config
    niching_config = NichingConfig(
        enabled=niching_enabled,
        use_optimal_pairing=niching_use_optimal_pairing,
        params_per_group=niching_params_per_group,
        sigma_share=niching_sigma_share,
        alpha=niching_alpha,
        optimal_pairing_metric=niching_optimal_pairing_metric,
    )

    config = {
        'num_generations': num_generations,
        'num_parents_mating': num_parents_mating,
        'sol_per_pop': sol_per_pop,
        'parent_selection_type': parent_selection_type,
        'K_tournament': k_tournament,
        'keep_elitism': keep_elitism,
        'crossover_type': crossover_type,
        'crossover_probability': crossover_probability,
        'mutation_type': mutation_type,
        'mutation_probability': mutation_probability,
        'save_best_solutions': save_best_solutions,
        'stop_criteria': stop_criteria,
        'niching_config': niching_config,
    }

    # Add any additional kwargs
    config.update(kwargs)

    return config


def load_ga_configuration_from_csv(
    file_path: Path,
    sort_by: str = 'best_fitness',
    ascending: bool = False,
    row_index: int = 0,
) -> dict[str, Any] | None:
    """
    Load GA configuration from CSV and return complete configuration dictionary.

    All CSV columns are optional. Missing parameters use defaults with warnings.

    Supported columns: num_generations, num_parents_mating, sol_per_pop,
    parent_selection_type, k_tournament, keep_elitism,
    crossover_type, crossover_probability,
    mutation_type, mutation_probability,
    save_best_solutions, stop_criteria,
    niching_enabled, niching_use_optimal_pairing, niching_params_per_group,
    niching_sigma_share, niching_alpha, niching_optimal_pairing_metric.

    Note: 'best_fitness' is informational only and not used in configuration.

    Parameters
    ----------
    file_path : Path
        Path to CSV file with GA configuration parameters
    sort_by : str, optional
        Column to sort by (default: 'best_fitness')
    ascending : bool, optional
        Sort order (default: False)
    row_index : int, optional
        Row index after sorting (default: 0)

    Returns
    -------
    dict[str, Any] | None
        Complete GA configuration ready for AdvancedGA, or None if file not found
    """
    if not file_path.exists():
        logger.warning(f'Configuration file not found: {file_path}')
        return None

    df = pd.read_csv(file_path)

    # Sort if sort_by column exists
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Get the specified row
    if row_index >= len(df):
        logger.warning(f'Row index {row_index} out of range. Using first row.')
        row_index = 0

    raw_config = df.iloc[row_index]
    default_config = create_ga_config()

    # Extract configuration from CSV
    config_kwargs, missing_params = _extract_config_from_csv(raw_config, default_config)

    # Warn about missing parameters
    _log_missing_params(missing_params, default_config)

    # Log loaded configuration
    _log_loaded_config(file_path, row_index, raw_config, config_kwargs)

    # Create config using the helper function
    return create_ga_config(**config_kwargs)


def _extract_config_from_csv(
    raw_config: pd.Series, default_config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Extract configuration parameters from CSV row.

    Returns
    -------
    config_kwargs : dict
        Extracted configuration parameters.
    missing_params : list
        List of parameter names that were missing from CSV.
    """
    missing_params: list[str] = []

    # Helper function to extract parameter from CSV or use default
    def get_param(param_name: str, param_type: type, default_value: Any) -> Any:
        """Extract parameter from CSV or return default."""
        if param_name in raw_config.index:
            value = raw_config[param_name]
            # Handle boolean strings from CSV (pd.read_csv doesn't auto-convert them)
            # Without this, CSV "False" → Python True (wrong!) because bool("False") = True
            if param_type is bool and isinstance(value, str):
                lower_val = value.lower()
                if lower_val in ['true', '1', 'yes']:
                    return True
                if lower_val in ['false', '0', 'no']:
                    return False
                # Unknown string, use default type conversion
                return param_type(value)
            return param_type(value)
        missing_params.append(param_name)
        return default_value

    # Extract values from CSV or use defaults
    config_kwargs: dict[str, Any] = {}

    # Standard GA parameters
    config_kwargs['num_generations'] = get_param(
        'num_generations', int, default_config['num_generations']
    )
    config_kwargs['num_parents_mating'] = get_param(
        'num_parents_mating', int, default_config['num_parents_mating']
    )
    config_kwargs['sol_per_pop'] = get_param('sol_per_pop', int, default_config['sol_per_pop'])
    config_kwargs['parent_selection_type'] = get_param(
        'parent_selection_type', str, default_config['parent_selection_type']
    )
    config_kwargs['K_tournament'] = get_param('K_tournament', int, default_config['K_tournament'])
    config_kwargs['keep_elitism'] = get_param('keep_elitism', int, default_config['keep_elitism'])
    config_kwargs['crossover_type'] = get_param(
        'crossover_type', str, default_config['crossover_type']
    )
    config_kwargs['crossover_probability'] = get_param(
        'crossover_probability', float, default_config['crossover_probability']
    )
    # Note: mutation_type from CSV would be a string, but we default to function
    # This is handled in create_ga_config if needed
    config_kwargs['mutation_probability'] = get_param(
        'mutation_probability', float, default_config['mutation_probability']
    )
    config_kwargs['save_best_solutions'] = get_param(
        'save_best_solutions', bool, default_config['save_best_solutions']
    )
    config_kwargs['stop_criteria'] = get_param(
        'stop_criteria', str, default_config['stop_criteria']
    )

    # Niching parameters
    config_kwargs['niching_enabled'] = get_param(
        'niching_enabled', bool, default_config['niching_config'].enabled
    )
    config_kwargs['niching_use_optimal_pairing'] = get_param(
        'niching_use_optimal_pairing',
        bool,
        default_config['niching_config'].use_optimal_pairing,
    )
    config_kwargs['niching_params_per_group'] = get_param(
        'niching_params_per_group',
        int,
        default_config['niching_config'].params_per_group,
    )
    config_kwargs['niching_sigma_share'] = get_param(
        'niching_sigma_share', float, default_config['niching_config'].sigma_share
    )
    config_kwargs['niching_alpha'] = get_param(
        'niching_alpha', float, default_config['niching_config'].alpha
    )
    config_kwargs['niching_optimal_pairing_metric'] = get_param(
        'niching_optimal_pairing_metric',
        str,
        default_config['niching_config'].optimal_pairing_metric,
    )

    return config_kwargs, missing_params


def _log_missing_params(missing_params: list[str], default_config: dict[str, Any]) -> None:
    """Log warnings for missing configuration parameters."""
    if not missing_params:
        return

    logger.warning('=' * 60)
    logger.warning('MISSING CONFIGURATION PARAMETERS - USING DEFAULTS')
    logger.warning('=' * 60)
    for param in missing_params:
        # Get default value (niching params are nested)
        if param.startswith('niching_'):
            attr_name = param.replace('niching_', '')  # Remove prefix
            default_value = getattr(default_config['niching_config'], attr_name)
        else:
            default_value = default_config[param]
        logger.warning(f"  ⚠️  '{param}' → default: {default_value}")
    logger.warning('=' * 60)


def _log_loaded_config(
    file_path: Path, row_index: int, raw_config: pd.Series, config_kwargs: dict[str, Any]
) -> None:
    """Log loaded configuration details."""
    logger.info(f'Loaded GA configuration from: {file_path}')
    logger.info(f'  Row index: {row_index}')
    if 'best_fitness' in raw_config:
        logger.info(f'  Best fitness: {raw_config["best_fitness"]:.4f}')

    logger.info('\n  Loaded parameters:')
    logger.info(f'    num_generations: {config_kwargs["num_generations"]}')
    logger.info(f'    num_parents_mating: {config_kwargs["num_parents_mating"]}')
    logger.info(f'    sol_per_pop: {config_kwargs["sol_per_pop"]}')
    logger.info(f'    parent_selection_type: {config_kwargs["parent_selection_type"]}')
    logger.info(f'    k_tournament: {config_kwargs["K_tournament"]}')
    logger.info(f'    keep_elitism: {config_kwargs["keep_elitism"]}')
    logger.info(f'    crossover_type: {config_kwargs["crossover_type"]}')
    logger.info(f'    crossover_probability: {config_kwargs["crossover_probability"]}')
    logger.info(f'    mutation_probability: {config_kwargs["mutation_probability"]}')
    logger.info(f'    save_best_solutions: {config_kwargs["save_best_solutions"]}')
    logger.info(f'    stop_criteria: {config_kwargs["stop_criteria"]}')
    logger.info(f'    niching_enabled: {config_kwargs["niching_enabled"]}')
    logger.info(f'    niching_sigma_share: {config_kwargs["niching_sigma_share"]}')
    logger.info(f'    niching_alpha: {config_kwargs["niching_alpha"]}')
    logger.info(
        f'    niching_optimal_pairing_metric: {config_kwargs["niching_optimal_pairing_metric"]}'
    )
