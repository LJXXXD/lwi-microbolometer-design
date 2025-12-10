"""
Mutation operators for genetic algorithm optimization.

This module provides custom mutation strategies that balance exploration and
exploitation while preserving population diversity. The mutations are designed
for multimodal optimization problems where maintaining diverse solutions is crucial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygad

logger = logging.getLogger(__name__)

# Implementation constants (not user-configurable)
TUPLE_LENGTH_FOR_RANGE = 2


@dataclass
class MutationConfig:
    """
    Configuration for diversity-preserving mutation operator.

    **Default values work well for most cases.** You typically only need to customize
    a few key parameters (like `stagnation_window_size` or `low_diversity_threshold`)
    or use one of the preset configurations:
    - `MutationConfig.balanced()` - Default, good for most problems
    - `MutationConfig.conservative()` - For fine-tuning and exploitation
    - `MutationConfig.aggressive()` - For exploration and multimodal problems

    All parameters are tunable to allow flexible mutation behavior customization
    when needed, but defaults provide sensible starting points.

    Parameters
    ----------
    stagnation_window_size : int
        Number of generations to look back when detecting stagnation.
        Default: 50
    stagnation_threshold : float
        Minimum fitness improvement required to avoid stagnation detection.
        Default: 1e-8
    min_fitness_history_for_stagnation : int
        Minimum number of fitness history entries required before stagnation
        detection can be applied.
        Default: 10
    low_diversity_threshold : float
        Diversity score below which diversity is considered "low" and triggers
        increased exploration.
        Default: 0.08
    early_step_fraction : float
        Step size fraction of gene range early in the run (higher exploration).
        Default: 0.15
    late_step_fraction : float
        Step size fraction of gene range late in the run (lower exploration).
        Default: 0.05
    stagnation_step_multiplier : float
        Multiplier for step size when stagnating.
        Default: 1.75
    low_diversity_step_multiplier : float
        Multiplier for step size when diversity is low.
        Default: 1.5
    early_dynamic_scale : float
        Additional scale factor for mutation probability early in the run.
        Default: 0.75
    stagnation_scale_multiplier : float
        Multiplier for mutation probability scale when stagnating.
        Default: 1.5
    low_diversity_scale_multiplier : float
        Multiplier for mutation probability scale when diversity is low.
        Default: 1.25
    min_mutation_probability : float
        Minimum allowed mutation probability (clipping lower bound).
        Default: 0.01
    max_mutation_probability : float
        Maximum allowed mutation probability (clipping upper bound).
        Default: 0.6
    restart_rate : float
        Base probability of completely re-sampling a gene from its space.
        Default: 0.05
    heavy_tail_rate : float
        Base probability of using heavy-tailed (Cauchy) mutation vs Gaussian.
        Default: 0.5
    directional_rate : float
        Base probability of applying directional push away from population mean.
        Default: 0.35
    restart_rate_multiplier : float
        Multiplier for restart_rate when stagnating or diversity is low.
        Default: 3.0
    restart_rate_max : float
        Maximum allowed restart_rate when boosted.
        Default: 0.25
    heavy_tail_rate_boosted : float
        Heavy tail rate when stagnating or diversity is low.
        Default: 0.7
    directional_rate_boosted : float
        Directional rate when stagnating or diversity is low.
        Default: 0.5
    directional_push_strength : float
        Strength of directional push away from population mean (as fraction of step_sigma).
        Default: 0.25
    coin_flip_probability : float
        Probability for random direction selection when current equals population mean.
        Default: 0.5
    discrete_index_scale_base : float
        Base scale for discrete value index steps.
        Default: 0.5
    discrete_index_scale_multiplier : float
        Multiplier for discrete value index steps based on progress.
        Default: 1.5
    discrete_max_jump_fraction : float
        Maximum jump as fraction of discrete value list length.
        Default: 0.25
    """

    # Stagnation detection parameters
    stagnation_window_size: int = 50
    stagnation_threshold: float = 1e-8
    min_fitness_history_for_stagnation: int = 10

    # Diversity threshold
    low_diversity_threshold: float = 0.08

    # Step size parameters
    early_step_fraction: float = 0.15
    late_step_fraction: float = 0.05
    stagnation_step_multiplier: float = 1.75
    low_diversity_step_multiplier: float = 1.5

    # Dynamic mutation probability scaling
    early_dynamic_scale: float = 0.75
    stagnation_scale_multiplier: float = 1.5
    low_diversity_scale_multiplier: float = 1.25
    min_mutation_probability: float = 0.01
    max_mutation_probability: float = 0.6

    # Operation mixture probabilities
    restart_rate: float = 0.05
    heavy_tail_rate: float = 0.5
    directional_rate: float = 0.35

    # Boosted rates when stagnating or low diversity
    restart_rate_multiplier: float = 3.0
    restart_rate_max: float = 0.25
    heavy_tail_rate_boosted: float = 0.7
    directional_rate_boosted: float = 0.5

    # Directional push parameters
    directional_push_strength: float = 0.25
    coin_flip_probability: float = 0.5

    # Discrete value mutation parameters
    discrete_index_scale_base: float = 0.5
    discrete_index_scale_multiplier: float = 1.5
    discrete_max_jump_fraction: float = 0.25

    @classmethod
    def conservative(cls) -> MutationConfig:
        """
        Create a conservative mutation configuration for fine-tuning and exploitation.

        Features:
        - Smaller step sizes
        - Lower mutation rates
        - Less aggressive exploration
        - Better for convergence to single optimum

        Returns
        -------
        MutationConfig
            Conservative configuration instance
        """
        return cls(
            early_step_fraction=0.08,
            late_step_fraction=0.02,
            stagnation_step_multiplier=1.3,
            low_diversity_step_multiplier=1.2,
            early_dynamic_scale=0.5,
            stagnation_scale_multiplier=1.2,
            low_diversity_scale_multiplier=1.1,
            restart_rate=0.02,
            heavy_tail_rate=0.3,
            directional_rate=0.2,
        )

    @classmethod
    def aggressive(cls) -> MutationConfig:
        """
        Create an aggressive mutation configuration for exploration and multimodal optimization.

        Features:
        - Larger step sizes
        - Higher mutation rates
        - More aggressive exploration
        - Better for finding multiple optima

        Returns
        -------
        MutationConfig
            Aggressive configuration instance
        """
        return cls(
            early_step_fraction=0.25,
            late_step_fraction=0.08,
            stagnation_step_multiplier=2.2,
            low_diversity_step_multiplier=2.0,
            early_dynamic_scale=1.0,
            stagnation_scale_multiplier=2.0,
            low_diversity_scale_multiplier=1.5,
            restart_rate=0.1,
            heavy_tail_rate=0.7,
            directional_rate=0.5,
            restart_rate_max=0.35,
            heavy_tail_rate_boosted=0.85,
            directional_rate_boosted=0.65,
        )

    @classmethod
    def balanced(cls) -> MutationConfig:
        """
        Create a balanced mutation configuration (same as default).

        This is the recommended starting point for most problems. Provides a good
        balance between exploration and exploitation.

        Returns
        -------
        MutationConfig
            Balanced configuration instance (default values)
        """
        return cls()


# Default configuration instance (balanced)
DEFAULT_MUTATION_CONFIG = MutationConfig.balanced()


def _extract_mutation_probability(ga_instance: pygad.GA) -> float:
    """Extract and normalize mutation probability from GA instance."""
    mutation_probability_param = getattr(ga_instance, 'mutation_probability', 0.1)
    try:
        if (
            isinstance(mutation_probability_param, (list, tuple, np.ndarray))
            and len(mutation_probability_param) > 0
        ):
            # Convert to array for np.mean (np.mean expects array, not list)
            prob_array = np.array([float(x) for x in mutation_probability_param])
            base_gene_mutation_probability = float(np.mean(prob_array))
        else:
            base_gene_mutation_probability = float(mutation_probability_param)
    except (TypeError, ValueError, AttributeError):
        logger.warning(
            f'Invalid mutation_probability: {mutation_probability_param}. Using default 0.1'
        )
        base_gene_mutation_probability = 0.1
    # Use Python min/max for scalar clipping (np.clip is for arrays)
    return float(max(0.0, min(1.0, base_gene_mutation_probability)))


def _calculate_progress(ga_instance: pygad.GA) -> float:
    """Calculate progress as fraction of generations completed."""
    total_gens = getattr(ga_instance, 'num_generations', 1) or 1
    gens_done = getattr(ga_instance, 'generations_completed', 0) or 0
    return min(1.0, max(0.0, gens_done / total_gens))


def _calculate_diversity_score(
    ga_instance: pygad.GA, num_genes: int, space_infos: list[dict[str, Any]]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate diversity score from population normalized by gene ranges.

    Returns
    -------
    diversity_score : float
        Normalized diversity score (mean of normalized standard deviations).
    population_mean : np.ndarray
        Mean of population across genes.
    ranges : np.ndarray
        Range for each gene (for normalization).
    """
    population = np.array(getattr(ga_instance, 'population', []), dtype=np.float64)
    population_mean = (
        np.mean(population, axis=0) if population.size else np.zeros(num_genes, dtype=np.float64)
    )

    # Calculate ranges for normalization
    ranges_list = []
    for info in space_infos:
        if info['is_discrete'] and info['values'] is not None:
            vals = info['values']
            try:
                num_vals = len(vals)
                rng = (
                    float(np.max(vals.astype(float)) - np.min(vals.astype(float)))
                    if num_vals > 0
                    else 1.0
                )
            except (TypeError, ValueError, AttributeError):
                rng = 1.0
        else:
            low = info['low'] if info['low'] is not None else 0.0
            high = info['high'] if info['high'] is not None else 1.0
            rng = float(max(high - low, 1e-12))
        ranges_list.append(rng)
    ranges = np.array(ranges_list, dtype=np.float64)

    pop_std = (
        np.std(population, axis=0) if population.size else np.zeros(num_genes, dtype=np.float64)
    )
    # Safe division: avoid division by zero manually
    # (mypy has issues with np.divide where parameter type)
    norm_std = np.zeros_like(pop_std)
    mask = ranges > 0
    norm_std[mask] = pop_std[mask] / ranges[mask]
    diversity_score = float(np.mean(norm_std))

    return diversity_score, population_mean, ranges


def _get_mutation_config(ga_instance: pygad.GA) -> MutationConfig:
    """Extract mutation configuration from GA instance with backward compatibility."""
    mutation_config = getattr(ga_instance, 'mutation_config', None)
    if mutation_config is None or not isinstance(mutation_config, MutationConfig):
        # Backward compatibility: check for individual attributes
        mutation_config = MutationConfig(
            stagnation_window_size=getattr(
                ga_instance,
                'mutation_stagnation_window_size',
                DEFAULT_MUTATION_CONFIG.stagnation_window_size,
            ),
            stagnation_threshold=getattr(
                ga_instance,
                'mutation_stagnation_threshold',
                DEFAULT_MUTATION_CONFIG.stagnation_threshold,
            ),
            low_diversity_threshold=getattr(
                ga_instance,
                'mutation_low_diversity_threshold',
                DEFAULT_MUTATION_CONFIG.low_diversity_threshold,
            ),
        )
    return mutation_config


def _detect_stagnation(ga_instance: pygad.GA, mutation_config: MutationConfig) -> bool:
    """Detect if the GA is stagnating based on fitness history."""
    best_fitness_history = getattr(ga_instance, 'best_solutions_fitness', None)
    if not (
        isinstance(best_fitness_history, list)
        and len(best_fitness_history) >= mutation_config.min_fitness_history_for_stagnation
    ):
        return False

    window = min(mutation_config.stagnation_window_size, len(best_fitness_history))
    improvement = float(best_fitness_history[-1]) - float(best_fitness_history[-window])
    return improvement < mutation_config.stagnation_threshold


def _calculate_adaptive_parameters(
    progress: float,
    stagnating: bool,
    diversity_score: float,
    mutation_config: MutationConfig,
    base_gene_mutation_probability: float,
) -> tuple[float, float, float, float, float]:
    """Calculate adaptive mutation parameters.

    Returns
    -------
    effective_gene_mutation_probability : float
        Effective mutation probability after scaling.
    base_step_frac : float
        Base step fraction of range.
    restart_rate : float
        Probability of restarting a gene.
    heavy_tail_rate : float
        Probability of using heavy-tailed mutation.
    directional_rate : float
        Probability of directional mutation.
    """
    # Base step fraction of range shrinks with progress
    base_step_frac = (
        mutation_config.early_step_fraction * (1.0 - progress)
        + mutation_config.late_step_fraction * progress
    )
    # Increase exploration if stagnating or diversity is low
    if stagnating:
        base_step_frac *= mutation_config.stagnation_step_multiplier
    if diversity_score < mutation_config.low_diversity_threshold:
        base_step_frac *= mutation_config.low_diversity_step_multiplier

    # Dynamic mutation probability scaling
    dynamic_scale = 1.0 + mutation_config.early_dynamic_scale * (1.0 - progress)
    if stagnating:
        dynamic_scale *= mutation_config.stagnation_scale_multiplier
    if diversity_score < mutation_config.low_diversity_threshold:
        dynamic_scale *= mutation_config.low_diversity_scale_multiplier
    # Use Python min/max for scalar clipping (np.clip is for arrays)
    prob_value = base_gene_mutation_probability * dynamic_scale
    effective_gene_mutation_probability = float(
        max(
            mutation_config.min_mutation_probability,
            min(mutation_config.max_mutation_probability, prob_value),
        )
    )

    # Operation mixture probabilities
    restart_rate = mutation_config.restart_rate
    heavy_tail_rate = mutation_config.heavy_tail_rate
    directional_rate = mutation_config.directional_rate
    if stagnating or diversity_score < mutation_config.low_diversity_threshold:
        restart_rate = min(
            mutation_config.restart_rate_max,
            restart_rate * mutation_config.restart_rate_multiplier,
        )
        heavy_tail_rate = mutation_config.heavy_tail_rate_boosted
        directional_rate = mutation_config.directional_rate_boosted

    return (
        effective_gene_mutation_probability,
        base_step_frac,
        restart_rate,
        heavy_tail_rate,
        directional_rate,
    )


def _mutate_discrete_gene(
    mutated: np.ndarray,
    r: int,
    g: int,
    info: dict[str, Any],
    mutation_config: MutationConfig,
    progress: float,
    restart_rate: float,
    random_generator: np.random.Generator,
) -> None:
    """Mutate a discrete/categorical gene."""
    vals = info['values']
    # With some rate, restart by drawing randomly from allowed values
    if random_generator.random() < restart_rate:
        mutated[r, g] = np.random.choice(vals)
        return

    # Move to a neighboring allowed value with heavy-tailed step in index space
    try:
        # Find nearest index of current value; fallback to random if not found
        current_val = mutated[r, g]
        try:
            current_idx = int(np.where(vals == current_val)[0][0])
        except (IndexError, TypeError):
            # Value not found in vals or comparison failed, use random index
            current_idx = int(random_generator.integers(0, len(vals)))

        # Heavy-tailed step in index domain
        idx_scale = max(
            1,
            round(
                mutation_config.discrete_index_scale_base
                + (mutation_config.discrete_index_scale_multiplier * (1.0 - progress))
            ),
        )
        # Cauchy-like integer step
        raw_step = np.random.standard_cauchy() * idx_scale
        # Cap extreme jumps to list length (use Python min/max for scalar clipping)
        max_jump = max(1, int(len(vals) * mutation_config.discrete_max_jump_fraction))
        rounded_step = round(raw_step)
        idx_step = int(max(-max_jump, min(max_jump, rounded_step)))
        if idx_step == 0:
            idx_step = (
                1 if random_generator.random() < mutation_config.coin_flip_probability else -1
            )
        new_idx = int(max(0, min(len(vals) - 1, current_idx + idx_step)))
        mutated[r, g] = vals[new_idx]
    except (IndexError, TypeError, ValueError):
        # Fallback to random choice if indexing or calculations fail
        mutated[r, g] = np.random.choice(vals)


def _mutate_continuous_gene(
    mutated: np.ndarray,
    r: int,
    g: int,
    info: dict[str, Any],
    mutation_config: MutationConfig,
    base_step_frac: float,
    gene_range: float,
    restart_rate: float,
    heavy_tail_rate: float,
    directional_rate: float,
    population_mean: np.ndarray,
    random_generator: np.random.Generator,
) -> None:
    """Mutate a continuous/ranged gene."""
    current = float(mutated[r, g])
    step_sigma = base_step_frac * gene_range

    # With some rate, completely re-sample from space
    if random_generator.random() < restart_rate:
        new_val = _random_from_space(info)
        mutated[r, g] = _clip_and_quantize(new_val, info)
        return

    # Choose mutation kernel: heavy-tailed Cauchy vs Gaussian
    use_heavy_tail = random_generator.random() < heavy_tail_rate
    if use_heavy_tail:
        step = float(np.random.standard_cauchy()) * step_sigma
    else:
        step = float(np.random.normal(loc=0.0, scale=step_sigma))

    # Optional directional push away from population mean to fight clustering
    if random_generator.random() < directional_rate and population_mean.size:
        # Use comparison instead of np.sign for scalar (np.sign is for arrays)
        diff = current - float(population_mean[g])
        if diff > 0:
            direction = 1.0
        elif diff < 0:
            direction = -1.0
        else:
            direction = (
                1.0 if random_generator.random() < mutation_config.coin_flip_probability else -1.0
            )
        step += direction * mutation_config.directional_push_strength * step_sigma

    new_val = current + step
    # Project back to feasible space
    mutated[r, g] = _clip_and_quantize(new_val, info)


def diversity_preserving_mutation(offspring: np.ndarray, ga_instance: pygad.GA) -> np.ndarray:
    """Balance exploration and exploitation with diversity control.

    Strategy
    --------
    - Adaptive step size: larger early, smaller later in the run.
    - Stagnation/low-diversity boost: escalate exploration via heavy-tailed noise
      and random re-sampling from gene space when progress stalls or diversity is low.
    - Directed diversification: small push away from population mean to reduce
      clustering without destroying elites.
    - Boundary-respecting: every change is projected back into `gene_space` using
      ranges, steps, or discrete values.

    Parameters
    ----------
    offspring : np.ndarray
        Offspring array of shape (num_offspring, num_genes).
    ga_instance : pygad.GA
        Current GA instance to access configuration, progress, and gene_space.
        Mutation behavior can be customized by setting a `mutation_config` attribute
        on the GA instance to a `MutationConfig` object. Alternatively, for backward
        compatibility, individual parameters can be set as attributes:
        - `mutation_stagnation_window_size` (int)
        - `mutation_stagnation_threshold` (float)
        - `mutation_low_diversity_threshold` (float)

    Returns
    -------
    np.ndarray
        Mutated offspring with the same shape as the input.

    Examples
    --------
    >>> # Use default mutation parameters (recommended for most cases)
    >>> ga = AdvancedGA(mutation_type=diversity_preserving_mutation, ...)
    >>>
    >>> # Use preset configurations (simplest customization)
    >>> from lwi_microbolometer_design.ga import MutationConfig
    >>> # For aggressive exploration (multimodal problems)
    >>> ga = AdvancedGA(
    ...     mutation_type=diversity_preserving_mutation,
    ...     mutation_config=MutationConfig.aggressive(),
    ...     ...
    ... )
    >>> # For conservative fine-tuning (single optimum)
    >>> ga = AdvancedGA(
    ...     mutation_type=diversity_preserving_mutation,
    ...     mutation_config=MutationConfig.conservative(),
    ...     ...
    ... )
    >>>
    >>> # Customize only a few key parameters (when presets aren't enough)
    >>> custom_config = MutationConfig(
    ...     stagnation_window_size=100,  # Look back further
    ...     low_diversity_threshold=0.1,  # More sensitive to low diversity
    ...     # All other parameters use defaults
    ... )
    >>> ga = AdvancedGA(
    ...     mutation_type=diversity_preserving_mutation,
    ...     mutation_config=custom_config,
    ...     ...
    ... )
    >>>
    >>> # Backward compatibility: set individual attributes (deprecated)
    >>> ga = AdvancedGA(
    ...     mutation_type=diversity_preserving_mutation,
    ...     mutation_stagnation_window_size=100,
    ...     ...
    ... )
    """
    if offspring is None or len(offspring) == 0:
        return offspring

    mutated = offspring.copy()
    num_offspring, num_genes = mutated.shape

    # Extract configuration and calculate signals
    base_gene_mutation_probability = _extract_mutation_probability(ga_instance)
    progress = _calculate_progress(ga_instance)

    # Build per-gene space info
    gene_space_raw = getattr(ga_instance, 'gene_space', None)
    gene_space_list = _ensure_gene_space_list(gene_space_raw, num_genes)
    space_infos = [_extract_gene_space_components(gene_space_list[g]) for g in range(num_genes)]

    # Calculate diversity score and get ranges
    diversity_score, population_mean, ranges = _calculate_diversity_score(
        ga_instance, num_genes, space_infos
    )

    # Get mutation configuration and detect stagnation
    mutation_config = _get_mutation_config(ga_instance)
    stagnating = _detect_stagnation(ga_instance, mutation_config)

    # Calculate adaptive parameters
    (
        effective_gene_mutation_probability,
        base_step_frac,
        restart_rate,
        heavy_tail_rate,
        directional_rate,
    ) = _calculate_adaptive_parameters(
        progress,
        stagnating,
        diversity_score,
        mutation_config,
        base_gene_mutation_probability,
    )

    # Apply mutations
    random_generator = np.random.default_rng()
    for r in range(num_offspring):
        for g in range(num_genes):
            if random_generator.random() > effective_gene_mutation_probability:
                continue

            info = space_infos[g]
            gene_range = ranges[g]

            # Dispatch to discrete or continuous mutation handler
            if info['is_discrete'] and info['values'] is not None:
                _mutate_discrete_gene(
                    mutated, r, g, info, mutation_config, progress, restart_rate, random_generator
                )
            else:
                _mutate_continuous_gene(
                    mutated,
                    r,
                    g,
                    info,
                    mutation_config,
                    base_step_frac,
                    gene_range,
                    restart_rate,
                    heavy_tail_rate,
                    directional_rate,
                    population_mean,
                    random_generator,
                )

    return mutated


def _extract_gene_space_components(space_entry: Any) -> dict[str, Any]:
    """Normalize a single gene_space entry to a common representation.

    Supports PyGAD formats:
    - Dict with keys 'low'/'high' and optional 'step'
    - Dict with key 'values' (discrete/categorical)
    - List/tuple/np.ndarray of allowed values (discrete)
    - Tuple (low, high)
    - Scalar value (fixed gene)

    Returns a dictionary with fields:
    - is_discrete: bool
    - values: Optional[np.ndarray]
    - low: Optional[float]
    - high: Optional[float]
    - step: Optional[float]
    - is_integer: bool (hint for casting)
    """
    result: dict[str, Any] = {
        'is_discrete': False,
        'values': None,
        'low': None,
        'high': None,
        'step': None,
        'is_integer': False,
    }

    if isinstance(space_entry, dict):
        if 'values' in space_entry and space_entry['values'] is not None:
            # dtype=object needed for mixed-type arrays (e.g., strings, numbers)
            # Convert to list first, then array to help mypy understand the type
            values_list = list(space_entry['values'])
            vals: np.ndarray = np.array(values_list, dtype=object)
            result['is_discrete'] = True
            result['values'] = vals
            # Infer integer hint if all values are ints
            result['is_integer'] = all(isinstance(v, (int, np.integer)) for v in vals)
            return result

        # Range-based
        low = space_entry.get('low', None)
        high = space_entry.get('high', None)
        step = space_entry.get('step', None)
        result['low'] = float(low) if low is not None else None
        result['high'] = float(high) if high is not None else None
        result['step'] = float(step) if step is not None else None

        # Infer integer if step is an integer step and bounds look integer
        if step is not None:
            result['is_integer'] = isinstance(step, (int, np.integer))
        if (
            low is not None
            and high is not None
            and all(isinstance(x, (int, np.integer)) for x in (low, high))
            and (step is None or isinstance(step, (int, np.integer)))
        ):
            result['is_integer'] = True
        return result

    if isinstance(space_entry, (list, tuple, np.ndarray)):
        # (low, high) tuple vs discrete list
        if len(space_entry) == TUPLE_LENGTH_FOR_RANGE and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in space_entry
        ):
            low, high = space_entry
            result['low'] = float(low)
            result['high'] = float(high)
            # Integer hint if both integer and range is small integer space
            result['is_integer'] = all(isinstance(x, (int, np.integer)) for x in (low, high))
            return result

        # dtype=object needed for mixed-type arrays
        # Convert to list first, then array to help mypy understand the type
        values_list = list(space_entry)
        vals_array: np.ndarray = np.array(values_list, dtype=object)
        result['is_discrete'] = True
        result['values'] = vals_array
        result['is_integer'] = all(isinstance(v, (int, np.integer)) for v in vals_array)
        return result

    # Scalar fixed value
    if isinstance(space_entry, (int, float, np.integer, np.floating)):
        # dtype=object needed for mixed-type arrays
        # Convert to list first, then array to help mypy understand the type
        vals_scalar: np.ndarray = np.array([space_entry], dtype=object)
        result['is_discrete'] = True
        result['values'] = vals_scalar
        result['is_integer'] = isinstance(space_entry, (int, np.integer))
        return result

    # Fallback: treat as unbounded float
    result['low'] = None
    result['high'] = None
    result['step'] = None
    result['is_integer'] = False
    return result


def _random_from_space(space_info: dict[str, Any]) -> int | float | Any:
    """Sample a random value from the given normalized space description."""
    if space_info['is_discrete'] and space_info['values'] is not None:
        return np.random.choice(space_info['values'])

    low = space_info['low']
    high = space_info['high']
    step = space_info['step']
    if low is None or high is None:
        # Unbounded fallback: standard normal
        val = float(np.random.normal(0.0, 1.0))
    elif step is None:
        # Use np.random.Generator for scalar uniform sampling (avoids mypy issues)
        rng = np.random.default_rng()
        val = float(rng.uniform(float(low), float(high)))
    else:
        # Sample from discretized grid uniformly
        num_steps = max(1, int(np.floor((high - low) / step)))
        idx = np.random.randint(0, num_steps + 1)
        val = low + idx * step

    if space_info['is_integer']:
        return round(val)
    return val


def _clip_and_quantize(value: int | float | Any, space_info: dict[str, Any]) -> int | float | Any:
    """Project a value back to the gene space, applying bounds and step/values.

    - For discrete/categorical: snap to the nearest allowed value.
    - For ranged genes: clip to [low, high] and snap to step if provided.
    - Preserve integer typing when hinted.
    """
    if space_info['is_discrete'] and space_info['values'] is not None:
        vals = space_info['values']
        # If value is already in the set, keep it.
        # Otherwise snap to nearest by numeric distance when possible.
        try:
            # Exact match fast path
            if any(value == v for v in vals):
                return value
        except (TypeError, ValueError):
            # Comparison failed (incompatible types), continue to numeric snapping
            pass

        # Numeric nearest neighbor if comparable, else random choice
        numeric_vals_list = []
        for v in vals:
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                numeric_vals_list.append(float(v))
        if isinstance(value, (int, float, np.integer, np.floating)) and numeric_vals_list:
            numeric_vals_array = np.array(numeric_vals_list, dtype=np.float64)
            value_float = float(value)
            idx = int(np.argmin(np.abs(numeric_vals_array - value_float)))
            nearest = float(numeric_vals_array[idx])
            # Return with original type when possible
            if space_info['is_integer']:
                return round(nearest)
            return float(nearest)
        return np.random.choice(vals)

    # Ranged
    low = space_info['low']
    high = space_info['high']
    step = space_info['step']
    if low is not None and high is not None:
        # Use Python min/max for scalar clipping (np.clip is for arrays)
        value = float(max(float(low), min(float(high), float(value))))
        if step is not None and step > 0:
            # Snap to nearest grid point
            k = round((value - low) / step)
            value = low + k * step
    if space_info['is_integer']:
        return round(value)
    return float(value)


def _ensure_gene_space_list(gene_space: Any, num_genes: int) -> list[Any]:
    """Return a per-gene list of gene_space entries regardless of original format."""
    if isinstance(gene_space, list) and len(gene_space) == num_genes:
        return gene_space
    # If a single dict/list provided, replicate across genes
    return [gene_space for _ in range(num_genes)]
