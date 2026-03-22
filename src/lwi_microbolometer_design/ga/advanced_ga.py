"""
PyGAD GA extended with fitness sharing/niching for population diversity.

Inherits from pygad.GA and adds optional niching to prevent premature convergence.
All PyGAD arguments are fully supported—see https://pygad.readthedocs.io/en/latest/

Example (Standard Euclidean - Single Parameters):
    # Use when each gene represents an independent parameter
    from lwi_microbolometer_design.ga import AdvancedGA, NichingConfig

    config = NichingConfig(
        enabled=True,
        sigma_share=1.0,
        alpha=1.0,
        use_optimal_pairing=False  # REQUIRED: Explicit choice
    )
    ga = AdvancedGA(
        # Required PyGAD args
        num_generations=1000,
        num_parents_mating=100,
        fitness_func=my_fitness_func,
        sol_per_pop=200,
        num_genes=8,
        # AdvancedGA-specific arg
        niching_config=config,
        # Optional PyGAD args (gene_space, mutation_type, crossover_type, etc.)
        gene_space=[{'low': 0, 'high': 10}] * 8,
        mutation_type='adaptive'
    )
    ga.run()

Example (Optimal Pairing - Grouped Parameters):
    # Use when genes are grouped (e.g., [mu1, sigma1, mu2, sigma2, ...])
    # and order of groups shouldn't matter
    config = NichingConfig(
        enabled=True,
        sigma_share=1.0,
        alpha=1.0,
        use_optimal_pairing=True,  # REQUIRED: Explicit choice
        params_per_group=2,  # Each sensor has 2 params (mu, sigma)
        optimal_pairing_metric='euclidean'
    )
    ga = AdvancedGA(
        num_generations=1000,
        num_parents_mating=100,
        fitness_func=my_fitness_func,
        sol_per_pop=200,
        num_genes=8,  # 4 sensors * 2 params each
        niching_config=config,
        gene_space=[{'low': 0, 'high': 10}] * 8,
        mutation_type='adaptive'
    )
    ga.run()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygad
from scipy.linalg import norm

from lwi_microbolometer_design.ga.diversity import compute_population_distance_matrix

# Optional Numba import check (for statistics reporting)
try:
    import numba  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedGA(pygad.GA):
    """PyGAD GA with optional fitness sharing/niching for population diversity.

    Drop-in replacement for pygad.GA that adds optional niching to prevent premature
    convergence. All PyGAD arguments supported—see https://pygad.readthedocs.io/en/latest/
    """

    def __init__(
        self,
        num_generations: int,
        num_parents_mating: int,
        fitness_func: Callable[[pygad.GA, np.ndarray, int], float],
        sol_per_pop: int,
        num_genes: int,
        niching_config: NichingConfig | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize AdvancedGA.

        Parameters
        ----------
        **Required PyGAD args:**

        num_generations : int
            Number of generations to evolve.
        num_parents_mating : int
            Number of parents to select for mating.
        fitness_func : Callable
            Fitness function: f(ga_instance, solution, solution_idx) -> float
        sol_per_pop : int
            Population size (solutions per generation).
        num_genes : int
            Number of genes per chromosome.

        **AdvancedGA-specific arg:**

        niching_config : NichingConfig, optional
            Enables fitness sharing. Default None (standard PyGAD behavior).

        **Additional PyGAD args:**

        **kwargs
            All other pygad.GA arguments (gene_space, mutation_type,
            crossover_type, parent_selection_type, keep_elitism, and many
            more). See PyGAD docs for full list.
        """
        # Niching configuration
        self.niching_config = niching_config

        # Dual fitness tracking (original for elitism, shared for selection)
        self.original_fitness_scores: np.ndarray | None = None
        self.shared_fitness_scores: np.ndarray | None = None
        self.original_fitness_history: list[float] = []
        self.shared_fitness_history: list[float] = []

        # Log niching configuration
        if self.niching_config and self.niching_config.enabled:
            if self.niching_config.use_optimal_pairing:
                # Check if Numba acceleration is available
                numba_status = (
                    "Numba-accelerated" if NUMBA_AVAILABLE else "sequential (Numba not available)"
                )
                logger.info(
                    "Advanced GA initialized with niching enabled "
                    f"(sigma_share={self.niching_config.sigma_share}, "
                    f"alpha={self.niching_config.alpha}, "
                    f"optimal_pairing=True, "
                    f"params_per_group={self.niching_config.params_per_group}, "
                    f"metric={self.niching_config.optimal_pairing_metric}, "
                    f"mode={numba_status})"
                )
                if not NUMBA_AVAILABLE:
                    logger.warning(
                        "Numba not available - optimal pairing will run ~2x slower. "
                        "Install numba for better performance: pip install numba"
                    )
            else:
                logger.info(
                    "Advanced GA initialized with niching enabled "
                    f"(sigma_share={self.niching_config.sigma_share}, "
                    f"alpha={self.niching_config.alpha}, "
                    f"metric={self.niching_config.distance_metric})"
                )
        else:
            logger.info("Advanced GA initialized without niching (standard PyGAD behavior)")

        # Call PyGAD constructor with all required and optional args
        super().__init__(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            **kwargs,
        )

    def _calculate_shared_fitness(self) -> np.ndarray:
        """Calculate fitness sharing: penalize crowded solutions, preserve originals.

        Core niching algorithm:
        1. Compute distance matrix
        2. Calculate niche count (sum of sharing coefficients)
        3. Divide fitness by niche count (crowded solutions get lower fitness)

        Note: Only called when niching is enabled.

        Returns
        -------
        np.ndarray
            Shared fitness scores (original fitness / niche_count)
        """
        if not self.niching_config or not self.niching_config.enabled:
            msg = "Method should only be called when niching is enabled"
            raise RuntimeError(msg)

        # Store original fitness for elitism
        self.original_fitness_scores = self.last_generation_fitness.copy()

        # Pre-compute distance matrix for efficiency
        # Use unified distance matrix computation from ga.diversity module
        if self.population is None or self.sol_per_pop == 0:
            distance_matrix = np.zeros((0, 0))
        else:
            distance_matrix = compute_population_distance_matrix(
                self.population, self.niching_config
            )

        # Apply fitness sharing: penalize each chromosome by its niche count
        shared_fitness = np.zeros_like(self.original_fitness_scores)
        for i in range(self.sol_per_pop):
            # Calculate niche count (sum of sharing coefficients from neighbors)
            niche_count = 1.0  # Start with self
            for j in range(self.sol_per_pop):
                if i != j:
                    sharing_coeff = niche_sharing_coefficient(
                        distance_matrix[i, j],
                        self.niching_config.sigma_share,
                        self.niching_config.alpha,
                    )
                    niche_count += sharing_coeff

            # Divide fitness by niche count (crowded solutions get penalized)
            shared_fitness[i] = self.original_fitness_scores[i] / niche_count

        # Store and track
        self.shared_fitness_scores = shared_fitness
        self.original_fitness_history.append(np.max(self.original_fitness_scores))
        self.shared_fitness_history.append(np.max(self.shared_fitness_scores))

        return shared_fitness

    def run_select_parents(self, call_on_parents: bool = True) -> None:
        """Override parent selection to optionally apply fitness sharing.

        If niching enabled: Apply fitness sharing for diversity
        If niching disabled: Standard PyGAD behavior (no overhead)

        Parameters
        ----------
        call_on_parents : bool
            Whether to call on_parents callback
        """
        # If niching disabled, use standard PyGAD behavior
        if not self.niching_config or not self.niching_config.enabled:
            super().run_select_parents(call_on_parents=call_on_parents)
            return

        # Apply fitness sharing for parent selection
        shared_fitness_scores = self._calculate_shared_fitness()
        self.last_generation_fitness = shared_fitness_scores

        # PyGAD parent selection with shared fitness
        super().run_select_parents(call_on_parents=call_on_parents)

        # Restore original fitness for elitism
        if self.original_fitness_scores is not None:
            self.last_generation_fitness = self.original_fitness_scores

    def get_statistics(self) -> dict[str, object]:
        """Get comprehensive statistics about the GA run and current population.

        This method provides convenient access to GA state information for analysis,
        logging, and visualization. It is optional - users can always extract data
        directly from GA attributes (population, last_generation_fitness, etc.) or
        implement custom analysis logic.

        Returns
        -------
        Dict[str, Any]
            Comprehensive statistics including:
            - Progress: generation count, completion ratio
            - Population: size, diversity metrics
            - Fitness: current generation stats (best/worst/mean/std)
            - Best solution: chromosome, fitness, index
            - GA config: selection type, mutation/crossover probabilities
            - Niching (if enabled): original vs shared fitness, niche parameters

        Notes
        -----
        - This method provides a standardized way to access GA statistics
        - For visualization, consider using `visualize_ga_results()` which accepts
          both this GA instance and custom result dictionaries
        - For custom analysis, you can always access GA attributes directly:
          `ga.population`, `ga.last_generation_fitness`, `ga.best_solutions_fitness`, etc.
        """
        # ===== PROGRESS TRACKING =====
        stats = {
            "generation": self.generations_completed,
            "total_generations": self.num_generations,
            "progress": (
                self.generations_completed / self.num_generations if self.num_generations > 0 else 0
            ),
        }

        # ===== POPULATION CONFIGURATION =====
        stats["population"] = {
            "size": self.sol_per_pop,
            "num_parents_mating": self.num_parents_mating,
            "num_genes": self.num_genes,
        }

        # ===== CURRENT GENERATION FITNESS =====
        if hasattr(self, "last_generation_fitness") and self.last_generation_fitness is not None:
            stats["fitness"] = {
                "best": float(np.max(self.last_generation_fitness)),
                "worst": float(np.min(self.last_generation_fitness)),
                "mean": float(np.mean(self.last_generation_fitness)),
                "std": float(np.std(self.last_generation_fitness)),
            }

        # ===== BEST SOLUTION EVER FOUND =====
        # Tracks the best fitness across ALL generations (not just current)
        if hasattr(self, "best_solutions_fitness") and len(self.best_solutions_fitness) > 0:
            best_fitness_ever = float(np.max(self.best_solutions_fitness))
            best_gen = int(np.argmax(self.best_solutions_fitness))
            stats["best_solution"] = {
                "fitness": best_fitness_ever,
                "generation_found": best_gen,
                "current_generation_best": float(self.best_solutions_fitness[-1]),
            }

        # ===== POPULATION DIVERSITY =====
        # Measures genetic diversity in the population (convergence indicator)
        if hasattr(self, "population") and self.population is not None:
            # Per-gene standard deviation
            population_std = np.std(self.population, axis=0)

            # Calculate gene ranges from current population for normalization
            gene_max = np.max(self.population, axis=0)
            gene_min = np.min(self.population, axis=0)
            gene_ranges = gene_max - gene_min
            # Avoid division by zero - replace near-zero ranges with 1.0
            gene_ranges = np.where(gene_ranges > np.finfo(float).eps, gene_ranges, 1.0)

            # Normalized diversity (scale-independent metric)
            normalized_std = population_std / gene_ranges

            stats["diversity"] = {
                "mean_gene_std": float(np.mean(population_std)),
                "normalized_mean_std": float(np.mean(normalized_std)),
                "min_gene_std": float(np.min(population_std)),
                "max_gene_std": float(np.max(population_std)),
            }

        # ===== GA CONFIGURATION =====
        # Extract operator names (handle custom functions gracefully)
        mutation_type_str = self.mutation_type if hasattr(self, "mutation_type") else None
        if callable(mutation_type_str):
            mutation_type_str = getattr(mutation_type_str, "__name__", "custom_function")

        crossover_type_str = self.crossover_type if hasattr(self, "crossover_type") else None
        if callable(crossover_type_str):
            crossover_type_str = getattr(crossover_type_str, "__name__", "custom_function")

        stats["config"] = {
            "parent_selection": self.parent_selection_type,
            "crossover_type": crossover_type_str,
            "mutation_type": mutation_type_str,
            "keep_elitism": self.keep_elitism if hasattr(self, "keep_elitism") else 0,
        }

        # ===== NICHING STATISTICS =====
        # Only populated when fitness sharing is enabled
        if self.niching_config and self.niching_config.enabled:
            stats["niching"] = {
                "enabled": True,
                "sigma_share": self.niching_config.sigma_share,
                "alpha": self.niching_config.alpha,
                "distance_metric": self.niching_config.distance_metric,
                "use_optimal_pairing": self.niching_config.use_optimal_pairing,
            }

            # Add optimal pairing parameters if enabled
            if self.niching_config.use_optimal_pairing:
                stats["niching"]["params_per_group"] = self.niching_config.params_per_group
                stats["niching"]["optimal_pairing_metric"] = (
                    self.niching_config.optimal_pairing_metric
                )
                stats["niching"]["numba_accelerated"] = NUMBA_AVAILABLE

            # Original fitness (before sharing)
            if self.original_fitness_scores is not None:
                stats["niching"]["original_fitness"] = {
                    "best": float(np.max(self.original_fitness_scores)),
                    "worst": float(np.min(self.original_fitness_scores)),
                    "mean": float(np.mean(self.original_fitness_scores)),
                    "std": float(np.std(self.original_fitness_scores)),
                }

            # Shared fitness (after niching penalty)
            if self.shared_fitness_scores is not None:
                stats["niching"]["shared_fitness"] = {
                    "best": float(np.max(self.shared_fitness_scores)),
                    "worst": float(np.min(self.shared_fitness_scores)),
                    "mean": float(np.mean(self.shared_fitness_scores)),
                    "std": float(np.std(self.shared_fitness_scores)),
                }

                # Fitness sharing impact metric
                if self.original_fitness_scores is not None:
                    sharing_ratio = np.mean(
                        self.shared_fitness_scores
                        / (self.original_fitness_scores + np.finfo(float).eps)
                    )
                    stats["niching"]["avg_sharing_penalty"] = float(1.0 - sharing_ratio)
        else:
            stats["niching"] = {"enabled": False}

        return stats


# ------------------------------------------------------------------------------
# NICHING
# ------------------------------------------------------------------------------
@dataclass
class NichingConfig:
    """Configuration for fitness sharing/niching to maintain diversity.

    Parameters
    ----------
    enabled : bool
        Whether to enable fitness sharing/niching
    use_optimal_pairing : bool
        **REQUIRED**: Whether to use optimal pairing distance for grouped parameters.
        - Set to True when genes are grouped (e.g., [mu1, sigma1, mu2, sigma2, ...])
          and order of groups shouldn't matter
        - Set to False for standard Euclidean distance when each gene is independent
        Must be explicitly specified to prevent accidental misuse.
    sigma_share : float
        Niche radius - solutions within this distance share fitness
        (typically 0.5-2.0)
    alpha : float
        Sharing power parameter - higher values create sharper boundaries
        (typically 1.0)
    distance_metric : str
        Distance metric to use when use_optimal_pairing=False (only 'euclidean' supported)
    params_per_group : int
        Number of parameters per group when using optimal pairing.
        For example, 2 for (mu, sigma) pairs. Only used when use_optimal_pairing=True.
    optimal_pairing_metric : str
        Metric to use for optimal pairing distance computation.
        Supports all metrics in calculate_optimal_pairing_distance.
        Only used when use_optimal_pairing=True.
    """

    # REQUIRED parameters (no defaults - must be explicitly specified)
    enabled: bool
    use_optimal_pairing: bool

    # Optional parameters with sensible defaults
    sigma_share: float = 1.0
    alpha: float = 1.0
    distance_metric: str = "euclidean"
    params_per_group: int = 2
    optimal_pairing_metric: str = "euclidean"

    def __post_init__(self) -> None:
        """Validate configuration values.

        Ensures positive parameters and supported metric choices.
        """
        if self.sigma_share <= 0:
            msg = "sigma_share must be > 0"
            raise ValueError(msg)
        if self.alpha <= 0:
            msg = "alpha must be > 0"
            raise ValueError(msg)
        if not self.use_optimal_pairing and self.distance_metric not in {"euclidean"}:
            msg = f"Unsupported distance_metric: {self.distance_metric!r}. Supported: 'euclidean'"
            raise ValueError(msg)
        if self.use_optimal_pairing and self.params_per_group <= 0:
            msg = "params_per_group must be > 0 when using optimal pairing"
            raise ValueError(msg)


def niche_sharing_coefficient(distance: float, sigma_share: float, alpha: float = 1.0) -> float:
    """Compute fitness sharing penalty based on distance between solutions.

    Determines the extent of fitness sharing between two solutions based on
    their distance. Solutions closer than sigma_share share more fitness;
    sharing decreases with distance.

    Formula: sh(d) = 1 - (d/sigma)^alpha if d < sigma, else 0

    Parameters
    ----------
    distance : float
        Distance between two chromosomes
    sigma_share : float
        Niche radius - solutions within this distance share fitness
    alpha : float
        Power parameter controlling sharing curve shape

    Returns
    -------
    float
        Sharing function value (0 to 1). Returns 0 if distance >= sigma_share.
    """
    if distance < sigma_share:
        coefficient: float = 1.0 - (distance / sigma_share) ** alpha
        return coefficient
    return 0.0


def compute_chromosome_distance(
    chromosome_a: np.ndarray,
    chromosome_b: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Compute distance between two chromosomes.

    Parameters
    ----------
    chromosome_a, chromosome_b : np.ndarray
        Chromosomes to compare (1D arrays of gene values)
    metric : str
        Distance metric (currently only 'euclidean' supported)

    Returns
    -------
    float
        Distance between chromosomes
    """
    if metric == "euclidean":
        distance = norm(chromosome_a - chromosome_b)
        return float(distance)
    msg = f"Unknown distance metric: {metric!r}"
    raise ValueError(msg)
