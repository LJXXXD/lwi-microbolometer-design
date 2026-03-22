"""
Automated hyperparameter tuning for Advanced GA.

This module provides systematic hyperparameter optimization using grid search
and statistical evaluation to find robust GA parameters.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from itertools import product as itertools_product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..analysis.distance_metrics import spectral_angle_mapper
from ..data.substance_atmosphere_data import load_substance_atmosphere_data
from ..simulation.gaussian_parameter_to_curves import gaussian_parameters_to_unit_amplitude_curves
from .advanced_ga import AdvancedGA
from .diversity import calculate_population_diversity
from .fitness import MinDissimilarityFitnessEvaluator
from .ga_configuration import create_ga_config
from .mutations import diversity_preserving_mutation
from .visualization import plot_top_sensor_designs

logger = logging.getLogger(__name__)


class GenerationTracker:
    """Tracks GA metrics across generations for a single run.

    This class holds the state for tracking fitness and diversity metrics,
    avoiding closure-based patterns that trigger B023 linting warnings.
    """

    def __init__(self) -> None:
        """Initialize empty tracking lists."""
        self.mean_fitness_history: list[float] = []
        self.diversity_history: list[float] = []
        self.best_fitness_history: list[float] = []

    def on_generation(self, ga_instance: AdvancedGA) -> None:
        """Track metrics per generation.

        This method is designed to be used as the `on_generation` callback
        for PyGAD's GA instances.

        Parameters
        ----------
        ga_instance : AdvancedGA
            The GA instance that triggered this callback
        """
        mean_fitness = float(np.mean(ga_instance.last_generation_fitness))
        self.mean_fitness_history.append(mean_fitness)

        diversity = calculate_population_diversity(ga_instance.population)
        self.diversity_history.append(diversity)

        if hasattr(ga_instance, "best_solutions_fitness") and ga_instance.best_solutions_fitness:
            self.best_fitness_history.append(float(ga_instance.best_solutions_fitness[-1]))
        else:
            self.best_fitness_history.append(float(np.max(ga_instance.last_generation_fitness)))


@dataclass
class HyperparameterSearchSpace:
    """Search space definition for hyperparameter tuning.

    Attributes
    ----------
    population_size : List[int]
        Population sizes to test
    num_generations : List[int]
        Number of generations to test
    num_parents_mating : List[int]
        Parent mating numbers to test
    mutation_rate_base : List[float]
        Base mutation rates to test
    crossover_rate : List[float]
        Crossover rates to test
    elitism_size : List[int]
        Elitism sizes to test
    niching_enabled : List[bool]
        Whether to enable niching
    sigma_share : List[float]
        Niche radii to test
    """

    population_size: list[int] | None = None
    num_generations: list[int] | None = None
    num_parents_mating: list[int] | None = None
    mutation_rate_base: list[float] | None = None
    crossover_rate: list[float] | None = None
    elitism_size: list[int] | None = None
    niching_enabled: list[bool] | None = None
    sigma_share: list[float] | None = None

    def __post_init__(self) -> None:
        """Set default values if not provided."""
        if self.population_size is None:
            self.population_size = [100, 200, 300]
        if self.num_generations is None:
            self.num_generations = [500, 1000]
        if self.num_parents_mating is None:
            self.num_parents_mating = [25, 50, 75]
        if self.mutation_rate_base is None:
            self.mutation_rate_base = [0.05, 0.1, 0.2]
        if self.crossover_rate is None:
            self.crossover_rate = [0.6, 0.8, 1.0]
        if self.elitism_size is None:
            self.elitism_size = [3, 5, 10]
        if self.niching_enabled is None:
            self.niching_enabled = [True, False]
        if self.sigma_share is None:
            self.sigma_share = [0.5, 1.0, 2.0]


@dataclass
class TuningResult:
    """Results from a single hyperparameter configuration run.

    Attributes
    ----------
    config : Dict[str, Any]
        Configuration parameters used
    best_fitness : float
        Best fitness achieved
    mean_fitness : float
        Mean fitness of final population
    diversity_score : float
        Final population diversity
    convergence_generation : int
        Generation when best fitness was first achieved
    high_quality_solutions : int
        Number of solutions with fitness > threshold
    """

    config: dict[str, Any]
    best_fitness: float
    mean_fitness: float
    diversity_score: float
    convergence_generation: int
    high_quality_solutions: int


def run_single_configuration(
    config_dict: dict[str, Any],
    fitness_func: Callable,
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    num_runs: int = 5,
    fitness_threshold: float = 45.0,
    random_seed_base: int = 42,
) -> TuningResult:
    """
    Run GA multiple times with a single configuration and aggregate results.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Configuration parameters (must include GA hyperparameters)
    fitness_func : Callable
        Fitness function compatible with PyGAD
    gene_space : list[dict[str, float]]
        Gene space bounds (list of dicts with 'low' and 'high' keys)
    params_per_basis_function : int
        Number of parameters per basis function (for diversity calculation)
    num_runs : int
        Number of independent runs per configuration
    fitness_threshold : float
        Threshold for high-quality solutions
    random_seed_base : int
        Base random seed (each run gets seed_base + run_idx)

    Returns
    -------
    TuningResult
        Aggregated results from multiple runs
    """
    best_fitnesses = []
    mean_fitnesses = []
    diversity_scores = []
    convergence_generations = []
    high_quality_counts = []
    random_seeds = []

    for run_idx in range(num_runs):
        random_seed = random_seed_base + run_idx
        random_seeds.append(random_seed)

        # CRITICAL: Set numpy random seed explicitly for reproducibility in multiprocessing
        # This ensures each worker process has a properly initialized random state
        np.random.seed(random_seed)

        # Build GA config using create_ga_config() helper
        # Extract parameters from config_dict, using defaults if missing
        ga_config = create_ga_config(
            num_generations=config_dict.get("num_generations", 2000),
            num_parents_mating=config_dict.get("num_parents_mating", 50),
            sol_per_pop=config_dict.get("sol_per_pop", 200),
            parent_selection_type=config_dict.get("parent_selection_type", "tournament"),
            K_tournament=config_dict.get("K_tournament", 3),
            keep_elitism=config_dict.get("keep_elitism", 5),
            crossover_type=config_dict.get("crossover_type", "uniform"),
            crossover_probability=config_dict.get("crossover_probability", 0.8),
            mutation_type=diversity_preserving_mutation,  # Use custom mutation
            mutation_probability=config_dict.get("mutation_probability", 0.1),
            save_best_solutions=True,
            stop_criteria=config_dict.get("stop_criteria", "saturate_200"),
            niching_enabled=config_dict.get("niching_enabled", True),
            niching_use_optimal_pairing=config_dict.get("niching_use_optimal_pairing", True),
            niching_params_per_group=config_dict.get(
                "niching_params_per_group", params_per_basis_function
            ),
            niching_sigma_share=config_dict.get("niching_sigma_share", 0.5),
            niching_alpha=config_dict.get("niching_alpha", 0.5),
            niching_optimal_pairing_metric=config_dict.get(
                "niching_optimal_pairing_metric", "euclidean"
            ),
            random_seed=random_seed,
        )

        # Track metrics per generation using a tracker class (avoids B023 closure warning)
        tracker = GenerationTracker()

        # Add runtime-specific parameters
        ga_params = ga_config.copy()
        ga_params["num_genes"] = len(gene_space)
        ga_params["gene_space"] = gene_space
        ga_params["fitness_func"] = fitness_func
        ga_params["on_generation"] = tracker.on_generation

        # Create and run GA
        ga = AdvancedGA(**ga_params)
        ga.run()

        # Extract results
        _best_chromosome, best_fitness, _best_idx = ga.best_solution()
        final_fitness_scores = ga.last_generation_fitness

        # Store metrics
        best_fitnesses.append(float(best_fitness))
        mean_fitnesses.append(float(np.mean(final_fitness_scores)))
        diversity_scores.append(tracker.diversity_history[-1] if tracker.diversity_history else 0.0)

        # Find convergence generation (when best fitness was first achieved)
        convergence_gen = 0
        if tracker.best_fitness_history:
            target_fitness = best_fitness * 0.99  # Within 1% of final best
            for gen, fitness in enumerate(tracker.best_fitness_history):
                if fitness >= target_fitness:
                    convergence_gen = gen
                    break
        convergence_generations.append(convergence_gen)

        # Count high-quality solutions
        high_quality_count = int(np.sum(final_fitness_scores >= fitness_threshold))
        high_quality_counts.append(high_quality_count)

    # Aggregate results (convert lists to arrays for np.mean)
    best_fitness_array = np.array(best_fitnesses, dtype=np.float64)
    mean_fitness_array = np.array(mean_fitnesses, dtype=np.float64)
    diversity_scores_array = np.array(diversity_scores, dtype=np.float64)
    convergence_generations_array = np.array(convergence_generations, dtype=np.float64)
    high_quality_counts_array = np.array(high_quality_counts, dtype=np.float64)

    return TuningResult(
        config=config_dict,
        best_fitness=float(np.mean(best_fitness_array)),
        mean_fitness=float(np.mean(mean_fitness_array)),
        diversity_score=float(np.mean(diversity_scores_array)),
        convergence_generation=int(np.mean(convergence_generations_array)),
        high_quality_solutions=int(np.mean(high_quality_counts_array)),
    )


class HyperparameterTuner:
    """
    Automated hyperparameter tuning for Advanced GA.

    This class provides systematic grid search over hyperparameter space
    with statistical evaluation and result logging.
    """

    def __init__(
        self,
        fitness_func: Callable,
        gene_space: list[dict[str, float]],
        search_space: HyperparameterSearchSpace,
        params_per_basis_function: int,
        num_runs: int = 5,
        fitness_threshold: float = 45.0,
        max_workers: int | None = None,
    ):
        """
        Initialize hyperparameter tuner.

        Parameters
        ----------
        fitness_func : Callable
            Fitness function for GA (compatible with PyGAD)
        gene_space : list[dict[str, float]]
            Gene space bounds (list of dicts with 'low' and 'high' keys)
        search_space : HyperparameterSearchSpace
            Search space definition
        params_per_basis_function : int
            Number of parameters per basis function (for diversity calculation)
        num_runs : int
            Number of independent runs per configuration
        fitness_threshold : float
            Threshold for high-quality solutions
        max_workers : int | None
            Maximum number of parallel workers (defaults to min(cpu_count(), 12))
        """
        self.fitness_func = fitness_func
        self.gene_space = gene_space
        self.search_space = search_space
        self.params_per_basis_function = params_per_basis_function
        self.num_runs = num_runs
        self.fitness_threshold = fitness_threshold
        self.max_workers = max_workers or mp.cpu_count()

        logger.info(f"Initialized HyperparameterTuner with {self.max_workers} workers")
        logger.info(f"Will run {num_runs} independent runs per configuration")

    def generate_configurations(self) -> list[dict[str, Any]]:
        """Generate all possible hyperparameter configurations."""
        configs = []

        # Core parameters from HyperparameterSearchSpace
        param_names = [
            "sol_per_pop",
            "num_generations",
            "num_parents_mating",
            "mutation_probability",
            "crossover_probability",
            "keep_elitism",
            "niching_enabled",
            "niching_sigma_share",
        ]

        param_values = [
            self.search_space.population_size,
            self.search_space.num_generations,
            self.search_space.num_parents_mating,
            self.search_space.mutation_rate_base,
            self.search_space.crossover_rate,
            self.search_space.elitism_size,
            self.search_space.niching_enabled,
            self.search_space.sigma_share,
        ]

        # Handle additional optional parameters if they exist in search_space
        # These come from YAML experiment configs
        if hasattr(self.search_space, "parent_selection_type"):
            param_names.append("parent_selection_type")
            param_values.append(self.search_space.parent_selection_type)
        if hasattr(self.search_space, "K_tournament"):
            param_names.append("K_tournament")
            param_values.append(self.search_space.K_tournament)
        if hasattr(self.search_space, "crossover_type"):
            param_names.append("crossover_type")
            param_values.append(self.search_space.crossover_type)
        if hasattr(self.search_space, "niching_alpha"):
            param_names.append("niching_alpha")
            param_values.append(self.search_space.niching_alpha)

        # itertools.product accepts any iterables, but mypy stubs are incomplete
        for combination in itertools_product(*param_values):  # type: ignore[arg-type]
            config = dict(zip(param_names, combination, strict=False))

            # Skip invalid configurations
            if config["num_parents_mating"] >= config["sol_per_pop"]:
                continue
            if config["keep_elitism"] >= config["sol_per_pop"]:
                continue

            # Handle conditional parameters
            # K_tournament only matters if using tournament selection
            parent_selection: Any = config.get("parent_selection_type")
            # Check if parent_selection is a string equal to 'tournament'
            # Note: K_tournament is kept in config but won't be used if not tournament selection
            # This is intentional - the config is valid even if K_tournament is ignored
            _is_tournament = isinstance(parent_selection, str) and parent_selection == "tournament"

            configs.append(config)

        logger.info(f"Generated {len(configs)} valid configurations")
        return configs

    def tune(self, output_dir: Path | None = None) -> pd.DataFrame:
        """
        Run hyperparameter tuning and return results.

        Parameters
        ----------
        output_dir : Path | None
            Directory to save results

        Returns
        -------
        pd.DataFrame
            Results dataframe with all configurations and metrics
        """
        if output_dir is None:
            output_dir = Path("outputs/tuning/hyperparameter_tuning_results")
        output_dir.mkdir(exist_ok=True)

        # Generate configurations
        configurations = self.generate_configurations()

        logger.info(f"Starting hyperparameter tuning with {len(configurations)} configurations")
        logger.info(f"Total runs: {len(configurations) * self.num_runs}")

        # Run configurations in parallel
        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(
                    run_single_configuration,
                    config,
                    self.fitness_func,
                    self.gene_space,
                    self.params_per_basis_function,
                    self.num_runs,
                    self.fitness_threshold,
                ): config
                for config in configurations
            }

            # Collect results
            for i, future in enumerate(as_completed(future_to_config)):
                try:
                    result = future.result()
                    results.append(result)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(configurations)} configurations")

                except Exception as e:
                    config = future_to_config[future]
                    logger.error(f"Configuration failed: {config}, Error: {e}")

        # Convert to DataFrame
        results_data = []
        for result in results:
            row = result.config.copy()
            row.update(
                {
                    "best_fitness": result.best_fitness,
                    "mean_fitness": result.mean_fitness,
                    "diversity_score": result.diversity_score,
                    "convergence_generation": result.convergence_generation,
                    "high_quality_solutions": result.high_quality_solutions,
                }
            )
            results_data.append(row)

        df_results = pd.DataFrame(results_data)

        # Sort by best fitness
        df_results = df_results.sort_values("best_fitness", ascending=False)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"tuning_results_{timestamp}.csv"
        df_results.to_csv(results_file, index=False)

        # Save summary
        summary_file = output_dir / f"tuning_summary_{timestamp}.json"
        summary = {
            "timestamp": timestamp,
            "total_configurations": len(configurations),
            "total_runs": len(configurations) * self.num_runs,
            "num_runs_per_config": self.num_runs,
            "fitness_threshold": self.fitness_threshold,
            "best_configuration": df_results.iloc[0].to_dict(),
            "top_5_configurations": df_results.head(5).to_dict("records"),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Tuning complete. Results saved to {results_file}")
        logger.info(
            f"Best configuration achieved fitness: {df_results.iloc[0]['best_fitness']:.4f}"
        )

        return df_results

    def analyze_results(self, results_df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze tuning results to identify patterns and recommendations.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results from tuning

        Returns
        -------
        dict[str, Any]
            Analysis results and recommendations
        """
        analysis: dict[str, Any] = {}

        # Top performers analysis
        top_10 = results_df.head(10)

        # Parameter importance analysis
        param_importance = {}
        for param in [
            "sol_per_pop",
            "num_generations",
            "num_parents_mating",
            "mutation_probability",
            "crossover_probability",
            "keep_elitism",
            "niching_enabled",
            "niching_sigma_share",
        ]:
            if param in top_10.columns:
                param_importance[param] = {
                    "top_10_mean": top_10[param].mean(),
                    "all_mean": results_df[param].mean(),
                    "top_10_std": top_10[param].std(),
                }

        analysis["parameter_importance"] = param_importance

        # Niching effectiveness
        niching_enabled = results_df[results_df["niching_enabled"]]
        niching_disabled = results_df[~results_df["niching_enabled"]]

        analysis["niching_effectiveness"] = {
            "enabled_mean_fitness": niching_enabled["best_fitness"].mean(),
            "disabled_mean_fitness": niching_disabled["best_fitness"].mean(),
            "enabled_mean_diversity": niching_enabled["diversity_score"].mean(),
            "disabled_mean_diversity": niching_disabled["diversity_score"].mean(),
        }

        # Convergence analysis (convert pandas Series results to float)
        convergence_mean = results_df["convergence_generation"].mean()
        convergence_q25 = results_df["convergence_generation"].quantile(0.25)
        convergence_q75 = results_df["convergence_generation"].quantile(0.75)
        convergence_analysis: dict[str, float] = {
            "mean_convergence_generation": float(convergence_mean)
            if pd.notna(convergence_mean)
            else 0.0,
            "fast_convergence_threshold": float(convergence_q25)
            if pd.notna(convergence_q25)
            else 0.0,
            "slow_convergence_threshold": float(convergence_q75)
            if pd.notna(convergence_q75)
            else 0.0,
        }
        analysis["convergence_analysis"] = convergence_analysis

        return analysis


def create_default_search_space() -> HyperparameterSearchSpace:
    """Create a reasonable default search space for initial tuning."""
    return HyperparameterSearchSpace(
        population_size=[100, 200, 300],
        num_generations=[500, 1000],
        num_parents_mating=[25, 50, 75],
        mutation_rate_base=[0.05, 0.1, 0.2],
        crossover_rate=[0.6, 0.8, 1.0],
        elitism_size=[3, 5, 10],
        niching_enabled=[True, False],
        sigma_share=[0.5, 1.0, 2.0],
    )


def create_focused_search_space() -> HyperparameterSearchSpace:
    """Create a focused search space around promising parameters."""
    return HyperparameterSearchSpace(
        population_size=[150, 200, 250],
        num_generations=[800, 1000, 1200],
        num_parents_mating=[40, 50, 60],
        mutation_rate_base=[0.08, 0.1, 0.12],
        crossover_rate=[0.7, 0.8, 0.9],
        elitism_size=[4, 5, 6],
        niching_enabled=[True],
        sigma_share=[0.8, 1.0, 1.2],
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration loaded from YAML.

    Attributes
    ----------
    name : str
        Experiment name
    description : str | None
        Optional experiment description
    data : dict[str, Any]
        Data loading configuration (file paths, parameters)
    sensor : dict[str, Any]
        Sensor configuration (num_basis_functions, param_bounds, etc.)
    search_space : dict[str, Any]
        Hyperparameter search space definition
    execution : dict[str, Any]
        Execution parameters (num_runs, seeds, workers, thresholds)
    validation : dict[str, Any] | None
        Validation mode configuration (if enabled)
    """

    name: str
    description: str | None
    data: dict[str, Any]
    sensor: dict[str, Any]
    search_space: dict[str, Any]
    execution: dict[str, Any]
    validation: dict[str, Any] | None


def load_experiment_config(yaml_path: Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Parameters
    ----------
    yaml_path : Path
        Path to YAML experiment definition file

    Returns
    -------
    ExperimentConfig
        Loaded experiment configuration

    Raises
    ------
    FileNotFoundError
        If YAML file doesn't exist
    ValueError
        If YAML structure is invalid or missing required fields
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment YAML file not found: {yaml_path}")

    with open(yaml_path) as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict) or "experiment" not in raw_config:
        raise ValueError('YAML must contain top-level "experiment" key')

    exp = raw_config["experiment"]

    # Validate required fields
    required_fields = ["name", "data", "sensor", "search_space", "execution"]
    missing_fields = [field for field in required_fields if field not in exp]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Resolve relative paths relative to project root (parent of experiments directory)
    # Paths in YAML are relative to project root, not YAML file location
    yaml_dir = yaml_path.parent
    # If YAML is in experiments/, project root is parent; otherwise assume current dir
    if yaml_dir.name == "experiments":
        project_root = yaml_dir.parent
    else:
        # Fallback: try to find project root by looking for src/ and data/ directories
        project_root = yaml_dir
        while project_root != project_root.parent:
            if (project_root / "src").exists() and (project_root / "data").exists():
                break
            project_root = project_root.parent

    if "spectral_data_file" in exp["data"]:
        exp["data"]["spectral_data_file"] = (
            project_root / exp["data"]["spectral_data_file"]
        ).resolve()
    if "air_transmittance_file" in exp["data"]:
        exp["data"]["air_transmittance_file"] = (
            project_root / exp["data"]["air_transmittance_file"]
        ).resolve()

    return ExperimentConfig(
        name=exp["name"],
        description=exp.get("description"),
        data=exp["data"],
        sensor=exp["sensor"],
        search_space=exp["search_space"],
        execution=exp["execution"],
        validation=exp.get("validation"),
    )


def create_fitness_evaluator_from_experiment(
    experiment: ExperimentConfig,
) -> Callable[[object, np.ndarray, int], float]:
    """Create fitness evaluator from experiment configuration.

    Parameters
    ----------
    experiment : ExperimentConfig
        Experiment configuration

    Returns
    -------
    Callable
        Fitness function compatible with PyGAD
    """
    # Load data
    loaded = load_substance_atmosphere_data(
        spectral_data_file=Path(experiment.data["spectral_data_file"]),
        air_transmittance_file=Path(experiment.data["air_transmittance_file"]),
        atmospheric_distance_ratio=experiment.data.get("atmospheric_distance_ratio", 0.11),
        temperature_kelvin=experiment.data.get("temperature_kelvin", 293.15),
        air_refractive_index=experiment.data.get("air_refractive_index", 1.0),
    )

    # Handle multi-condition data (should be single condition for tuning)
    if isinstance(loaded, list):
        if len(loaded) > 1:
            logger.warning("Multi-condition data detected, using first condition only")
        scene = loaded[0]
    else:
        scene = loaded

    evaluator = MinDissimilarityFitnessEvaluator(
        scene=scene,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=experiment.sensor["params_per_basis_function"],
        distance_metric=spectral_angle_mapper,
    )

    return evaluator.fitness_func


def create_search_space_from_experiment(experiment: ExperimentConfig) -> HyperparameterSearchSpace:
    """Create HyperparameterSearchSpace from experiment configuration.

    Parameters
    ----------
    experiment : ExperimentConfig
        Experiment configuration

    Returns
    -------
    HyperparameterSearchSpace
        Search space for tuning (with additional attributes for optional parameters)
    """
    search_space_dict = experiment.search_space

    # Create base search space
    search_space = HyperparameterSearchSpace(
        population_size=search_space_dict.get("sol_per_pop", [100, 200, 300]),
        num_generations=search_space_dict.get("num_generations", [1000, 2000]),
        num_parents_mating=search_space_dict.get("num_parents_mating", [40, 50, 60]),
        mutation_rate_base=search_space_dict.get("mutation_probability", [0.05, 0.1, 0.15]),
        crossover_rate=search_space_dict.get("crossover_probability", [0.7, 0.8, 0.9]),
        elitism_size=search_space_dict.get("keep_elitism", [3, 5, 10]),
        niching_enabled=search_space_dict.get("niching_enabled", [True, False]),
        sigma_share=search_space_dict.get("niching_sigma_share", [0.5, 1.0, 2.0]),
    )

    # Add optional parameters as attributes
    if "parent_selection_type" in search_space_dict:
        search_space.parent_selection_type = search_space_dict["parent_selection_type"]
    if "K_tournament" in search_space_dict:
        search_space.K_tournament = search_space_dict["K_tournament"]
    if "crossover_type" in search_space_dict:
        search_space.crossover_type = search_space_dict["crossover_type"]
    if "niching_alpha" in search_space_dict:
        search_space.niching_alpha = search_space_dict["niching_alpha"]

    return search_space


def create_gene_space_from_experiment(experiment: ExperimentConfig) -> list[dict[str, float]]:
    """Create gene space from experiment configuration.

    Parameters
    ----------
    experiment : ExperimentConfig
        Experiment configuration

    Returns
    -------
    list[dict[str, float]]
        Gene space bounds (list of dicts with 'low' and 'high' keys)
    """
    num_basis_functions = experiment.sensor["num_basis_functions"]
    param_bounds = experiment.sensor["param_bounds"]

    # Repeat param_bounds for each basis function
    gene_space: list[dict[str, float]] = param_bounds * num_basis_functions

    return gene_space


def visualize_top_configurations(
    results_df: pd.DataFrame,
    experiment: ExperimentConfig,
    fitness_func: Callable,
    gene_space: list[dict[str, float]],
    params_per_basis_function: int,
    output_dir: Path,
    top_k: int = 5,
    num_generations_override: int | None = None,
) -> None:
    """Generate visualizations for top K configurations.

    For each top configuration, runs a GA and generates plot_top_sensor_designs visualization.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from tuning (sorted by best_fitness descending)
    experiment : ExperimentConfig
        Experiment configuration (for data loading)
    fitness_func : Callable
        Fitness function for GA
    gene_space : list[dict[str, float]]
        Gene space bounds
    params_per_basis_function : int
        Number of parameters per basis function
    output_dir : Path
        Output directory for visualizations
    top_k : int
        Number of top configurations to visualize (default: 5)
    num_generations_override : int | None
        Override number of generations for visualization runs (default: None, uses config value)
    """
    logger.info(f"\n=== Generating Visualizations for Top {top_k} Configurations ===")

    # Get top K configurations
    top_configs = results_df.head(top_k)

    # Load data to get wavelengths for visualization
    loaded = load_substance_atmosphere_data(
        spectral_data_file=Path(experiment.data["spectral_data_file"]),
        air_transmittance_file=Path(experiment.data["air_transmittance_file"]),
        atmospheric_distance_ratio=experiment.data.get("atmospheric_distance_ratio", 0.11),
        temperature_kelvin=experiment.data.get("temperature_kelvin", 293.15),
        air_refractive_index=experiment.data.get("air_refractive_index", 1.0),
    )

    # Handle multi-condition data
    if isinstance(loaded, list):
        if len(loaded) > 1:
            logger.warning("Multi-condition data detected, using first condition only")
        scene = loaded[0]
    else:
        scene = loaded

    wavelengths = scene.wavelengths
    fitness_threshold = experiment.execution.get("fitness_threshold", 45.0)

    for idx, (_, config_row) in enumerate(top_configs.iterrows(), 1):
        logger.info(f"\nVisualizing configuration {idx}/{top_k} (rank {idx})...")

        try:
            # Extract configuration from row (exclude result columns)
            result_columns = [
                "best_fitness",
                "mean_fitness",
                "diversity_score",
                "convergence_generation",
                "high_quality_solutions",
            ]
            config_dict = {k: v for k, v in config_row.items() if k not in result_columns}

            # Override generations if specified (for faster visualization runs)
            if num_generations_override:
                config_dict["num_generations"] = num_generations_override
                logger.info(f"  Using {num_generations_override} generations for visualization")

            # Run single GA run with this configuration
            random_seed = experiment.execution.get("random_seed_base", 42)

            # Build GA config
            ga_config = create_ga_config(
                num_generations=config_dict.get("num_generations", 2000),
                num_parents_mating=config_dict.get("num_parents_mating", 50),
                sol_per_pop=config_dict.get("sol_per_pop", 200),
                parent_selection_type=config_dict.get("parent_selection_type", "tournament"),
                K_tournament=config_dict.get("K_tournament", 3),
                keep_elitism=config_dict.get("keep_elitism", 5),
                crossover_type=config_dict.get("crossover_type", "uniform"),
                crossover_probability=config_dict.get("crossover_probability", 0.8),
                mutation_type=diversity_preserving_mutation,
                mutation_probability=config_dict.get("mutation_probability", 0.1),
                save_best_solutions=True,
                stop_criteria=config_dict.get("stop_criteria", "saturate_200"),
                niching_enabled=config_dict.get("niching_enabled", True),
                niching_use_optimal_pairing=config_dict.get("niching_use_optimal_pairing", True),
                niching_params_per_group=config_dict.get(
                    "niching_params_per_group", params_per_basis_function
                ),
                niching_sigma_share=config_dict.get("niching_sigma_share", 0.5),
                niching_alpha=config_dict.get("niching_alpha", 0.5),
                niching_optimal_pairing_metric=config_dict.get(
                    "niching_optimal_pairing_metric", "euclidean"
                ),
                random_seed=random_seed,
            )

            # Track metrics using a tracker class (avoids B023 closure warning)
            diversity_tracker = GenerationTracker()

            # Add runtime-specific parameters
            ga_params = ga_config.copy()
            ga_params["num_genes"] = len(gene_space)
            ga_params["gene_space"] = gene_space
            ga_params["fitness_func"] = fitness_func
            ga_params["on_generation"] = diversity_tracker.on_generation

            # Create and run GA
            ga = AdvancedGA(**ga_params)
            ga.run()

            # Extract results
            final_fitness_scores = ga.last_generation_fitness
            final_population = ga.population

            # Filter high-quality solutions
            high_quality_mask = final_fitness_scores >= fitness_threshold
            high_quality_population = final_population[high_quality_mask]
            high_quality_fitness = final_fitness_scores[high_quality_mask]

            if len(high_quality_population) == 0:
                logger.warning(
                    f"  No high-quality solutions found (threshold: {fitness_threshold}). "
                    f"Best fitness: {np.max(final_fitness_scores):.4f}"
                )
                continue

            logger.info(
                f"  Found {len(high_quality_population)} high-quality solutions "
                f"(best: {np.max(high_quality_fitness):.4f})"
            )

            # Create output directory for this configuration
            config_output_dir = output_dir / f"top_{idx}_config"
            config_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate visualization (scene wavelengths are canonical 1D)
            wavelengths_1d = np.asarray(wavelengths)

            plot_top_sensor_designs(
                high_quality_population=high_quality_population,
                high_quality_fitness=high_quality_fitness,
                wavelengths=wavelengths_1d,
                fitness_threshold=fitness_threshold,
                output_dir=config_output_dir,
            )

            logger.info(f"  ✓ Visualization saved to {config_output_dir}")

        except Exception as e:
            logger.error(f"  ✗ Failed to visualize configuration {idx}: {e}")
            logger.debug(traceback.format_exc())
            continue

    logger.info(f"\n✓ Completed visualizations for top {top_k} configurations")
