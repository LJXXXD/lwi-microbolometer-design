"""
YAML experiment configuration and wiring for GA hyperparameter tuning.

This module loads experiment definitions and builds search space, gene space,
and fitness callables used by :mod:`lwi_microbolometer_design.ga.tuning`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from ..analysis.distance_metrics import spectral_angle_mapper
from ..data.substance_atmosphere_data import load_substance_atmosphere_data
from ..simulation.gaussian_parameter_to_curves import gaussian_parameters_to_unit_amplitude_curves
from .fitness import MinDissimilarityFitnessEvaluator

if TYPE_CHECKING:
    from .tuning import HyperparameterSearchSpace

logger = logging.getLogger(__name__)


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


def create_search_space_from_experiment(
    experiment: ExperimentConfig,
) -> HyperparameterSearchSpace:
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
    # Local import avoids circular import with tuning (search space class lives there).
    from .tuning import HyperparameterSearchSpace

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
