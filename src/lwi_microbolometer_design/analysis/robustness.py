"""Environmental robustness evaluation for optimised sensor designs.

Evaluates how sensor fitness degrades under varying environmental
conditions (temperature, atmospheric distance, noise).  This module
implements **Phase 1** of the robustness roadmap: non-invasive testing
of existing elite designs across a grid of environmental parameters.

The core data structure is :class:`RobustnessResult`, a per-elite record
that stores fitness values across all tested conditions alongside summary
statistics (mean, min, std, coefficient of variation).

References
----------
Roadmap item 1 in ``docs/CODEBASE_WALKTHROUGH_AND_STRATEGY.md``, lines 226-244.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from lwi_microbolometer_design.analysis import (
    compute_distance_matrix,
    min_based_dissimilarity_score,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data.scene_config import SceneConfig
from lwi_microbolometer_design.simulation import simulate_sensor_output

logger = logging.getLogger(__name__)


@dataclass
class ConditionLabel:
    """Human-readable label for a single environmental condition.

    Parameters
    ----------
    temperature_k : float
        Scene temperature in Kelvin.
    atmospheric_distance_ratio : float
        Atmospheric distance ratio.
    air_refractive_index : float
        Refractive index of air.
    """

    temperature_k: float
    atmospheric_distance_ratio: float
    air_refractive_index: float

    @classmethod
    def from_scene(cls, scene: SceneConfig) -> ConditionLabel:
        """Create a label from a :class:`SceneConfig`."""
        return cls(
            temperature_k=scene.temperature_k,
            atmospheric_distance_ratio=scene.atmospheric_distance_ratio,
            air_refractive_index=scene.air_refractive_index,
        )

    def short_str(self) -> str:
        """Compact string for plot tick labels."""
        return (
            f"T={self.temperature_k:.0f}K, "
            f"d={self.atmospheric_distance_ratio:.2f}, "
            f"n={self.air_refractive_index:.2f}"
        )

    def __str__(self) -> str:
        return self.short_str()


@dataclass
class RobustnessResult:
    """Per-elite robustness evaluation results.

    Parameters
    ----------
    elite_id : int
        Identifier for this elite (rank or archive index).
    chromosome : np.ndarray
        Sensor parameter vector.
    nominal_fitness : float
        Fitness under the nominal (original) condition.
    condition_labels : list[ConditionLabel]
        Description of each tested condition.
    fitness_per_condition : np.ndarray
        Fitness value under each condition, shape ``(num_conditions,)``.
    """

    elite_id: int
    chromosome: np.ndarray
    nominal_fitness: float
    condition_labels: list[ConditionLabel] = field(repr=False)
    fitness_per_condition: np.ndarray = field(repr=False)

    @property
    def mean_fitness(self) -> float:
        """Mean fitness across all conditions."""
        return float(np.mean(self.fitness_per_condition))

    @property
    def min_fitness(self) -> float:
        """Worst-case fitness across all conditions."""
        return float(np.min(self.fitness_per_condition))

    @property
    def max_fitness(self) -> float:
        """Best-case fitness across all conditions."""
        return float(np.max(self.fitness_per_condition))

    @property
    def std_fitness(self) -> float:
        """Standard deviation of fitness across conditions."""
        return float(np.std(self.fitness_per_condition))

    @property
    def cv_fitness(self) -> float:
        """Coefficient of variation (std / mean), dimensionless spread metric."""
        mean = self.mean_fitness
        if mean == 0.0:
            return float("inf")
        return self.std_fitness / mean

    @property
    def retention_ratio(self) -> float:
        """Fraction of nominal fitness retained in the worst case."""
        if self.nominal_fitness == 0.0:
            return 0.0
        return self.min_fitness / self.nominal_fitness

    @property
    def worst_condition_index(self) -> int:
        """Index of the condition that yields the lowest fitness."""
        return int(np.argmin(self.fitness_per_condition))


def evaluate_elite_fitness(
    chromosome: np.ndarray,
    scene: SceneConfig,
    parameters_to_curves: Callable[[list[tuple], np.ndarray], np.ndarray],
    params_per_basis_function: int,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
) -> float:
    """Evaluate a single chromosome's fitness under one SceneConfig.

    Parameters
    ----------
    chromosome : np.ndarray
        Sensor parameter vector.
    scene : SceneConfig
        Environmental condition.
    parameters_to_curves : callable
        Basis-function callback (e.g. ``gaussian_parameters_to_unit_amplitude_curves``).
    params_per_basis_function : int
        Genes per basis function (2 for Gaussian: mu, sigma).
    distance_metric : callable
        Pairwise distance function; default SAM.

    Returns
    -------
    float
        Scalar fitness (min-based dissimilarity score).
    """
    basis_params = [
        tuple(chromosome[i : i + params_per_basis_function])
        for i in range(0, len(chromosome), params_per_basis_function)
    ]
    basis_functions = parameters_to_curves(basis_params, scene.wavelengths)
    sensor_outputs = simulate_sensor_output(
        wavelengths=scene.wavelengths,
        substances_emissivity=scene.emissivity_curves,
        basis_functions=basis_functions,
        temperature_k=scene.temperature_k,
        atmospheric_distance_ratio=scene.atmospheric_distance_ratio,
        air_refractive_index=scene.air_refractive_index,
        air_transmittance=scene.air_transmittance,
    )
    distance_matrix = compute_distance_matrix(sensor_outputs, distance_func=distance_metric, axis=1)
    return float(min_based_dissimilarity_score(distance_matrix=distance_matrix))


def evaluate_elite_robustness(
    elite_id: int,
    chromosome: np.ndarray,
    nominal_fitness: float,
    scenes: Sequence[SceneConfig],
    parameters_to_curves: Callable[[list[tuple], np.ndarray], np.ndarray],
    params_per_basis_function: int,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
) -> RobustnessResult:
    """Evaluate one elite across multiple environmental conditions.

    Parameters
    ----------
    elite_id : int
        Identifier for this elite.
    chromosome : np.ndarray
        Sensor parameter vector.
    nominal_fitness : float
        Fitness under the original optimisation condition.
    scenes : Sequence[SceneConfig]
        Environmental conditions to evaluate.
    parameters_to_curves : callable
        Basis-function callback.
    params_per_basis_function : int
        Genes per basis function.
    distance_metric : callable
        Pairwise distance function.

    Returns
    -------
    RobustnessResult
        Fitness values and summary statistics across all conditions.
    """
    condition_labels = [ConditionLabel.from_scene(s) for s in scenes]
    fitnesses = np.array(
        [
            evaluate_elite_fitness(
                chromosome=chromosome,
                scene=scene,
                parameters_to_curves=parameters_to_curves,
                params_per_basis_function=params_per_basis_function,
                distance_metric=distance_metric,
            )
            for scene in scenes
        ],
        dtype=np.float64,
    )

    return RobustnessResult(
        elite_id=elite_id,
        chromosome=chromosome.copy(),
        nominal_fitness=nominal_fitness,
        condition_labels=condition_labels,
        fitness_per_condition=fitnesses,
    )


def evaluate_solutions_robustness(
    solutions: Sequence[dict[str, Any]],
    scenes: Sequence[SceneConfig],
    parameters_to_curves: Callable[[list[tuple], np.ndarray], np.ndarray],
    params_per_basis_function: int,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
    top_n: int | None = None,
) -> list[RobustnessResult]:
    """Evaluate environmental robustness for a collection of sensor designs.

    This is the **generic entry point** that works with any optimiser output
    (GA, MAP-Elites, CMA-ME, etc.) as long as solutions carry ``"chromosome"``
    and ``"fitness"`` keys.

    Parameters
    ----------
    solutions : Sequence[dict[str, Any]]
        Each dict must contain at minimum ``"chromosome"`` (np.ndarray) and
        ``"fitness"`` (float).
    scenes : Sequence[SceneConfig]
        Environmental conditions to test.
    parameters_to_curves : callable
        Basis-function callback.
    params_per_basis_function : int
        Genes per basis function.
    distance_metric : callable
        Pairwise distance function.
    top_n : int | None
        If set, evaluate only the *top_n* highest-fitness solutions.
        ``None`` evaluates all solutions.

    Returns
    -------
    list[RobustnessResult]
        One result per evaluated solution, sorted by nominal fitness descending.
    """
    sorted_solutions = sorted(solutions, key=lambda x: x["fitness"], reverse=True)
    if top_n is not None:
        sorted_solutions = sorted_solutions[:top_n]

    logger.info(
        "Evaluating %d solutions across %d environmental conditions",
        len(sorted_solutions),
        len(scenes),
    )

    results: list[RobustnessResult] = []
    for rank, sol in enumerate(
        tqdm(sorted_solutions, desc="Robustness evaluation", unit="solution")
    ):
        result = evaluate_elite_robustness(
            elite_id=rank,
            chromosome=sol["chromosome"],
            nominal_fitness=sol["fitness"],
            scenes=scenes,
            parameters_to_curves=parameters_to_curves,
            params_per_basis_function=params_per_basis_function,
            distance_metric=distance_metric,
        )
        results.append(result)

    return results


def evaluate_archive_robustness(
    archive: dict[tuple[int, int], dict[str, Any]],
    scenes: Sequence[SceneConfig],
    parameters_to_curves: Callable[[list[tuple], np.ndarray], np.ndarray],
    params_per_basis_function: int,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
    top_n: int | None = None,
) -> list[RobustnessResult]:
    """Evaluate environmental robustness for elites in a MAP-Elites archive.

    Convenience wrapper around :func:`evaluate_solutions_robustness` that
    accepts the MAP-Elites archive dict format directly.

    Parameters
    ----------
    archive : dict
        MAP-Elites archive mapping ``(x_bin, y_bin) -> individual dict``.
    scenes : Sequence[SceneConfig]
        Environmental conditions to test.
    parameters_to_curves : callable
        Basis-function callback.
    params_per_basis_function : int
        Genes per basis function.
    distance_metric : callable
        Pairwise distance function.
    top_n : int | None
        If set, evaluate only the *top_n* highest-fitness elites.
        ``None`` evaluates the entire archive.

    Returns
    -------
    list[RobustnessResult]
        One result per evaluated elite, sorted by nominal fitness descending.
    """
    return evaluate_solutions_robustness(
        solutions=list(archive.values()),
        scenes=scenes,
        parameters_to_curves=parameters_to_curves,
        params_per_basis_function=params_per_basis_function,
        distance_metric=distance_metric,
        top_n=top_n,
    )


def summarise_robustness(results: list[RobustnessResult]) -> dict[str, Any]:
    """Compute aggregate statistics across all evaluated elites.

    Parameters
    ----------
    results : list[RobustnessResult]
        Per-elite robustness results.

    Returns
    -------
    dict[str, Any]
        Summary with keys ``num_elites``, ``num_conditions``,
        ``mean_retention_ratio``, ``worst_retention_ratio``,
        ``mean_cv``, ``mean_nominal``, ``mean_worst_case``,
        ``mean_best_case``.
    """
    retentions = np.array([r.retention_ratio for r in results])
    cvs = np.array([r.cv_fitness for r in results])
    nominals = np.array([r.nominal_fitness for r in results])
    worst_cases = np.array([r.min_fitness for r in results])
    best_cases = np.array([r.max_fitness for r in results])

    return {
        "num_elites": len(results),
        "num_conditions": len(results[0].fitness_per_condition) if results else 0,
        "mean_retention_ratio": float(np.mean(retentions)),
        "worst_retention_ratio": float(np.min(retentions)),
        "mean_cv": float(np.mean(cvs)),
        "mean_nominal": float(np.mean(nominals)),
        "mean_worst_case": float(np.mean(worst_cases)),
        "mean_best_case": float(np.mean(best_cases)),
    }
