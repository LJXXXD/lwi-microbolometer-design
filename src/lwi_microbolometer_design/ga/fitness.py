"""
Fitness function builders for GA optimization.

These utilities construct PyGAD-compatible fitness functions that wire together
simulation, distance computation, and scoring in a reproducible, testable way.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

from lwi_microbolometer_design.analysis import (
    compute_distance_matrix,
    min_based_dissimilarity_score,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data.scene_config import SceneConfig
from lwi_microbolometer_design.simulation import simulate_sensor_output

logger = logging.getLogger(__name__)

SUPPORTED_AGGREGATIONS = ("single", "min", "mean")


class MinDissimilarityFitnessEvaluator:
    """Fitness evaluator for sensor optimisation (GA, MAP-Elites, CMA-ME).

    Converts chromosomes to basis functions, simulates sensor output, and
    computes min-based dissimilarity scores.

    Supports **single-condition** and **multi-condition** (robust) evaluation.
    When constructed with a list of :class:`SceneConfig` objects, the evaluator
    computes fitness under every condition and aggregates using the chosen
    strategy (``"min"`` for worst-case / minimax, ``"mean"`` for expected
    performance).  Single-condition mode (``aggregation="single"``) is the
    default and is fully backward-compatible.

    Why a class instead of a factory function?  Serialization for
    multiprocessing.  Factory functions create closures that capture external
    variables, making them difficult to pickle.  This class stores parameters
    as attributes, enabling proper serialization for parallel execution.

    Supports any basis function type via *parameters_to_curves* callback.
    """

    def __init__(
        self,
        scene: SceneConfig | Sequence[SceneConfig],
        parameters_to_curves: Callable[[list[tuple], np.ndarray], np.ndarray],
        params_per_basis_function: int,
        distance_metric: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
        aggregation: str = "single",
    ):
        """
        Initialize fitness evaluator with all required parameters.

        Parameters
        ----------
        scene : SceneConfig | Sequence[SceneConfig]
            One or more immutable scene snapshots.  A single ``SceneConfig``
            activates single-condition mode (``aggregation`` is forced to
            ``"single"``).  A list/sequence enables multi-condition robust
            evaluation.
        parameters_to_curves : callable
            Function that converts parameter tuples to basis function curves.
            ``list[tuple[float, ...]]`` -> ``np.ndarray``
        params_per_basis_function : int
            Number of parameters per basis function.
            Example: 2 for Gaussian (mu, sigma), 3 for Lorentzian (mu, sigma, gamma).
        distance_metric : callable, optional
            Pairwise distance function; default is SAM.
        aggregation : str, optional
            How to combine fitness values across conditions.

            * ``"single"`` -- single-condition mode (default, backward-compatible).
            * ``"min"`` -- worst-case (minimax robust optimisation).
            * ``"mean"`` -- average across conditions.

        Raises
        ------
        ValueError
            If *aggregation* is not one of the supported values, or if
            ``"single"`` is requested with multiple scenes.
        """
        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {SUPPORTED_AGGREGATIONS}, got {aggregation!r}"
            )

        if isinstance(scene, SceneConfig):
            self._scenes: list[SceneConfig] = [scene]
            self.aggregation = "single"
            if aggregation not in ("single", "min", "mean"):
                raise ValueError(
                    f"aggregation must be one of {SUPPORTED_AGGREGATIONS}, got {aggregation!r}"
                )
        else:
            self._scenes = list(scene)
            if len(self._scenes) == 0:
                raise ValueError("At least one SceneConfig is required.")
            if aggregation == "single" and len(self._scenes) > 1:
                logger.warning(
                    "Multiple scenes provided with aggregation='single'; "
                    "defaulting to aggregation='min' (worst-case)."
                )
                aggregation = "min"
            self.aggregation = aggregation

        self.distance_metric = distance_metric
        self.params_per_basis_function = params_per_basis_function
        self.parameters_to_curves = parameters_to_curves

    @property
    def scene(self) -> SceneConfig:
        """Primary (first) scene — backward-compatible accessor."""
        return self._scenes[0]

    @property
    def scenes(self) -> list[SceneConfig]:
        """All scenes used for evaluation."""
        return self._scenes

    @property
    def num_conditions(self) -> int:
        """Number of environmental conditions."""
        return len(self._scenes)

    def _evaluate_single_scene(
        self,
        basis_params: list[tuple],
        scene: SceneConfig,
    ) -> float:
        """Compute fitness for one scene."""
        basis_functions = self.parameters_to_curves(basis_params, scene.wavelengths)

        sensor_outputs = simulate_sensor_output(
            wavelengths=scene.wavelengths,
            substances_emissivity=scene.emissivity_curves,
            basis_functions=basis_functions,
            temperature_k=scene.temperature_k,
            atmospheric_distance_ratio=scene.atmospheric_distance_ratio,
            air_refractive_index=scene.air_refractive_index,
            air_transmittance=scene.air_transmittance,
        )

        distance_matrix = compute_distance_matrix(
            sensor_outputs,
            distance_func=self.distance_metric,
            axis=1,
        )
        return float(min_based_dissimilarity_score(distance_matrix=distance_matrix))

    def fitness_func(
        self, _ga_instance: object, chromosome: np.ndarray, _chromosome_idx: int
    ) -> float:
        """Compute fitness for a given chromosome.

        When multiple scenes are configured, evaluates the chromosome under
        each scene and aggregates according to ``self.aggregation``.

        Parameters
        ----------
        _ga_instance : object
            GA instance (unused, for PyGAD compatibility).
        chromosome : np.ndarray
            Chromosome genes representing sensor parameters.
            Length should be ``num_basis_functions * params_per_basis_function``.
        _chromosome_idx : int
            Chromosome index (unused, for PyGAD compatibility).

        Returns
        -------
        float
            Aggregated fitness score (min-based dissimilarity).

        Notes
        -----
        Chromosome parsing example:

        * ``params_per_basis_function=2``:
          ``[p1, p2, p3, p4, ...]`` -> ``[(p1, p2), (p3, p4), ...]``
        * ``params_per_basis_function=3``:
          ``[p1, p2, p3, p4, p5, p6, ...]`` -> ``[(p1, p2, p3), (p4, p5, p6), ...]``
        """
        num_genes = len(chromosome)

        if num_genes % self.params_per_basis_function != 0:
            raise ValueError(
                f"Chromosome length ({num_genes}) must be divisible by "
                f"params_per_basis_function ({self.params_per_basis_function})"
            )

        basis_params = [
            tuple(chromosome[i : i + self.params_per_basis_function])
            for i in range(0, num_genes, self.params_per_basis_function)
        ]

        if self.aggregation == "single":
            return self._evaluate_single_scene(basis_params, self._scenes[0])

        fitnesses = [self._evaluate_single_scene(basis_params, scene) for scene in self._scenes]

        if self.aggregation == "min":
            return float(min(fitnesses))
        return float(np.mean(fitnesses))
