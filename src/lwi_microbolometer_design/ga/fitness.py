"""
Fitness function builders for GA optimization.

These utilities construct PyGAD-compatible fitness functions that wire together
simulation, distance computation, and scoring in a reproducible, testable way.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lwi_microbolometer_design.analysis import (
    compute_distance_matrix,
    min_based_dissimilarity_score,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.data.scene_config import SceneConfig
from lwi_microbolometer_design.simulation import simulate_sensor_output


class MinDissimilarityFitnessEvaluator:
    """
    GA fitness evaluator for sensor optimization.

    Converts chromosomes to basis functions, simulates sensor output, and computes
    min-based dissimilarity scores.

    Why a class instead of a factory function? Serialization for multiprocessing.
    Factory functions create closures that capture external variables, making them
    difficult to pickle. This class stores parameters as attributes, enabling
    proper serialization for parallel GA execution.

    Supports any basis function type via parameters_to_curves callback.
    """

    def __init__(
        self,
        scene: SceneConfig,
        parameters_to_curves: Callable[[list[tuple], np.ndarray], np.ndarray],
        params_per_basis_function: int,
        distance_metric: Callable[[np.ndarray, np.ndarray], float] = spectral_angle_mapper,
    ):
        """
        Initialize fitness evaluator with all required parameters.

        Parameters
        ----------
        scene : SceneConfig
            Immutable scene data (wavelength grid, emissivity, atmosphere, temperature).
        parameters_to_curves : callable
            Function that converts parameter tuples to basis function curves.
            Input: List[Tuple[float, ...]] -> Output: np.ndarray
            Example: gaussian_parameters_to_unit_amplitude_curves
        params_per_basis_function : int
            Number of parameters per basis function.
            Used to extract parameters from the chromosome.
            Example: 2 for Gaussian (mu, sigma), 3 for Lorentzian (mu, sigma, gamma).
        distance_metric : callable, optional
            Function to compute pairwise distances; default is SAM.
        """
        self.scene = scene
        self.distance_metric = distance_metric
        self.params_per_basis_function = params_per_basis_function
        self.parameters_to_curves = parameters_to_curves

    def fitness_func(
        self, _ga_instance: object, chromosome: np.ndarray, _chromosome_idx: int
    ) -> float:
        """
        Compute fitness for a given chromosome.

        Parameters
        ----------
        ga_instance : object
            GA instance (unused, for PyGAD compatibility)
        chromosome : np.ndarray
            Chromosome genes representing sensor parameters.
            Length should be num_basis_functions * params_per_basis_function
        chromosome_idx : int
            Chromosome index (unused, for PyGAD compatibility)

        Returns
        -------
        float
            Fitness score (min-based dissimilarity score)

        Notes
        -----
        The chromosome is parsed into parameters based on params_per_basis_function.
        For example:
        - If params_per_basis_function=2: [p1, p2, p3, p4, ...] -> [(p1, p2), (p3, p4), ...]
        - If params_per_basis_function=3:
          [p1, p2, p3, p4, p5, p6, ...] -> [(p1, p2, p3), (p4, p5, p6), ...]
        """
        # Extract parameters from chromosome based on params_per_basis_function
        num_genes = len(chromosome)

        if num_genes % self.params_per_basis_function != 0:
            raise ValueError(
                f"Chromosome length ({num_genes}) must be divisible by "
                f"params_per_basis_function ({self.params_per_basis_function})"
            )

        # Convert chromosome array to list of parameter tuples
        # e.g., [mu1, sigma1, mu2, sigma2] -> [(mu1, sigma1), (mu2, sigma2)]
        basis_params = [
            tuple(chromosome[i : i + self.params_per_basis_function])
            for i in range(0, num_genes, self.params_per_basis_function)
        ]

        # Convert parameters to basis function curves
        basis_functions = self.parameters_to_curves(basis_params, self.scene.wavelengths)

        sensor_outputs = simulate_sensor_output(
            wavelengths=self.scene.wavelengths,
            substances_emissivity=self.scene.emissivity_curves,
            basis_functions=basis_functions,
            temperature_k=self.scene.temperature_k,
            atmospheric_distance_ratio=self.scene.atmospheric_distance_ratio,
            air_refractive_index=self.scene.air_refractive_index,
            air_transmittance=self.scene.air_transmittance,
        )

        distance_matrix = compute_distance_matrix(
            sensor_outputs,
            distance_func=self.distance_metric,
            axis=1,  # Compare columns (substances)
        )
        return float(min_based_dissimilarity_score(distance_matrix=distance_matrix))
