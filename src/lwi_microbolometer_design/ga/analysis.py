"""
Population diversity analysis for optimization results.

This module contains functions for analyzing the diversity and quality of
solutions in optimization populations, including clustering and family analysis.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from lwi_microbolometer_design.analysis import calculate_optimal_pairing_distance
from lwi_microbolometer_design.ga.diversity import compute_population_distance_matrix


@dataclass
class AnalysisConfig:
    """
    Configuration for population diversity analysis.

    **Default values work well for most cases.** You typically only need to customize
    a few key parameters (like `top_n_percentage` or `fitness_threshold_percentage`)
    or use one of the preset configurations:
    - `AnalysisConfig.balanced()` - Default, good for most problems
    - `AnalysisConfig.quick()` - For fast analysis of large populations
    - `AnalysisConfig.thorough()` - For detailed population analysis

    All parameters are tunable to allow flexible analysis customization when needed,
    but defaults provide sensible starting points.

    Parameters
    ----------
    top_n_percentage : float
        Percentage of top solutions to analyze (e.g., 0.5 = top 50%).
        Default: 0.5
    min_top_n : int
        Minimum number of solutions to analyze regardless of percentage.
        Default: 10
    max_top_n : int
        Maximum number of solutions to analyze regardless of percentage.
        Default: 1000
    fitness_threshold_percentage : float
        Percentage of fitness range to use as threshold for high-quality solutions
        (e.g., 0.1 = top 10% of fitness range).
        Default: 0.1
    segment_percentages : list[float]
        List of percentages for segment analysis (e.g., [0.1, 0.25, 0.5] for
        top 10%, 25%, 50% segments).
        Default: [0.1, 0.25, 0.5]
    use_adaptive_clustering : bool
        Whether to use adaptive clustering methods.
        Default: True
    min_top_n_for_analysis : int
        Minimum number of solutions required for clustering analysis.
        Default: 2
    min_solutions_for_clustering : int
        Minimum number of solutions required for clustering operations.
        Default: 2
    min_inertias_for_elbow : int
        Minimum number of inertia values required for elbow method.
        Default: 3
    default_cluster_count : int
        Default number of clusters when elbow method cannot be applied.
        Default: 2
    min_cluster_count : int
        Minimum number of clusters allowed.
        Default: 2
    max_cluster_count_threshold : int
        Maximum number of clusters before warning about too many families.
        Default: 5
    low_diversity_threshold : float
        Threshold below which diversity is considered low (for recommendations).
        Default: 0.1
    """

    # Top N analysis parameters
    top_n_percentage: float = 0.5
    min_top_n: int = 10
    max_top_n: int = 1000

    # Fitness threshold parameters
    fitness_threshold_percentage: float = 0.1

    # Segment analysis parameters
    segment_percentages: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])

    # Clustering parameters
    use_adaptive_clustering: bool = True
    min_top_n_for_analysis: int = 2
    min_solutions_for_clustering: int = 2
    min_inertias_for_elbow: int = 3
    default_cluster_count: int = 2
    min_cluster_count: int = 2
    max_cluster_count_threshold: int = 5
    low_diversity_threshold: float = 0.1

    @classmethod
    def quick(cls) -> "AnalysisConfig":
        """
        Create a quick analysis configuration for fast analysis of large populations.

        Features:
        - Analyzes smaller top percentage (top 20%)
        - Fewer segments
        - Faster clustering thresholds
        - Good for quick insights on large populations

        Returns
        -------
        AnalysisConfig
            Quick configuration instance
        """
        return cls(
            top_n_percentage=0.2,
            min_top_n=5,
            max_top_n=500,
            fitness_threshold_percentage=0.05,
            segment_percentages=[0.05, 0.1, 0.2],
            max_cluster_count_threshold=3,
        )

    @classmethod
    def thorough(cls) -> "AnalysisConfig":
        """
        Create a thorough analysis configuration for detailed population analysis.

        Features:
        - Analyzes larger top percentage (top 75%)
        - More segments
        - More sensitive clustering
        - Good for detailed understanding of population structure

        Returns
        -------
        AnalysisConfig
            Thorough configuration instance
        """
        return cls(
            top_n_percentage=0.75,
            min_top_n=20,
            max_top_n=2000,
            fitness_threshold_percentage=0.15,
            segment_percentages=[0.05, 0.1, 0.25, 0.5, 0.75],
            max_cluster_count_threshold=10,
            low_diversity_threshold=0.05,
        )

    @classmethod
    def balanced(cls) -> "AnalysisConfig":
        """
        Create a balanced analysis configuration (same as default).

        This is the recommended starting point for most cases. Provides a good
        balance between analysis depth and speed.

        Returns
        -------
        AnalysisConfig
            Balanced configuration instance (default values)
        """
        return cls()


# Default configuration instance (balanced)
DEFAULT_ANALYSIS_CONFIG = AnalysisConfig.balanced()


def analyze_population_diversity(
    final_population: list[Any],
    top_n: int | None = None,
    fitness_threshold: float | None = None,
    clustering_radius: float | None = None,
    distance_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    niching_config: Any | None = None,
    analysis_config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """
    Analyzes population diversity with autonomous default analysis plus optional custom analysis.

    This function ALWAYS performs a comprehensive default analysis using sensible autonomous
    parameters (top 50% of population, top 10% fitness threshold, adaptive clustering).
    Additionally, if the user provides specific parameters, it performs a SECOND analysis
    with those custom parameters.

    The function returns both analyses, ensuring users always get meaningful insights
    while allowing customization for specific needs.

    Parameters
    ----------
    final_population : List[Any]
        The list of solution objects in the final generation. Each solution should have
        'genes' (parameter array) and 'fitness' attributes.
    top_n : int, optional
        Number of top solutions to analyze in CUSTOM analysis. If None, only default analysis.
        Default: None (only default analysis)
    fitness_threshold : float, optional
        Minimum fitness for high-quality solutions in CUSTOM analysis.
        If None, only default analysis.
        Default: None (only default analysis)
    clustering_radius : float, optional
        Distance radius for clustering in CUSTOM analysis. If None, only default analysis.
        Default: None (only default analysis)
    distance_func : Callable[[np.ndarray, np.ndarray], float], optional
        Distance function for parameter comparison. If None and niching_config
        is provided with optimal pairing enabled, uses optimal pairing mode automatically.
        Default: None (uses standard Euclidean if no niching_config)
    niching_config : Optional[Any], optional
        Niching configuration from AdvancedGA. If provided, population spread metrics
        will use the same distance calculation as the GA (respects optimal pairing).
        Default: None (uses standard Euclidean distance)
    analysis_config : AnalysisConfig, optional
        Configuration object for analysis parameters. If None, uses DEFAULT_ANALYSIS_CONFIG.
        Allows full customization of all analysis thresholds and parameters.
        Default: None (uses default configuration)

    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis report containing:
        - population_summary: Overall population statistics
        - default_analysis: Autonomous analysis with sensible defaults
        - custom_analysis: User-specified analysis (if parameters provided)
    """
    if not final_population:
        return _empty_analysis_report()

    # Use provided config or default
    config = analysis_config if analysis_config is not None else DEFAULT_ANALYSIS_CONFIG

    # Sort solutions by fitness (descending)
    final_population.sort(key=lambda x: x.fitness, reverse=True)
    population_size = len(final_population)

    # Extract fitness values and population genes
    fitness_values = np.array([sol.fitness for sol in final_population])
    # population_genes: 2D array where rows=individuals, columns=gene positions
    population_genes = np.array([sol.genes for sol in final_population])

    # Always perform default autonomous analysis
    default_params = _extend_analysis_config_with_calculated_values(
        fitness_values, population_size, config
    )

    default_segment_analyses = _analyze_fitness_segments(
        final_population, fitness_values, default_params, config
    )

    # Determine distance function for clustering
    clustering_distance_func = _determine_clustering_distance_func(distance_func, niching_config)

    default_clustering_results = _perform_clustering_analysis(
        final_population, population_genes, default_params, clustering_distance_func, config
    )

    default_diversity_metrics = _calculate_diversity_metrics(
        population_genes, fitness_values, default_params, niching_config
    )

    default_recommendations = _generate_recommendations(
        default_segment_analyses, default_clustering_results, default_diversity_metrics, config
    )

    # Additionally perform user-specified analysis if parameters provided
    custom_analysis = None
    if top_n is not None or fitness_threshold is not None or clustering_radius is not None:
        custom_params = _determine_custom_analysis_parameters(
            fitness_values, population_size, top_n, fitness_threshold, clustering_radius, config
        )

        custom_segment_analyses = _analyze_fitness_segments(
            final_population, fitness_values, custom_params, config
        )

        custom_clustering_results = _perform_clustering_analysis(
            final_population, population_genes, custom_params, clustering_distance_func, config
        )

        custom_diversity_metrics = _calculate_diversity_metrics(
            population_genes, fitness_values, custom_params, niching_config
        )

        custom_recommendations = _generate_recommendations(
            custom_segment_analyses, custom_clustering_results, custom_diversity_metrics, config
        )

        custom_analysis = {
            "analysis_parameters": custom_params,
            "fitness_segments": custom_segment_analyses,
            "clustering_analysis": custom_clustering_results,
            "diversity_metrics": custom_diversity_metrics,
            "recommendations": custom_recommendations,
        }

    # Compile comprehensive report
    report = {
        "population_summary": {
            "total_solutions": population_size,
            "fitness_range": {
                "min": float(np.min(fitness_values)),
                "max": float(np.max(fitness_values)),
                "mean": float(np.mean(fitness_values)),
                "std": float(np.std(fitness_values)),
            },
        },
        "default_analysis": {
            "analysis_parameters": default_params,
            "fitness_segments": default_segment_analyses,
            "clustering_analysis": default_clustering_results,
            "diversity_metrics": default_diversity_metrics,
            "recommendations": default_recommendations,
        },
    }

    # Add custom analysis if performed
    if custom_analysis is not None:
        report["custom_analysis"] = custom_analysis

    return report


# Helper functions for configuration and analysis


def _calculate_top_n(population_size: int, percentage: float, min_n: int, max_n: int) -> int:
    """Calculate top_n with percentage and bounds."""
    return max(min_n, min(max_n, int(population_size * percentage)))


def _extend_analysis_config_with_calculated_values(
    fitness_values: np.ndarray, population_size: int, analysis_config: AnalysisConfig
) -> dict[str, Any]:
    """
    Extend analysis config with calculated values based on population data.

    Takes the base configuration and adds calculated fields like actual top_n values,
    fitness thresholds, etc. that depend on the specific population.
    """
    params: dict[str, Any] = {}

    # Add calculated top_n
    params["top_n"] = _calculate_top_n(
        population_size,
        analysis_config.top_n_percentage,
        analysis_config.min_top_n,
        analysis_config.max_top_n,
    )

    # Add calculated fitness threshold
    fitness_range = np.max(fitness_values) - np.min(fitness_values)
    params["fitness_threshold"] = (
        np.max(fitness_values) - analysis_config.fitness_threshold_percentage * fitness_range
    )

    # Add clustering parameters
    params["clustering_radius"] = None  # Use adaptive clustering
    params["segment_percentages"] = analysis_config.segment_percentages

    return params


def _determine_custom_analysis_parameters(
    fitness_values: np.ndarray,
    population_size: int,
    top_n: int | None,
    fitness_threshold: float | None,
    clustering_radius: float | None,
    analysis_config: AnalysisConfig,
) -> dict[str, Any]:
    """Determine custom analysis parameters based on user specifications."""
    params: dict[str, Any] = {}

    # Use user-specified top_n or default to percentage-based calculation
    if top_n is not None:
        params["top_n"] = min(top_n, population_size)
    else:
        params["top_n"] = _calculate_top_n(
            population_size,
            analysis_config.top_n_percentage,
            analysis_config.min_top_n,
            analysis_config.max_top_n,
        )

    # Use user-specified fitness threshold or auto-determine
    if fitness_threshold is not None:
        params["fitness_threshold"] = fitness_threshold
    else:
        fitness_range = np.max(fitness_values) - np.min(fitness_values)
        params["fitness_threshold"] = (
            np.max(fitness_values) - analysis_config.fitness_threshold_percentage * fitness_range
        )

    # Use user-specified clustering radius or adaptive clustering
    if clustering_radius is not None:
        params["clustering_radius"] = clustering_radius
        params["use_adaptive_clustering"] = False
    else:
        params["clustering_radius"] = None
        params["use_adaptive_clustering"] = analysis_config.use_adaptive_clustering

    params["segment_percentages"] = analysis_config.segment_percentages

    return params


def _determine_clustering_distance_func(
    distance_func: Callable | None,
    niching_config: Any | None,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Determine distance function for clustering based on user input and niching config.

    If distance_func is provided, use it. Otherwise, check niching_config for
    optimal pairing settings and create appropriate distance function.
    """
    if distance_func is not None:
        return distance_func

    # Check if niching_config suggests optimal pairing
    if niching_config and niching_config.enabled and niching_config.use_optimal_pairing:
        # Create optimal pairing distance function
        params_per_group = niching_config.params_per_group
        metric = niching_config.optimal_pairing_metric or "euclidean"

        def optimal_pairing_dist_func(array_a: np.ndarray, array_b: np.ndarray) -> float:
            """Reshape arrays and use optimal pairing."""
            # Reshape into groups
            groups_a = array_a.reshape(-1, params_per_group)
            groups_b = array_b.reshape(-1, params_per_group)

            # Convert to list of tuples
            items_a = [tuple(group) for group in groups_a]
            items_b = [tuple(group) for group in groups_b]

            # Compute optimal pairing distance
            return calculate_optimal_pairing_distance(items_a, items_b, metric=metric)

        return optimal_pairing_dist_func

    # Default: Euclidean distance
    # Wrap euclidean to ensure proper type signature
    def euclidean_wrapper(array_a: np.ndarray, array_b: np.ndarray) -> float:
        """Wrap scipy euclidean distance."""
        return float(euclidean(array_a, array_b))

    return euclidean_wrapper


def _empty_analysis_report() -> dict[str, Any]:
    """Return an empty analysis report for empty populations."""
    return {
        "population_summary": {
            "total_solutions": 0,
            "fitness_range": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            "analysis_parameters": {},
        },
        "fitness_segments": {},
        "clustering_analysis": {"clusters": [], "method": "none"},
        "diversity_metrics": {},
        "recommendations": ["No solutions to analyze"],
    }


# Analysis functions


def _analyze_fitness_segments(
    final_population: list[Any],
    fitness_values: np.ndarray,
    analysis_params: dict[str, Any],
    _analysis_config: AnalysisConfig,
) -> dict[str, Any]:
    """Analyze diversity across different fitness segments of the population."""
    segments = {}

    for percentage in analysis_params["segment_percentages"]:
        segment_size = int(len(final_population) * percentage)
        if segment_size == 0:
            continue

        segment_solutions = final_population[:segment_size]
        segment_fitness = fitness_values[:segment_size]

        # Calculate diversity metrics for this segment
        segment_params = np.array([sol.genes for sol in segment_solutions])

        # Parameter space diversity (standard deviation across dimensions)
        param_diversity = np.std(segment_params, axis=0)

        # Fitness diversity
        fitness_diversity = np.std(segment_fitness)

        segments[f"top_{int(percentage * 100)}_percent"] = {
            "size": segment_size,
            "fitness_range": {
                "min": float(np.min(segment_fitness)),
                "max": float(np.max(segment_fitness)),
                "mean": float(np.mean(segment_fitness)),
                "std": float(fitness_diversity),
            },
            "parameter_diversity": {
                "mean_std": float(np.mean(param_diversity)),
                "max_std": float(np.max(param_diversity)),
                "min_std": float(np.min(param_diversity)),
            },
            "solutions": segment_solutions,
        }

    return segments


def _perform_clustering_analysis(
    final_population: list[Any],
    population_genes: np.ndarray,
    analysis_params: dict[str, Any],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    analysis_config: AnalysisConfig,
) -> dict[str, Any]:
    """Perform clustering analysis using adaptive methods."""
    top_n = analysis_params["top_n"]
    top_solutions = final_population[:top_n]
    top_genes = population_genes[:top_n]

    if len(top_solutions) < analysis_config.min_top_n_for_analysis:
        return {
            "method": "insufficient_data",
            "clusters": [],
            "cluster_count": 0,
            "silhouette_score": 0.0,
        }

    # Standardize parameters for clustering
    scaler = StandardScaler()
    scaled_genes = scaler.fit_transform(top_genes)

    # Try different clustering methods
    clustering_results = {}

    # Method 1: K-means with optimal k
    optimal_k = _find_optimal_k(scaled_genes, min(10, len(top_solutions) // 2), analysis_config)
    if optimal_k > 1:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(scaled_genes)
        clustering_results["kmeans"] = {
            "labels": kmeans_labels,
            "n_clusters": optimal_k,
            "inertia": kmeans.inertia_,
        }

    # Method 2: DBSCAN with adaptive parameters
    dbscan_results = _adaptive_dbscan(scaled_genes)
    if dbscan_results["n_clusters"] > 0:
        clustering_results["dbscan"] = dbscan_results

    # Choose best clustering method
    best_method = _select_best_clustering(clustering_results, scaled_genes)

    if best_method is None:
        return {
            "method": "no_clusters_found",
            "clusters": [],
            "cluster_count": 0,
            "silhouette_score": 0.0,
        }

    # Generate cluster reports
    labels = clustering_results[best_method]["labels"]
    cluster_reports = _generate_cluster_reports(top_solutions, labels, distance_func)

    return {
        "method": best_method,
        "clusters": cluster_reports,
        "cluster_count": len(cluster_reports),
        "silhouette_score": clustering_results[best_method].get("silhouette_score", 0.0),
    }


def _calculate_diversity_metrics(
    population_genes: np.ndarray,
    fitness_values: np.ndarray,
    _analysis_params: dict[str, Any],
    niching_config: Any | None = None,
) -> dict[str, Any]:
    """
    Calculate quantitative diversity metrics for the population.

    Uses unified diversity calculation that respects GA's niching configuration
    for population spread metrics, ensuring consistency with GA behavior.
    """
    metrics = {}

    # Parameter space diversity (independent of distance metric)
    gene_std = np.std(population_genes, axis=0)
    metrics["parameter_diversity"] = {
        "mean_std": float(np.mean(gene_std)),
        "max_std": float(np.max(gene_std)),
        "min_std": float(np.min(gene_std)),
        "total_variance": float(np.sum(np.var(population_genes, axis=0))),
    }

    # Fitness diversity (independent of distance metric)
    metrics["fitness_diversity"] = {
        "std": float(np.std(fitness_values)),
        "range": float(np.max(fitness_values) - np.min(fitness_values)),
        "coefficient_of_variation": float(np.std(fitness_values) / np.mean(fitness_values)),
    }

    # Population spread (uses unified diversity calculation that respects niching config)
    if len(population_genes) > 1:
        # Use unified distance matrix calculation (respects optimal pairing if configured)
        distance_matrix = compute_population_distance_matrix(population_genes, niching_config)

        # Extract upper triangle (excluding diagonal)
        n = len(population_genes)
        upper_triangle_indices = np.triu_indices(n, k=1)
        distances = distance_matrix[upper_triangle_indices]

        metrics["population_spread"] = {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
        }

    return metrics


def _generate_recommendations(
    segment_analyses: dict[str, Any],
    clustering_results: dict[str, Any],
    diversity_metrics: dict[str, Any],
    analysis_config: AnalysisConfig,
) -> list[str]:
    """Generate actionable recommendations based on analysis results."""
    recommendations = []

    # Analyze clustering results
    cluster_count = clustering_results.get("cluster_count", 0)
    if cluster_count == 0:
        recommendations.append(
            "No distinct solution families found - consider increasing population diversity"
        )
    elif cluster_count == 1:
        recommendations.append(
            "All solutions belong to single family - population may lack diversity"
        )
    elif cluster_count > analysis_config.max_cluster_count_threshold:
        recommendations.append(
            "Many solution families found - consider focusing on top-performing families"
        )

    # Analyze fitness segments
    top_10_segment = segment_analyses.get("top_10_percent", {})
    if top_10_segment:
        fitness_std = top_10_segment["fitness_range"]["std"]
        if fitness_std < analysis_config.low_diversity_threshold:
            recommendations.append(
                "Top solutions have very similar fitness - "
                "consider exploring different solution approaches"
            )

    # Analyze parameter diversity
    param_diversity = diversity_metrics.get("parameter_diversity", {})
    if param_diversity.get("mean_std", 0) < analysis_config.low_diversity_threshold:
        recommendations.append(
            "Low parameter diversity - consider increasing mutation rates or population size"
        )

    # Analyze population spread
    population_spread = diversity_metrics.get("population_spread", {})
    if population_spread:
        mean_dist = population_spread.get("mean_distance", 0)
        std_dist = population_spread.get("std_distance", 0)
        if mean_dist > 0 and std_dist / mean_dist < analysis_config.low_diversity_threshold:
            recommendations.append(
                "Solutions are clustered tightly - consider increasing exploration"
            )

    if not recommendations:
        recommendations.append("Population shows good diversity and structure")

    return recommendations


# Clustering helper functions


def _find_optimal_k(data: np.ndarray, max_k: int, analysis_config: AnalysisConfig) -> int:
    """Find optimal number of clusters using elbow method."""
    if max_k < analysis_config.min_cluster_count:
        return 1

    inertias = []
    k_range = range(1, min(max_k + 1, len(data)))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Simple elbow detection
    if len(inertias) < analysis_config.min_inertias_for_elbow:
        return analysis_config.default_cluster_count

    # Find elbow point
    # Convert inertias list to array
    inertias_array = np.array(inertias, dtype=np.float64)
    diffs = np.diff(inertias_array)
    second_diffs = np.diff(diffs)
    elbow_idx = int(np.argmax(second_diffs)) + 1

    result = max(analysis_config.min_cluster_count, min(elbow_idx + 1, max_k))
    return int(result)


def _adaptive_dbscan(data: np.ndarray) -> dict[str, Any]:
    """Perform DBSCAN clustering with adaptive parameter selection."""
    # Try different eps values
    eps_values = np.linspace(0.1, 2.0, 10)
    best_score = -1
    best_labels = None
    best_eps = None

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(data)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_eps = eps

    return {
        "labels": best_labels if best_labels is not None else np.zeros(len(data)),
        "n_clusters": len(set(best_labels)) - (1 if -1 in best_labels else 0)
        if best_labels is not None
        else 0,
        "eps": best_eps,
        "silhouette_score": best_score,
    }


def _select_best_clustering(clustering_results: dict[str, Any], _data: np.ndarray) -> str | None:
    """Select the best clustering method based on silhouette score."""
    if not clustering_results:
        return None

    best_method = None
    best_score = -1

    for method, results in clustering_results.items():
        if "silhouette_score" in results and results["silhouette_score"] > best_score:
            best_score = results["silhouette_score"]
            best_method = method

    return best_method


def _generate_cluster_reports(
    solutions: list[Any],
    labels: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
) -> list[dict[str, Any]]:
    """Generate detailed reports for each cluster."""
    unique_labels = np.unique(labels)
    cluster_reports = []

    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue

        cluster_mask = labels == cluster_id
        cluster_solutions = [sol for i, sol in enumerate(solutions) if cluster_mask[i]]

        if not cluster_solutions:
            continue

        # Find representative solution (highest fitness)
        cluster_solutions.sort(key=lambda x: x.fitness, reverse=True)
        representative = cluster_solutions[0]

        # Calculate intra-cluster distances
        cluster_params = np.array([sol.genes for sol in cluster_solutions])
        if len(cluster_params) > 1:
            distances = []
            for i in range(len(cluster_params)):
                for j in range(i + 1, len(cluster_params)):
                    dist = distance_func(cluster_params[i], cluster_params[j])
                    distances.append(dist)
            distances_array = (
                np.array(distances, dtype=np.float64)
                if distances
                else np.array([0.0], dtype=np.float64)
            )
            avg_distance = float(np.mean(distances_array))
        else:
            avg_distance = 0.0

        cluster_reports.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(cluster_solutions),
                "representative_fitness": representative.fitness,
                "representative_parameters": representative.genes.tolist(),
                "fitness_range": {
                    "min": min(sol.fitness for sol in cluster_solutions),
                    "max": max(sol.fitness for sol in cluster_solutions),
                    "mean": float(
                        np.mean(
                            np.array([sol.fitness for sol in cluster_solutions], dtype=np.float64)
                        )
                    ),
                },
                "avg_intra_cluster_distance": float(avg_distance),
                "solutions": cluster_solutions,
            }
        )

    # Sort by representative fitness
    cluster_reports.sort(key=lambda x: x["representative_fitness"], reverse=True)

    return cluster_reports
