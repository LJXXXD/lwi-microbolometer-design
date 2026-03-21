#!/usr/bin/env python3
"""Population Diversity Analysis Script.

Loads a population pickle file, runs DBSCAN clustering to identify distinct solution families,
and reports the number of distinct families found.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_population_diversity(
    population: np.ndarray,
    eps: float = 2.0,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> dict[str, object]:
    """Analyze population diversity using DBSCAN clustering.

    Parameters
    ----------
    population : np.ndarray
        Population array of shape (n_individuals, n_genes)
    eps : float
        Maximum distance between samples for DBSCAN clustering (default: 2.0)
    min_samples : int
        Minimum number of samples in a neighborhood for DBSCAN (default: 5)
    metric : str
        Distance metric for DBSCAN (default: 'euclidean')

    Returns
    -------
    dict[str, object]
        Analysis results including:
        - n_families: Number of distinct families (clusters)
        - n_noise: Number of noise points (outliers)
        - labels: Cluster labels for each individual
        - cluster_sizes: Sizes of each cluster
    """
    if len(population) < min_samples:
        logger.warning(
            f"Population size ({len(population)}) is smaller than min_samples ({min_samples}). "
            "Adjusting min_samples."
        )
        min_samples = max(2, len(population) // 4)

    # Run DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = clustering.fit_predict(population)

    # Count distinct families (excluding noise points labeled as -1)
    unique_labels = set(labels)
    n_families = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(np.sum(labels == -1))

    # Calculate cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        if label == -1:
            cluster_sizes["noise"] = n_noise
        else:
            cluster_sizes[f"family_{label}"] = int(np.sum(labels == label))

    return {
        "n_families": n_families,
        "n_noise": n_noise,
        "labels": labels,
        "cluster_sizes": cluster_sizes,
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
    }


def main() -> None:
    """Analyze population diversity from pickle file."""
    parser = argparse.ArgumentParser(
        description="Analyze population diversity using DBSCAN clustering to identify solution families."
    )
    parser.add_argument(
        "population_file",
        type=Path,
        help="Path to population pickle file (e.g., niching_pop_run_1.pkl)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=2.0,
        help="DBSCAN eps parameter - maximum distance between samples (default: 2.0)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter - minimum samples in neighborhood (default: 5)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for DBSCAN (default: euclidean)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Population Diversity Analysis")
    logger.info("=" * 60)
    logger.info(f"Population file: {args.population_file}")
    logger.info(
        f"DBSCAN parameters: eps={args.eps}, min_samples={args.min_samples}, metric={args.metric}"
    )

    # Load population
    if not args.population_file.exists():
        logger.error(f"Population file not found: {args.population_file}")
        return

    try:
        with open(args.population_file, "rb") as f:
            population = pickle.load(f)
        logger.info(
            f"Loaded population: {len(population)} individuals, {population.shape[1]} genes"
        )
    except Exception as e:
        logger.error(f"Failed to load population file: {e}")
        return

    # Analyze diversity
    logger.info("\nRunning DBSCAN clustering...")
    results = analyze_population_diversity(
        population, eps=args.eps, min_samples=args.min_samples, metric=args.metric
    )

    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("Analysis Results")
    logger.info("=" * 60)
    logger.info(f"Found {results['n_families']} distinct families")
    logger.info(f"Noise points (outliers): {results['n_noise']}")

    if results["cluster_sizes"]:
        logger.info("\nCluster sizes:")
        for cluster_name, size in sorted(results["cluster_sizes"].items()):
            logger.info(f"  {cluster_name}: {size} individuals")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
