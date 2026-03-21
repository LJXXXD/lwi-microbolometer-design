#!/usr/bin/env python3
"""Module 2 Plotting Script.

Generates visualizations for multimodal optimization analysis:
1. t-SNE scatter plots comparing Basic and Niching GA populations
2. Representative designs plot showing 4-channel Gaussian curves for cluster medoids
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from lwi_microbolometer_design.analysis.distance_metrics import spectral_angle_mapper
from lwi_microbolometer_design.data import load_substance_atmosphere_data
from lwi_microbolometer_design.ga import MinDissimilarityFitnessEvaluator
from lwi_microbolometer_design.simulation.gaussian_parameter_to_curves import (
    gaussian_parameters_to_unit_amplitude_curves,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Plotting style
plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def load_population(filepath: Path) -> np.ndarray:
    """Load population from pickle file.

    Parameters
    ----------
    filepath : Path
        Path to pickle file

    Returns
    -------
    np.ndarray
        Population array
    """
    with open(filepath, "rb") as f:
        population = pickle.load(f)
    return population


def compute_tsne_embedding(
    population: np.ndarray, n_components: int = 2, random_state: int = 42
) -> np.ndarray:
    """Compute t-SNE embedding for population visualization.

    Parameters
    ----------
    population : np.ndarray
        Population array of shape (n_individuals, n_genes)
    n_components : int
        Number of dimensions for t-SNE (default: 2)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    np.ndarray
        t-SNE embedding of shape (n_individuals, n_components)
    """
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=30, n_iter=1000)
    embedding = tsne.fit_transform(population)
    return embedding


def find_cluster_medoids(
    population: np.ndarray,
    labels: np.ndarray,
    fitness_evaluator: MinDissimilarityFitnessEvaluator,
) -> dict[int, tuple[int, np.ndarray]]:
    """Find the best-scoring member (medoid) of each cluster.

    Parameters
    ----------
    population : np.ndarray
        Population array
    labels : np.ndarray
        Cluster labels from DBSCAN
    fitness_evaluator : MinDissimilarityFitnessEvaluator
        Fitness evaluator to score chromosomes

    Returns
    -------
    dict[int, tuple[int, np.ndarray]]
        Dictionary mapping cluster label to (index, chromosome) of best-scoring member
    """
    medoids = {}
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue

        # Get all members of this cluster
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]
        cluster_members = population[cluster_mask]

        # Calculate fitness for all cluster members
        fitness_scores = []
        for idx, chromosome in zip(cluster_indices, cluster_members):
            try:
                fitness = fitness_evaluator.fitness_func(None, chromosome, idx)
                fitness_scores.append((idx, fitness))
            except Exception as e:
                logger.warning(f"Error computing fitness for chromosome {idx}: {e}")
                continue

        if not fitness_scores:
            continue

        # Find best-scoring member (highest fitness)
        best_idx, best_fitness = max(fitness_scores, key=lambda x: x[1])
        medoids[label] = (best_idx, population[best_idx])
        logger.info(f"Cluster {label}: best fitness = {best_fitness:.2f}")

    return medoids


def plot_tsne_comparison(
    basic_population: np.ndarray,
    niching_population: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate 2-panel t-SNE scatter plot comparing Basic and Niching populations.

    Parameters
    ----------
    basic_population : np.ndarray
        Basic GA population
    niching_population : np.ndarray
        Niching GA population
    output_dir : Path
        Output directory for saving plots
    """
    logger.info("Computing t-SNE embeddings...")
    basic_embedding = compute_tsne_embedding(basic_population)
    niching_embedding = compute_tsne_embedding(niching_population)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel (a): Basic GA
    ax = axes[0]
    ax.scatter(
        basic_embedding[:, 0],
        basic_embedding[:, 1],
        alpha=0.6,
        s=20,
        c="#2E86AB",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_title("(a) t-SNE of Basic GA Population", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")

    # Panel (b): Niching GA
    ax = axes[1]
    ax.scatter(
        niching_embedding[:, 0],
        niching_embedding[:, 1],
        alpha=0.6,
        s=20,
        c="#A23B72",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_title("(b) t-SNE of Niching GA Population", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_file = output_dir / "module2_tsne_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"t-SNE comparison plot saved to: {output_file}")
    plt.close()


def plot_representative_designs(
    niching_population: np.ndarray,
    wavelengths: np.ndarray,
    fitness_evaluator: MinDissimilarityFitnessEvaluator,
    output_dir: Path,
    eps: float = 2.0,
    min_samples: int = 5,
) -> None:
    """Generate plot showing 4-channel Gaussian curves for cluster medoids.

    Parameters
    ----------
    niching_population : np.ndarray
        Niching GA population
    wavelengths : np.ndarray
        Wavelength array for plotting curves
    fitness_evaluator : MinDissimilarityFitnessEvaluator
        Fitness evaluator to score chromosomes
    output_dir : Path
        Output directory for saving plots
    eps : float
        DBSCAN eps parameter (default: 2.0)
    min_samples : int
        DBSCAN min_samples parameter (default: 5)
    """
    logger.info("Running DBSCAN clustering...")
    if len(niching_population) < min_samples:
        min_samples = max(2, len(niching_population) // 4)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clustering.fit_predict(niching_population)

    logger.info("Finding cluster medoids...")
    medoids = find_cluster_medoids(niching_population, labels, fitness_evaluator)

    if not medoids:
        logger.warning("No clusters found. Cannot generate representative designs plot.")
        return

    logger.info(f"Found {len(medoids)} clusters with medoids")

    # Generate plot
    num_clusters = len(medoids)
    fig, axes = plt.subplots(1, num_clusters, figsize=(5 * num_clusters, 6))
    if num_clusters == 1:
        axes = [axes]

    for idx, (cluster_label, (medoid_idx, medoid_chromosome)) in enumerate(sorted(medoids.items())):
        ax = axes[idx]

        # Convert chromosome to Gaussian parameters
        num_genes = len(medoid_chromosome)
        params_per_basis_function = 2  # mu, sigma
        num_basis_functions = num_genes // params_per_basis_function

        gaussian_params = [
            tuple(medoid_chromosome[i : i + params_per_basis_function])
            for i in range(0, num_genes, params_per_basis_function)
        ]

        # Generate curves
        curves = gaussian_parameters_to_unit_amplitude_curves(gaussian_params, wavelengths)

        # Plot each channel
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for channel_idx in range(num_basis_functions):
            ax.plot(
                wavelengths,
                curves[:, channel_idx],
                label=f"Channel {channel_idx + 1}",
                color=colors[channel_idx % len(colors)],
                linewidth=2,
            )

        # Formatting
        ax.set_xlabel("Wavelength (µm)", fontsize=12)
        ax.set_ylabel("Response", fontsize=12)
        ax.set_title(f"Cluster {cluster_label} Medoid", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_ylim([0, 1.1])

    plt.tight_layout()
    output_file = output_dir / "module2_representative_designs.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Representative designs plot saved to: {output_file}")
    plt.close()


def main() -> None:
    """Generate Module 2 plots."""
    parser = argparse.ArgumentParser(description="Generate Module 2 visualization plots.")
    parser.add_argument(
        "--basic-pop-file",
        type=Path,
        default=Path("outputs/module2/populations/basic_pop_run_1.pkl"),
        help="Path to Basic GA population pickle file (default: outputs/module2/populations/basic_pop_run_1.pkl)",
    )
    parser.add_argument(
        "--niching-pop-file",
        type=Path,
        default=Path("outputs/module2/populations/niching_pop_run_1.pkl"),
        help="Path to Niching GA population pickle file (default: outputs/module2/populations/niching_pop_run_1.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/module2"),
        help="Output directory for plots (default: outputs/module2)",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=2.0,
        help="DBSCAN eps parameter for clustering (default: 2.0)",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter (default: 5)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Module 2: Plotting Multimodal Optimization Results")
    logger.info("=" * 60)

    # Load populations
    logger.info("\n=== Loading Populations ===")
    if not args.basic_pop_file.exists():
        logger.error(f"Basic GA population file not found: {args.basic_pop_file}")
        return
    if not args.niching_pop_file.exists():
        logger.error(f"Niching GA population file not found: {args.niching_pop_file}")
        return

    basic_population = load_population(args.basic_pop_file)
    niching_population = load_population(args.niching_pop_file)
    logger.info(f"Loaded Basic GA population: {len(basic_population)} individuals")
    logger.info(f"Loaded Niching GA population: {len(niching_population)} individuals")

    # Load data for fitness evaluation (needed for representative designs)
    logger.info("\n=== Loading Data for Fitness Evaluation ===")
    spectral_data_file = Path("data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")
    air_transmittance_file = Path("data/Test 3 - 4 White Powers/Air transmittance.xlsx")
    atmospheric_distance_ratio = 0.11
    temperature_kelvin = 293.15
    air_refractive_index = 1.0

    try:
        data = load_substance_atmosphere_data(
            spectral_data_file=spectral_data_file,
            air_transmittance_file=air_transmittance_file,
            atmospheric_distance_ratio=atmospheric_distance_ratio,
            temperature_kelvin=temperature_kelvin,
            air_refractive_index=air_refractive_index,
        )
        logger.info("✓ Data loaded successfully.")

        # Create fitness evaluator
        fitness_evaluator = MinDissimilarityFitnessEvaluator(
            wavelengths=data["wavelengths"],
            emissivity_curves=data["emissivity_curves"],
            temperature_K=data["temperature_K"],
            atmospheric_distance_ratio=data["atmospheric_distance_ratio"],
            air_refractive_index=data["air_refractive_index"],
            air_transmittance=data["air_transmittance"],
            parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
            params_per_basis_function=2,
            distance_metric=spectral_angle_mapper,
        )
        wavelengths = data["wavelengths"]
        if wavelengths.ndim > 1:
            wavelengths = wavelengths.flatten()

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    logger.info("\n=== Generating Plots ===")
    plot_tsne_comparison(basic_population, niching_population, args.output_dir)
    plot_representative_designs(
        niching_population,
        wavelengths,
        fitness_evaluator,
        args.output_dir,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Plotting complete!")
    logger.info(f"Plots saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
