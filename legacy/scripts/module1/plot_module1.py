#!/usr/bin/env python3
"""Module 1 Plotting Script.

Generates comparison plots:
1. Solution Quality (GA vs. Grid Search) - two-bar chart
2. Computation Time (GA vs. Grid Search) - two-bar chart with log scale
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Plotting style
plt.style.use("default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# Baseline values from v1 Grid Search
V1_GRID_SEARCH_BEST_FITNESS = 47.41
V1_GRID_SEARCH_TIME_SECONDS = 12600.0  # 3.5 hours


def plot_quality_comparison(df_results: pd.DataFrame, output_dir: Path) -> None:
    """Generate solution quality comparison plot.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results DataFrame with best_fitness_score column
    output_dir : Path
        Output directory for saving plots
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate mean fitness from GA runs
    ga_mean_fitness = df_results["best_fitness_score"].mean()

    # Create bars
    categories = ["v1 Grid Search", "Basic GA"]
    values = [V1_GRID_SEARCH_BEST_FITNESS, ga_mean_fitness]
    colors = ["#2E86AB", "#A23B72"]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Formatting
    ax.set_ylabel("Best Differentiability Score", fontsize=14, fontweight="bold")
    ax.set_title("Solution Quality (GA vs. Grid Search)", fontsize=16, fontweight="bold")
    ax.set_ylim([0, max(values) * 1.15])  # Add 15% padding at top
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_file = output_dir / "module1_quality_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Quality comparison plot saved to: {output_file}")
    plt.close()


def plot_efficiency_comparison(df_results: pd.DataFrame, output_dir: Path) -> None:
    """Generate computation time comparison plot.

    Parameters
    ----------
    df_results : pd.DataFrame
        Results DataFrame with total_runtime_seconds column
    output_dir : Path
        Output directory for saving plots
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate mean runtime from GA runs
    ga_mean_runtime = df_results["total_runtime_seconds"].mean()

    # Create bars
    categories = ["v1 Grid Search", "Basic GA"]
    values = [V1_GRID_SEARCH_TIME_SECONDS, ga_mean_runtime]
    colors = ["#2E86AB", "#A23B72"]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels on bars (format time nicely)
    labels = []
    for value in values:
        if value >= 3600:
            hours = value / 3600
            labels.append(f"{hours:.1f}h")
        elif value >= 60:
            minutes = value / 60
            labels.append(f"{minutes:.1f}m")
        else:
            labels.append(f"{value:.1f}s")

    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Formatting
    ax.set_ylabel("Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_title("Computation Time (GA vs. Grid Search)", fontsize=16, fontweight="bold")
    ax.set_yscale("log")  # Log scale for better visualization
    ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")

    plt.tight_layout()
    output_file = output_dir / "module1_efficiency_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Efficiency comparison plot saved to: {output_file}")
    plt.close()


def main() -> None:
    """Generate Module 1 plots."""
    logger.info("=" * 60)
    logger.info("Module 1: Plotting Comparison Results")
    logger.info("=" * 60)

    # Load results
    results_file = Path("module1_baseline_results.csv")
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Please run run_module1_baseline.py first")
        return

    df_results = pd.read_csv(results_file)
    logger.info(f"Loaded {len(df_results)} results from {results_file}")

    # Create output directory
    output_dir = Path("outputs/module1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    logger.info("\nGenerating plots...")
    plot_quality_comparison(df_results, output_dir)
    plot_efficiency_comparison(df_results, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Plotting complete!")
    logger.info(f"Plots saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
