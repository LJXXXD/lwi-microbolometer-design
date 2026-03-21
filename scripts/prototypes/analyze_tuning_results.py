#!/usr/bin/env python3
"""Analyze GA hyperparameter tuning results."""

import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)


def analyze_results(csv_path: Path, json_path: Path) -> None:
    """Analyze tuning results and print comprehensive insights."""

    # Load data
    df = pd.read_csv(csv_path)
    with open(json_path, "r") as f:
        _ = json.load(f)

    print("=" * 80)
    print("GA HYPERPARAMETER TUNING RESULTS ANALYSIS")
    print("=" * 80)
    print(f"\nTotal configurations tested: {len(df):,}")
    print(f"Timestamp: {csv_path.stem.split('_')[-2:][0]}_{csv_path.stem.split('_')[-1]}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    print(f"Best Fitness Achieved: {df['best_fitness'].max():.4f}")
    print(f"Mean Fitness (all configs): {df['best_fitness'].mean():.4f}")
    print(f"Median Fitness: {df['best_fitness'].median():.4f}")
    print(f"Std Dev: {df['best_fitness'].std():.4f}")
    print(f"25th Percentile: {df['best_fitness'].quantile(0.25):.4f}")
    print(f"75th Percentile: {df['best_fitness'].quantile(0.75):.4f}")

    # Top configurations
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    top10 = df.nlargest(10, "best_fitness")
    for rank, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"\nRank {rank}: Fitness = {row['best_fitness']:.4f}")
        print(f"  Population: {row['sol_per_pop']}, Parents: {row['num_parents_mating']}")
        print(
            f"  Mutation: {row['mutation_probability']:.2f}, Crossover: {row['crossover_probability']:.2f}"
        )
        print(f"  Elitism: {row['keep_elitism']}, Selection: {row['parent_selection_type']}")
        print(f"  Crossover Type: {row['crossover_type']}, K_tournament: {row['K_tournament']}")
        print(
            f"  Niching: {row['niching_enabled']}, Sigma: {row['niching_sigma_share']}, Alpha: {row['niching_alpha']}"
        )
        print(
            f"  Diversity: {row['diversity_score']:.2f}, Convergence Gen: {row['convergence_generation']}"
        )
        print(f"  High-quality solutions: {row['high_quality_solutions']:.0f}")

    # Parameter importance analysis
    print("\n" + "=" * 80)
    print("PARAMETER IMPORTANCE (Top 10 vs All)")
    print("=" * 80)
    top10_configs = df.nlargest(10, "best_fitness")

    for param in [
        "sol_per_pop",
        "num_parents_mating",
        "mutation_probability",
        "crossover_probability",
        "keep_elitism",
    ]:
        top10_mean = top10_configs[param].mean()
        all_mean = df[param].mean()
        top10_mode = (
            top10_configs[param].mode()[0] if len(top10_configs[param].mode()) > 0 else "N/A"
        )
        all_mode = df[param].mode()[0] if len(df[param].mode()) > 0 else "N/A"

        print(f"\n{param}:")
        print(f"  Top 10 mean: {top10_mean:.2f} | All mean: {all_mean:.2f}")
        print(f"  Top 10 mode: {top10_mode} | All mode: {all_mode}")

    # Niching analysis
    print("\n" + "=" * 80)
    print("NICHING EFFECTIVENESS")
    print("=" * 80)
    niching_mask = df["niching_enabled"].astype(bool)
    niching_enabled = df[niching_mask]
    niching_disabled = df[~niching_mask]

    if len(niching_enabled) > 0 and len(niching_disabled) > 0:
        print(f"Niching ENABLED ({len(niching_enabled)} configs):")
        print(f"  Mean fitness: {niching_enabled['best_fitness'].mean():.4f}")
        print(f"  Mean diversity: {niching_enabled['diversity_score'].mean():.4f}")
        print(f"  Best fitness: {niching_enabled['best_fitness'].max():.4f}")

        print(f"\nNiching DISABLED ({len(niching_disabled)} configs):")
        print(f"  Mean fitness: {niching_disabled['best_fitness'].mean():.4f}")
        print(f"  Mean diversity: {niching_disabled['diversity_score'].mean():.4f}")
        print(f"  Best fitness: {niching_disabled['best_fitness'].max():.4f}")

        print("\nDifference:")
        print(
            f"  Fitness: {niching_disabled['best_fitness'].mean() - niching_enabled['best_fitness'].mean():.4f}"
        )
        print(
            f"  Diversity: {niching_enabled['diversity_score'].mean() - niching_disabled['diversity_score'].mean():.4f}"
        )

    # Convergence analysis
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)
    print(f"Mean convergence generation: {df['convergence_generation'].mean():.1f}")
    print(f"Median convergence generation: {df['convergence_generation'].median():.1f}")
    print(f"Fastest convergence: {df['convergence_generation'].min()} generations")
    print(f"Slowest convergence: {df['convergence_generation'].max()} generations")

    # Best configuration details
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION DETAILS")
    print("=" * 80)
    best = df.loc[df["best_fitness"].idxmax()]
    print(f"Fitness: {best['best_fitness']:.4f}")
    print("\nHyperparameters:")
    for col in [
        "sol_per_pop",
        "num_generations",
        "num_parents_mating",
        "mutation_probability",
        "crossover_probability",
        "keep_elitism",
        "parent_selection_type",
        "K_tournament",
        "crossover_type",
        "niching_enabled",
        "niching_sigma_share",
        "niching_alpha",
    ]:
        print(f"  {col}: {best[col]}")
    print("\nPerformance Metrics:")
    print(f"  Mean fitness: {best['mean_fitness']:.4f}")
    print(f"  Diversity score: {best['diversity_score']:.4f}")
    print(f"  Convergence generation: {best['convergence_generation']}")
    print(f"  High-quality solutions: {best['high_quality_solutions']:.0f}")

    # High-quality solutions analysis
    print("\n" + "=" * 80)
    print("HIGH-QUALITY SOLUTIONS (fitness >= 50.0)")
    print("=" * 80)
    high_quality = df[df["high_quality_solutions"] > 0]
    if len(high_quality) > 0:
        print(f"Configurations producing high-quality solutions: {len(high_quality)}")
        print(
            f"Mean high-quality solutions per config: {high_quality['high_quality_solutions'].mean():.1f}"
        )
        print(f"Max high-quality solutions: {high_quality['high_quality_solutions'].max():.0f}")
    else:
        print("No configurations produced high-quality solutions (fitness >= 50.0)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_tuning_results.py <results_directory>")
        print("Example: python analyze_tuning_results.py outputs/tuning/full_grid_search")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    csv_files = list(results_dir.glob("tuning_results_*.csv"))
    json_files = list(results_dir.glob("analysis_*.json"))

    if not csv_files:
        print(f"Error: No CSV files found in {results_dir}")
        sys.exit(1)

    if not json_files:
        print(f"Error: No JSON analysis files found in {results_dir}")
        sys.exit(1)

    # Use most recent files
    csv_file = sorted(csv_files)[-1]
    json_file = sorted(json_files)[-1]

    print(f"Analyzing: {csv_file.name}")
    print(f"Analysis file: {json_file.name}\n")

    analyze_results(csv_file, json_file)
