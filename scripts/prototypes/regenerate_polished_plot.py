#!/usr/bin/env python3
"""Regenerate the polished elites plot with improved styling without re-running optimization."""

import pickle
import sys
from pathlib import Path


# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from polish_map_elites import plot_polished_elites


def main() -> None:
    """Regenerate plots from saved results."""
    output_dir = Path("outputs/map_elites/hc")
    results_file = output_dir / "map_elites_hc_results.pkl"

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run polish_map_elites.py first to generate results.")
        return

    # Load saved results
    print(f"Loading results from: {results_file}")
    with open(results_file, "rb") as f:
        saved_data = pickle.load(f)

    polished_results = saved_data["polished_results"]
    top_elites_initial = saved_data["top_elites_initial"]
    wavelengths_array = saved_data["wavelengths"]

    print(f"Loaded {len(polished_results)} polished results")

    # Generate top 10 plot with improved styling
    plot_path_top10 = output_dir / "polished_top10_elites_fixed.png"
    print(f"\nGenerating top 10 plot: {plot_path_top10}")
    plot_polished_elites(
        initial_elites=top_elites_initial,
        polished_elites=polished_results,
        wavelengths=wavelengths_array,
        output_path=plot_path_top10,
        top_n=10,
    )

    # Generate top 20 plot with improved styling
    plot_path_top20 = output_dir / "polished_top20_elites_fixed.png"
    print(f"Generating top 20 plot: {plot_path_top20}")
    plot_polished_elites(
        initial_elites=top_elites_initial,
        polished_elites=polished_results,
        wavelengths=wavelengths_array,
        output_path=plot_path_top20,
        top_n=20,
    )

    print("\n✓ Plots regenerated successfully with improved styling!")


if __name__ == "__main__":
    main()
