#!/usr/bin/env python3
"""Phase 1 Environmental Robustness Test for optimised sensor designs.

Loads existing optimisation results (MAP-Elites archive or GA output) and
evaluates top solutions under a grid of environmental conditions
(temperature x atmospheric distance).  Produces diagnostic plots that
answer: "How badly does fitness degrade when we deviate from nominal
conditions?"

Usage
-----
Run from the project root::

    python scripts/robustness/run_robustness_test.py

Outputs are written to ``outputs/robustness/``.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.analysis.robustness import (
    evaluate_archive_robustness,
    summarise_robustness,
)
from lwi_microbolometer_design.data import load_substance_atmosphere_data
from lwi_microbolometer_design.visualization.robustness_visualization import (
    plot_condition_sensitivity,
    plot_fitness_degradation_heatmap,
    plot_fitness_distribution_by_condition,
    plot_retention_histogram,
    plot_worst_case_vs_nominal,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPECTRAL_DATA_FILE = Path("data/Test 3 - 4 White Powers/white_powders_with_labels.xlsx")
AIR_TRANSMITTANCE_FILE = Path("data/Test 3 - 4 White Powers/Air transmittance.xlsx")

TEMPERATURES_K = [273.15, 283.15, 293.15, 303.15, 313.15]
DISTANCE_RATIOS = [0.05, 0.08, 0.11, 0.15, 0.20]
REFRACTIVE_INDICES = [1.0]

NOMINAL_TEMPERATURE_K = 293.15
NOMINAL_DISTANCE_RATIO = 0.11
NOMINAL_REFRACTIVE_INDEX = 1.0

NUM_PARAMS_PER_BASIS_FUNCTION = 2
TOP_N_ELITES = 50


def _load_archive(archive_path: Path) -> dict:
    """Load a pickled MAP-Elites archive."""
    logger.info("Loading archive from %s", archive_path)
    with archive_path.open("rb") as f:
        archive = pickle.load(f)  # noqa: S301
    logger.info("Archive contains %d elites", len(archive))
    return archive


def _find_best_archive() -> Path:
    """Locate the best available archive in outputs/map_elites/."""
    candidates = [
        Path("outputs/map_elites/cma_me/cma_me_archive.pkl"),
        Path("outputs/map_elites/raw/map_elites_raw_archive.pkl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    logger.error("No MAP-Elites archive found. Run a MAP-Elites script first.")
    sys.exit(1)


def main() -> None:
    """Run Phase 1 environmental robustness test."""
    print("=" * 80)
    print("PHASE 1: ENVIRONMENTAL ROBUSTNESS TEST")
    print("=" * 80)
    print("Evaluating how elite sensor designs perform under varying")
    print("temperature and atmospheric conditions.\n")

    # ------------------------------------------------------------------
    # [1/5] Load archive
    # ------------------------------------------------------------------
    archive_path = _find_best_archive()
    archive = _load_archive(archive_path)
    print(f"[1/5] Loaded archive: {archive_path.name} ({len(archive)} elites)")

    # ------------------------------------------------------------------
    # [2/5] Build environmental condition grid
    # ------------------------------------------------------------------
    print("\n[2/5] Building environmental condition grid...")
    print(f"  Temperatures (K):     {TEMPERATURES_K}")
    print(f"  Distance ratios:      {DISTANCE_RATIOS}")
    print(f"  Refractive indices:   {REFRACTIVE_INDICES}")

    scenes = load_substance_atmosphere_data(
        spectral_data_file=SPECTRAL_DATA_FILE,
        air_transmittance_file=AIR_TRANSMITTANCE_FILE,
        atmospheric_distance_ratio=DISTANCE_RATIOS,
        temperature_kelvin=TEMPERATURES_K,
        air_refractive_index=REFRACTIVE_INDICES,
    )
    if not isinstance(scenes, list):
        scenes = [scenes]

    num_conditions = len(scenes)
    print(f"  Total conditions: {num_conditions}")

    # Identify which scene index corresponds to nominal
    nominal_idx = None
    for i, s in enumerate(scenes):
        if (
            np.isclose(s.temperature_k, NOMINAL_TEMPERATURE_K)
            and np.isclose(s.atmospheric_distance_ratio, NOMINAL_DISTANCE_RATIO)
            and np.isclose(s.air_refractive_index, NOMINAL_REFRACTIVE_INDEX)
        ):
            nominal_idx = i
            break
    if nominal_idx is not None:
        print(
            f"  Nominal condition at index {nominal_idx}: "
            f"T={NOMINAL_TEMPERATURE_K}K, d={NOMINAL_DISTANCE_RATIO}"
        )
    else:
        logger.warning("Nominal condition not found in grid; using archive fitness as nominal.")

    # ------------------------------------------------------------------
    # [3/5] Evaluate robustness
    # ------------------------------------------------------------------
    print(f"\n[3/5] Evaluating top {TOP_N_ELITES} elites across {num_conditions} conditions...")
    results = evaluate_archive_robustness(
        archive=archive,
        scenes=scenes,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=NUM_PARAMS_PER_BASIS_FUNCTION,
        distance_metric=spectral_angle_mapper,
        top_n=TOP_N_ELITES,
    )

    # ------------------------------------------------------------------
    # [4/5] Summary statistics
    # ------------------------------------------------------------------
    summary = summarise_robustness(results)
    print("\n[4/5] Robustness Summary:")
    print(f"  Elites evaluated:       {summary['num_elites']}")
    print(f"  Conditions tested:      {summary['num_conditions']}")
    print(f"  Mean nominal fitness:   {summary['mean_nominal']:.2f}°")
    print(f"  Mean worst-case:        {summary['mean_worst_case']:.2f}°")
    print(f"  Mean best-case:         {summary['mean_best_case']:.2f}°")
    print(f"  Mean retention ratio:   {summary['mean_retention_ratio']:.2%}")
    print(f"  Worst retention ratio:  {summary['worst_retention_ratio']:.2%}")
    print(f"  Mean CV:                {summary['mean_cv']:.4f}")

    # Per-elite top 5
    print("\n  Top 5 most robust elites (by retention ratio):")
    sorted_by_retention = sorted(results, key=lambda r: r.retention_ratio, reverse=True)
    for i, r in enumerate(sorted_by_retention[:5], 1):
        print(
            f"    {i}. Elite {r.elite_id}: nominal={r.nominal_fitness:.2f}°, "
            f"worst={r.min_fitness:.2f}°, retention={r.retention_ratio:.2%}"
        )

    print("\n  Top 5 least robust elites:")
    for i, r in enumerate(sorted_by_retention[-5:], 1):
        print(
            f"    {i}. Elite {r.elite_id}: nominal={r.nominal_fitness:.2f}°, "
            f"worst={r.min_fitness:.2f}°, retention={r.retention_ratio:.2%}"
        )

    # ------------------------------------------------------------------
    # [5/5] Generate plots
    # ------------------------------------------------------------------
    output_dir = Path("outputs/robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[5/5] Generating plots to {output_dir}/...")

    plot_fitness_degradation_heatmap(
        results, output_path=output_dir / "fitness_degradation_heatmap.png"
    )
    print("  - fitness_degradation_heatmap.png")

    plot_fitness_distribution_by_condition(
        results, output_path=output_dir / "fitness_distribution_by_condition.png"
    )
    print("  - fitness_distribution_by_condition.png")

    plot_worst_case_vs_nominal(results, output_path=output_dir / "worst_case_vs_nominal.png")
    print("  - worst_case_vs_nominal.png")

    plot_condition_sensitivity(results, output_path=output_dir / "condition_sensitivity.png")
    print("  - condition_sensitivity.png")

    plot_retention_histogram(results, output_path=output_dir / "retention_histogram.png")
    print("  - retention_histogram.png")

    # Save JSON summary
    summary_path = output_dir / "robustness_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  - {summary_path.name}")

    # Save full results as pickle for downstream analysis
    results_path = output_dir / "robustness_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump(results, f)
    print(f"  - {results_path.name}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ROBUSTNESS TEST COMPLETE")
    print("=" * 80)
    retention_pct = summary["mean_retention_ratio"] * 100
    if retention_pct >= 90:
        verdict = "EXCELLENT - designs are highly robust to environmental variation"
    elif retention_pct >= 75:
        verdict = "GOOD - moderate degradation; robust optimisation (Phase 2) recommended"
    elif retention_pct >= 50:
        verdict = "CONCERNING - significant degradation; Phase 2 robust optimisation needed"
    else:
        verdict = "POOR - severe degradation; fundamental approach review recommended"
    print(f"\nVerdict: {verdict}")
    print(f"Mean fitness retention: {retention_pct:.1f}%")
    print(f"\nOutputs saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
