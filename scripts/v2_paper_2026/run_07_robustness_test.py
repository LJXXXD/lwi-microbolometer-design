#!/usr/bin/env python3
"""v2_paper_2026 step 07: Phase 1 environmental robustness on the hero MAP-Elites+HC archive."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

_SUITE_DIR = Path(__file__).resolve().parent
if str(_SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUITE_DIR))

from tqdm import tqdm

import _artifacts
import _budgets
import _experiment_body
import _paths
import _plotting
from lwi_microbolometer_design import (
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.analysis.robustness import (
    align_nominal_fitness_to_training_scene,
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
logger = logging.getLogger("v2_paper.run_07")


def main() -> None:
    parser = argparse.ArgumentParser(description="v2_paper_2026 step 07: robustness grid")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help=(
            "QD archive pickle (default: step 05 MAP-Elites+HC map_elites_archive.pkl; "
            "override for e.g. step 04/06 pickles)."
        ),
    )
    _budgets.add_quickrun_arg(parser)
    args = parser.parse_args()

    archive_path = args.archive or _paths.default_robustness_archive_path()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if args.quick:
        top_n = 5
        temps, dists, refidx = _budgets.quickrun_robustness_physics_grid()
    else:
        cap = _budgets.ROBUSTNESS_TOP_N
        top_n = None if cap <= 0 else cap
        temps = _budgets.ROBUSTNESS_TEMPERATURES_K
        dists = _budgets.ROBUSTNESS_DISTANCE_RATIOS
        refidx = _budgets.ROBUSTNESS_REFRACTIVE_INDICES

    out_dir = _paths.step_output_dir(_paths.STEP_ROBUSTNESS)

    with archive_path.open("rb") as f:
        archive = pickle.load(f)  # noqa: S301

    nominal_scene = _experiment_body.load_nominal_scene()
    scenes = load_substance_atmosphere_data(
        spectral_data_file=_experiment_body.spectral_data_path(),
        air_transmittance_file=_experiment_body.air_transmittance_path(),
        atmospheric_distance_ratio=list(dists),
        temperature_kelvin=list(temps),
        air_refractive_index=list(refidx),
    )
    if not isinstance(scenes, list):
        scenes = [scenes]

    n_cond = len(scenes)
    grid_lbl = f"{len(temps)}×{len(dists)}×{len(refidx)}"
    top_lbl = "all" if top_n is None else str(top_n)
    tqdm.write(
        f"\n{'=' * 72}\n"
        f"  STEP 07 | Robustness Phase 1 | archive={archive_path.name} | elites={top_lbl} | "
        f"conditions={n_cond} ({grid_lbl} physics grid)\n"
        f"{'=' * 72}\n"
    )
    _plotting.apply_presentation_style()
    t0 = time.perf_counter()
    results = evaluate_archive_robustness(
        archive=archive,
        scenes=scenes,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=_experiment_body.num_params_per_basis_function(),
        distance_metric=spectral_angle_mapper,
        top_n=top_n,
        show_progress=True,
        progress_desc=f"[07] Robustness | elites (×{n_cond} conditions each)",
    )
    wall_s = time.perf_counter() - t0

    align_nominal_fitness_to_training_scene(results, scenes, nominal_scene)
    summary = summarise_robustness(results)

    plot_fitness_degradation_heatmap(
        results, output_path=out_dir / "fitness_degradation_heatmap.png"
    )
    plot_fitness_distribution_by_condition(
        results, output_path=out_dir / "fitness_distribution_by_condition.png"
    )
    plot_worst_case_vs_nominal(results, output_path=out_dir / "worst_case_vs_nominal.png")
    plot_condition_sensitivity(results, output_path=out_dir / "condition_sensitivity.png")
    plot_retention_histogram(results, output_path=out_dir / "retention_histogram.png")

    summary_path = out_dir / "robustness_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    results_path = out_dir / "robustness_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump(results, f)

    _artifacts.save_run_summary(
        out_dir / "run_summary.json",
        {
            "step": "07_robustness_test",
            "method": "environmental_robustness_phase_1",
            "archive_source": str(archive_path.resolve()),
            "budgets": {
                "top_n_elites": top_n,
                "num_conditions": len(scenes),
                "physics_grid_shape": {
                    "n_temperatures": len(temps),
                    "n_distance_ratios": len(dists),
                    "n_refractive_indices": len(refidx),
                },
            },
            "wall_time_seconds": wall_s,
            "mean_retention_ratio": summary.get("mean_retention_ratio"),
            "mean_nominal_fitness": summary.get("mean_nominal"),
            "mean_worst_case_fitness": summary.get("mean_worst_case"),
            "output_files": [
                "robustness_summary.json",
                "robustness_results.pkl",
                "fitness_degradation_heatmap.png",
                "fitness_distribution_by_condition.png",
                "worst_case_vs_nominal.png",
                "condition_sensitivity.png",
                "retention_histogram.png",
            ],
        },
    )
    logger.info("Robustness complete; outputs in %s", out_dir)


if __name__ == "__main__":
    main()
