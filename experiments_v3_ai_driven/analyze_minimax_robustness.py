#!/usr/bin/env python3
"""Evaluate the step-08 Minimax archive on the configured robustness grid.

Uses the evaluator, scene grid, top-N selection and nominal-scene alignment
from ``scripts/v2_paper_2026/run_07_robustness_test.py``. The default output is
``results/minimax_robustness.json`` beside this script; ``--output`` selects a
separate verification file. The comparison reads the stored step-07 nominal
summary, whose source archive and generation settings require independent
provenance. This is in-grid surrogate evaluation, not off-grid validation or
measured detector performance.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SUITE = _REPO / "scripts" / "v2_paper_2026"
if str(_SUITE) not in sys.path:
    sys.path.insert(0, str(_SUITE))

import _budgets  # noqa: E402
import _experiment_body  # noqa: E402

from lwi_microbolometer_design import (  # noqa: E402
    gaussian_parameters_to_unit_amplitude_curves,
    spectral_angle_mapper,
)
from lwi_microbolometer_design.analysis.robustness import (  # noqa: E402
    align_nominal_fitness_to_training_scene,
    evaluate_archive_robustness,
    summarise_robustness,
)
from lwi_microbolometer_design.data import load_substance_atmosphere_data  # noqa: E402

_MINIMAX_ARCHIVE = (
    _REPO / "outputs" / "v2_paper_2026" / "08_minimax_optimization" / "map_elites_archive.pkl"
)
_OUT = Path(__file__).resolve().parent / "results" / "minimax_robustness.json"
_NOMINAL_SUMMARY = (
    _REPO / "outputs" / "v2_paper_2026" / "07_robustness_test" / "robustness_summary.json"
)


def main() -> None:
    """Evaluate the existing archive and write a summary to the selected path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_OUT, help="Output JSON path.")
    args = parser.parse_args()
    cap = _budgets.ROBUSTNESS_TOP_N
    top_n = None if cap <= 0 else cap
    temps = _budgets.ROBUSTNESS_TEMPERATURES_K
    dists = _budgets.ROBUSTNESS_DISTANCE_RATIOS
    refidx = _budgets.ROBUSTNESS_REFRACTIVE_INDICES

    if not _MINIMAX_ARCHIVE.is_file():
        raise FileNotFoundError(f"Minimax archive not found: {_MINIMAX_ARCHIVE}")

    with _MINIMAX_ARCHIVE.open("rb") as f:
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

    t0 = time.perf_counter()
    results = evaluate_archive_robustness(
        archive=archive,
        scenes=scenes,
        parameters_to_curves=gaussian_parameters_to_unit_amplitude_curves,
        params_per_basis_function=_experiment_body.num_params_per_basis_function(),
        distance_metric=spectral_angle_mapper,
        top_n=top_n,
        show_progress=True,
        progress_desc="[minimax-robustness] elites (x25 scenes each)",
    )
    wall_s = time.perf_counter() - t0

    align_nominal_fitness_to_training_scene(results, scenes, nominal_scene)
    summary = summarise_robustness(results)
    summary["archive_source"] = str(_MINIMAX_ARCHIVE.resolve())
    summary["archive_label"] = (
        "08_minimax_optimization MAP-Elites (worst-case-over-25-scenes objective)"
    )
    summary["grid"] = {
        "temperatures_k": list(temps),
        "distance_ratios": list(dists),
        "refractive_indices": list(refidx),
        "num_conditions": len(scenes),
        "top_n_elites": top_n,
    }
    summary["wall_time_seconds"] = wall_s

    if _NOMINAL_SUMMARY.is_file():
        with _NOMINAL_SUMMARY.open("r", encoding="utf-8") as f:
            nominal = json.load(f)
        summary["nominal_archive_comparison"] = {
            "source": "07_robustness_test/robustness_summary.json (step-05 nominal MAP-Elites+HC archive)",
            "mean_retention_ratio": nominal.get("mean_retention_ratio"),
            "worst_retention_ratio": nominal.get("worst_retention_ratio"),
            "mean_nominal": nominal.get("mean_nominal"),
            "mean_worst_case": nominal.get("mean_worst_case"),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
