"""Default evaluation budgets and argparse helpers for presentation runs."""

from __future__ import annotations

import argparse
import os


def _cpu_count() -> int:
    """Usable CPU count (at least 1)."""
    return max(1, os.cpu_count() or 1)


# --- Multiprocessing caps (defaults use all cores up to these limits) ---
# Multi-start: many full-GA processes — allow up to 200 workers on large machines.
MULTI_START_MAX_WORKERS_CAP = int(os.environ.get("PRES_MULTI_START_WORKERS_CAP", "200"))
# HC polish: heavier per process — keep default pool under 50 cores to limit RAM pressure.
HC_MAX_WORKERS_CAP = int(os.environ.get("PRES_HC_WORKERS_CAP", "49"))

MULTI_START_MAX_WORKERS_DEFAULT = min(_cpu_count(), MULTI_START_MAX_WORKERS_CAP)
HC_MAX_WORKERS_DEFAULT = min(_cpu_count(), HC_MAX_WORKERS_CAP)

# --- Production-oriented defaults (hours + strong CPU). Override via CLI / env. ---

# Step 01 (standard GA) — tuned up vs earlier suite (step 02 uses separate ADVANCED constants).
GA_NUM_GENERATIONS_STANDARD = int(os.environ.get("PRES_GA_GENERATIONS", "5000"))
GA_SOL_PER_POP_STANDARD = int(os.environ.get("PRES_GA_POP", "240"))
GA_PARENTS_MATING_FRACTION_STANDARD = float(
    os.environ.get("PRES_GA_PARENTS_FRACTION_STANDARD", "0.52")
)
GA_STOP_CRITERIA = "saturate_10000"

# Top-elite spectrum PNGs (steps 01–08): graph single-linkage on optimal-pairing *D* at τ.
ELITE_GRAPH_TAU = float(os.environ.get("PRES_ELITE_GRAPH_TAU", "0.5"))

# Step 02 (advanced GA / niching) — intentionally separate; do not inherit 01 bumps.
GA_NUM_GENERATIONS_ADVANCED = int(os.environ.get("PRES_GA_GENERATIONS_ADVANCED", "4000"))
GA_SOL_PER_POP_ADVANCED = int(os.environ.get("PRES_GA_POP_ADVANCED", "200"))

MULTI_START_NUM_RUNS = int(os.environ.get("PRES_MULTI_START_RUNS", "200"))
# Per-run defaults match step 01 unless PRES_MULTI_START_* override.
MULTI_START_GENERATIONS = int(
    os.environ.get("PRES_MULTI_START_GENS", str(GA_NUM_GENERATIONS_STANDARD))
)
MULTI_START_POP = int(os.environ.get("PRES_MULTI_START_POP", str(GA_SOL_PER_POP_STANDARD)))
MULTI_START_MAX_WORKERS = int(
    os.environ.get("PRES_MULTI_START_WORKERS", str(MULTI_START_MAX_WORKERS_DEFAULT))
)

# Steps 04–05: match step 01 total fitness evaluations (archive seed + mutation loop).
MAP_ELITES_TOTAL_FITNESS_EVALS = int(
    os.environ.get(
        "PRES_MAP_ELITES_TOTAL_EVALS",
        str(GA_NUM_GENERATIONS_STANDARD * GA_SOL_PER_POP_STANDARD),
    )
)
MAP_ELITES_NUM_INITIAL = int(
    os.environ.get("PRES_MAP_ELITES_INITIAL", str(GA_SOL_PER_POP_STANDARD))
)
MAP_ELITES_ITERATIONS = max(0, MAP_ELITES_TOTAL_FITNESS_EVALS - MAP_ELITES_NUM_INITIAL)

# Step 06: same nominal fitness-call budget as steps 01 / 04 (single-scene evals).
CMA_ME_TOTAL_EVALS = int(
    os.environ.get(
        "PRES_CMA_ME_EVALS",
        str(GA_NUM_GENERATIONS_STANDARD * GA_SOL_PER_POP_STANDARD),
    )
)
CMA_ME_NUM_EMITTERS = int(os.environ.get("PRES_CMA_ME_EMITTERS", "6"))
CMA_ME_NUM_INITIAL = int(os.environ.get("PRES_CMA_ME_INITIAL", str(GA_SOL_PER_POP_STANDARD)))

# Minimax: each fitness call evaluates all grid conditions (~25 forward passes).
MINIMAX_CMA_ME_TOTAL_EVALS = int(os.environ.get("PRES_MINIMAX_EVALS", "200000"))
MINIMAX_BUDGET_MULTIPLIER = float(os.environ.get("PRES_MINIMAX_MULT", "1.0"))

# Step 07: archive elites to evaluate (0 = entire archive). Smaller default keeps
# heatmaps/ticks readable; scale up via PRES_ROBUSTNESS_TOP_N.
ROBUSTNESS_TOP_N = int(os.environ.get("PRES_ROBUSTNESS_TOP_N", "64"))

# Environmental grid for step 07 — compact 5×5×1 = 25 conditions for clearer plots.
# Must include training nominal from ``_experiment_body.load_nominal_scene``
# (293.15 K, 0.11, 1.0) for ``align_nominal_fitness_to_training_scene``.
ROBUSTNESS_TEMPERATURES_K: tuple[float, ...] = (
    273.15,
    283.15,
    293.15,
    303.15,
    313.15,
)
ROBUSTNESS_DISTANCE_RATIOS: tuple[float, ...] = (
    0.05,
    0.08,
    0.11,
    0.15,
    0.20,
)
ROBUSTNESS_REFRACTIVE_INDICES: tuple[float, ...] = (1.0,)

HC_MAX_ELITES = int(os.environ.get("PRES_HC_MAX_ELITES", "160"))
HC_FITNESS_THRESHOLD = float(os.environ.get("PRES_HC_THRESHOLD", "45.0"))
HC_MAX_WORKERS = int(os.environ.get("PRES_HC_WORKERS", str(HC_MAX_WORKERS_DEFAULT)))


def add_quickrun_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--quick`` to slash budgets for smoke tests."""
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny budgets for smoke testing (overrides default scales).",
    )


def quickrun_ga_baseline() -> tuple[int, int]:
    return 8, 24


def quickrun_multi_start() -> tuple[int, int, int]:
    return 3, 12, 20


def quickrun_map_elites() -> tuple[int, int]:
    """Return ``(num_iterations, num_initial)`` with same total evals as :func:`quickrun_ga_baseline`."""
    gens, pop = quickrun_ga_baseline()
    total = gens * pop
    num_initial = pop
    return max(0, total - num_initial), num_initial


def quickrun_cma_me() -> tuple[int, int]:
    """Return ``(total_evals, num_initial)`` matching :func:`quickrun_ga_baseline` total."""
    gens, pop = quickrun_ga_baseline()
    return gens * pop, pop


def quickrun_minimax() -> int:
    return 800


def quickrun_robustness_physics_grid() -> tuple[
    tuple[float, ...], tuple[float, ...], tuple[float, ...]
]:
    """Small T × d × n grid for ``run_07 --quick`` (includes nominal)."""
    return (
        (273.15, 293.15, 313.15),
        (0.08, 0.11, 0.15),
        (1.0,),
    )


def apply_quickrun_multipliers(quick: bool) -> dict[str, int | float]:
    """Return overrides when --quick is set (scripts apply locally)."""
    if not quick:
        return {}
    cma_q = quickrun_cma_me()
    return {
        "ga_generations": quickrun_ga_baseline()[0],
        "ga_pop": quickrun_ga_baseline()[1],
        "multi_runs": quickrun_multi_start()[0],
        "multi_gens": quickrun_multi_start()[1],
        "multi_pop": quickrun_multi_start()[2],
        "map_iters": quickrun_map_elites()[0],
        "map_initial": quickrun_map_elites()[1],
        "cma_evals": cma_q[0],
        "cma_initial": cma_q[1],
        "minimax_evals": quickrun_minimax(),
    }
