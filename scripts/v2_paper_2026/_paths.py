"""Project and output paths for the v2_paper_2026 numbered script suite."""

from __future__ import annotations

from pathlib import Path

# Subdirectory names under outputs/v2_paper_2026/
STEP_STANDARD_GA = "01_standard_ga"
STEP_ADVANCED_GA = "02_advanced_ga_niching"
STEP_MULTI_START_GA = "03_multi_start_ga"
STEP_MAP_ELITES = "04_map_elites"
STEP_MAP_ELITES_HC = "05_map_elites_hc"
STEP_CMA_ME = "06_cma_me"
STEP_ROBUSTNESS = "07_robustness_test"
STEP_MINIMAX = "08_minimax_optimization"


def project_root() -> Path:
    """Repository root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parents[2]


def v2_paper_2026_output_root() -> Path:
    """Unified output root for all v2_paper_2026 suite runs."""
    return project_root() / "outputs" / "v2_paper_2026"


def step_output_dir(step_key: str) -> Path:
    """Return (and create) the output directory for a suite step."""
    path = v2_paper_2026_output_root() / step_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_cma_me_archive_path() -> Path:
    """Archive produced by run_06 (CMA-ME only)."""
    return v2_paper_2026_output_root() / STEP_CMA_ME / "cma_me_archive.pkl"


def default_robustness_archive_path() -> Path:
    """Pre-HC QD archive from run_05; default input for run_07 robustness phase."""
    return v2_paper_2026_output_root() / STEP_MAP_ELITES_HC / "map_elites_archive.pkl"
