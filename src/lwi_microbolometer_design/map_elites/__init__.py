"""MAP-Elites quality-diversity algorithm for sensor design exploration.

This package provides a complete MAP-Elites system including:
- Archive management (feature extraction, binning, initialisation)
- Core MAP-Elites loop with Gaussian mutation
- CMA-ME (Covariance Matrix Adaptation MAP-Elites) with emitter-based search
- Post-hoc polishing strategies (hill climbing, CMA-ES)
- Visualisation utilities (heatmaps, elite plots, convergence curves)

Fitness evaluation is passed in as a callable; the ``ga`` package provides
``MinDissimilarityFitnessEvaluator`` which is the standard choice.
"""

from .algorithm import mutate_chromosome, run_map_elites
from .archive import (
    archive_coverage_pct,
    bin_coordinates,
    extract_features,
    initialize_archive,
    reachable_cell_count,
)
from .cma_me import run_cma_me
from .emitters import EmitterBase, OptimizingEmitter
from .polish import polish_single_elite_cma, polish_single_elite_hc
from .visualization import (
    plot_cma_me_progress,
    plot_map_elites_heatmap,
    plot_polished_elites,
    plot_top_elites,
)

__all__ = [
    # Archive
    "archive_coverage_pct",
    "bin_coordinates",
    "extract_features",
    "initialize_archive",
    "reachable_cell_count",
    # Algorithm
    "mutate_chromosome",
    "run_map_elites",
    # CMA-ME
    "EmitterBase",
    "OptimizingEmitter",
    "run_cma_me",
    # Polish
    "polish_single_elite_cma",
    "polish_single_elite_hc",
    # Visualization
    "plot_cma_me_progress",
    "plot_map_elites_heatmap",
    "plot_polished_elites",
    "plot_top_elites",
]
