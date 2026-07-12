"""Consistent matplotlib styling and small presentation figures."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np

import _budgets

from lwi_microbolometer_design.analysis import (
    compute_distance_matrix,
    family_labels_from_optimal_pairing_d,
)
from lwi_microbolometer_design.map_elites.archive import bin_coordinates, extract_features
from lwi_microbolometer_design.map_elites.visualization import plot_top_individual_curves

logger = logging.getLogger(__name__)

# Top-N spectra: family colours = graph single-linkage on optimal-pairing *D* at ``τ`` below.
SUITE_TOP_ELITE_COUNTS: tuple[int, ...] = (10, 20, 50)

# Top-elite spectrum PNGs: emit one file per size; stem ends with ``_{W}x{H}`` (e.g. ``_8x6``, ``_4x4``).
ELITE_SPECTRA_FIGSIZES: tuple[tuple[float, float], ...] = (
    (8.0, 6.0),
    (4.0, 4.0),
)
# First entry is the primary / legacy reference size.
ELITE_SPECTRA_FIGSIZE: tuple[float, float] = ELITE_SPECTRA_FIGSIZES[0]
# Legend text (pt): keep constant across compact figsizes so the small layout does not shrink it.
ELITE_SPECTRA_LEGEND_FONTSIZE: float = 9.0

# **τ** in optimal-pairing *D* units. Lower *τ* → more families; if all *D* are below *τ*, one family.
ELITE_SPECTRA_GRAPH_DIST_THRESHOLD: float = _budgets.ELITE_GRAPH_TAU

# Genes per Gaussian (μ, σ); must match the GA / evaluators.
ELITE_SPECTRA_PARAMS_PER_GROUP: int = 2


class _EliteSpec(TypedDict, total=False):
    """``file_suffix`` empty → ``{basename}_top{{N}}_elites_{{W}}x{{H}}.png`` per :data:`ELITE_SPECTRA_FIGSIZES`."""

    file_suffix: str
    tau: float


# Kept as a 1-tuple for ``ELITE_SPECTRA_FILE_SUFFIXES`` / run_summary lists; extend if needed.
ELITE_SPECTRA_CLUSTERING_PROFILES: tuple[_EliteSpec, ...] = (
    {"file_suffix": "", "tau": ELITE_SPECTRA_GRAPH_DIST_THRESHOLD},
)

ELITE_SPECTRA_FILE_SUFFIXES: tuple[str, ...] = tuple(
    str(p.get("file_suffix", "")) for p in ELITE_SPECTRA_CLUSTERING_PROFILES
)


def _figsize_filename_suffix(figsize: tuple[float, float]) -> str:
    """``8x6`` / ``5x3`` style tag for PNG stems."""

    def _part(x: float) -> str:
        xf = float(x)
        if xf == int(xf):
            return str(int(xf))
        return format(xf, "g").replace(".", "p")

    return f"{_part(figsize[0])}x{_part(figsize[1])}"


def elite_spectrum_png_filenames(elites_basename: str) -> list[str]:
    """Basenames written by :func:`export_presentation_top_elite_spectra_sweep` (for run summaries)."""
    names: list[str] = []
    for n in SUITE_TOP_ELITE_COUNTS:
        for suf in ELITE_SPECTRA_FILE_SUFFIXES:
            for fig in ELITE_SPECTRA_FIGSIZES:
                names.append(
                    f"{elites_basename}_top{n}_elites{suf}_{_figsize_filename_suffix(fig)}.png"
                )
    return names


def _labels_for_spec(p: _EliteSpec, dist: np.ndarray, fit_list: list[float]) -> np.ndarray:
    tau = float(p.get("tau", ELITE_SPECTRA_GRAPH_DIST_THRESHOLD))
    lab, _ = family_labels_from_optimal_pairing_d(dist, fit_list, tau=tau)
    return lab


def _pool_distance_matrix(items: list[np.ndarray]) -> np.ndarray:
    return compute_distance_matrix(
        items,
        metric="euclidean",
        use_optimal_pairing=True,
        params_per_group=ELITE_SPECTRA_PARAMS_PER_GROUP,
    )


def apply_presentation_style() -> None:
    """Apply shared rcParams before any suite plotting."""
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "image.cmap": "viridis",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def export_presentation_top_elite_spectra(
    individuals: Sequence[dict[str, Any]],
    wavelengths: np.ndarray,
    parameters_to_curves: Callable[..., np.ndarray],
    out_dir: Path,
    stem: str,
    title_prefix: str,
    top_n: int,
) -> None:
    """One top-*N* PNG per entry in :data:`ELITE_SPECTRA_FIGSIZES` (graph-τ families)."""
    apply_presentation_style()
    ranked = sorted(individuals, key=lambda x: float(x["fitness"]), reverse=True)
    take = min(len(ranked), int(top_n))
    if take == 0:
        return
    pool = ranked[:take]
    items = [np.asarray(p["chromosome"], dtype=np.float64) for p in pool]
    dist = _pool_distance_matrix(items)
    fits = [float(p["fitness"]) for p in pool]
    prof0 = ELITE_SPECTRA_CLUSTERING_PROFILES[0]
    full_l = _labels_for_spec(prof0, dist, fits)
    for fig in ELITE_SPECTRA_FIGSIZES:
        tag = _figsize_filename_suffix(fig)
        out_path_sz = out_dir / f"{stem}_{tag}.png"
        plot_top_individual_curves(
            individuals,
            wavelengths,
            parameters_to_curves,
            out_path_sz,
            top_n,
            title_prefix,
            figsize=fig,
            color_by_family=True,
            show_rank_labels=False,
            fontsize_scale=1.0,
            legend_fontsize=ELITE_SPECTRA_LEGEND_FONTSIZE,
            family_labels=full_l,
        )


def export_presentation_top_elite_spectra_sweep(
    individuals: Sequence[dict[str, Any]],
    wavelengths: np.ndarray,
    parameters_to_curves: Callable[..., np.ndarray],
    out_dir: Path,
    elites_basename: str,
    title_prefix: str,
    *,
    top_ns: tuple[int, ...] | None = None,
    elite_figsizes: tuple[tuple[float, float], ...] | None = None,
    legend_ncol: int | None = None,
) -> None:
    """Top-*N* PNGs (one per figure size); shared *D* pool at ``τ``.

    Parameters
    ----------
    elite_figsizes :
        Override :data:`ELITE_SPECTRA_FIGSIZES` (e.g. tall canvases for very large *N*).
    legend_ncol :
        Passed to :func:`plot_top_individual_curves` when not ``None`` (multi-column legend).
    """
    apply_presentation_style()
    counts = top_ns or SUITE_TOP_ELITE_COUNTS
    figsizes = elite_figsizes if elite_figsizes is not None else ELITE_SPECTRA_FIGSIZES
    max_n = max(counts)
    ranked = sorted(individuals, key=lambda x: float(x["fitness"]), reverse=True)
    n_pool = min(len(ranked), max_n)
    if n_pool == 0:
        return
    ranked_pool = ranked[:n_pool]
    items = [np.asarray(p["chromosome"], dtype=np.float64) for p in ranked_pool]
    dist = _pool_distance_matrix(items)
    fits = [float(p["fitness"]) for p in ranked_pool]

    for prof in ELITE_SPECTRA_CLUSTERING_PROFILES:
        full_labs = _labels_for_spec(prof, dist, fits)
        suffix = str(prof.get("file_suffix", ""))
        n_fam = int(len(np.unique(full_labs)))
        for top_n in counts:
            take = min(int(top_n), n_pool)
            if take == 0:
                continue
            stem = f"{elites_basename}_top{top_n}_elites{suffix}"
            for fig in figsizes:
                tag = _figsize_filename_suffix(fig)
                out_path = out_dir / f"{stem}_{tag}.png"
                plot_top_individual_curves(
                    individuals,
                    wavelengths,
                    parameters_to_curves,
                    out_path,
                    top_n,
                    title_prefix,
                    figsize=fig,
                    color_by_family=True,
                    show_rank_labels=False,
                    fontsize_scale=1.0,
                    legend_fontsize=ELITE_SPECTRA_LEGEND_FONTSIZE,
                    family_labels=full_labs[:take],
                    legend_ncol=legend_ncol,
                )
        logger.info(
            "Elite graph-τ (suffix=%r): %d families (n_pool=%d) for %s",
            suffix or "primary",
            n_fam,
            n_pool,
            elites_basename,
        )


def plot_descriptor_scatter(
    population: np.ndarray,
    fitness: np.ndarray,
    mu_range: tuple[float, float],
    grid_resolution: int,
    output_path: Path,
    title: str,
) -> None:
    """Scatter smallest vs second-smallest mu coloured by fitness (mode-collapse narrative)."""
    apply_presentation_style()
    m1 = np.empty(len(population))
    m2 = np.empty(len(population))
    for i, chrom in enumerate(population):
        u, v = extract_features(np.asarray(chrom, dtype=float))
        m1[i], m2[i] = u, v

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sc = ax.scatter(m1, m2, c=fitness, cmap="viridis", s=28, alpha=0.85, edgecolors="none")
    ax.set_xlim(mu_range[0], mu_range[1])
    ax.set_ylim(mu_range[0], mu_range[1])
    ax.set_aspect("equal")
    ax.set_xlabel("Smallest μ (µm)")
    ax.set_ylabel("Second-smallest μ (µm)")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("Fitness (min SAM angle, °)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def count_unique_descriptor_bins(
    population: np.ndarray,
    mu_range: tuple[float, float],
    grid_resolution: int,
) -> int:
    """Number of distinct MAP-Elites bins occupied by a population."""
    keys: set[tuple[int, int]] = set()
    for chrom in population:
        u, v = extract_features(np.asarray(chrom, dtype=float))
        keys.add(bin_coordinates(u, v, grid_resolution, mu_range))
    return len(keys)
