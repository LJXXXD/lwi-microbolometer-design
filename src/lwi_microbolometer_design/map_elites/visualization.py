"""Visualisation utilities for MAP-Elites archives and polished results."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from lwi_microbolometer_design.analysis import (
    clodd_partition,
    compute_distance_matrix,
    family_labels_from_optimal_pairing_d,
    graph_component_labels_from_distance_matrix,
    ivat_transform,
    map_labels_to_original_order,
    normalize_ordered_dissimilarity,
    vat_reorder,
)

from lwi_microbolometer_design.map_elites.archive import extract_features, reachable_cell_count

logger = logging.getLogger(__name__)

# Back-compat: tests and older call sites
_graph_component_labels_from_distance_matrix = graph_component_labels_from_distance_matrix

# Distinct colors for family-legend stacked spectra (presentation).
_FAMILY_COLORS = plt.cm.tab10(np.linspace(0, 0.9, 10))


def _chromosome_family_labels(
    chromosomes: np.ndarray,
    *,
    params_per_group: int = 2,
    degenerate_atol: float = 1e-7,
    clodd_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    """Partition designs into families via optimal-pairing distance + iVAT + CLODD.

    Pairwise distances use :func:`~lwi_microbolometer_design.analysis.compute_distance_matrix`
    with ``use_optimal_pairing=True`` so permutations of (μ, σ) groups do not
    inflate distance. Ordered iVAT is passed to :func:`clodd_partition` (Havens
    et al., 2009) to obtain a crisp partition along the VAT order.

    Parameters
    ----------
    chromosomes
        Shape ``(n_individuals, n_genes)``. Length divisible by *params_per_group*.
    params_per_group
        Genes per basis group (default 2: μ and σ per Gaussian).
    degenerate_atol
        If all off-diagonal optimal-pairing distances are below this, return a
        single family (avoids spurious multi-cluster splits on near-zero D*).
    clodd_options
        Optional kwargs forwarded to :func:`clodd_partition` (e.g. ``c_max``,
        ``alpha``, ``gamma``). CLODD does not use a single “distance threshold”;
        it maximizes a blockiness objective on the ordered iVAT matrix.

    Returns
    -------
    labels
        Integer family id per row (reordered by :func:`_reorder_families_by_fitness`
        in the plotter so family ``0`` has best mean fitness).
    n_families
        Number of distinct families.
    """
    n = int(chromosomes.shape[0])
    if n <= 0:
        return np.array([], dtype=int), 0
    if n == 1:
        return np.zeros(1, dtype=int), 1

    items = [np.asarray(chromosomes[i], dtype=np.float64) for i in range(n)]
    dist = compute_distance_matrix(
        items,
        metric="euclidean",
        use_optimal_pairing=True,
        params_per_group=int(params_per_group),
    )
    d_max = float(np.max(dist[np.triu_indices(n, k=1)]))
    if d_max <= degenerate_atol:
        return np.zeros(n, dtype=int), 1
    if n == 2:
        return np.array([0, 1], dtype=int), 2

    vat_matrix, reorder = vat_reorder(np.asarray(dist, dtype=np.float64))
    ivat = ivat_transform(vat_matrix)
    d_star = normalize_ordered_dissimilarity(ivat)
    co = dict(clodd_options) if clodd_options else {}
    cld = clodd_partition(d_star, mode="auto", **co)
    labels = map_labels_to_original_order(cld.labels_ordered, reorder)
    return labels.astype(int), int(len(np.unique(labels)))


def _reorder_families_by_fitness(labels: np.ndarray, fitnesses: Sequence[float]) -> np.ndarray:
    """Permute family ids so ``0`` is the family with highest mean fitness."""
    unique = np.unique(labels)
    means: dict[int, float] = {}
    for c in unique:
        idx = np.nonzero(labels == c)[0]
        means[int(c)] = float(np.mean([fitnesses[i] for i in idx]))
    order = sorted(unique, key=lambda c: -means[int(c)])
    remap = {int(old): new for new, old in enumerate(order)}
    return np.array([remap[int(c)] for c in labels], dtype=int)


def family_labels_for_top_elites(
    fitness_ordered_individuals: Sequence[dict[str, Any]],
    *,
    clodd_options: dict[str, Any] | None = None,
    params_per_group: int = 2,
) -> np.ndarray:
    """Assign family ids with **one** CLODD run on this elite set (best individual first).

    Use the same *fitness_ordered_individuals* list you pass to plotting as the
    top-:math:`K` (e.g. :math:`K=50`); then slice the returned array for
    :math:`K'\\!\\in\\{10,20,50\\}` so 10/20/50 figures share a consistent
    partition.

    Parameters
    ----------
    fitness_ordered_individuals
        Elite dicts with ``chromosome`` and ``fitness``, **sorted by fitness
        descending** (same order as the stacked elite plot, rank 0 = best).
    clodd_options
        Optional kwargs for :func:`clodd_partition` (e.g. ``c_max``, ``gamma``).
    params_per_group
        Genes per basis (default 2 for μ, σ per Gaussian).

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` with ``n = len(fitness_ordered_individuals)``. Family ``0``
        has the highest mean fitness among families.
    """
    n = len(fitness_ordered_individuals)
    if n == 0:
        return np.array([], dtype=int)
    chrom_stack = np.stack(
        [np.asarray(ind["chromosome"], dtype=float) for ind in fitness_ordered_individuals]
    )
    raw, _ = _chromosome_family_labels(
        chrom_stack,
        clodd_options=clodd_options,
        params_per_group=int(params_per_group),
    )
    fits = [float(ind["fitness"]) for ind in fitness_ordered_individuals]
    return _reorder_families_by_fitness(raw, fits)


def family_labels_for_top_elites_graph_threshold(
    fitness_ordered_individuals: Sequence[dict[str, Any]],
    *,
    params_per_group: int = 2,
    degenerate_atol: float = 1e-7,
    distance_threshold: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Graph single-linkage on optimal-pairing *D*; see
    :func:`lwi_microbolometer_design.analysis.elite_clustering.family_labels_from_optimal_pairing_d`.
    """
    n = len(fitness_ordered_individuals)
    if n == 0:
        return np.array([], dtype=int), 0.0
    if n == 1:
        return np.zeros(1, dtype=int), float(distance_threshold)

    items = [
        np.asarray(fitness_ordered_individuals[i]["chromosome"], dtype=np.float64) for i in range(n)
    ]
    dist = compute_distance_matrix(
        items,
        metric="euclidean",
        use_optimal_pairing=True,
        params_per_group=int(params_per_group),
    )
    off = dist[np.triu_indices(n, k=1)]
    d_max = float(np.max(off)) if off.size else 0.0
    if d_max <= degenerate_atol:
        return np.zeros(n, dtype=int), float(d_max if np.isfinite(d_max) else 0.0)

    tau = float(distance_threshold)
    fits = [float(fitness_ordered_individuals[i]["fitness"]) for i in range(n)]
    out, _note = family_labels_from_optimal_pairing_d(dist, fits, tau=tau)
    return out, tau


def set_tight_ylim_stacked_spectra(
    ax: Axes,
    y_min_data: float,
    y_max_data: float,
    *,
    pad_fraction: float = 0.032,
    pad_floor: float = 0.014,
) -> None:
    """Set y-limits from observed stack extrema with padding ∝ vertical span.

    Avoids large fixed margins (e.g. +0.5) that waste headroom when few tiers
    are plotted while staying stable for 10 vs 50 elites.

    Parameters
    ----------
    ax :
        Axes containing offset stacked curves.
    y_min_data, y_max_data :
        Minimum and maximum *y* actually reached by plotted lines.
    pad_fraction :
        Padding added below *y_min_data* and above *y_max_data*, as a fraction
        of ``y_max_data - y_min_data``.
    pad_floor :
        Minimum padding (data units) when the span is very small.
    """
    span = float(y_max_data - y_min_data)
    if not np.isfinite(span):
        span = 0.0
    span = max(span, 1e-12)
    pad = max(pad_fraction * span, pad_floor)
    ax.set_ylim(y_min_data - pad, y_max_data + pad)


def plot_map_elites_heatmap(
    archive: dict[tuple[int, int], dict[str, Any]],
    grid_resolution: int,
    mu_range: tuple[float, float],
    output_path: Path,
) -> None:
    """Render the MAP-Elites archive as a fitness heatmap.

    Parameters
    ----------
    archive : dict
        ``(x_bin, y_bin) -> individual dict``.
    grid_resolution : int
        Bins per dimension.
    mu_range : tuple[float, float]
        (min, max) for both axes.
    output_path : Path
        Where to save the figure.
    """
    mu_min, mu_max = mu_range
    fitness_grid = np.full((grid_resolution, grid_resolution), np.nan)
    for (x_bin, y_bin), individual in archive.items():
        fitness_grid[x_bin, y_bin] = individual["fitness"]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        fitness_grid.T,
        origin="lower",
        extent=[mu_min, mu_max, mu_min, mu_max],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("Smallest \u03bc (\u00b5m)", fontsize=14)
    ax.set_ylabel("Second Smallest \u03bc (\u00b5m)", fontsize=14)
    ax.set_title(
        f"MAP-Elites Fitness Landscape\n"
        f"Reachable Coverage: {len(archive)}/{reachable_cell_count(grid_resolution)} cells",
        fontsize=16,
        fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fitness", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved heatmap to: {output_path}")
    plt.close()


def plot_top_individual_curves(
    individuals: Sequence[dict[str, Any]],
    wavelengths: np.ndarray,
    parameters_to_curves: Callable[..., np.ndarray],
    output_path: Path,
    top_n: int,
    title_prefix: str,
    *,
    figsize: tuple[float, float] = (14.0, 10.0),
    color_by_family: bool = False,
    show_rank_labels: bool | None = None,
    fontsize_scale: float = 1.0,
    legend_fontsize: float | None = None,
    tick_labelsize: float | None = None,
    legend_ncol: int | None = None,
    clodd_options: dict[str, Any] | None = None,
    family_labels: np.ndarray | None = None,
    title_family_note: str = "",
) -> None:
    """Plot the top-*N* individuals by fitness as stacked Gaussian-basis curves.

    Each dict must contain ``chromosome`` and ``fitness`` (same keys as MAP-Elites
    archive members). Used for MAP-Elites archives, GA final populations, and
    multi-start champion pools.

    Parameters
    ----------
    individuals :
        Candidates to rank by fitness (descending).
    wavelengths : np.ndarray
        Wavelength axis (µm).
    parameters_to_curves : callable
        ``f(gaussian_param_pairs, wavelengths) -> curves``.
    output_path : Path
        Output PNG path.
    top_n : int
        Maximum number of curves to draw.
    title_prefix : str
        Figure title prefix.
    figsize :
        Figure size in inches.
    color_by_family :
        If True, colour by *family_labels* (or CLODD in *_chromosome_family_labels* when
        *family_labels* is None), and add a legend.
    show_rank_labels :
        Per-rank text on the left. If None, defaults to True unless
        ``color_by_family`` is True.
    fontsize_scale :
        Multiplier for title, axis, rank, and base legend font size.
    legend_fontsize :
        Legend text size; default ``max(5, 9*fontsize_scale)``.
    tick_labelsize :
        Tick label size; default ``max(5, 0.78*fs_axis)`` with ``fs_axis=14*fontsize_scale``.
    legend_ncol :
        Columns in the family legend. Default: 1. Pass ``2`` (etc.) for a multi-column layout.
    clodd_options :
        If ``color_by_family`` and *family_labels* is None, optional kwargs for
        :func:`clodd_partition` (e.g. ``c_max``, ``gamma``, ``alpha``). Ignored if
        *family_labels* is set.
    family_labels :
        If set, skip internal clustering: pass the length-``N`` label vector for
        the ``N`` plotted elites. Must have length
        equal to the number of plotted elites.
    title_family_note :
        Unused; kept for backward compatibility (titles no longer append τ or family counts).
    """
    ranked = sorted(individuals, key=lambda x: float(x["fitness"]), reverse=True)
    top_elites = ranked[: int(top_n)]
    if not top_elites:
        logger.warning("No individuals to plot; skipping %s", output_path)
        return

    if show_rank_labels is None:
        show_rank_labels = not color_by_family

    num_individuals = len(top_elites)
    fig, ax = plt.subplots(figsize=figsize)
    offset_step = 0.2
    y_mins: list[float] = []
    y_maxs: list[float] = []

    n_families = 0
    if color_by_family:
        # Do not reassign the name ``family_labels`` to None here: that would shadow
        # the function parameter and ignore precomputed suite sweep labels.
        if family_labels is not None:
            family_labels = np.asarray(family_labels, dtype=int)
            if int(family_labels.shape[0]) != num_individuals:
                raise ValueError(
                    f"family_labels length {family_labels.shape[0]} != plotted elites {num_individuals}"
                )
            n_families = int(len(np.unique(family_labels)))
        else:
            chrom_stack = np.stack(
                [np.asarray(ind["chromosome"], dtype=float) for ind in top_elites]
            )
            raw_labels, _ = _chromosome_family_labels(chrom_stack, clodd_options=clodd_options)
            fits = [float(ind["fitness"]) for ind in top_elites]
            family_labels = _reorder_families_by_fitness(raw_labels, fits)
            n_families = int(len(np.unique(family_labels)))

    lw = 1.2 if figsize[0] <= 9.0 else 1.5

    for i, individual in enumerate(top_elites):
        chromosome = individual["chromosome"]
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = parameters_to_curves(gaussian_params, wavelengths)
        vertical_offset = (num_individuals - 1 - i) * offset_step
        if color_by_family and family_labels is not None:
            color = _FAMILY_COLORS[int(family_labels[i]) % len(_FAMILY_COLORS)]
            alpha = 0.88
        else:
            color = "red"
            alpha = 0.7

        for basis_func in basis_functions.T:
            scaled_basis = basis_func * 0.3
            stacked = scaled_basis + vertical_offset
            y_mins.append(float(np.min(stacked)))
            y_maxs.append(float(np.max(stacked)))
            ax.plot(
                wavelengths,
                stacked,
                color=color,
                alpha=alpha,
                linewidth=lw,
            )

    if y_mins and y_maxs:
        set_tight_ylim_stacked_spectra(ax, min(y_mins), max(y_maxs))
    else:
        set_tight_ylim_stacked_spectra(ax, 0.0, 1.0)

    fs_axis = 14 * fontsize_scale
    fs_title = 16 * fontsize_scale
    ax.set_xlabel("Wavelength (\u00b5m)", fontsize=fs_axis)
    ax.set_ylabel("Spectral Responsivity (dimensionless)", fontsize=fs_axis)
    _tick = tick_labelsize if tick_labelsize is not None else max(5.0, 0.78 * fs_axis)
    ax.tick_params(axis="x", which="major", labelsize=_tick)
    # Stacked curves use arbitrary vertical offsets; numeric y ticks are misleading.
    ax.set_yticks([])
    _title = f"{title_prefix}: Top {num_individuals}"
    ax.set_title(
        _title,
        fontsize=fs_title,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)

    if color_by_family and family_labels is not None and n_families > 0:
        fits = [float(ind["fitness"]) for ind in top_elites]
        legend_handles: list[plt.Line2D] = []
        legend_labels: list[str] = []
        # Labels may be non-consecutive when precomputed on a larger elite pool then sliced
        # (e.g. top-50 graph components; top-20 plot). Must iterate actual ids, not range(K).
        present = sorted(int(x) for x in np.unique(family_labels))
        for fam in present:
            idx = [j for j in range(num_individuals) if int(family_labels[j]) == fam]
            if not idx:
                continue
            top_fit = float(max(fits[j] for j in idx))
            c = _FAMILY_COLORS[fam % len(_FAMILY_COLORS)]
            legend_handles.append(plt.Line2D([0], [0], color=c, linewidth=2.5))
            legend_labels.append(f"score: {top_fit:.2f}")
        _leg_fs = (
            float(legend_fontsize)
            if legend_fontsize is not None
            else max(5.0, 9.0 * fontsize_scale)
        )
        _leg_base: dict[str, Any] = {
            "fontsize": _leg_fs,
            "framealpha": 0.92,
            "handlelength": 1.6,
        }
        _ncol = int(legend_ncol) if legend_ncol is not None else 1
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            ncol=_ncol,
            **_leg_base,
        )

    if show_rank_labels:
        trans_axes_x_data_y = blended_transform_factory(ax.transAxes, ax.transData)
        fs_rank = 8 * fontsize_scale
        for i, individual in enumerate(top_elites):
            y_base = (num_individuals - 1 - i) * offset_step
            mu_1, mu_2 = extract_features(individual["chromosome"])
            ax.text(
                0.01,
                y_base,
                f"Rank {i + 1}: F={individual['fitness']:.2f} "
                f"(\u03bc\u2081={mu_1:.1f}, \u03bc\u2082={mu_2:.1f})",
                transform=trans_axes_x_data_y,
                fontsize=fs_rank,
                verticalalignment="baseline",
                horizontalalignment="left",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
                clip_on=False,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved top individuals plot to %s", output_path)
    plt.close(fig)


def plot_top_from_population(
    population: np.ndarray,
    fitness: np.ndarray,
    wavelengths: np.ndarray,
    parameters_to_curves: Callable[..., np.ndarray],
    output_path: Path,
    top_n: int,
    title_prefix: str,
    **kwargs: Any,
) -> None:
    """Rank *population* by *fitness* and plot the top spectral curves."""
    pop = np.asarray(population, dtype=float)
    fit = np.asarray(fitness, dtype=float)
    individuals = [
        {"chromosome": np.asarray(pop[i], dtype=float), "fitness": float(fit[i])}
        for i in range(pop.shape[0])
    ]
    plot_top_individual_curves(
        individuals,
        wavelengths,
        parameters_to_curves,
        output_path,
        top_n,
        title_prefix,
        **kwargs,
    )


def plot_top_elites(
    archive: dict[tuple[int, int], dict[str, Any]],
    wavelengths: np.ndarray,
    parameters_to_curves: Callable[..., np.ndarray],
    output_path: Path,
    top_n: int = 10,
    title_prefix: str = "MAP-Elites Raw",
    **kwargs: Any,
) -> None:
    """Plot the top-*N* elite response curves from the archive.

    Parameters
    ----------
    archive : dict
        ``(x_bin, y_bin) -> individual dict``.
    wavelengths : np.ndarray
        Wavelength axis.
    parameters_to_curves : callable
        ``f(params, wavelengths) -> curves``.
    output_path : Path
        Where to save the figure.
    top_n : int
        How many elites to show.
    title_prefix : str
        Title label.
    """
    plot_top_individual_curves(
        list(archive.values()),
        wavelengths,
        parameters_to_curves,
        output_path,
        top_n,
        title_prefix,
        **kwargs,
    )


def plot_polished_elites(
    initial_elites: list[dict[str, Any]],
    polished_elites: list[dict[str, Any]],
    wavelengths: np.ndarray,
    parameters_to_curves: Callable[..., np.ndarray],
    output_path: Path,
    top_n: int = 20,
    title_prefix: str = "MAP-Elites Polished",
) -> None:
    """Plot polished top-*N* elites with before/after fitness labels.

    Parameters
    ----------
    initial_elites : list[dict]
        Pre-polish elite dicts, retained for caller compatibility. Labels use
        each polish record's own initial_fitness, independent of completion order.
    polished_elites : list[dict]
        Post-polish result dicts.
    wavelengths : np.ndarray
        Wavelength axis.
    parameters_to_curves : callable
        ``f(params, wavelengths) -> curves``.
    output_path : Path
        Where to save the figure.
    top_n : int
        How many to show.
    title_prefix : str
        Title label.
    """
    sorted_indices = np.argsort([e["polished_fitness"] for e in polished_elites])[::-1]
    sorted_polished = [polished_elites[i] for i in sorted_indices]

    top_n_actual = min(top_n, len(sorted_polished))
    sorted_polished = sorted_polished[:top_n_actual]

    num_individuals = len(sorted_polished)
    fig, ax = plt.subplots(figsize=(14, 10))
    offset_step = 0.2
    y_mins: list[float] = []
    y_maxs: list[float] = []

    for i, polished in enumerate(sorted_polished):
        chromosome = polished["polished_chromosome"]
        gaussian_params = [(chromosome[j], chromosome[j + 1]) for j in range(0, len(chromosome), 2)]
        basis_functions = parameters_to_curves(gaussian_params, wavelengths)
        vertical_offset = (num_individuals - 1 - i) * offset_step

        for basis_func in basis_functions.T:
            scaled_basis = basis_func * 0.3
            stacked = scaled_basis + vertical_offset
            y_mins.append(float(np.min(stacked)))
            y_maxs.append(float(np.max(stacked)))
            ax.plot(
                wavelengths,
                stacked,
                color="red",
                alpha=0.7,
                linewidth=1.5,
            )

    if y_mins and y_maxs:
        set_tight_ylim_stacked_spectra(ax, min(y_mins), max(y_maxs))
    else:
        set_tight_ylim_stacked_spectra(ax, 0.0, 1.0)
    ax.set_xlabel("Wavelength (\u00b5m)", fontsize=14)
    ax.set_ylabel("Spectral Responsivity (scaled, offset applied)", fontsize=14)
    ax.set_title(
        f"{title_prefix}: Top {top_n_actual} Refined Solutions",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    trans_axes_x_data_y = blended_transform_factory(ax.transAxes, ax.transData)
    for i, polished in enumerate(sorted_polished):
        y_base = (num_individuals - 1 - i) * offset_step
        ax.text(
            0.01,
            y_base,
            f"Rank {i + 1}: {polished['initial_fitness']:.2f} \u2192 {polished['polished_fitness']:.2f} "
            f"(+{polished['fitness_gain']:.2f})",
            transform=trans_axes_x_data_y,
            fontsize=8,
            verticalalignment="baseline",
            horizontalalignment="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
            clip_on=False,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved polished elites plot to %s", output_path)
    plt.close(fig)


def plot_cma_me_progress(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Plot CMA-ME convergence curves (coverage and best fitness).

    Parameters
    ----------
    history : dict[str, list[float]]
        Must contain keys ``"evals"``, ``"archive_size"``,
        ``"best_fitness"``, and ``"coverage_pct"`` as returned by
        :func:`~.cma_me.run_cma_me` metadata.
    output_path : Path
        Where to save the figure.
    """
    evals = history["evals"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(evals, history["coverage_pct"], color="steelblue", linewidth=2)
    ax1.set_ylabel("Reachable Coverage (%)", fontsize=12)
    ax1.set_title("CMA-ME Convergence", fontsize=16, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(evals, history["best_fitness"], color="firebrick", linewidth=2)
    ax2.set_xlabel("Fitness Evaluations", fontsize=12)
    ax2.set_ylabel("Best Fitness", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved CMA-ME progress plot to: {output_path}")
    plt.close()
