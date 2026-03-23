"""Visualisation utilities for environmental robustness analysis.

Provides publication-ready plots that answer the key Phase 1 question:
"How badly does fitness degrade when conditions deviate from nominal?"

All functions accept a list of :class:`RobustnessResult` objects produced by
:func:`~lwi_microbolometer_design.analysis.robustness.evaluate_archive_robustness`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lwi_microbolometer_design.analysis.robustness import (
    RobustnessResult,
)

_STYLE_DEFAULTS = {
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE_DEFAULTS)


def plot_fitness_degradation_heatmap(
    results: list[RobustnessResult],
    output_path: Path | None = None,
    figsize: tuple[float, float] = (14, 8),
    cmap: str = "RdYlGn",
) -> plt.Figure:
    """Heatmap of fitness values: elites (rows) x conditions (columns).

    Rows are sorted by nominal fitness (best at top).  Colour encodes
    absolute fitness so cool-spot regions of the map immediately reveal
    which condition-elite combinations are problematic.

    Parameters
    ----------
    results : list[RobustnessResult]
        Per-elite robustness results.
    output_path : Path | None
        If provided, save figure to this path.
    figsize : tuple[float, float]
        Figure size in inches.
    cmap : str
        Matplotlib colour map.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    _apply_style()

    sorted_results = sorted(results, key=lambda r: r.nominal_fitness, reverse=True)
    matrix = np.array([r.fitness_per_condition for r in sorted_results])

    condition_labels = [lbl.short_str() for lbl in sorted_results[0].condition_labels]
    elite_labels = [f"Elite {r.elite_id} (nom={r.nominal_fitness:.1f})" for r in sorted_results]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Fitness (min SAM angle, degrees)")

    ax.set_xticks(range(len(condition_labels)))
    ax.set_xticklabels(condition_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(elite_labels)))
    ax.set_yticklabels(elite_labels, fontsize=8)

    ax.set_xlabel("Environmental Condition")
    ax.set_ylabel("Elite (sorted by nominal fitness)")
    ax.set_title("Fitness Across Environmental Conditions")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_fitness_distribution_by_condition(
    results: list[RobustnessResult],
    output_path: Path | None = None,
    figsize: tuple[float, float] = (14, 7),
) -> plt.Figure:
    """Box-plot of fitness distributions grouped by environmental condition.

    Each box shows the spread of fitness values across all evaluated
    elites under one condition.  A narrow, high box means the condition
    is benign; a low, wide box means high sensitivity.

    Parameters
    ----------
    results : list[RobustnessResult]
        Per-elite robustness results.
    output_path : Path | None
        If provided, save figure to this path.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    _apply_style()

    matrix = np.array([r.fitness_per_condition for r in results])
    condition_labels = [lbl.short_str() for lbl in results[0].condition_labels]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        matrix,
        labels=condition_labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)

    ax.set_xlabel("Environmental Condition")
    ax.set_ylabel("Fitness (min SAM angle, degrees)")
    ax.set_title("Fitness Distribution per Environmental Condition")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_worst_case_vs_nominal(
    results: list[RobustnessResult],
    output_path: Path | None = None,
    figsize: tuple[float, float] = (9, 8),
) -> plt.Figure:
    """Scatter plot comparing each elite's nominal fitness to its worst-case.

    Points on the diagonal ``y = x`` are perfectly robust.  Points below
    the diagonal lost fitness; the vertical distance from the diagonal
    quantifies the robustness gap.

    Parameters
    ----------
    results : list[RobustnessResult]
        Per-elite robustness results.
    output_path : Path | None
        If provided, save figure to this path.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    _apply_style()

    nominals = np.array([r.nominal_fitness for r in results])
    worst = np.array([r.min_fitness for r in results])

    fig, ax = plt.subplots(figsize=figsize)

    lo = min(nominals.min(), worst.min()) * 0.95
    hi = max(nominals.max(), worst.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="Perfect robustness (y = x)")

    retention = np.array([r.retention_ratio for r in results])
    scatter = ax.scatter(
        nominals, worst, c=retention, cmap="RdYlGn", s=50, edgecolors="k", linewidths=0.5
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Retention Ratio (worst / nominal)")

    ax.set_xlabel("Nominal Fitness (degrees)")
    ax.set_ylabel("Worst-Case Fitness (degrees)")
    ax.set_title("Nominal vs. Worst-Case Fitness")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_condition_sensitivity(
    results: list[RobustnessResult],
    output_path: Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Bar chart showing mean fitness per condition, with error bars.

    Highlights which conditions cause the largest fitness drops
    relative to the overall mean.

    Parameters
    ----------
    results : list[RobustnessResult]
        Per-elite robustness results.
    output_path : Path | None
        If provided, save figure to this path.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    _apply_style()

    matrix = np.array([r.fitness_per_condition for r in results])
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    condition_labels = [lbl.short_str() for lbl in results[0].condition_labels]

    overall_mean = means.mean()
    colours = ["#2ca02c" if m >= overall_mean else "#d62728" for m in means]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(means))
    ax.bar(x, means, yerr=stds, color=colours, edgecolor="k", linewidth=0.5, capsize=3)
    ax.axhline(overall_mean, color="gray", linestyle="--", alpha=0.7, label="Overall mean")

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Environmental Condition")
    ax.set_ylabel("Mean Fitness (degrees)")
    ax.set_title("Condition Sensitivity Analysis")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_retention_histogram(
    results: list[RobustnessResult],
    output_path: Path | None = None,
    figsize: tuple[float, float] = (9, 6),
) -> plt.Figure:
    """Histogram of worst-case retention ratios across all elites.

    A retention ratio of 1.0 means the elite keeps 100% of its
    nominal fitness; 0.5 means the worst condition halves its fitness.

    Parameters
    ----------
    results : list[RobustnessResult]
        Per-elite robustness results.
    output_path : Path | None
        If provided, save figure to this path.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    _apply_style()

    retentions = np.array([r.retention_ratio for r in results])

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(retentions, bins=25, color="#4C72B0", edgecolor="k", linewidth=0.5, alpha=0.8)
    ax.axvline(
        retentions.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {retentions.mean():.2%}",
    )

    ax.set_xlabel("Worst-Case Retention Ratio")
    ax.set_ylabel("Number of Elites")
    ax.set_title("Distribution of Worst-Case Fitness Retention")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig
